import argparse
import zlib
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_raw, save_raw
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader_test import create_dataloader_test
from inference_utils import (
    fuse_native_xy_residual,
    resample_prediction_to_dense_z,
    resize_volume_to_shape,
    sliding_window_predict,
)
from main import gpu_setting, prepare_data_dict
from trainers import BaseI2ITrainer, build_trainer

tf.config.run_functions_eagerly(True)


def load_checkpoint(checkpoint_path) -> tuple[BaseI2ITrainer, int]:
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    step = model.optimizer.iterations.numpy()
    return model, step


def rescale_pred_to_org(pred, out_size_zyx):
    """正規化スペーシングの予測画像を元画像のクロップ領域サイズに戻す"""
    return resize_volume_to_shape(pred, out_size_zyx, order=1, anti_alias=True)


def _native_xy_residual_settings(cfg, force_enabled=False, target_z_override=None):
    cfg_native = cfg.get("inference", {}).get("native_xy_residual", {})
    enabled = bool(cfg_native.get("enabled", False)) or bool(force_enabled)
    target_z = cfg_native.get("target_z_spacing_mm", None)
    if target_z_override is not None:
        target_z = target_z_override
    if target_z is None:
        target_z = float(cfg.aug.affine.norm_spacing_zyx[0])
    return enabled, float(target_z), cfg_native


def _output_grid_settings(cfg, output_grid_override=None, target_z_override=None):
    """通常予測の保存gridを解決する。self_srは既定でdense_zを使用する。"""
    cfg_infer = cfg.get("inference", {})
    output_grid = (
        output_grid_override
        if output_grid_override is not None
        else cfg_infer.get("output_grid", "auto")
    )
    output_grid = str(output_grid).lower().replace("-", "_")
    if output_grid == "auto":
        output_grid = "dense_z" if str(cfg.data.mode) == "self_sr" else "original"
    if output_grid not in ("original", "dense_z"):
        raise ValueError(
            "inference.output_gridはauto/original/dense_zを指定してください: "
            f"{output_grid}"
        )

    cfg_dense = cfg_infer.get("dense_z", {})
    target_z = cfg_dense.get("target_z_spacing_mm", None)
    if target_z_override is not None:
        target_z = target_z_override
    if target_z is None:
        target_z = float(cfg.aug.affine.norm_spacing_zyx[0])
    target_z = float(target_z)
    if not np.isfinite(target_z) or target_z <= 0.0:
        raise ValueError(f"target_z_spacing_mmは正にしてください: {target_z}")
    return output_grid, target_z, cfg_dense


def _validate_native_xy_residual_mode(cfg, target_z_spacing_mm):
    algorithm_name = str(cfg.algorithm.name)
    algorithm_cfg = cfg.algorithm.get(algorithm_name, {})
    output_mode = str(algorithm_cfg.get("output_mode", ""))
    if output_mode != "residual":
        raise ValueError(
            "native XY residual推論はoutput_mode=residualのmodel専用です: "
            f"algorithm={algorithm_name}, output_mode={output_mode or '(なし)'}"
        )
    if not bool(cfg.image.share_normalization):
        raise ValueError(
            "native XY residual推論ではsource/targetの線形強度差を一致させるため"
            "image.share_normalization=Trueが必要です"
        )
    if not np.isfinite(target_z_spacing_mm) or target_z_spacing_mm <= 0.0:
        raise ValueError(
            f"target_z_spacing_mmは正にしてください: {target_z_spacing_mm}"
        )


def _to_int16(volume):
    return np.clip(np.rint(volume), -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str, help="gpu num (default 0)")
    parser.add_argument(
        "--native-xy-residual",
        action="store_true",
        help="1 mm残差だけをnative XYへ加えるthrough-plane SR推論を有効化",
    )
    parser.add_argument(
        "--target-z-spacing-mm",
        type=float,
        default=None,
        help="dense-z/native XY residual出力のz spacing。省略時は学習時norm spacing z",
    )
    parser.add_argument(
        "--output-grid",
        choices=("auto", "original", "dense-z"),
        default=None,
        help="通常予測の保存grid。autoではself_srのみdense-z、それ以外はoriginal",
    )
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path

    # 実験時の設定ファイルを読み込む
    cfg_path = checkpoint_path.parents[1] / "output.yaml"
    cfg = OmegaConf.load(cfg_path)

    # 保存場所を作成
    save_dir = checkpoint_path.parents[1] / "preds"
    save_dir.mkdir(exist_ok=True)

    # テスト時に使うGPUを設定
    gpu_setting(args.gpu, cfg.gpu_allow_growth)
    policy_name = str(cfg.get("mixed_precision_policy", "float32"))
    keras.mixed_precision.set_global_policy(policy_name)

    # データを準備
    val_dict = prepare_data_dict(cfg)[1]
    test_loader = create_dataloader_test(val_dict, cfg)

    # モデルを読み込む
    _model, step = load_checkpoint(checkpoint_path)
    logging.info(f"Loaded from: {checkpoint_path} (step: {step})")

    cfg_infer = cfg.get("inference", {})
    native_xy_enabled, target_z_spacing_mm, cfg_native_xy = (
        _native_xy_residual_settings(
            cfg,
            force_enabled=args.native_xy_residual,
            target_z_override=args.target_z_spacing_mm,
        )
    )
    output_grid, dense_z_spacing_mm, cfg_dense_z = _output_grid_settings(
        cfg,
        output_grid_override=args.output_grid,
        target_z_override=args.target_z_spacing_mm,
    )
    if native_xy_enabled:
        _validate_native_xy_residual_mode(cfg, target_z_spacing_mm)
        logging.info(
            "Native XY residual inference enabled: target_z_spacing_mm=%.4f",
            target_z_spacing_mm,
        )
    elif output_grid == "dense_z":
        logging.info(
            "Dense-z self-SR output enabled: target_z_spacing_mm=%.4f",
            dense_z_spacing_mm,
        )
    use_sliding_window = bool(cfg_infer.get("sliding_window", True))
    configured_patch = cfg_infer.get("patch_size_zyx", None)
    patch_size_zyx = (
        tuple(cfg.aug.crop_size_zyx)
        if configured_patch is None
        else tuple(configured_patch)
    )
    model_shape = patch_size_zyx + (1,) if use_sliding_window else (None, None, None, 1)
    model: BaseI2ITrainer = build_trainer(cfg, model_shape)
    # 読み込んだコピーをセットする
    model.set_weights(_model.get_weights())

    for data in tqdm(test_loader):
        key = data["img_key"].numpy().decode()
        cfg_repro = cfg.get("reproducibility", {})
        base_seed = int(cfg_repro.get("seed", 0))
        case_seed = (base_seed + zlib.crc32(key.encode())) & 0x7FFFFFFF
        if use_sliding_window:
            pred = sliding_window_predict(
                model,
                data,
                patch_size_zyx,
                overlap=float(cfg_infer.get("overlap", 0.5)),
                batch_size=int(cfg_infer.get("batch_size", 1)),
                base_seed=case_seed,
            )
        else:
            # batch dimを追加（targetは推論に不要なので渡さない）
            _data = dict(
                src_imgs=data["img"][None],
                img_msks=data["img_msk"][None],
                src_min_clip_vals=data["src_min_clip_val"][None],
                src_max_clip_vals=data["src_max_clip_val"][None],
                sample_seeds=tf.constant([[base_seed, case_seed]], tf.int32),
            )
            pred = model.predict_step(_data)
            pred = pred.numpy()[0, :, :, :, 0]

        img_size_zyx = data["img_size_zyx"].numpy()
        spacing_zyx = data["spacing_zyx"].numpy()
        crop_zyxzyx = data["crop_zyxzyx"].numpy()
        tgt_min_val = data["tgt_min_clip_val"].numpy()
        tgt_max_val = data["tgt_max_clip_val"].numpy()

        if native_xy_enabled:
            native_path = Path(data["img_path"].numpy().decode())
            native_img = read_raw(native_path).astype(np.float32)
            out, output_spacing, _ = fuse_native_xy_residual(
                native_img,
                spacing_zyx,
                data["img"].numpy(),
                pred,
                float(data["src_min_clip_val"].numpy()),
                float(data["src_max_clip_val"].numpy()),
                target_z_spacing_mm,
                base_interpolation_order=int(
                    cfg_native_xy.get("base_interpolation_order", 1)
                ),
                residual_interpolation_order=int(
                    cfg_native_xy.get("residual_interpolation_order", 1)
                ),
            )
            suffix = str(
                cfg_native_xy.get("filename_suffix", ".native_xy_residual.pred")
            )
            save_path = save_dir / f"{key}{suffix}.hdr"
            save_raw(_to_int16(out), output_spacing, save_path)
        else:
            # [0,1] -> targetの強度レンジに戻す
            pred = pred * (tgt_max_val - tgt_min_val) + tgt_min_val

            if output_grid == "dense_z":
                out, output_spacing = resample_prediction_to_dense_z(
                    pred,
                    img_size_zyx,
                    spacing_zyx,
                    dense_z_spacing_mm,
                    interpolation_order=int(
                        cfg_dense_z.get("interpolation_order", 1)
                    ),
                    never_downsample=bool(
                        cfg_dense_z.get("never_downsample", True)
                    ),
                )
                suffix = str(cfg_dense_z.get("filename_suffix", ".sr.pred"))
                save_path = save_dir / f"{key}{suffix}.hdr"
                save_raw(_to_int16(out), output_spacing, save_path)
            else:
                # 従来動作：結果を元画像の粗いgridへ戻す
                out_size_zyx = crop_zyxzyx[3:6] - crop_zyxzyx[:3]
                pred = rescale_pred_to_org(pred, out_size_zyx)

                out = np.full(list(img_size_zyx), tgt_min_val, np.float32)
                out[
                    crop_zyxzyx[0] : crop_zyxzyx[3],
                    crop_zyxzyx[1] : crop_zyxzyx[4],
                    crop_zyxzyx[2] : crop_zyxzyx[5],
                ] = pred

                save_path = save_dir / f"{key}.pred.hdr"
                save_raw(_to_int16(out), spacing_zyx, save_path)
