import argparse
import zlib
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from irg import save_raw
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader_test import create_dataloader_test
from inference_utils import resize_volume_to_shape, sliding_window_predict
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


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str, help="gpu num (default 0)")
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

        # [0,1] -> targetの強度レンジに戻す
        pred = pred * (tgt_max_val - tgt_min_val) + tgt_min_val

        # 結果を元画像空間に戻す
        out_size_zyx = crop_zyxzyx[3:6] - crop_zyxzyx[:3]
        pred = rescale_pred_to_org(pred, out_size_zyx)

        out = np.full(list(img_size_zyx), tgt_min_val, np.float32)
        out[
            crop_zyxzyx[0] : crop_zyxzyx[3],
            crop_zyxzyx[1] : crop_zyxzyx[4],
            crop_zyxzyx[2] : crop_zyxzyx[5],
        ] = pred

        save_path = save_dir / f"{key}.pred.hdr"
        save_raw(out.astype(np.int16), spacing_zyx, save_path)
