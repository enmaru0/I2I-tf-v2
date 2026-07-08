import argparse
from pathlib import Path

import commonlib
import keras
import numpy as np
import tensorflow as tf
from absl import logging
from irg import save_raw
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataloader_test import create_dataloader_test
from main import gpu_setting, prepare_data_dict
from trainers import BaseI2ITrainer, build_trainer

tf.config.run_functions_eagerly(True)


def load_checkpoint(checkpoint_path) -> tuple[BaseI2ITrainer, int]:
    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    step = model.optimizer.iterations.numpy()
    return model, step


def rescale_pred_to_org(pred, out_size_zyx):
    """正規化スペーシングの予測画像を元画像のクロップ領域サイズに戻す"""
    transform = commonlib.RescaleTransform3DWithFilter(
        commonlib.ImageFilter.LINEAR,
        commonlib.ImageFilter.LINEAR,
        commonlib.ImageFilter.LINEAR,
        commonlib.ImageFilter.DEFAULT_OUT(np.float32),
        commonlib.ImageFilter.DEFAULT_IN,
        commonlib.ImageFilter.ZERO_OVER,
    )
    transform.SetOrgImageSize(*pred.shape[::-1])
    transform.SetResultImageSize(*out_size_zyx[::-1])
    transform.SetOrgImage(pred)
    return transform.Transform()


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

    # データを準備
    val_dict = prepare_data_dict(cfg)[1]
    test_loader = create_dataloader_test(val_dict, cfg)

    # モデルを読み込む
    _model, step = load_checkpoint(checkpoint_path)
    logging.info(f"Loaded from: {checkpoint_path} (step: {step})")

    # テスト時は入力サイズが可変なので可変に対応したモデルを作成
    model: BaseI2ITrainer = build_trainer(cfg, (None, None, None, 1))
    # 読み込んだコピーをセットする
    model.set_weights(_model.get_weights())

    for data in tqdm(test_loader):
        # batch dimを追加（targetは推論に不要なので渡さない）
        _data = dict(
            src_imgs=data["img"][None],
            img_msks=data["img_msk"][None],
            src_min_clip_vals=data["src_min_clip_val"][None],
            src_max_clip_vals=data["src_max_clip_val"][None],
        )
        pred = model.predict_step(_data)  # (1, z, y, x, 1) 値域は[0,1]
        pred = pred.numpy()[0, :, :, :, 0]

        key = data["img_key"].numpy().decode()
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
