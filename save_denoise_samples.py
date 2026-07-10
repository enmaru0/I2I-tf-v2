"""
学習済みモデルで検証パッチをデノイズし、
noisy入力 / denoise結果 / clean正解 をraw画像として保存する。
パッチ単位の推論結果を保存する目視確認用スクリプト。

使い方:
    python save_denoise_samples.py results/mr_denoise_reg/checkpoints/model_best.keras \
        --num_batches 2
"""

import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from irg import save_raw
from omegaconf import OmegaConf

from data.dataloader import create_dataloader
from main import gpu_setting, prepare_data_dict
from trainers import attach_aux_optimizers

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--num_batches", default=2, type=int)
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path
    cfg = OmegaConf.load(checkpoint_path.parents[1] / "output.yaml")
    gpu_setting(args.gpu, cfg.gpu_allow_growth)
    tf.config.run_functions_eagerly(True)

    save_dir = checkpoint_path.parents[1] / "denoise_samples"
    save_dir.mkdir(exist_ok=True)
    spacing = np.array(cfg.aug.affine.norm_spacing_zyx, np.float32)

    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    model.cfg = cfg
    attach_aux_optimizers(model, cfg)

    val_dict = prepare_data_dict(cfg)[1]
    val_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    def to_u16(img01):
        # [0,1] -> 0-4095に伸ばして保存（相対的な見やすさ優先）
        return (np.clip(img01, 0, 1) * 4095).astype(np.int16)

    n = 0
    for bi, batch in enumerate(val_loader):
        if bi >= args.num_batches:
            break
        _, preds, src_imgs, tgt_imgs, img_msks = model.predict_step(
            batch, return_aux=True
        )
        preds = preds.numpy()
        src_imgs = src_imgs.numpy()
        tgt_imgs = tgt_imgs.numpy()
        for i in range(preds.shape[0]):
            save_raw(to_u16(src_imgs[i, ..., 0]), spacing, save_dir / f"s{n}_noisy.hdr")
            save_raw(to_u16(preds[i, ..., 0]), spacing, save_dir / f"s{n}_denoised.hdr")
            save_raw(to_u16(tgt_imgs[i, ..., 0]), spacing, save_dir / f"s{n}_clean.hdr")
            print(f"saved sample {n}")
            n += 1
    print(f"saved {n} samples to {save_dir}")
