"""
知覚指標での手法比較。PSNR/SSIMに加えて以下を計算する:
- LPIPS: 学習済みAlexNet特徴空間での距離。低いほど知覚的に自然（人間の知覚と相関）。
         PSNR/SSIMと違い、ぼけた画像を高く評価しにくい。3Dは軸方向スライスごとに計算し平均。
- sharpness_ratio: 出力の勾配エネルギー / 正解の勾配エネルギー。
                   1に近いほど正解と同程度の鮮鋭さ、<1はぼけ、>1は過剰なノイズ/エッジ。

使い方（複数チェックポイントを並べて比較。生成系はpath@num_stepsでステップ指定可）:
    python eval_perceptual.py \
        results/deg_reg/checkpoints/model_best.keras \
        results/deg_rfr/checkpoints/model_best.keras@2
"""

import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from omegaconf import OmegaConf

from data.dataloader import create_dataloader
from losses import masked_psnr, ssim
from main import gpu_setting, prepare_data_dict
from trainers import attach_aux_optimizers

GENERATIVE = {"edm", "rectified_flow", "i2i_rfr", "i2i_rfr_x0", "resshift", "split_mean_flow"}


def sharpness(vol_zyx, msk_zyx):
    """マスク内の平均勾配強度（鮮鋭さの指標）"""
    gz, gy, gx = np.gradient(vol_zyx.astype(np.float32))
    gmag = np.sqrt(gz**2 + gy**2 + gx**2)
    m = msk_zyx > 0.5
    return float(gmag[m].mean()) if m.any() else 0.0


def collect_slices(vol_bzyxc, msk_bzyxc, min_cov=0.1):
    """(b,z,y,x,1)からマスク被覆が十分な軸(z)スライスを(H,W)で取り出す"""
    slices = []
    for i in range(vol_bzyxc.shape[0]):
        for z in range(vol_bzyxc.shape[1]):
            if msk_bzyxc[i, z, :, :, 0].mean() >= min_cov:
                slices.append(vol_bzyxc[i, z, :, :, 0])
    return slices


def to_lpips_batch(slices, torch, device):
    """[0,1]の(H,W)スライス列を LPIPS入力 (N,3,H,W) の[-1,1]テンソルに変換"""
    arr = np.stack(slices, axis=0)[:, None, :, :]  # (N,1,H,W)
    arr = np.repeat(arr, 3, axis=1)  # グレースケール->3ch
    arr = np.clip(arr, 0, 1) * 2.0 - 1.0
    return torch.from_numpy(arr.astype(np.float32)).to(device)


if __name__ == "__main__":
    logging.set_verbosity(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoints", nargs="+", help="model_best.keras[@num_steps] を並べる"
    )
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--max_batches", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    first_cfg = OmegaConf.load(
        Path(args.checkpoints[0].split("@")[0]).parents[1] / "output.yaml"
    )
    gpu_setting(args.gpu, first_cfg.gpu_allow_growth)
    tf.config.run_functions_eagerly(True)

    import lpips
    import torch

    # LPIPSはCPUで実行（TFがGPUメモリを確保するため競合回避）
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to("cpu")

    print(
        f"\n{'method':>16} | {'PSNR':>6} | {'SSIM':>6} | "
        f"{'LPIPS_down':>10} | {'sharp_ratio':>11}"
    )
    print("-" * 64)

    for entry in args.checkpoints:
        ckpt_str, _, ns = entry.partition("@")
        ckpt = Path(ckpt_str)
        cfg = OmegaConf.load(ckpt.parents[1] / "output.yaml")
        name = cfg.algorithm.name
        model = keras.models.load_model(ckpt, safe_mode=False)
        model.cfg = cfg
        attach_aux_optimizers(model, cfg)
        if ns and name in GENERATIVE:
            cfg.algorithm[name].num_steps = int(ns)

        val_dict = prepare_data_dict(cfg)[1]
        val_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

        tf.random.set_seed(args.seed)
        psnrs, ssims, lpips_vals, sharp_ratios = [], [], [], []
        for bi, batch in enumerate(val_loader):
            if args.max_batches and bi >= args.max_batches:
                break
            _, preds, _, tgt, msk = model.predict_step(batch, return_aux=True)
            preds_np, tgt_np, msk_np = preds.numpy(), tgt.numpy(), msk.numpy()

            psnrs.append(float(masked_psnr(tgt, preds, msk)))
            ssims.append(float(ssim(tgt, preds)))

            for i in range(preds_np.shape[0]):
                st = sharpness(tgt_np[i, ..., 0], msk_np[i, ..., 0])
                sp = sharpness(preds_np[i, ..., 0], msk_np[i, ..., 0])
                if st > 1e-6:
                    sharp_ratios.append(sp / st)

            ps = collect_slices(preds_np, msk_np)
            ts = collect_slices(tgt_np, msk_np)
            if ps:
                with torch.no_grad():
                    d = lpips_model(
                        to_lpips_batch(ps, torch, "cpu"),
                        to_lpips_batch(ts, torch, "cpu"),
                    )
                lpips_vals.append(float(d.mean()))

        label = name if not ns else f"{name}@{ns}"
        print(
            f"{label:>16} | {np.mean(psnrs):>6.2f} | {np.mean(ssims):>6.3f} | "
            f"{np.mean(lpips_vals):>10.4f} | {np.mean(sharp_ratios):>11.3f}"
        )
