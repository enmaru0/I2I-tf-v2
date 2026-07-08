import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from omegaconf import OmegaConf

from data.dataloader import create_dataloader
from losses import masked_l1_loss, masked_psnr, ssim
from main import gpu_setting, prepare_data_dict
from trainers import attach_aux_optimizers

# サンプリングステップ数(num_steps)を持つ生成系アルゴリズム
GENERATIVE = {"edm", "rectified_flow", "i2i_rfr"}


def evaluate_once(model, val_loader, max_batches: int, seed: int):
    """検証データ全体（最大max_batches）でPSNR/SSIM/MAEを集計する"""
    tf.random.set_seed(seed)  # 生成系のサンプリング初期ノイズを揃えて比較可能にする
    psnr_list, ssim_list, mae_list = [], [], []
    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        # predict_stepは[0,1]・マスク済みのpreds/tgtを返す
        _, preds, _, tgt_imgs, img_msks = model.predict_step(batch, return_aux=True)
        psnr_list.append(float(masked_psnr(tgt_imgs, preds, img_msks)))
        ssim_list.append(float(ssim(tgt_imgs, preds)))
        mae_list.append(float(masked_l1_loss(tgt_imgs, preds, img_msks)))
    return (
        float(np.mean(psnr_list)),
        float(np.mean(ssim_list)),
        float(np.mean(mae_list)),
        len(psnr_list),
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser(
        description="検証パッチでPSNR/SSIM/MAEを計測する。"
        "生成系は--num_stepsでサンプリングステップ数をスイープできる。"
    )
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--num_steps",
        default="",
        type=str,
        help="生成系のサンプリングステップ数をカンマ区切りで指定 (例: 1,2,4,8,16)。"
        "省略時はcfgの値を使う",
    )
    parser.add_argument(
        "--max_batches",
        default=0,
        type=int,
        help="評価に使うバッチ数の上限 (0で全バッチ)",
    )
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    checkpoint_path: Path = args.checkpoint_path
    cfg = OmegaConf.load(checkpoint_path.parents[1] / "output.yaml")
    gpu_setting(args.gpu, cfg.gpu_allow_growth)

    # 推論を安定させるためeagerで実行する（サンプリングのpythonループ対策）
    tf.config.run_functions_eagerly(True)

    model = keras.models.load_model(checkpoint_path, safe_mode=False)
    model.cfg = cfg
    attach_aux_optimizers(model, cfg)
    name = cfg.algorithm.name
    step = int(model.optimizer.iterations.numpy())
    logging.info(f"Loaded {name} from {checkpoint_path} (step: {step})")

    # 検証用データローダー（学習と同じ固定サイズパッチ。commonlib不要）
    val_dict = prepare_data_dict(cfg)[1]
    val_loader = create_dataloader(val_dict, is_training=False, cfg=cfg)

    # スイープするnum_stepsのリストを決める
    if args.num_steps and name in GENERATIVE:
        steps_list = [int(s) for s in args.num_steps.split(",")]
    elif name in GENERATIVE:
        steps_list = [int(cfg.algorithm[name].num_steps)]
    else:
        steps_list = [None]  # 生成系でない場合はnum_stepsは無関係

    print(f"\n=== Evaluation: {name} (step {step}) ===")
    header = f"{'num_steps':>10} | {'PSNR':>7} | {'SSIM':>7} | {'MAE':>7} | {'#batch':>6}"
    print(header)
    print("-" * len(header))
    results = []
    for ns in steps_list:
        if ns is not None:
            cfg.algorithm[name].num_steps = ns
        psnr, ssim_v, mae, n = evaluate_once(
            model, val_loader, args.max_batches, args.seed
        )
        results.append((ns, psnr, ssim_v, mae, n))
        label = "-" if ns is None else str(ns)
        print(f"{label:>10} | {psnr:>7.3f} | {ssim_v:>7.4f} | {mae:>7.4f} | {n:>6}")

    # 最良のnum_steps（PSNR基準）を表示
    if len(results) > 1:
        best = max(results, key=lambda r: r[1])
        print(f"\nbest num_steps (by PSNR): {best[0]} (PSNR={best[1]:.3f})")
