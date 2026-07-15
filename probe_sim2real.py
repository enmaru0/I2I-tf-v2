"""
シミュレーション劣化と実劣化の分布ギャップを判別器プローブで定量化する。

- simパッチ: 学習データ(クリーン画像)に学習時と同一のランダム劣化
  (data.modeに応じてself_sr / self_noise)を適用して作る
- realパッチ: 実劣化画像フォルダ(--real_dir)からサンプリングする
両者を同一の決定的な前処理（回転なし・ランダム中心クロップ・
norm_spacing_zyxへのリサンプル・clip値正規化）に通し、
PatchGAN判別器で sim vs real の2値分類を学習する。

ホールドアウト症例での分類精度が
  ~50%: 分布ギャップほぼなし（判別不能）
  60-75%: 軽微なギャップ（校正やランダム化拡大を検討）
  85%以上: 明確なギャップ（劣化生成の学習や実データDC損失を検討）
の目安になる。--examples_dir を指定するとスコア付きの
サンプルスライスPNGを保存する（何がずれているかの目視診断用）。

使い方:
  python probe_sim2real.py --real_dir /path/to/real_degraded \
      --overrides data.mode=self_sr
"""

import argparse
import random
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from absl import logging
from omegaconf import OmegaConf

from data.dataloader import apply_random_self_noise, apply_random_self_sr, get_clip_vals
from data.dataloader_utils import save_intensity
from data.utils import (
    AffineTransform,
    calc_img_crop_region,
    random_crop_center_within_bb,
)
from irg import read_hdr, read_raw
from main import gpu_setting, prepare_data_dict
from models import build_patch_discriminator


def collect_hdr_list(directory: Path, max_volumes: int) -> list[Path]:
    hdr_list = [p for p in sorted(directory.glob("*.hdr")) if ".mask" not in p.name]
    if max_volumes > 0:
        hdr_list = hdr_list[:max_volumes]
    assert len(hdr_list) > 0, f"画像(.hdr)が見つかりません: {directory}"
    return hdr_list


def extract_patches(hdr_path: Path, cfg, patch_size_zyx, n_patches, degrade_fn=None):
    """
    1ボリュームからランダム中心のパッチをn_patches枚取り出す。
    回転・反転・スケールなしの決定的アフィン（norm_spacingへのリサンプルのみ）。
    degrade_fnが与えられた場合はリサンプル後のパッチに劣化を適用する
    （学習時と同じ「アフィン後に劣化」の順序）。
    戻り値: [0,1]正規化・マスク済みの (n, z, y, x, 1) float32
    """
    patch_size_zyx = np.array(patch_size_zyx)
    img_size_zyx, src_dtype, spacing_zyx = read_hdr(hdr_path)
    min_val, max_val = get_clip_vals(hdr_path, cfg)
    intensity_range = float(max_val - min_val)

    patches = []
    box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    for _ in range(n_patches):
        crop_center_zyx = random_crop_center_within_bb(
            box_zyxzyx,
            img_size_zyx,
            patch_size_zyx,
            spacing_zyx,
            cfg.aug.affine.norm_spacing_zyx,
            [0, 0, 0],
        )
        affine_transform = AffineTransform(
            crop_size_zyx=patch_size_zyx, **cfg.aug.affine
        )
        affine_matrix = affine_transform.get_affine(
            spacing_zyx, crop_center_zyx, is_training=False
        )
        img_region_zyxzyx, shift_start = calc_img_crop_region(
            patch_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
        )
        affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

        img = read_raw(
            hdr_path,
            clip_zyxzyx=img_region_zyxzyx,
            img_dtype=src_dtype,
            size_zyx=img_size_zyx,
            use_memmap=True,
        )
        ones = np.ones(img.shape, np.uint8)
        msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)
        img = affine_transform.apply(img, affine_matrix, order=1).astype(np.float32)

        if degrade_fn is not None:
            img = degrade_fn(img, intensity_range)

        img01 = np.clip((img - min_val) / (intensity_range + 1e-6), 0.0, 1.0)
        img01 = (img01 * msk).astype(np.float32)
        patches.append(img01[..., None])
    return np.stack(patches)


def build_sim_degrade_fn(cfg):
    """学習時と同一のランダム劣化関数を返す"""
    mode = cfg.data.mode
    if mode == "self_sr":

        def degrade(img, intensity_range):
            return apply_random_self_sr(
                img, cfg.aug.affine.norm_spacing_zyx, cfg.data.self_sr
            )

    elif mode in ("denoise", "self_noise"):

        def degrade(img, intensity_range):
            return apply_random_self_noise(
                img,
                intensity_range,
                cfg.data.self_noise,
                norm_spacing_zyx=cfg.aug.affine.norm_spacing_zyx,
            )

    else:
        raise ValueError(
            f"probeはdata.mode=self_sr/denoise/self_noiseで使ってください（現在: {mode}）"
        )
    return degrade


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """rank統計によるAUC（sklearn不要）"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_example_png(patch, score, out_path: Path):
    """パッチの中央zスライスをスコア付きファイル名でPNG保存する"""
    slc = patch[patch.shape[0] // 2, :, :, 0]
    img = np.clip(slc * 255.0, 0, 255).astype(np.uint8)[..., None]
    png = tf.io.encode_png(tf.constant(img))
    tf.io.write_file(str(out_path), png)


def main():
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser(description="sim/real劣化分布の判別器プローブ")
    parser.add_argument(
        "--real_dir", type=Path, required=True, help="実劣化画像(.hdr/.raw)のフォルダ"
    )
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--max_volumes", default=32, type=int, help="各サイドで使う最大症例数"
    )
    parser.add_argument("--patches_per_volume", default=8, type=int)
    parser.add_argument(
        "--patch_size", nargs=3, type=int, default=[64, 64, 64], metavar=("Z", "Y", "X")
    )
    parser.add_argument(
        "--steps", default=1000, type=int, help="判別器の学習ステップ数"
    )
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--base_ch", default=16, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument(
        "--holdout_ratio",
        default=0.25,
        type=float,
        help="評価用にホールドアウトする症例の割合",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--examples_dir",
        type=Path,
        default=None,
        help="指定すると評価パッチのスコア付きPNGを保存する",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    cfg.data_dir = Path(cfg.data_dir)
    gpu_setting(args.gpu, cfg.gpu_allow_growth)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    degrade_fn = build_sim_degrade_fn(cfg)

    # sim側: 学習データのクリーン画像（prepare_data_dictと同じ列挙）
    train_dict, _ = prepare_data_dict(cfg)
    sim_hdr_list = []
    for value in train_dict.values():
        sim_hdr_list += list(value["img_hdr_list"])
    sim_hdr_list = sorted(sim_hdr_list)[: args.max_volumes]
    real_hdr_list = collect_hdr_list(args.real_dir, args.max_volumes)
    logging.info(
        f"sim volumes: {len(sim_hdr_list)}, real volumes: {len(real_hdr_list)}"
    )

    # MRは正規化用のintensityファイルを事前計算する
    if cfg.image.modality == "MR":
        for hdr_path in sim_hdr_list + real_hdr_list:
            save_intensity(
                str(hdr_path),
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )

    # 症例単位でtrain/evalに分割（パッチ単位のリークを防ぐ）
    def split_volumes(hdr_list):
        hdr_list = list(hdr_list)
        random.shuffle(hdr_list)
        n_eval = max(1, int(len(hdr_list) * args.holdout_ratio))
        return hdr_list[n_eval:], hdr_list[:n_eval]

    sim_train, sim_eval = split_volumes(sim_hdr_list)
    real_train, real_eval = split_volumes(real_hdr_list)

    def build_side(hdr_list, degrade):
        chunks = []
        for hdr_path in hdr_list:
            chunks.append(
                extract_patches(
                    hdr_path,
                    cfg,
                    args.patch_size,
                    args.patches_per_volume,
                    degrade_fn=degrade,
                )
            )
            logging.info(f"  patches from {hdr_path.name}")
        return np.concatenate(chunks)

    logging.info("extracting sim(train) patches...")
    x_sim_tr = build_side(sim_train, degrade_fn)
    logging.info("extracting real(train) patches...")
    x_real_tr = build_side(real_train, None)
    logging.info("extracting sim(eval) patches...")
    x_sim_ev = build_side(sim_eval, degrade_fn)
    logging.info("extracting real(eval) patches...")
    x_real_ev = build_side(real_eval, None)

    x_train = np.concatenate([x_sim_tr, x_real_tr])
    y_train = np.concatenate(
        [np.zeros(len(x_sim_tr), np.float32), np.ones(len(x_real_tr), np.float32)]
    )
    x_eval = np.concatenate([x_sim_ev, x_real_ev])
    y_eval = np.concatenate(
        [np.zeros(len(x_sim_ev), np.float32), np.ones(len(x_real_ev), np.float32)]
    )
    logging.info(f"train patches: {len(x_train)}, eval patches: {len(x_eval)}")

    # PatchGAN判別器（pix2pixのものを流用。2入力は同じパッチを渡す）
    input_shape = tuple(args.patch_size) + (1,)
    disc = build_patch_discriminator(
        input_shape, base_ch=args.base_ch, num_layers=args.num_layers
    )
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=args.seed)
        .repeat()
        .batch(args.batch_size)
        .prefetch(2)
    )

    @tf.function
    def train_step(x, label):
        with tf.GradientTape() as tape:
            logits = disc([x, x], training=True)
            label_map = tf.reshape(label, (-1, 1, 1, 1, 1)) * tf.ones_like(logits)
            loss = bce(label_map, logits)
        grads = tape.gradient(loss, disc.trainable_variables)
        optimizer.apply_gradients(zip(grads, disc.trainable_variables))
        return loss

    @tf.function
    def score_step(x):
        logits = disc([x, x], training=False)
        return tf.sigmoid(tf.reduce_mean(logits, axis=(1, 2, 3, 4)))

    data_iter = iter(dataset)
    for step in range(1, args.steps + 1):
        x, label = next(data_iter)
        loss = train_step(x, label)
        if step % 100 == 0 or step == args.steps:
            logging.info(f"step {step}/{args.steps} loss={float(loss):.4f}")

    # ホールドアウト症例で評価
    scores = []
    for i in range(0, len(x_eval), args.batch_size):
        scores.append(score_step(x_eval[i : i + args.batch_size]).numpy())
    scores = np.concatenate(scores)
    pred = (scores > 0.5).astype(np.float32)
    acc = float((pred == y_eval).mean())
    auc = auc_score(scores, y_eval)
    sim_mean = float(scores[y_eval < 0.5].mean())
    real_mean = float(scores[y_eval > 0.5].mean())

    print("\n=== sim2real gap probe (held-out volumes) ===")
    print(f"  mode: {cfg.data.mode}")
    print(f"  accuracy: {acc:.3f}  (0.5に近いほどギャップ小)")
    print(f"  AUC:      {auc:.3f}")
    print(
        f"  mean score  sim: {sim_mean:.3f} / real: {real_mean:.3f} (real=1が正解ラベル)"
    )
    if acc < 0.6:
        print(
            "  → 分布ギャップはほぼありません。シミュレーションは実劣化をよく再現しています"
        )
    elif acc < 0.85:
        print(
            "  → 軽微なギャップがあります。calibrate_degradation.pyでの校正や"
            "劣化パラメータ範囲の拡大を検討してください"
        )
    else:
        print(
            "  → 明確なギャップがあります。校正で改善しない場合は劣化生成の学習や"
            "実データのdata consistency損失の導入を検討してください"
        )

    if args.examples_dir is not None:
        args.examples_dir.mkdir(parents=True, exist_ok=True)
        # 判別器が最も「らしい」と判定したパッチと誤判定パッチを保存する
        for side, label_val in (("sim", 0.0), ("real", 1.0)):
            idx = np.where(y_eval == label_val)[0]
            order = idx[np.argsort(scores[idx])]
            picks = list(order[:4]) + list(order[-4:])  # スコア両端
            for j in picks:
                save_example_png(
                    x_eval[j],
                    scores[j],
                    args.examples_dir / f"{side}_score{scores[j]:.3f}_{j:04d}.png",
                )
        print(f"  examples saved to: {args.examples_dir}")


if __name__ == "__main__":
    main()
