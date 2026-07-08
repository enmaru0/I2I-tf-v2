"""
デノイズ・ぼかし修正の動作確認用に合成ペアデータセットを作成するスクリプト。

target: ランダムな楕円体を配置した滑らかなボリューム（クリーン画像）
source: targetにガウシアンノイズやぼかしを加えた劣化画像

使い方:
    python utils/make_synthetic_dataset.py --out_dir datasets_i2i \
        --num_train 6 --num_val 2 --size 96 --degradation noise
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).parent.parent))
from irg import save_raw  # noqa: E402


def make_clean_volume(rng, size_zyx, num_blobs=12):
    """ランダムな楕円体を重ねたクリーン画像を作る。値はCTライク(-240～360 HU)"""
    vol = np.full(size_zyx, -150.0, np.float32)  # 背景

    zz, yy, xx = np.meshgrid(
        np.arange(size_zyx[0]),
        np.arange(size_zyx[1]),
        np.arange(size_zyx[2]),
        indexing="ij",
    )
    for _ in range(num_blobs):
        center = rng.uniform(0.15, 0.85, 3) * np.array(size_zyx)
        radii = rng.uniform(0.05, 0.3, 3) * np.array(size_zyx)
        value = rng.uniform(-100, 300)
        dist = (
            ((zz - center[0]) / radii[0]) ** 2
            + ((yy - center[1]) / radii[1]) ** 2
            + ((xx - center[2]) / radii[2]) ** 2
        )
        vol = np.where(dist <= 1.0, value, vol).astype(np.float32)

    # 境界を少し滑らかにして自然な構造にする
    vol = gaussian_filter(vol, sigma=1.0)
    return vol


def degrade(rng, clean, mode, noise_std, blur_sigma):
    """クリーン画像から劣化画像(source)を作る"""
    src = clean.copy()
    if mode in ("blur", "both"):
        src = gaussian_filter(src, sigma=blur_sigma)
    if mode in ("noise", "both"):
        src = src + rng.normal(0, noise_std, src.shape).astype(np.float32)
    return src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("datasets_i2i"))
    parser.add_argument("--dataset_name", default="synth")
    parser.add_argument("--num_train", type=int, default=6)
    parser.add_argument("--num_val", type=int, default=2)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument(
        "--degradation", choices=["noise", "blur", "both"], default="noise"
    )
    parser.add_argument("--noise_std", type=float, default=40.0)
    parser.add_argument("--blur_sigma", type=float, default=2.0)
    parser.add_argument("--target_suffix", default=".target")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    size_zyx = (args.size,) * 3
    spacing_zyx = np.array([1.0, 1.0, 1.0], np.float32)

    for split, num in [("train", args.num_train), ("val", args.num_val)]:
        save_dir = args.out_dir / split / args.dataset_name
        save_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num):
            clean = make_clean_volume(rng, size_zyx)
            src = degrade(rng, clean, args.degradation, args.noise_std, args.blur_sigma)

            key = f"case{i:03d}"
            save_raw(
                src.astype(np.int16), spacing_zyx, save_dir / f"{key}.hdr"
            )
            save_raw(
                clean.astype(np.int16),
                spacing_zyx,
                save_dir / f"{key}{args.target_suffix}.hdr",
            )
            print(f"saved {split}/{args.dataset_name}/{key} (+ target)")


if __name__ == "__main__":
    main()
