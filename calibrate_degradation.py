"""
実劣化画像からシミュレーションパラメータを推定し、
self_noise / self_sr の推奨設定値を出力する校正スクリプト。

- noise校正: Immerkær法（3x3ラプラシアン応答の総和）でスライスごとの
  ノイズσを推定し、正規化強度レンジに対する相対値(noise_std_rel)へ換算する。
  テクスチャの影響を抑えるため前景マスク内で計算し、症例内は中央値で集約する
- sr校正: ヘッダのスペーシングから厚肉軸とスライス間隔の分布を集計する。
  スライス厚は別途計測できないため間隔と同じ値を提案する（要確認）

出力は config.yaml に貼れるYAMLスニペットと --overrides 文字列。
TensorFlow不要なのでどの環境でも実行できる。

使い方:
  python calibrate_degradation.py --real_dir /path/to/real --task noise
  python calibrate_degradation.py --real_dir /path/to/real --task sr
"""

import argparse
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import convolve

from data.utils.data_utils import calculate_intensity
from irg import read_hdr, read_raw

# Immerkær "Fast Noise Variance Estimation" (1996) のラプラシアン差分カーネル
IMMERKAER_KERNEL = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], np.float32
)

AXIS_NAMES = {0: "AX", 1: "COR", 2: "SAG"}


def estimate_noise_sigma_slice(slc: np.ndarray, mask: np.ndarray) -> float | None:
    """1スライスのノイズσをImmerkær法で推定する（前景マスク内のみ）"""
    n_valid = int(mask.sum())
    if n_valid < 1000:  # 前景が少なすぎるスライスはスキップ
        return None
    response = np.abs(convolve(slc.astype(np.float32), IMMERKAER_KERNEL, mode="reflect"))
    total = float(response[mask].sum())
    return math.sqrt(math.pi / 2.0) * total / (6.0 * n_valid)


def get_intensity_range(img: np.ndarray, cfg) -> tuple[float, float] | None:
    """学習時の正規化と同じ規約でclip値を求める（MR:percentile / CT:window）"""
    if cfg.image.modality == "MR":
        if not np.any(img):
            return None  # 全ゼロボリューム（データ不良）はスキップ
        return calculate_intensity(
            img.astype(np.float32),
            cfg.image.MR.min_percentile,
            cfg.image.MR.max_percentile,
        )
    level = float(cfg.image.CT.window_level)
    width = float(cfg.image.CT.window_width)
    return level - width / 2.0, level + width / 2.0


def calibrate_noise(hdr_list, cfg, foreground_threshold_rel, slice_stride):
    print("\n--- noise calibration (Immerkær, foreground median) ---")
    per_volume = []
    for hdr_path in hdr_list:
        img, _spacing = _read_volume(hdr_path)
        if img is None:
            continue
        clip = get_intensity_range(img, cfg)
        if clip is None:
            print(f"  {hdr_path.stem}: 全ゼロのためスキップ")
            continue
        min_val, max_val = clip
        intensity_range = float(max_val - min_val)
        threshold = min_val + foreground_threshold_rel * intensity_range

        sigmas = []
        for z in range(0, img.shape[0], slice_stride):
            slc = img[z].astype(np.float32)
            sigma = estimate_noise_sigma_slice(slc, slc > threshold)
            if sigma is not None:
                sigmas.append(sigma)
        if not sigmas:
            print(f"  {hdr_path.stem}: 前景スライスが無いためスキップ")
            continue
        sigma_med = float(np.median(sigmas))
        std_rel = sigma_med / (intensity_range + 1e-6)
        per_volume.append(std_rel)
        print(
            f"  {hdr_path.stem}: sigma={sigma_med:.2f} "
            f"range={intensity_range:.1f} -> std_rel={std_rel:.4f}"
        )

    assert per_volume, "有効な症例がありませんでした"
    vals = np.array(per_volume)
    lo, med, hi = (
        float(np.percentile(vals, 10)),
        float(np.median(vals)),
        float(np.percentile(vals, 90)),
    )
    lo, med, hi = round(lo, 3), round(med, 3), round(hi, 3)
    print(f"\n  症例数: {len(vals)}  std_rel p10/median/p90 = {lo}/{med}/{hi}")
    print("\n=== 推奨設定 (conf/config.yaml の data.self_noise) ===")
    print(f"    noise_std_rel_range: [{lo}, {hi}]")
    print(f"    val_noise_std_rel: {med}")
    print("\n=== --overrides 文字列 ===")
    print(
        f'  "data.self_noise.noise_std_rel_range=[{lo},{hi}]" '
        f"data.self_noise.val_noise_std_rel={med}"
    )
    print("\n注意: Immerkær法はテクスチャや平滑化の影響を受けます。実データが"
          "既にノイズ除去済み・圧縮済みの場合は過小評価になります。"
          "probe_sim2real.pyで校正後のギャップを確認してください。")


def calibrate_sr(hdr_list, cfg):
    print("\n--- through-plane SR calibration (header spacing) ---")
    axis_counts = {0: 0, 1: 0, 2: 0}
    intervals = []
    for hdr_path in hdr_list:
        size_zyx, _dtype, spacing_zyx = read_hdr(hdr_path)
        spacing_zyx = [float(v) for v in spacing_zyx]
        axis = int(np.argmax(spacing_zyx))
        axis_counts[axis] += 1
        interval = spacing_zyx[axis]
        intervals.append(interval)
        print(
            f"  {hdr_path.stem}: size={tuple(int(v) for v in size_zyx)} "
            f"spacing={tuple(round(v, 3) for v in spacing_zyx)} "
            f"-> 厚肉軸={AXIS_NAMES[axis]} 間隔={interval:.2f}mm"
        )

    assert intervals, "有効な症例がありませんでした"
    vals = np.array(intervals)
    axes = [AXIS_NAMES[a] for a, c in axis_counts.items() if c > 0]
    lo = round(float(vals.min()), 1)
    hi = round(float(vals.max()), 1)
    med = round(float(np.median(vals)), 1)
    val_axis = AXIS_NAMES[max(axis_counts, key=axis_counts.get)]
    print(f"\n  症例数: {len(vals)}  軸分布: "
          + ", ".join(f"{AXIS_NAMES[a]}={c}" for a, c in axis_counts.items()))
    print(f"  間隔[mm] min/median/max = {lo}/{med}/{hi}")
    print("\n=== 推奨設定 (conf/config.yaml の data.self_sr) ===")
    print(f"    axes: [{', '.join(axes)}]")
    print(f"    slice_interval_mm_range: [{lo}, {hi}]")
    print(f"    slice_thickness_mm_range: [{lo}, {hi}] # 厚みは間隔と同値と仮定（要確認）")
    print(f"    val_axis: {val_axis}")
    print(f"    val_slice_interval_mm: {med}")
    print(f"    val_slice_thickness_mm: {med}")
    print("\n=== --overrides 文字列 ===")
    print(
        f'  "data.self_sr.axes=[{",".join(axes)}]" '
        f'"data.self_sr.slice_interval_mm_range=[{lo},{hi}]" '
        f'"data.self_sr.slice_thickness_mm_range=[{lo},{hi}]" '
        f"data.self_sr.val_axis={val_axis} "
        f"data.self_sr.val_slice_interval_mm={med} "
        f"data.self_sr.val_slice_thickness_mm={med}"
    )
    print("\n注意: スキャナ側で薄スライスへ補間済みのデータはスペーシングが実効"
          "間隔と一致しません（その場合は撮像プロトコルの値を使ってください）。"
          "スライス厚（プロファイル幅）はヘッダから分からないため間隔と同値を"
          "仮定しています。ギャップ縮小の確認にはprobe_sim2real.pyを使ってください。")


def _read_volume(hdr_path: Path):
    img, spacing_zyx = read_raw(hdr_path, return_spacing_zyx=True)
    return np.asarray(img), spacing_zyx


def main():
    parser = argparse.ArgumentParser(description="実劣化画像からのシミュレーション校正")
    parser.add_argument("--real_dir", type=Path, required=True,
                        help="実劣化画像(.hdr/.raw)のフォルダ")
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--task", choices=["auto", "noise", "sr"], default="auto",
                        help="auto: cfg.data.modeから決める")
    parser.add_argument("--max_volumes", default=64, type=int)
    parser.add_argument("--foreground_threshold_rel", default=0.05, type=float,
                        help="前景マスクの閾値（clipレンジに対する相対値）")
    parser.add_argument("--slice_stride", default=2, type=int,
                        help="ノイズ推定に使うスライスの間引き幅")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    task = args.task
    if task == "auto":
        task = {"self_noise": "noise", "self_sr": "sr"}.get(str(cfg.data.mode))
        assert task is not None, (
            f"data.mode={cfg.data.mode}からタスクを決められません。--taskを指定してください"
        )

    hdr_list = [
        p for p in sorted(args.real_dir.glob("*.hdr")) if ".mask" not in p.name
    ]
    if args.max_volumes > 0:
        hdr_list = hdr_list[: args.max_volumes]
    assert len(hdr_list) > 0, f"画像(.hdr)が見つかりません: {args.real_dir}"
    print(f"real volumes: {len(hdr_list)} ({args.real_dir})")

    if task == "noise":
        calibrate_noise(
            hdr_list, cfg, args.foreground_threshold_rel, max(1, args.slice_stride)
        )
    else:
        calibrate_sr(hdr_list, cfg)


if __name__ == "__main__":
    main()
