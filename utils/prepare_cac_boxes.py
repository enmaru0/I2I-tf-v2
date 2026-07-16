"""位置合わせ済み実ペアのgated targetからCAC crop sidecarを作る。"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from data.cac_motion import make_soft_heart_mask, resolve_heart_mask_path  # noqa: E402
from data.pairing import resolve_target_hdr_path  # noqa: E402
from config_utils import load_config_with_extends  # noqa: E402
from irg import read_hdr, read_raw  # noqa: E402


def _write_box(path: Path, mask: np.ndarray):
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError(f"空maskです: {path}")
    start = indices.min(axis=0)
    stop = indices.max(axis=0) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(map(str, [*start, *stop])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config_cac.yaml"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--heart-mask-dir", type=Path, default=None)
    parser.add_argument("--threshold-hu", type=float, default=130.0)
    args = parser.parse_args()

    config = load_config_with_extends(args.config)
    source_root = args.source_dir.resolve()
    target_root = args.target_dir.resolve()
    config.data_dir = source_root
    config.data.target_data_dir = target_root
    source_headers = [
        path
        for path in sorted(source_root.glob("**/*.hdr"))
        if ".mask." not in path.name and not path.name.endswith(".target.hdr")
    ]
    if not source_headers:
        raise FileNotFoundError(f"sourceが見つかりません: {source_root}")

    for source_path in tqdm(source_headers, desc="preparing CAC boxes"):
        if ".mask." in source_path.name:
            continue
        target_path = resolve_target_hdr_path(source_path, config)
        size, _, spacing = read_hdr(target_path)
        target = read_raw(target_path).astype(np.float32)
        heart_mask = None
        # gated targetのsidecarを優先し、なければsource側を参照する。
        mask_path = resolve_heart_mask_path(target_path, image_root=target_root)
        if mask_path is None:
            mask_path = resolve_heart_mask_path(source_path, image_root=source_root)
        if mask_path is None and args.heart_mask_dir is not None:
            mask_path = resolve_heart_mask_path(
                target_path,
                image_root=target_root,
                mask_root=args.heart_mask_dir,
                include_local=False,
                required=True,
            )
        if mask_path is not None:
            heart_mask = read_raw(mask_path) > 0
            if tuple(heart_mask.shape) != tuple(target.shape):
                raise ValueError(
                    "heart mask shape mismatch: "
                    f"target={tuple(target.shape)}, mask={heart_mask.shape}"
                )
        soft_heart = make_soft_heart_mask(
            size, spacing, config.data.cac_motion, heart_mask
        )
        candidates = (target >= args.threshold_hu) & (soft_heart >= 0.25)
        crop_mask = candidates if np.any(candidates) else soft_heart >= 0.5
        _write_box(source_path.with_suffix(".cac.box.txt"), crop_mask)


if __name__ == "__main__":
    main()
