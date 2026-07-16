"""位置合わせ済みCAC pairのmotion/degradation統計をJSONへ集計する。"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config_with_extends  # noqa: E402
from data.cac_motion import (  # noqa: E402
    compare_cac_pair,
    make_soft_heart_mask,
    resolve_heart_mask_path,
)
from data.pairing import resolve_target_hdr_path  # noqa: E402
from irg import read_hdr, read_raw  # noqa: E402


def _summary(cases):
    paths = [
        "threshold_dice",
        "centroid_distance_mm",
        "heart_mae_hu",
        "target_cac_mae_hu",
        "volume_ratio",
        "peak_hu_ratio",
    ]
    output = {}
    for key in paths:
        values = [case[key] for case in cases if case[key] is not None]
        if not values:
            output[key] = None
            continue
        values = np.asarray(values, dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10.0)),
            "p90": float(np.percentile(values, 90.0)),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config_cac.yaml"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--heart-mask-dir", type=Path, default=None)
    parser.add_argument("--threshold-hu", type=float, default=130.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config_with_extends(args.config)
    source_root = args.source_dir.resolve()
    target_root = args.target_dir.resolve()
    config.data_dir = source_root
    config.data.target_data_dir = target_root
    headers = [
        path
        for path in sorted(source_root.glob("**/*.hdr"))
        if ".mask." not in path.name
    ]
    if not headers:
        raise FileNotFoundError(f"sourceが見つかりません: {source_root}")

    cases = []
    for source_path in tqdm(headers, desc="summarizing CAC pairs"):
        relative = source_path.relative_to(source_root)
        target_path = resolve_target_hdr_path(source_path, config)
        source_size, _, source_spacing = read_hdr(source_path)
        target_size, _, target_spacing = read_hdr(target_path)
        if tuple(source_size) != tuple(target_size) or not np.allclose(
            source_spacing, target_spacing, atol=1e-3
        ):
            raise ValueError(f"size/spacing mismatch: {relative}")
        source = read_raw(source_path).astype(np.float32)
        target = read_raw(target_path).astype(np.float32)
        external_mask = None
        # ROIはgated targetに対応するmaskを優先する。
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
            external_mask = read_raw(mask_path) > 0
            if tuple(external_mask.shape) != tuple(target.shape):
                raise ValueError(
                    "heart mask shape mismatch: "
                    f"target={tuple(target.shape)}, mask={external_mask.shape}"
                )
        soft_mask = make_soft_heart_mask(
            source.shape, source_spacing, config.data.cac_motion, external_mask
        )
        metrics = compare_cac_pair(
            source,
            target,
            source_spacing,
            heart_mask=soft_mask >= 0.25,
            threshold=args.threshold_hu,
        )
        metrics["case"] = str(relative)
        metrics["target_case"] = str(target_path.relative_to(target_root))
        cases.append(metrics)

    result = {"num_cases": len(cases), "summary": _summary(cases), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
