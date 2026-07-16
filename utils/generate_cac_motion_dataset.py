"""Gated CAC CTから合成non-gated/sourceとclean/targetのpaired_dirを作る。"""

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config_utils import load_config_with_extends  # noqa: E402
from data.cac_motion import (  # noqa: E402
    make_soft_heart_mask,
    resolve_heart_mask_path,
    simulate_cac_motion,
)
from irg import read_hdr, read_raw, save_raw, read_re4  # noqa: E402


def _write_box(path: Path, mask: np.ndarray):
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError(f"空のCAC/heart maskからboxを作成できません: {path}")
    start = indices.min(axis=0)
    stop = indices.max(axis=0) + 1
    path.write_text(",".join(map(str, [*start, *stop])))


def _load_optional_mask(
    image_path: Path, image_root: Path, mask_root: Path | None, shape
):
    mask_path = resolve_heart_mask_path(
        image_path,
        image_root=image_root,
        mask_root=mask_root,
        required=mask_root is not None,
    )
    if mask_path is None:
        return None
    mask = read_re4(mask_path)
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(
            f"heart mask shape mismatch: image={tuple(shape)}, mask={mask.shape}"
        )
    return mask > 0


def _collect_clean_headers(input_dir: Path):
    return [
        path
        for path in sorted(input_dir.glob("**/*.hdr"))
        if ".mask." not in path.name and not path.name.endswith(".target.hdr")
    ]


def _to_int16_hu(volume):
    return np.clip(np.rint(volume), -32768, 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("conf/config_cac.yaml"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heart-mask-dir", type=Path, default=None)
    parser.add_argument("--variants-per-case", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--simulator",
        choices=[
            "image_blend",
            "parallel_fbp",
            "elastic_parallel_fbp",
            "calcium_local_fbp",
        ],
        default=None,
    )
    args = parser.parse_args()

    config = load_config_with_extends(args.config)
    cfg_cac = config.data.cac_motion
    if args.simulator is not None:
        cfg_cac.simulator = args.simulator
    variants = (
        int(args.variants_per_case)
        if args.variants_per_case is not None
        else int(cfg_cac.variants_per_case)
    )
    if variants < 1:
        raise ValueError("variants-per-caseは1以上にしてください")
    base_seed = int(config.reproducibility.seed if args.seed is None else args.seed)

    input_dir = args.input_dir.resolve()
    source_root = args.output_dir.resolve() / "source"
    target_root = args.output_dir.resolve() / "target"
    headers = _collect_clean_headers(input_dir)
    if not headers:
        raise FileNotFoundError(f"gated CAC volumeが見つかりません: {input_dir}")

    progress = tqdm(headers, desc="generating CAC motion pairs")
    for hdr_path in progress:
        relative = hdr_path.relative_to(input_dir)
        size_zyx, _, spacing_zyx = read_hdr(hdr_path)
        clean = read_raw(hdr_path).astype(np.float32)
        if tuple(clean.shape) != tuple(size_zyx):
            raise ValueError(f"header/raw shape mismatch: {hdr_path}")
        heart_mask = _load_optional_mask(
            hdr_path, input_dir, args.heart_mask_dir, clean.shape
        )
        soft_heart = make_soft_heart_mask(clean.shape, spacing_zyx, cfg_cac, heart_mask)
        cac_mask = (clean >= 130.0) & (soft_heart >= 0.25)
        crop_mask = cac_mask if np.any(cac_mask) else soft_heart >= 0.5

        for variant in range(variants):
            progress.set_postfix_str(
                f"{relative} variant={variant + 1}/{variants} "
                f"simulator={cfg_cac.simulator}",
                refresh=True,
            )
            key = f"{relative.stem}__sim{variant:03d}"
            relative_variant = relative.with_name(key + ".hdr")
            seed_key = f"{relative}:{variant}".encode()
            sample_seed = (base_seed + zlib.crc32(seed_key)) & 0xFFFFFFFF
            started_at = time.perf_counter()
            simulated, metadata = simulate_cac_motion(
                clean,
                spacing_zyx,
                cfg_cac,
                rng=np.random.default_rng(sample_seed),
                heart_mask=heart_mask,
            )
            simulation_seconds = time.perf_counter() - started_at
            metadata.update(
                {
                    "seed": int(sample_seed),
                    "input": str(relative),
                    "spacing_zyx": [float(value) for value in spacing_zyx],
                    "simulation_seconds": float(simulation_seconds),
                }
            )

            source_path = source_root / relative_variant
            target_path = target_root / relative_variant
            source_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            save_raw(_to_int16_hu(simulated), spacing_zyx, source_path)
            save_raw(_to_int16_hu(clean), spacing_zyx, target_path)
            source_path.with_suffix(".json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2)
            )
            _write_box(source_path.with_suffix(".cac.box.txt"), crop_mask)
            _write_box(target_path.with_suffix(".cac.box.txt"), crop_mask)

    print(f"source: {source_root}")
    print(f"target: {target_root}")


if __name__ == "__main__":
    main()
