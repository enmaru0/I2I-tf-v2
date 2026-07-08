import random
import sys
from pathlib import Path

import numpy as np
from absl import logging
from irg import read_raw, read_re4

from .utils import (
    calculate_intensity,
    center_within_image,
    extract_bb,
    random_crop_center_within_bb,
    random_crop_organ_center_with_crop,
)


def load_organ_box(organ_box_path):
    with open(organ_box_path, "r") as f:
        organ_box_str = f.read().strip()
    return np.array(list(map(int, organ_box_str.split(","))))


def save_organ_box(img_hdr_path: str, suffix):
    assert suffix.startswith("."), "suffix must start with ."
    img_hdr_path = Path(img_hdr_path)
    msk_hdr_path = img_hdr_path.with_suffix(suffix + ".mask.hdr")
    save_path = msk_hdr_path.parent / msk_hdr_path.name.replace(".mask.hdr", ".box.txt")
    if save_path.exists():
        return save_path

    if not msk_hdr_path.exists():
        logging.warning(f"{msk_hdr_path} not found")
        return
    msk = read_re4(msk_hdr_path, src_dst_bit_dict={0: 0})
    if msk.sum() == 0:
        logging.error(f"Empty Mask: {msk_hdr_path}")
        sys.exit()
    organ_box_zyxzyx = extract_bb(msk > 0, to_open=True)
    with open(save_path, "w") as f:
        f.write(",".join(map(str, organ_box_zyxzyx)))


def save_intensity(img_hdr_path: str, min_percentile: float, max_percentile: float):
    img_hdr_path = Path(img_hdr_path)
    save_path = img_hdr_path.with_suffix(
        f".intensity-{min_percentile}-{max_percentile}.txt"
    )
    if save_path.exists():
        return save_path
    img = read_raw(img_hdr_path).astype(np.float32)
    min_intensity, max_intensity = calculate_intensity(
        img, min_percentile, max_percentile
    )
    with open(save_path, "w") as f:
        f.write(f"{min_intensity},{max_intensity}")


def load_intensity(intensity_path):
    with open(intensity_path, "r") as f:
        txt = f.read().strip().split(",")
        min_intensity, max_intensity = map(float, txt)
    return min_intensity, max_intensity


def get_center(
    img_size_zyx,
    img_spacing_zyx,
    is_training,
    body_box_zyxzyx,
    organ_box_zyxzyx,
    random_crop_method,
    crop_size_zyx,
    target_spacing_zyx,
    margin,
    crop_keep_ratio,
):
    if not is_training:
        mode = "organ_center"
    else:
        mode_list = list(random_crop_method.keys())
        weight = list(random_crop_method.values())
        mode = random.choices(mode_list, weight, k=1)[0]

    if mode == "body":
        # 体表マスク内でランダムクロップ
        center_zyx = random_crop_center_within_bb(
            body_box_zyxzyx,
            img_size_zyx,
            crop_size_zyx,
            img_spacing_zyx,
            target_spacing_zyx,
            [0, 0, 0],
        )
    elif mode == "organ":
        # 臓器内でランダムクロップ
        center_zyx = random_crop_center_within_bb(
            organ_box_zyxzyx,
            img_size_zyx,
            crop_size_zyx,
            img_spacing_zyx,
            target_spacing_zyx,
            [margin] * 3,
        )
    elif mode == "organ_crop":
        # 臓器が見切れるようにクロップする
        center_zyx = random_crop_organ_center_with_crop(
            organ_box_zyxzyx,
            img_size_zyx,
            crop_size_zyx,
            img_spacing_zyx,
            target_spacing_zyx,
            keep_ratio=crop_keep_ratio,
        )
    elif mode == "image":
        center_zyx = random_crop_center_within_bb(
            tuple([0, 0, 0]) + tuple(img_size_zyx),
            img_size_zyx,
            crop_size_zyx,
            img_spacing_zyx,
            target_spacing_zyx,
            [0, 0, 0],
        )
    elif mode == "organ_center":
        # 臓器の中心でクロップ
        center_zyx = (organ_box_zyxzyx[0:3] + organ_box_zyxzyx[3:6]) / 2

        center_zyx = center_within_image(
            center_zyx, img_size_zyx, crop_size_zyx, img_spacing_zyx, target_spacing_zyx
        )
    else:
        raise NotImplementedError(mode)
    return center_zyx
