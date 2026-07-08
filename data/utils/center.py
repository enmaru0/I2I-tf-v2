import random

import numpy as np

from .data_utils import add_margin


def center_within_image(
    center_zyx: np.ndarray,
    src_range_zyxzyx: np.ndarray,
    dst_size_zyx: np.ndarray,
    src_spacing_zyx: np.ndarray,
    dst_spacing_zyx: np.ndarray,
) -> np.ndarray:
    """
    Adjust the center coordinates within an image to ensure that a cropped region can fit within the image.
    It will also adjust the center so that cropped image will be at the top corner (0,0,0).
    Parameters:
    - center_zyx (np.ndarray): Array of shape (3,) representing the center coordinates.
    - src_range_zyxzyx (np.ndarray): Array of shape (6,) representing the range of the source image.
    - dst_size_zyx (np.ndarray): Array of shape (3,) representing the size of the destination image.
    - src_spacing_zyx (np.ndarray): Array of shape (3,) representing the spacing of the source image.
    - dst_spacing_zyx (np.ndarray): Array of shape (3,) representing the spacing of the destination image.

    Returns:
    - center_zyx (np.ndarray): Array of shape (3,) representing the adjusted center coordinates.
    """

    if len(src_range_zyxzyx) == 3:
        src_range_zyxzyx = np.concatenate([[0, 0, 0], src_range_zyxzyx])
    center_zyx = center_zyx.copy()
    # crop後に画像内にできるだけ収まるように調整

    for axis in range(3):
        norm_scale = src_spacing_zyx[axis] / dst_spacing_zyx[axis]
        crop_size_norm_half = dst_size_zyx[axis] / norm_scale / 2

        start = center_zyx[axis] - crop_size_norm_half
        end = center_zyx[axis] + crop_size_norm_half

        if start < src_range_zyxzyx[axis] and end > src_range_zyxzyx[axis + 3]:
            # startとendが画像を囲んでいる。画像が左上（0,0,0）に来るように調整する
            center_zyx[axis] += src_range_zyxzyx[axis] - start
        elif start > src_range_zyxzyx[axis] and end > src_range_zyxzyx[axis + 3]:
            # startが画像内にある、endは画像外にある
            shift_start = min(
                start - src_range_zyxzyx[axis], end - src_range_zyxzyx[axis + 3]
            )
            center_zyx[axis] -= shift_start
        elif end < src_range_zyxzyx[axis + 3] and start < src_range_zyxzyx[axis]:
            # endが画像内にある、startは画像外にある
            shift_end = src_range_zyxzyx[axis] - start
            center_zyx[axis] += shift_end
        else:
            # start and/or endが画像外にあるが、シフトさせる余裕がない
            pass
    return center_zyx


def random_crop_center_within_bb(
    bb_zyxzyx, src_size_zyx, dst_size_zyx, src_spacing_zyx, dst_spacing_zyx, margin_zyx
):
    bb_zyxzyx = np.array(bb_zyxzyx)
    bb_zyxzyx = add_margin(bb_zyxzyx, src_size_zyx, margin_zyx, src_spacing_zyx)

    random_center_zyx = [
        random.uniform(bb_zyxzyx[axis], bb_zyxzyx[axis + 3]) for axis in range(3)
    ]

    center_zyx = center_within_image(
        np.array(random_center_zyx),
        bb_zyxzyx,
        dst_size_zyx,
        src_spacing_zyx,
        dst_spacing_zyx,
    )

    center_zyx = center_within_image(
        center_zyx, src_size_zyx, dst_size_zyx, src_spacing_zyx, dst_spacing_zyx
    )
    return center_zyx


def random_crop_center_with_tumor(
    tumor_zyxzyx,
    organ_zyxzyx,
    src_size_zyx,
    dst_size_zyx,
    src_spacing_zyx,
    dst_spacing_zyx,
    margin_zyx,
):
    tumor_zyxzyx = np.array(tumor_zyxzyx)
    tumor_zyxzyx = add_margin(tumor_zyxzyx, src_size_zyx, margin_zyx, src_spacing_zyx)

    if organ_zyxzyx is not None:
        organ_zyxzyx = add_margin(
            organ_zyxzyx, src_size_zyx, margin_zyx, src_spacing_zyx
        )

    random_center_zyx = [
        random.uniform(tumor_zyxzyx[axis], tumor_zyxzyx[axis + 3]) for axis in range(3)
    ]

    # 可能な限り腫瘍が入るように調整
    center_zyx = center_within_image(
        np.array(random_center_zyx),
        tumor_zyxzyx,
        dst_size_zyx,
        src_spacing_zyx,
        dst_spacing_zyx,
    )
    # 可能な限り臓器内に収まるように調整
    if organ_zyxzyx is not None:
        center_zyx = center_within_image(
            center_zyx, organ_zyxzyx, dst_size_zyx, src_spacing_zyx, dst_spacing_zyx
        )
    # 可能な限り画像内に収まるように調整
    center_zyx = center_within_image(
        center_zyx, src_size_zyx, dst_size_zyx, src_spacing_zyx, dst_spacing_zyx
    )
    return center_zyx


def random_crop_organ_center_with_crop(
    organ_zyxzyx,
    size_zyx,
    crop_size_zyx,
    src_spacing_zyx,
    dst_spacing_zyx,
    keep_ratio=0.3,
):
    """
    臓器BBの6方向のうち1方向が見切れるようになる画像中心を計算する
    ただし、すでに臓器が見切れている場合は臓器が可能な限り入るような中心を返す。
    keep_ratio: 少なくともkeep_ratioだけは残るようにする
    """
    organ_zyxzyx = organ_zyxzyx.copy()
    organ_center_zyx = (organ_zyxzyx[:3] + organ_zyxzyx[3:6]) / 2

    if np.any(organ_zyxzyx[:3] <= 0) or np.any(organ_zyxzyx[3:] >= size_zyx):
        # すでに見切れている
        center_zyx = center_within_image(
            organ_center_zyx,
            np.concatenate([[0, 0, 0], size_zyx]),
            crop_size_zyx,
            src_spacing_zyx,
            dst_spacing_zyx,
        )
        return center_zyx

    choice = random.choice([0, 1, 2, 3, 4, 5])
    organ_size = (organ_zyxzyx[3:6] - organ_zyxzyx[:3])[choice % 3]
    src_spacing = src_spacing_zyx[choice % 3]
    dst_spacing = dst_spacing_zyx[choice % 3]
    crop_size = crop_size_zyx[choice % 3]
    crop_size_half_norm = crop_size / 2.0 * dst_spacing / src_spacing

    crop_len = organ_size * random.random() * (1 - keep_ratio)
    center_zyx = (organ_zyxzyx[:3] + organ_zyxzyx[3:6]) / 2
    if choice < 3:
        center_zyx[choice % 3] = organ_zyxzyx[choice] + crop_len + crop_size_half_norm
    else:
        center_zyx[choice % 3] = organ_zyxzyx[choice] - crop_len - crop_size_half_norm

    return center_zyx
