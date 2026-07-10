import random

import numpy as np
from einops import rearrange
from scipy.ndimage import zoom

from .data_utils import should_apply_condition


def prepare_thin2thick(
    src_spacing_zyx: np.ndarray,
    dst_spacing_zyx: np.ndarray,
    crop_size_zyx: np.ndarray,
    thick2thin_rate_zyx: float,
    is_training: bool,
    spacing_max_val: float,
    thickness_range=[2, 6],
    thickness=None,
    rng=None,
):
    rng = random if rng is None else rng
    # if thickness is specified ignore thickness_range
    # 3方向がspacing_max_valよりも大きくないと実行しない
    if max(src_spacing_zyx) > spacing_max_val or not is_training:
        extra_slice = 0
        apply_thin_thick = False
        thickness = 0
        axis = 0
    elif all(rate <= 0 for rate in thick2thin_rate_zyx):
        # すべてのrateが0の場合はthin->thick変換を行わない
        extra_slice = 0
        apply_thin_thick = False
        thickness = 0
        axis = 0
    else:
        axis_cand = [axis for axis in range(3) if thick2thin_rate_zyx[axis] > 0]
        axis = rng.choice(axis_cand)
        if should_apply_condition(thick2thin_rate_zyx[axis], is_training, rng=rng):
            if thickness is None:
                thickness = rng.randint(thickness_range[0], thickness_range[1])
            thickness = max(1, int((thickness / dst_spacing_zyx[axis]) + 0.5))
            extra_slice = thickness - crop_size_zyx[axis] % thickness + thickness
            apply_thin_thick = True
        else:
            extra_slice = 0
            apply_thin_thick = False
            thickness = 0
            axis = 0
    return dict(
        extra_slice=extra_slice,
        apply_thin_thick=apply_thin_thick,
        thickness=thickness,
        axis=axis,
    )


def virtual_thick_generator(img, thickness, order=1, axis=0):
    z, y, x = img.shape[:3]
    if axis == 0:
        assert (z % thickness) == 0, (z, thickness)
        img_reshape = rearrange(
            img, "(d1 thickness) y x -> d1 y x thickness", thickness=thickness
        )
        zoom_list = (thickness, 1, 1)
    elif axis == 1:
        assert (y % thickness) == 0, (y, thickness)
        img_reshape = rearrange(
            img, "z (d1 thickness) x -> z d1 x thickness", thickness=thickness
        )
        zoom_list = (1, thickness, 1)
    elif axis == 2:
        assert (x % thickness) == 0, (x, thickness)
        img_reshape = rearrange(
            img, "z y (d1 thickness) -> z y d1 thickness", thickness=thickness
        )
        zoom_list = (1, 1, thickness)
    else:
        raise ValueError(axis)
    # 単純平均の場合
    # img_mean = img_reshape.mean(3, img.dtype)

    # 単純平均ではなくスライス感度プロフィールに近い形で計算したい場合
    SSP_range = np.arange(thickness)
    SSP_mu = thickness / 2.0
    SSP_sigma = thickness / 6.0
    SSP = np.exp(-((SSP_range - SSP_mu) ** 2) / (2 * SSP_sigma**2)) / SSP_sigma
    SSP = SSP / np.sum(SSP)
    img_mean = img_reshape.dot(SSP)

    # 元サイズに戻す
    zoom(img_mean, zoom_list, order=order, output=img)
    return img
