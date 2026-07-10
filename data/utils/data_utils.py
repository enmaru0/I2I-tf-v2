import random

import numpy as np


def should_apply_condition(freq, is_training, rng=None):
    if not is_training:
        return False
    rng = random if rng is None else rng
    if rng.random() < freq:
        return True
    else:
        return False


def crop_input(input, bb_zyxzyx):
    input = input[
        bb_zyxzyx[0] : bb_zyxzyx[3],
        bb_zyxzyx[1] : bb_zyxzyx[4],
        bb_zyxzyx[2] : bb_zyxzyx[5],
    ]
    return input


def normalize_img(img, wl: tuple[int, float], ww: tuple[int, float]) -> None:
    min_val = wl - ww / 2
    max_val = wl + ww / 2
    img -= min_val
    img /= max_val - min_val
    np.clip(img, 0, 1, out=img)


def clip_bbs(bbs, start_zyx, end_zyx):
    bbs = bbs.copy()
    if bbs.ndim == 3:
        bbs[:, :, 0] = np.clip(bbs[:, :, 0], start_zyx[0], end_zyx[0])
        bbs[:, :, 1] = np.clip(bbs[:, :, 1], start_zyx[1], end_zyx[1])
        bbs[:, :, 2] = np.clip(bbs[:, :, 2], start_zyx[2], end_zyx[2])
    elif bbs.ndim == 2:
        if bbs.shape[1] == 6:
            bbs[:, [0, 3]] = np.clip(bbs[:, [0, 3]], start_zyx[0], end_zyx[0])
            bbs[:, [1, 4]] = np.clip(bbs[:, [1, 4]], start_zyx[1], end_zyx[1])
            bbs[:, [2, 5]] = np.clip(bbs[:, [2, 5]], start_zyx[2], end_zyx[2])
        elif bbs.shape[1] == 3:
            bbs[:, 0] = np.clip(bbs[:, 0], start_zyx[0], end_zyx[0])
            bbs[:, 1] = np.clip(bbs[:, 1], start_zyx[1], end_zyx[1])
            bbs[:, 2] = np.clip(bbs[:, 2], start_zyx[2], end_zyx[2])
        else:
            raise ValueError(bbs)
    elif bbs.ndim == 1:
        bbs[[0, 3]] = np.clip(bbs[[0, 3]], start_zyx[0], end_zyx[0])
        bbs[[1, 4]] = np.clip(bbs[[1, 4]], start_zyx[1], end_zyx[1])
        bbs[[2, 5]] = np.clip(bbs[[2, 5]], start_zyx[2], end_zyx[2])
    return bbs


def calc_img_crop_region(
    crop_size_zyx: tuple[int, int, int],
    affine,
    min_zyx: tuple[int, int, int],
    max_zyx: tuple[int, int, int],
    extra_slice: int = 0,
):
    crop_size_extra = crop_size_zyx.copy()
    crop_size_extra[0] += extra_slice
    crop_zyx = np.array([[0, 0, 0], crop_size_extra], np.float32)
    input_8 = eight_point_extractor(crop_zyx[None])
    output_8 = input_8 @ affine.T
    clip_zyx = two_point_extractor(output_8)[0]
    clip_zyx[0] = np.floor(clip_zyx[0])
    clip_zyx[1] = np.ceil(clip_zyx[1])
    clip_zyxzyx = clip_zyx.reshape(-1)
    clip_zyxzyx = clip_zyxzyx.astype(np.int16)
    clip_zyxzyx[3:] += 1
    shift_start = np.maximum(clip_zyxzyx[:3], 0)
    clip_zyxzyx = clip_bbs(clip_zyxzyx, min_zyx, max_zyx)
    return clip_zyxzyx, shift_start


def eight_point_extractor(BB):
    BB_num = BB.shape[0]
    bb_output = np.ones([BB_num, 8, 4], dtype=np.float32)

    bb_output[:, 0, 0] = np.max(BB[:, :, 0], 1)
    bb_output[:, 0, 1] = np.max(BB[:, :, 1], 1)
    bb_output[:, 0, 2] = np.max(BB[:, :, 2], 1)

    bb_output[:, 1, 0] = np.min(BB[:, :, 0], 1)
    bb_output[:, 1, 1] = np.max(BB[:, :, 1], 1)
    bb_output[:, 1, 2] = np.max(BB[:, :, 2], 1)

    bb_output[:, 2, 0] = np.max(BB[:, :, 0], 1)
    bb_output[:, 2, 1] = np.min(BB[:, :, 1], 1)
    bb_output[:, 2, 2] = np.max(BB[:, :, 2], 1)

    bb_output[:, 3, 0] = np.max(BB[:, :, 0], 1)
    bb_output[:, 3, 1] = np.max(BB[:, :, 1], 1)
    bb_output[:, 3, 2] = np.min(BB[:, :, 2], 1)

    bb_output[:, 4, 0] = np.min(BB[:, :, 0], 1)
    bb_output[:, 4, 1] = np.min(BB[:, :, 1], 1)
    bb_output[:, 4, 2] = np.max(BB[:, :, 2], 1)

    bb_output[:, 5, 0] = np.min(BB[:, :, 0], 1)
    bb_output[:, 5, 1] = np.max(BB[:, :, 1], 1)
    bb_output[:, 5, 2] = np.min(BB[:, :, 2], 1)

    bb_output[:, 6, 0] = np.max(BB[:, :, 0], 1)
    bb_output[:, 6, 1] = np.min(BB[:, :, 1], 1)
    bb_output[:, 6, 2] = np.min(BB[:, :, 2], 1)

    bb_output[:, 7, 0] = np.min(BB[:, :, 0], 1)
    bb_output[:, 7, 1] = np.min(BB[:, :, 1], 1)
    bb_output[:, 7, 2] = np.min(BB[:, :, 2], 1)

    return bb_output


def two_point_extractor(bb_output):
    BB_num = bb_output.shape[0]
    BB = np.zeros([BB_num, 2, 3], dtype=np.float32)
    BB[:, 0] = np.min(bb_output[:, :, :3], 1)
    BB[:, 1] = np.max(bb_output[:, :, :3], 1)
    return BB


def six_point_extractor(BB):
    BB_num = BB.shape[0]
    bb_output = np.ones([BB_num, 6, 4], dtype=np.float32)
    bb_eight = eight_point_extractor(BB)
    # depth, height, width = BB[:,1] - BB[:,0]
    bb_output[:, 0] = (
        bb_eight[:, 0] + bb_eight[:, 2] + bb_eight[:, 3] + bb_eight[:, 6]
    ) / 4
    bb_output[:, 1] = (
        bb_eight[:, 0] + bb_eight[:, 1] + bb_eight[:, 3] + bb_eight[:, 5]
    ) / 4
    bb_output[:, 2] = (
        bb_eight[:, 0] + bb_eight[:, 1] + bb_eight[:, 2] + bb_eight[:, 4]
    ) / 4
    bb_output[:, 3] = (
        bb_eight[:, 1] + bb_eight[:, 4] + bb_eight[:, 5] + bb_eight[:, 7]
    ) / 4
    bb_output[:, 4] = (
        bb_eight[:, 2] + bb_eight[:, 4] + bb_eight[:, 6] + bb_eight[:, 7]
    ) / 4
    bb_output[:, 5] = (
        bb_eight[:, 3] + bb_eight[:, 5] + bb_eight[:, 6] + bb_eight[:, 7]
    ) / 4
    return bb_output


def extract_bb(msk, to_open):
    assert msk.dtype == np.bool_, msk.dtype
    assert msk.ndim in [3, 4], msk.shape
    if msk.ndim == 4:
        assert msk.shape[-1] == 1, msk.shape
    z, y, x = msk.shape[:3]
    z = np.any(msk, axis=(1, 2))
    z_min, z_max = np.nonzero(z)[0][[0, -1]]
    y = np.any(msk[z_min : z_max + 1], axis=(0, 2))
    y_min, y_max = np.nonzero(y)[0][[0, -1]]
    x = np.any(msk[z_min : z_max + 1, y_min : y_max + 1], axis=(0, 1))
    x_min, x_max = np.nonzero(x)[0][[0, -1]]

    if to_open:
        z_max += 1
        y_max += 1
        x_max += 1
    return np.array([z_min, y_min, x_min, z_max, y_max, x_max])


def add_channel_dim(input):
    if input is None:
        return None
    elif input.ndim == 3:
        return input[:, :, :, None]
    else:
        assert input.ndim == 4, input.ndim
        return input


def add_margin(
    bb, size_zyx, margin_mm_zyx, src_spacing_zyx, round_to_int=False, pad_remain=True
):
    bb = np.array(bb, np.float32)
    if not isinstance(margin_mm_zyx, np.ndarray):
        margin_mm_zyx = np.array(margin_mm_zyx)
    if not isinstance(src_spacing_zyx, np.ndarray):
        src_spacing_zyx = np.array(src_spacing_zyx)
    if not isinstance(size_zyx, np.ndarray):
        size_zyx = np.array(size_zyx)
    margin_zyx = margin_mm_zyx / src_spacing_zyx
    margin_zyx = np.floor(margin_zyx)

    bb_org_shape = bb.shape
    if bb.shape == (6,):
        bb = bb.reshape(1, 6)
    elif bb.shape == (2, 3):
        bb = bb.reshape(1, 6)
    elif bb.ndim == 2 and bb.shape[1] == 6:
        pass
    else:
        raise NotImplementedError(bb.shape)
    if pad_remain:
        # 後側が足りないときは前側に足す
        pad_remain_end = size_zyx[None] - (bb[:, 3:6] + margin_zyx[None])
        # 前側の処理
        pad_remain_end = np.maximum(-pad_remain_end, 0)
        bb[:, :3] -= margin_zyx[None] + pad_remain_end
        # 後ろ側の処理
        pad_remain_start = np.maximum(0, -bb[:, :3])
        bb[:, 3:] += margin_zyx[None] + pad_remain_start
    else:
        bb[:, :3] -= margin_zyx[None]
        bb[:, 3:] += margin_zyx[None]
    if round_to_int:
        bb = (bb + 0.5).astype(np.int32)
    bb = clip_bbs(bb, [0, 0, 0], size_zyx)
    return bb.reshape(bb_org_shape)


def calculate_intensity(image, min_percentile=0.5, max_percentile=99.5):
    """
    Calculate the intensity of a 3D medical image to [min_percentile, max_percentile] percentiles
    for non-zero voxels.

    Parameters:
        image (ndarray): The input 3D medical image
        min_percentile (float): The minimum percentile to clip the intensities (default: 0.5)
        max_percentile (float): The maximum percentile to clip the intensities (default: 99.5)

    Returns:
        ndarray: The clipped 3D medical image
    """
    assert image.dtype.kind == "f", "image must be float type"

    # Compute the non-zero voxels

    # Flatten the image and filter non-zero values
    image_flat = image.ravel()
    non_zero_value = image_flat[image_flat != 0]

    # Compute the intensity percentiles for the non-zero voxels
    min_intensity, max_intensity = np.percentile(
        non_zero_value, [min_percentile, max_percentile]
    )

    return min_intensity, max_intensity


def get_pad_for_margin(
    crop_size_zyxzyx,
    dilation_mm_zyx,
    size_zyx,
    src_spacing_zyx,
    dst_spacing,
    num_pool,
    return_dst_size=False,
):
    if not isinstance(crop_size_zyxzyx, np.ndarray):
        crop_size_zyxzyx = np.array(crop_size_zyxzyx)
    if not isinstance(dilation_mm_zyx, np.ndarray):
        dilation_mm_zyx = np.array(dilation_mm_zyx)
    if not isinstance(size_zyx, np.ndarray):
        size_zyx = np.array(size_zyx)
    if not isinstance(src_spacing_zyx, np.ndarray):
        src_spacing_zyx = np.array(src_spacing_zyx)
    if not isinstance(dst_spacing, np.ndarray):
        dst_spacing = np.array(dst_spacing)
    # 正規化後のクロップサイズを計算
    clip_size_zyx = crop_size_zyxzyx[3:6] - crop_size_zyxzyx[:3] - 1
    clip_size_zyx = clip_size_zyx * src_spacing_zyx / dst_spacing
    current_crop_size = np.ceil(clip_size_zyx + 2 * dilation_mm_zyx / dst_spacing) + 1

    # dilationによって、画像外まで広がる可能性があるのでclipする。
    current_crop_size = np.minimum(
        current_crop_size, np.ceil((size_zyx - 1) * src_spacing_zyx / dst_spacing) + 1
    )
    current_crop_size = current_crop_size.astype(np.int32)
    # UNetに対応させるためにサイズを整える。この操作で画像サイズよりも若干大きくなる可能性があるがそこは諦める
    feat_stride = (1 << num_pool) - 1
    resized_zyx = (current_crop_size + feat_stride) & ~feat_stride
    pad_zyx = resized_zyx - current_crop_size

    if return_dst_size:
        return dilation_mm_zyx + pad_zyx / 2 * dst_spacing, resized_zyx
    else:
        return dilation_mm_zyx + pad_zyx / 2 * dst_spacing


def check_mask_bit_number(msk):
    if msk.dtype == np.uint8:
        return 8
    if msk.dtype == np.uint16:
        return 16
    if msk.dtype == np.uint32:
        return 32
    if msk.dtype == np.uint64:
        return 64
    raise ValueError(f"Unsupported mask dtype: {msk.dtype}")
