import itertools
import zlib

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter, zoom


def resize_volume_to_shape(
    volume: np.ndarray,
    out_shape_zyx,
    order: int,
    anti_alias: bool = True,
) -> np.ndarray:
    """3D volumeを指定shapeへリサンプルし、端数をcrop/padで厳密に合わせる。"""
    volume = np.asarray(volume, np.float32)
    out_shape = np.asarray(out_shape_zyx, np.int64)
    in_shape = np.asarray(volume.shape[:3], np.int64)
    scale = out_shape / in_shape
    if anti_alias and np.any(scale < 1.0):
        sigma = np.where(scale < 1.0, 0.5 * (1.0 / scale - 1.0), 0.0)
        volume = gaussian_filter(volume, sigma=sigma, mode="nearest")

    output = zoom(
        volume,
        tuple(scale),
        order=int(order),
        prefilter=int(order) > 1,
        mode="nearest",
    ).astype(np.float32, copy=False)

    slices = tuple(slice(0, min(output.shape[i], int(out_shape[i]))) for i in range(3))
    output = output[slices]
    pad_width = [
        (0, max(0, int(out_shape[i]) - output.shape[i])) for i in range(3)
    ]
    if any(after > 0 for _, after in pad_width):
        output = np.pad(output, pad_width, mode="edge")
    return output


def _sliding_starts(size: int, patch: int, overlap: float) -> list[int]:
    if size <= patch:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _blend_window(patch_size_zyx) -> np.ndarray:
    window = np.ones(tuple(patch_size_zyx), np.float32)
    for axis, size in enumerate(patch_size_zyx):
        weights = (
            np.hanning(size).astype(np.float32)
            if size > 1
            else np.ones(1, np.float32)
        )
        weights = np.maximum(weights, 1e-3)
        shape = [1, 1, 1]
        shape[axis] = size
        window *= weights.reshape(shape)
    return window[..., None]


def sliding_window_predict(
    model,
    data,
    patch_size_zyx,
    overlap: float,
    batch_size: int,
    base_seed: int,
):
    """3D volumeをoverlap付きパッチで推論し、Hann重みで結合する。"""
    patch_size = np.asarray(patch_size_zyx, np.int32)
    if np.any(patch_size <= 0):
        raise ValueError(f"inference.patch_size_zyx must be positive: {patch_size_zyx}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"inference.overlap must be in [0, 1): {overlap}")

    src = data["img"].numpy().astype(np.float32, copy=False)
    msk = data["img_msk"].numpy().astype(np.float32, copy=False)
    original_shape = np.asarray(src.shape[:3], np.int32)
    padded_shape = np.maximum(original_shape, patch_size)
    pad_after = padded_shape - original_shape
    min_val = float(data["src_min_clip_val"].numpy())
    pad_width = (
        (0, int(pad_after[0])),
        (0, int(pad_after[1])),
        (0, int(pad_after[2])),
        (0, 0),
    )
    src = np.pad(
        src, pad_width, mode="constant", constant_values=min_val
    )
    msk = np.pad(msk, pad_width, mode="constant")

    starts = [
        _sliding_starts(int(size), int(patch), overlap)
        for size, patch in zip(padded_shape, patch_size)
    ]
    coordinates = list(itertools.product(*starts))
    blend = _blend_window(patch_size)
    pred_sum = np.zeros(src.shape, np.float32)
    weight_sum = np.zeros(src.shape, np.float32)

    batch_size = max(1, int(batch_size))
    for batch_start in range(0, len(coordinates), batch_size):
        batch_coords = coordinates[batch_start : batch_start + batch_size]
        src_batch, msk_batch, seeds = [], [], []
        for z, y, x in batch_coords:
            end = np.array([z, y, x], np.int32) + patch_size
            src_batch.append(src[z : end[0], y : end[1], x : end[2]])
            msk_batch.append(msk[z : end[0], y : end[1], x : end[2]])
            tile_hash = zlib.crc32(f"{z}:{y}:{x}".encode()) & 0x7FFFFFFF
            seeds.append([int(base_seed), tile_hash])

        model_data = {
            "src_imgs": tf.constant(np.stack(src_batch), tf.float32),
            "img_msks": tf.constant(np.stack(msk_batch), tf.float32),
            "src_min_clip_vals": tf.repeat(
                data["src_min_clip_val"][None], len(batch_coords), axis=0
            ),
            "src_max_clip_vals": tf.repeat(
                data["src_max_clip_val"][None], len(batch_coords), axis=0
            ),
            "sample_seeds": tf.constant(seeds, tf.int32),
        }
        pred_batch = model.predict_step(model_data).numpy()
        for pred_patch, mask_patch, (z, y, x) in zip(
            pred_batch, msk_batch, batch_coords
        ):
            end = np.array([z, y, x], np.int32) + patch_size
            weight = blend * mask_patch
            pred_sum[z : end[0], y : end[1], x : end[2]] += pred_patch * weight
            weight_sum[z : end[0], y : end[1], x : end[2]] += weight

    pred = np.divide(
        pred_sum,
        weight_sum,
        out=np.zeros_like(pred_sum),
        where=weight_sum > 0,
    )
    return pred[
        : original_shape[0], : original_shape[1], : original_shape[2], 0
    ]
