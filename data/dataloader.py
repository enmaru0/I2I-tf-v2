import multiprocessing
import random
import zlib
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw
from scipy.ndimage import gaussian_filter, uniform_filter1d, zoom
from scipy.signal import fftconvolve
from tqdm import tqdm

from .pairing import resolve_target_hdr_path
from .dataloader_utils import (
    load_intensity,
    load_organ_box,
    save_intensity,
    save_organ_box,
)
from .utils import (
    AffineTransform,
    add_channel_dim,
    calc_img_crop_region,
    center_within_image,
    crop_input,
    prepare_thin2thick,
    random_crop_center_within_bb,
    virtual_thick_generator,
)


def make_motion_blur_kernel(direction_zyx, length: float) -> np.ndarray:
    """
    手ブレ様の直線PSFカーネルを作る。
    direction_zyx: ブレ方向（voxel空間。正規化不要）
    length: ブレの長さ（voxel）
    直線上に等間隔にサンプル点を置き、トリリニアに堆積して正規化する。
    """
    d = np.asarray(direction_zyx, np.float32)
    d = d / (np.linalg.norm(d) + 1e-8)
    ksize = max(int(np.ceil(length)) | 1, 3)  # 奇数サイズ
    kernel = np.zeros((ksize, ksize, ksize), np.float32)
    center = (ksize - 1) / 2.0

    half = (length - 1.0) / 2.0
    num_taps = max(int(np.ceil(length) * 4), 8)
    for t in np.linspace(-half, half, num_taps):
        p = center + d * t
        i0 = np.floor(p).astype(np.int64)
        f = p - i0
        for dz in (0, 1):
            wz = f[0] if dz else 1.0 - f[0]
            for dy in (0, 1):
                wy = f[1] if dy else 1.0 - f[1]
                for dx in (0, 1):
                    wx = f[2] if dx else 1.0 - f[2]
                    z, y, x = i0[0] + dz, i0[1] + dy, i0[2] + dx
                    if 0 <= z < ksize and 0 <= y < ksize and 0 <= x < ksize:
                        kernel[z, y, x] += wz * wy * wx
    kernel /= kernel.sum() + 1e-8
    return kernel


def apply_motion_blur(img: np.ndarray, direction_zyx, length: float) -> np.ndarray:
    """直線PSFをFFT畳み込みで適用する（境界はreflectで暗化を防ぐ）"""
    kernel = make_motion_blur_kernel(direction_zyx, length)
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode="reflect")
    out = fftconvolve(padded, kernel, mode="same")
    out = out[pad:-pad, pad:-pad, pad:-pad]
    return out.astype(np.float32, copy=False)


def apply_random_contrast_augmentation(
    images: list[np.ndarray],
    cfg_contrast,
    rng=None,
    reference_index: int = 0,
    mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """同じ単調contrast変換をclean/source/targetへ共有して適用する。"""
    rng = np.random.default_rng() if rng is None else rng
    outputs = [np.asarray(image, dtype=np.float32) for image in images]
    if not bool(cfg_contrast.get("enabled", False)):
        return outputs
    if rng.random() >= float(cfg_contrast.get("p_apply", 0.5)):
        return outputs

    reference = outputs[reference_index]
    valid = np.isfinite(reference)
    if mask is not None:
        valid &= np.asarray(mask) > 0
    finite = reference[valid]
    if finite.size == 0:
        return outputs
    percentile_range = cfg_contrast.get("percentile_range", [1.0, 99.0])
    lo, hi = np.percentile(finite, percentile_range)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return outputs

    gamma = float(rng.uniform(*cfg_contrast.get("gamma_range", [0.75, 1.35])))
    scale = float(rng.uniform(*cfg_contrast.get("scale_range", [0.85, 1.2])))
    shift = float(rng.uniform(*cfg_contrast.get("shift_range", [-0.08, 0.08])))
    invert = rng.random() < float(cfg_contrast.get("invert_prob", 0.0))

    bias_field = None
    if rng.random() < float(cfg_contrast.get("bias_field_prob", 0.0)):
        sigma = float(
            rng.uniform(*cfg_contrast.get("bias_field_sigma_range", [16.0, 64.0]))
        )
        smooth = gaussian_filter(rng.normal(size=reference.shape), sigma=sigma)
        smooth = smooth - float(smooth.mean())
        smooth_std = float(smooth.std())
        if smooth_std > 1e-6:
            smooth = smooth / smooth_std
        smooth = np.clip(smooth / 3.0, -1.0, 1.0)
        bias_scale = float(
            rng.uniform(*cfg_contrast.get("bias_field_scale_range", [0.9, 1.1]))
        )
        log_amplitude = abs(np.log(bias_scale))
        if log_amplitude > 0.0:
            bias_field = np.exp(smooth * log_amplitude).astype(np.float32)

    intensity_range = max(float(hi - lo), 1e-6)
    transformed = []
    for image in outputs:
        value = np.clip((image - lo) / intensity_range, 0.0, 1.0)
        if invert:
            value = 1.0 - value
        value = np.power(value, gamma)
        if bias_field is not None and bias_field.shape == value.shape:
            value = value * bias_field
        value = np.clip((value - 0.5) * scale + 0.5 + shift, 0.0, 1.0)
        transformed.append((value * intensity_range + lo).astype(np.float32))
    return transformed


def _correlated_noise(shape, sigma: float, grain_sigma: float, rng) -> np.ndarray:
    noise = rng.normal(size=shape).astype(np.float32)
    if grain_sigma > 0.01:
        noise = gaussian_filter(noise, sigma=(0.0, grain_sigma, grain_sigma))
        noise_std = float(noise.std())
        if noise_std > 1e-10:
            noise = noise / noise_std
    return (noise * sigma).astype(np.float32, copy=False)


def simulate_rician_noise(
    clean: np.ndarray, sigma: float, grain_sigma: float = 0.0, rng=None
) -> np.ndarray:
    """MRI magnitude画像のRician noiseを生成する。"""
    rng = np.random.default_rng() if rng is None else rng
    n1 = _correlated_noise(clean.shape, sigma, grain_sigma, rng)
    n2 = _correlated_noise(clean.shape, sigma, grain_sigma, rng)
    magnitude = np.clip(clean, 0.0, None)
    return np.sqrt((magnitude + n1) ** 2 + n2**2).astype(np.float32)


def _kspace_truncate_2d(image: np.ndarray, keep_fraction: float) -> np.ndarray:
    if keep_fraction >= 1.0:
        return image.astype(np.float32, copy=False)
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    ny, nx = image.shape
    keep_y = max(1, int(ny * keep_fraction / 2.0))
    keep_x = max(1, int(nx * keep_fraction / 2.0))
    center_y, center_x = ny // 2, nx // 2
    mask = np.zeros((ny, nx), dtype=bool)
    mask[
        center_y - keep_y : center_y + keep_y, center_x - keep_x : center_x + keep_x
    ] = True
    spectrum[~mask] = 0
    return np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum))).astype(np.float32)


def simulate_kspace_noise(
    clean: np.ndarray, sigma_image: float, resolution_reduction: float = 0.0, rng=None
) -> np.ndarray:
    """slice-wise 2D FFT上で帯域制限とcomplex Gaussian noiseを適用する。"""
    rng = np.random.default_rng() if rng is None else rng
    keep_fraction = float(np.clip(1.0 - resolution_reduction, 1e-3, 1.0))
    outputs = []
    for image in clean.astype(np.float32, copy=False):
        if keep_fraction < 1.0 - 1e-4:
            image = _kspace_truncate_2d(image, keep_fraction)
        spectrum = np.fft.fft2(image.astype(np.float64))
        ny, nx = image.shape
        sigma_k = float(sigma_image) * np.sqrt(ny * nx)
        complex_noise = (
            rng.normal(size=(ny, nx)) + 1j * rng.normal(size=(ny, nx))
        ) * sigma_k
        outputs.append(np.abs(np.fft.ifft2(spectrum + complex_noise)))
    return np.stack(outputs).astype(np.float32)


def _sample_config_value(value, rng) -> float:
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__len__") and not isinstance(value, str)
    ):
        if len(value) != 2:
            raise ValueError(f"rangeは[min, max]で指定してください: {value}")
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def _resize_2d(image: np.ndarray, shape_yx) -> np.ndarray:
    """bilinear resize後にcrop/padして出力shapeを厳密に合わせる。"""
    shape_yx = tuple(int(v) for v in shape_yx)
    output = zoom(
        image,
        (shape_yx[0] / image.shape[0], shape_yx[1] / image.shape[1]),
        order=1,
        prefilter=False,
        mode="nearest",
    )
    output = output[: shape_yx[0], : shape_yx[1]]
    pad_y = shape_yx[0] - output.shape[0]
    pad_x = shape_yx[1] - output.shape[1]
    if pad_y > 0 or pad_x > 0:
        output = np.pad(output, ((0, pad_y), (0, pad_x)), mode="edge")
    return output.astype(np.float32, copy=False)


def _sample_lowfield_parameters(cfg_lowfield, rng, profile_name=None):
    profiles = list(cfg_lowfield.get("profiles", []))
    if profiles:
        candidates = profiles
        if profile_name:
            candidates = [
                profile
                for profile in profiles
                if str(profile.get("name", "")) == str(profile_name)
            ]
            if not candidates:
                raise ValueError(f"lowfield profileが見つかりません: {profile_name}")
        weights = np.asarray(
            [float(profile.get("weight", 1.0)) for profile in candidates],
            dtype=np.float64,
        )
        weights = weights / weights.sum()
        profile = candidates[int(rng.choice(len(candidates), p=weights))]
        return (
            _sample_config_value(profile.get("target_snr"), rng),
            _sample_config_value(profile.get("downsample"), rng),
            _sample_config_value(profile.get("noise_corr_mm"), rng),
            _sample_config_value(profile.get("kspace_keep"), rng),
        )
    return (
        float(rng.uniform(cfg_lowfield.target_snr_min, cfg_lowfield.target_snr_max)),
        float(rng.uniform(cfg_lowfield.downsample_min, cfg_lowfield.downsample_max)),
        float(rng.uniform(0.0, cfg_lowfield.noise_corr_mm_max)),
        float(rng.uniform(cfg_lowfield.kspace_keep_min, cfg_lowfield.kspace_keep_max)),
    )


def simulate_lowfield_noise(
    clean: np.ndarray, spacing_zyx, cfg_lowfield, rng=None, profile_name=None
) -> np.ndarray:
    """SNR・面内解像度・noise相関・k-space帯域を連動させるlow-field模擬。"""
    rng = np.random.default_rng() if rng is None else rng
    target_snr, downsample, noise_corr_mm, kspace_keep = _sample_lowfield_parameters(
        cfg_lowfield, rng, profile_name=profile_name
    )
    target_snr = max(float(target_snr), 1e-3)
    downsample = float(np.clip(downsample, 1e-3, 1.0))
    kspace_keep = float(np.clip(kspace_keep, 1e-3, 1.0))

    positive = np.clip(clean, 0.0, None)
    threshold = np.percentile(positive, 60.0)
    foreground = positive[positive > threshold]
    signal = (
        float(np.median(foreground))
        if foreground.size > 0
        else float(positive.max() + 1e-8)
    )
    sigma = signal / target_snr
    spacing_xy = float(np.mean(np.asarray(spacing_zyx, dtype=np.float32)[1:]))
    corr_fwhm_px = max(float(noise_corr_mm), 0.0) / (spacing_xy + 1e-8)

    outputs = []
    for image in clean.astype(np.float32, copy=False):
        if kspace_keep < 1.0 - 1e-4:
            image = _kspace_truncate_2d(image, kspace_keep)
        if downsample < 1.0 - 1e-4:
            anti_alias_sigma = (1.0 / downsample) / 2.3548
            image = gaussian_filter(image, sigma=anti_alias_sigma)
            small_shape = (
                max(1, int(round(image.shape[0] * downsample))),
                max(1, int(round(image.shape[1] * downsample))),
            )
            small = _resize_2d(image, small_shape)
            grain_sigma = corr_fwhm_px * downsample / 2.3548
            n1 = _correlated_noise((1,) + small.shape, sigma, grain_sigma, rng)[0]
            n2 = _correlated_noise((1,) + small.shape, sigma, grain_sigma, rng)[0]
            small = np.sqrt((np.clip(small, 0.0, None) + n1) ** 2 + n2**2)
            image = _resize_2d(small, clean.shape[1:])
        else:
            grain_sigma = corr_fwhm_px / 2.3548
            image = simulate_rician_noise(
                image[None], sigma, grain_sigma=grain_sigma, rng=rng
            )[0]
        outputs.append(image)
    return np.stack(outputs).astype(np.float32)


# through-plane SRの劣化軸（データはz,y,x順）: AX=z / COR=y / SAG=x
SR_AXIS_TO_DIM = {"AX": 0, "COR": 1, "SAG": 2}
CONTINUOUS_SR_SIGMA_FACTOR = 0.32


def _normalize_sr_interpolation(
    slice_interpolation: str, b_spline_order: int = 3
) -> tuple[int, bool]:
    """self_srの補間設定をscipy.zoomのorder/prefilterへ変換する。"""
    method = str(slice_interpolation).lower().replace("_", "-")
    if method == "linear":
        return 1, False
    if method == "spline":
        return 3, True
    if method in ("b-spline", "bspline", "b-spine", "bspine"):
        return int(b_spline_order), False
    raise NotImplementedError(slice_interpolation)


def _resize_axis0(
    vol: np.ndarray, out_n: int, order: int, prefilter: bool = True
) -> np.ndarray:
    """軸0のスライス数をout_nに変える（scipy.zoom。端数はcrop/padで厳密に合わせる）"""
    out_n = int(out_n)
    if vol.shape[0] == out_n:
        return vol.astype(np.float32, copy=False)
    order = int(order)
    if order > 1 and vol.shape[0] < 4:
        order = 1  # スライス数が少なすぎるときはスプラインを使わない
    out = zoom(
        vol,
        (out_n / vol.shape[0], 1.0, 1.0),
        order=order,
        prefilter=bool(prefilter) and order > 1,
        mode="nearest",
    )
    if out.shape[0] > out_n:
        start = (out.shape[0] - out_n) // 2
        out = out[start : start + out_n]
    elif out.shape[0] < out_n:
        pad_before = (out_n - out.shape[0]) // 2
        pad_after = out_n - out.shape[0] - pad_before
        out = np.pad(out, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return out.astype(np.float32, copy=False)


def _resize_axis0_pytorch_compatible(
    vol: np.ndarray, out_n: int, order: int = 1, prefilter: bool = True
) -> np.ndarray:
    """torch/data.pyのThroughPlaneSuperResolution._resize_zと同じzoom処理。"""
    out_n = int(out_n)
    if out_n < 1:
        raise ValueError("out_n must be >= 1")
    if vol.shape[0] == out_n:
        return vol.astype(np.float32, copy=False)

    order = int(order)
    if order > 1 and vol.shape[0] < 4:
        order = 1
    # modeを指定しないこともPyTorch版と同じ。scipy.zoom既定の
    # mode="constant", cval=0.0, grid_mode=Falseを使用する。
    out = zoom(
        vol, (out_n / vol.shape[0], 1.0, 1.0), order=order, prefilter=bool(prefilter)
    )
    if out.shape[0] > out_n:
        start = (out.shape[0] - out_n) // 2
        out = out[start : start + out_n]
    elif out.shape[0] < out_n:
        pad_before = (out_n - out.shape[0]) // 2
        pad_after = out_n - out.shape[0] - pad_before
        out = np.pad(out, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return out.astype(np.float32, copy=False)


def _simulate_pytorch_box_profile_axis0(
    axis_first: np.ndarray,
    interval_px: float,
    thickness_px: float,
    interpolation_order: int,
    interpolation_prefilter: bool,
    harmonize_slice_thickness_to_interval: bool = False,
) -> np.ndarray:
    """PyTorch版physical reductionのbox profile経路をそのまま再現する。"""
    profiled = axis_first.astype(np.float32, copy=False)
    if thickness_px > 1.0:
        size = max(1, int(round(float(thickness_px))))
        if size > 1:
            profiled = uniform_filter1d(
                profiled, size=size, axis=0, mode="nearest"
            ).astype(np.float32, copy=False)

    if bool(harmonize_slice_thickness_to_interval):
        interval_px = float(interval_px)
        thickness_px = float(thickness_px)
        if interval_px > thickness_px + 1e-6:
            add_fwhm_px = np.sqrt(
                max(interval_px * interval_px - thickness_px * thickness_px, 0.0)
            )
            harmonize_sigma_px = float(add_fwhm_px / 2.3548)
            if harmonize_sigma_px > 0.0:
                profiled = gaussian_filter(
                    profiled, sigma=(harmonize_sigma_px, 0.0, 0.0)
                ).astype(np.float32, copy=False)

    n_axis = int(axis_first.shape[0])
    low_axis = int(round((n_axis - 1) / float(interval_px))) + 1 if n_axis > 1 else 1
    low_axis = max(1, min(n_axis, low_axis))
    lowres = _resize_axis0_pytorch_compatible(
        profiled, low_axis, order=interpolation_order, prefilter=interpolation_prefilter
    )
    return _resize_axis0_pytorch_compatible(
        lowres, n_axis, order=interpolation_order, prefilter=interpolation_prefilter
    )


def _sample_axis0(
    vol: np.ndarray, interval_px: float, phase_px: int = 0, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """PSF適用済みvolumeを規則的なスライス中心で標本化する。"""
    n_axis = int(vol.shape[0])
    phase_px = int(np.clip(phase_px, 0, max(n_axis - 1, 0)))
    positions = np.arange(phase_px, n_axis, interval_px, dtype=np.float32)
    if positions.size == 0:
        positions = np.array([float(phase_px)], np.float32)

    if int(order) == 0:
        indices = np.clip(np.rint(positions).astype(np.int64), 0, n_axis - 1)
        sampled = vol[indices]
    elif int(order) == 1:
        lower = np.floor(positions).astype(np.int64)
        upper = np.minimum(lower + 1, n_axis - 1)
        weight = (positions - lower).reshape((-1, 1, 1))
        sampled = (1.0 - weight) * vol[lower] + weight * vol[upper]
    else:
        raise ValueError("self_sr.acquisition_orderは0または1を指定してください")
    return sampled.astype(np.float32, copy=False), positions


def _reconstruct_axis0(
    samples: np.ndarray, positions: np.ndarray, out_n: int, order: int, prefilter: bool
) -> np.ndarray:
    """低解像度スライス列をzoomで密なグリッドへ戻し、観測外は端値で補う。"""
    out_n = int(out_n)
    if samples.shape[0] == 1:
        return np.repeat(samples, out_n, axis=0).astype(np.float32, copy=False)

    start = int(round(float(positions[0])))
    span_n = max(2, int(round(float(positions[-1] - positions[0]))) + 1)
    dense = _resize_axis0(samples, span_n, order, prefilter)

    output = np.empty((out_n,) + samples.shape[1:], np.float32)
    output[:start] = dense[0]
    copy_end = min(start + span_n, out_n)
    output[start:copy_end] = dense[: copy_end - start]
    output[copy_end:] = dense[min(copy_end - start - 1, dense.shape[0] - 1)]
    return output


def simulate_through_plane_sr(
    img: np.ndarray,
    axis_dim: int,
    axis_spacing_mm: float,
    slice_interval_mm: float,
    slice_thickness_mm: float,
    slice_profile: str = "gaussian",
    interpolation_order: int = 1,
    interpolation_prefilter: bool = True,
    continuous_sigma_k: float = CONTINUOUS_SR_SIGMA_FACTOR,
    acquisition_order: int = 1,
    slice_phase_px: int = 0,
    box_profile_implementation: str = "tensorflow",
    pytorch_box_interpolation_prefilter: bool = True,
    pytorch_box_harmonize_thickness_to_interval: bool = False,
) -> np.ndarray:
    """
    スライス方向の低解像度撮像をシミュレートする（edm-torchの
    ThroughPlaneSuperResolution physicalモード相当）。
    1. スライスプロファイルぼかし
       - gaussian: FWHM=スライス厚
       - box: 幅=スライス厚
       - continuous: sigma=スライス間隔*k
    2. 標準TF方式はスライス間隔・位相に合わせて物理位置で標本化
       PyTorch互換box方式はtorch/data.pyと同じ枚数へzoom downsample
    3. 元グリッドへ補間で戻す（出力サイズは入力と同じ）
    """
    axis_first = np.moveaxis(img.astype(np.float32, copy=False), axis_dim, 0)
    interval_px = max(float(slice_interval_mm) / float(axis_spacing_mm), 1.0)
    thickness_px = max(float(slice_thickness_mm) / float(axis_spacing_mm), 1.0)

    slice_profile = str(slice_profile).lower()
    box_profile_implementation = str(box_profile_implementation).lower()
    if box_profile_implementation not in ("tensorflow", "pytorch"):
        raise ValueError(
            "box_profile_implementationはtensorflow/pytorchです: "
            f"{box_profile_implementation}"
        )
    if slice_profile == "box" and box_profile_implementation == "pytorch":
        upsampled = _simulate_pytorch_box_profile_axis0(
            axis_first,
            interval_px,
            thickness_px,
            interpolation_order,
            pytorch_box_interpolation_prefilter,
            pytorch_box_harmonize_thickness_to_interval,
        )
        return np.moveaxis(upsampled, 0, axis_dim).astype(np.float32, copy=False)

    if slice_profile == "none":
        pass
    elif slice_profile == "gaussian" and thickness_px <= 1.0:
        pass
    elif slice_profile == "gaussian":
        sigma = thickness_px / 2.3548  # FWHM -> sigma
        axis_first = gaussian_filter(axis_first, sigma=(sigma, 0.0, 0.0))
    elif slice_profile == "box" and thickness_px <= 1.0:
        pass
    elif slice_profile == "box":
        size = max(1, int(round(thickness_px)))
        if size > 1:
            axis_first = uniform_filter1d(axis_first, size=size, axis=0, mode="nearest")
    elif slice_profile == "continuous":
        sigma = interval_px * float(continuous_sigma_k)
        axis_first = gaussian_filter(axis_first, sigma=(sigma, 0.0, 0.0))
    else:
        raise NotImplementedError(slice_profile)

    n_axis = axis_first.shape[0]
    lowres, positions = _sample_axis0(
        axis_first, interval_px, phase_px=slice_phase_px, order=acquisition_order
    )
    upsampled = _reconstruct_axis0(
        lowres, positions, n_axis, interpolation_order, interpolation_prefilter
    )
    return np.moveaxis(upsampled, 0, axis_dim).astype(np.float32, copy=False)


def _apply_self_noise(
    clean: np.ndarray,
    intensity_range: float,
    cfg_sn,
    rng,
    is_training: bool,
    norm_spacing_zyx=(1.0, 1.0, 1.0),
) -> np.ndarray:
    degraded = clean.astype(np.float32, copy=False)
    motion_length = (
        float(rng.uniform(*cfg_sn.motion_blur.length_range))
        if is_training and rng.random() < float(cfg_sn.motion_blur.prob)
        else float(cfg_sn.motion_blur.get("val_length", 0.0))
        if not is_training
        else 0.0
    )
    if motion_length > 0.0:
        degraded = apply_motion_blur(degraded, rng.normal(size=3), motion_length)

    blur_sigma = (
        float(rng.uniform(*cfg_sn.blur_sigma_range))
        if is_training and rng.random() < float(cfg_sn.blur_prob)
        else float(cfg_sn.get("val_blur_sigma", 0.0))
        if not is_training
        else 0.0
    )
    if blur_sigma > 0.0:
        degraded = gaussian_filter(degraded, sigma=blur_sigma)

    noise_std_rel = (
        float(rng.uniform(*cfg_sn.noise_std_rel_range))
        if is_training
        else float(cfg_sn.val_noise_std_rel)
    )
    simulation = str(cfg_sn.get("simulation", "gaussian")).lower()
    if simulation == "gaussian":
        noise = rng.normal(0.0, noise_std_rel * intensity_range, degraded.shape).astype(
            np.float32
        )
        return degraded + noise
    if simulation == "rician":
        cfg_rician = cfg_sn.get("rician", {})
        signal_range = float(degraded.max() - degraded.min())
        if signal_range < 1e-8:
            signal_range = 1.0
        grain_sigma = (
            float(rng.uniform(*cfg_rician.get("grain_sigma_range", [0.0, 0.0])))
            if is_training
            else float(cfg_rician.get("val_grain_sigma", 0.0))
        )
        return simulate_rician_noise(
            degraded, noise_std_rel * signal_range, grain_sigma=grain_sigma, rng=rng
        )
    if simulation == "kspace":
        cfg_kspace = cfg_sn.get("kspace", {})
        signal_range = float(degraded.max() - degraded.min())
        if signal_range < 1e-8:
            signal_range = 1.0
        reduction = (
            float(
                rng.uniform(*cfg_kspace.get("resolution_reduction_range", [0.0, 0.0]))
            )
            if is_training
            else float(cfg_kspace.get("val_resolution_reduction", 0.0))
        )
        return simulate_kspace_noise(
            degraded,
            noise_std_rel * signal_range,
            resolution_reduction=reduction,
            rng=rng,
        )
    if simulation == "lowfield":
        cfg_lowfield = cfg_sn.lowfield
        profile_name = None
        if not is_training:
            profile_name = cfg_lowfield.get("val_profile", None) or None
        return simulate_lowfield_noise(
            degraded, norm_spacing_zyx, cfg_lowfield, rng=rng, profile_name=profile_name
        )
    raise NotImplementedError(f"未知のdenoise simulation: {simulation}")


def apply_random_self_noise(
    clean: np.ndarray,
    intensity_range: float,
    cfg_sn,
    rng=None,
    norm_spacing_zyx=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """denoise/self_noiseの学習時ランダム劣化を適用する。"""
    rng = np.random.default_rng() if rng is None else rng
    return _apply_self_noise(
        clean,
        intensity_range,
        cfg_sn,
        rng,
        is_training=True,
        norm_spacing_zyx=norm_spacing_zyx,
    )


def apply_validation_self_noise(
    clean: np.ndarray,
    intensity_range: float,
    cfg_sn,
    rng=None,
    norm_spacing_zyx=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """固定validationパラメータと症例seedで決定的なdenoise劣化を適用する。"""
    rng = np.random.default_rng(0) if rng is None else rng
    return _apply_self_noise(
        clean,
        intensity_range,
        cfg_sn,
        rng,
        is_training=False,
        norm_spacing_zyx=norm_spacing_zyx,
    )


def apply_random_self_sr(
    clean: np.ndarray, norm_spacing_zyx, cfg_sr, py_rng=None, np_rng=None
) -> np.ndarray:
    """
    self_srの学習時ランダム劣化（軸・スライス間隔・厚みをサンプリング）。
    学習ローダとprobe_sim2real.pyの両方から使う。
    """
    py_rng = random if py_rng is None else py_rng
    np_rng = np.random.default_rng() if np_rng is None else np_rng
    interpolation_order, interpolation_prefilter = _normalize_sr_interpolation(
        cfg_sr.slice_interpolation, cfg_sr.get("b_spline_order", 3)
    )
    protocols = list(cfg_sr.get("protocols", []))
    if protocols:
        weights = [float(protocol.get("weight", 1.0)) for protocol in protocols]
        protocol = py_rng.choices(protocols, weights=weights, k=1)[0]
        axis_name = str(protocol.axis).upper()
        slice_interval_mm = float(protocol.slice_interval_mm)
        slice_thickness_mm = float(
            protocol.get("slice_thickness_mm", slice_interval_mm)
        )
    else:
        axis_name = str(py_rng.choice(list(cfg_sr.axes))).upper()
        slice_interval_mm = np_rng.uniform(*cfg_sr.slice_interval_mm_range)
        slice_thickness_mm = np_rng.uniform(*cfg_sr.slice_thickness_mm_range)
    axis_dim = SR_AXIS_TO_DIM[axis_name]
    if cfg_sr.clamp_thickness_to_interval:
        slice_thickness_mm = min(slice_thickness_mm, slice_interval_mm)
    interval_px = max(float(slice_interval_mm) / float(norm_spacing_zyx[axis_dim]), 1.0)
    phase_px = 0
    pytorch_box = (
        str(cfg_sr.slice_profile).lower() == "box"
        and str(cfg_sr.get("box_profile_implementation", "tensorflow")).lower()
        == "pytorch"
    )
    if bool(cfg_sr.get("random_slice_phase", False)) and not pytorch_box:
        phase_px = int(np_rng.integers(0, max(1, int(np.ceil(interval_px)))))
    return simulate_through_plane_sr(
        clean,
        axis_dim,
        float(norm_spacing_zyx[axis_dim]),
        slice_interval_mm,
        slice_thickness_mm,
        slice_profile=cfg_sr.slice_profile,
        interpolation_order=interpolation_order,
        interpolation_prefilter=interpolation_prefilter,
        continuous_sigma_k=cfg_sr.get("continuous_sigma_k", CONTINUOUS_SR_SIGMA_FACTOR),
        acquisition_order=int(cfg_sr.get("acquisition_order", 1)),
        slice_phase_px=phase_px,
        box_profile_implementation=cfg_sr.get(
            "box_profile_implementation", "tensorflow"
        ),
        pytorch_box_interpolation_prefilter=cfg_sr.get(
            "pytorch_box_interpolation_prefilter", True
        ),
        pytorch_box_harmonize_thickness_to_interval=cfg_sr.get(
            "pytorch_box_harmonize_thickness_to_interval", False
        ),
    )


def get_clip_vals(img_hdr_path: Path, cfg) -> tuple[float, float]:
    """正規化に使うmin/max値をモダリティに応じて取得する"""
    if cfg.image.modality == "MR":
        intensity_path = img_hdr_path.with_suffix(
            f".intensity-{cfg.image.MR.min_percentile}-{cfg.image.MR.max_percentile}.txt"
        )  # save_intensityで作成
        min_val, max_val = load_intensity(intensity_path)
    else:
        window_level = float(cfg.image.CT.window_level)
        window_width = float(cfg.image.CT.window_width)
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2
    return min_val, max_val


def _apply_crop_exclude_start_slices(box_zyxzyx, img_size_zyx, cfg_aug):
    """学習時ランダムcrop候補から、元画像z方向の先頭nスライスを外す。"""
    exclude_start = int(cfg_aug.get("crop_exclude_start_slices", 0))
    box_zyxzyx = np.array(box_zyxzyx, dtype=np.float32, copy=True)
    if exclude_start <= 0:
        return box_zyxzyx

    img_z = int(img_size_zyx[0])
    if img_z <= 1:
        return box_zyxzyx

    z_min = min(max(exclude_start, 0), img_z - 1)
    box_zyxzyx[0] = max(box_zyxzyx[0], z_min)
    if box_zyxzyx[3] <= box_zyxzyx[0]:
        box_zyxzyx[3] = min(float(img_z), box_zyxzyx[0] + 1.0)
    return box_zyxzyx


def get_crop_center(
    img_hdr_path: Path,
    img_size_zyx,
    spacing_zyx,
    is_training: bool,
    cfg,
    rng=None,
    validation_patch_index: int = 0,
    validation_patch_count: int = 1,
):
    """クロップ中心を決める。検証時は画像中心、学習時はランダム"""
    crop_size_zyx = cfg.aug.crop_size_zyx
    norm_spacing_zyx = cfg.aug.affine.norm_spacing_zyx

    if not is_training:
        center = np.array(img_size_zyx, dtype=np.float32) / 2.0
        if validation_patch_count > 1:
            z_min = float(cfg.aug.get("crop_exclude_start_slices", 0))
            z_max = float(img_size_zyx[0])
            fraction = (validation_patch_index + 1.0) / (validation_patch_count + 1.0)
            center[0] = z_min + fraction * max(z_max - z_min, 0.0)
        # 固定位置（クロップが画像内に収まるように調整）
        return center_within_image(
            center, img_size_zyx, crop_size_zyx, spacing_zyx, norm_spacing_zyx
        )

    rng = random if rng is None else rng
    mode_list = list(cfg.aug.random_crop_method.keys())
    weight = list(cfg.aug.random_crop_method.values())
    mode = rng.choices(mode_list, weight, k=1)[0]

    if mode == "body":
        # 体表マスクの矩形内でランダムクロップ（.body.box.txtはsave_organ_boxで作成）
        body_box_path = img_hdr_path.with_suffix(".body.box.txt")
        box_zyxzyx = load_organ_box(body_box_path)
    elif mode == "cac":
        # CAC専用configでのみ使用するopt-in動作。gated targetから事前作成した
        # sidecarをsourceと同じ場所へ置き、陽性病変周囲をoversampleする。
        cac_box_path = img_hdr_path.with_suffix(".cac.box.txt")
        if not cac_box_path.exists():
            raise FileNotFoundError(
                f"CAC crop用sidecarがありません: {cac_box_path}\n"
                "utils/prepare_cac_boxes.pyで事前作成してください。"
            )
        box_zyxzyx = load_organ_box(cac_box_path)
    elif mode == "image":
        # 画像全体からランダムクロップ
        box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    else:
        raise NotImplementedError(mode)

    box_zyxzyx = _apply_crop_exclude_start_slices(box_zyxzyx, img_size_zyx, cfg.aug)
    return random_crop_center_within_bb(
        box_zyxzyx,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        norm_spacing_zyx,
        [0, 0, 0],
        rng=rng,
    )


def preprocess_image_np(
    img_hdr_list_with_data_name: list[bytes], sample_index: int, is_training: bool, cfg
):
    src_hdr_path, dataname = img_hdr_list_with_data_name
    dataname = dataname.decode()
    src_hdr_path = Path(src_hdr_path.decode())

    cfg_repro = cfg.get("reproducibility", {})
    base_seed = int(cfg_repro.get("seed", 0))
    path_seed = zlib.crc32(f"{dataname}:{src_hdr_path}".encode())
    sample_seed = (base_seed + path_seed + int(sample_index) * 0x9E3779B1) & 0xFFFFFFFF
    py_rng = random.Random(sample_seed)
    np_rng = np.random.default_rng(sample_seed)
    cfg_eval = cfg.get("evaluation", {})
    val_patch_count = max(1, int(cfg_eval.get("val_patches_per_volume", 1)))
    val_patch_index = 0 if is_training else int(sample_index) % val_patch_count

    # denoise/self_noiseモード: クリーン画像を1枚だけ読み込み、targetはクリーン、
    #   sourceはクリーン+合成ノイズとする（自己教師デノイジング）
    # self_srモード: クリーン画像を1枚だけ読み込み、targetはクリーン、
    #   sourceはスライス方向の低解像度シミュレーション（through-plane SR）
    # pairedモード: source(xxx.hdr)とtarget(xxx.target.hdr)を同一フォルダから読む
    # paired_dirモード: source(data_dir)とtarget(target_data_dir)を別フォルダの同名ファイルから読む
    self_noise = cfg.data.mode in ("self_noise", "denoise")
    self_sr = cfg.data.mode == "self_sr"

    # 画像リサンプルの補間次数（回転による面内ボケを抑えるためcubic推奨）。マスクは常に0
    img_interp_order = int(cfg.aug.image_interpolation_order)
    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, src_dtype, spacing_zyx = read_hdr(src_hdr_path)

    if not (self_noise or self_sr):
        tgt_hdr_path = resolve_target_hdr_path(src_hdr_path, cfg)
        tgt_size_zyx, tgt_dtype, tgt_spacing_zyx = read_hdr(tgt_hdr_path)
        # source/targetは位置合わせ済み（同一サイズ・同一スペーシング）が前提
        assert tuple(img_size_zyx) == tuple(tgt_size_zyx), (
            f"size mismatch: {src_hdr_path} {img_size_zyx} vs {tgt_size_zyx}"
        )
        assert np.allclose(spacing_zyx, tgt_spacing_zyx, atol=1e-3), (
            f"spacing mismatch: {src_hdr_path} {spacing_zyx} vs {tgt_spacing_zyx}"
        )

    # クロップ中心を決める
    crop_center_zyx = get_crop_center(
        src_hdr_path,
        img_size_zyx,
        spacing_zyx,
        is_training,
        cfg,
        rng=py_rng,
        validation_patch_index=val_patch_index,
        validation_patch_count=val_patch_count,
    )

    # アフィン変換のためのインスタンスを作成
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)

    # アフィン行列を計算（source/targetで同一の行列を使うのが最重要ポイント）
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training, rng=py_rng
    )

    # 必要な画像領域を計算
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    # 画像などは切り取って読み込むのでその分アフィン行列をシフトさせる
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    # 正規化パラメータを取得（ノイズ量の基準にも使う）
    src_min_val, src_max_val = get_clip_vals(src_hdr_path, cfg)

    # ここでは画像は読み込まずメモリマッピングをするだけ。アフィン変換で初めて画像を読む
    src_img = read_raw(
        src_hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=src_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )

    # 画像の有効領域マスク：全1配列を同じアフィン変換にかけ、回転などで
    # 生じるパディング領域(0)を検出する。BatchRenormの統計範囲や損失計算の
    # 有効領域として使う
    ones = np.ones(src_img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)

    if self_noise:
        # クリーン画像を1回だけアフィン変換し、targetとする
        clean = affine_transform.apply(src_img, affine_matrix, order=img_interp_order)
        if is_training and cfg.image.modality == "MR":
            clean = apply_random_contrast_augmentation(
                [clean],
                cfg.data.get("contrast_augmentation", {}),
                rng=np_rng,
                mask=img_msk,
            )[0]
        tgt_img = clean
        # sourceは劣化画像: モーションブラー→ガウシアンぼかし→ノイズの順に加える
        # ノイズ量は強度レンジ(正規化min-max)に対する相対量で指定する
        cfg_sn = cfg.data.self_noise
        intensity_range = float(src_max_val - src_min_val)
        norm_spacing_zyx = cfg.aug.affine.norm_spacing_zyx
        if is_training:
            # 学習時は劣化をランダムに適用（probe_sim2real.pyと共通の関数）
            src_img = apply_random_self_noise(
                clean,
                intensity_range,
                cfg_sn,
                rng=np_rng,
                norm_spacing_zyx=norm_spacing_zyx,
            )
        else:
            # 検証はエポック間・実行間で再現するようファイル名でシードを固定する
            # （組み込みhashはプロセス毎に変わるためcrc32を使う）
            seed = zlib.crc32(src_hdr_path.stem.encode())
            rng = np.random.default_rng(seed)
            src_img = apply_validation_self_noise(
                clean,
                intensity_range,
                cfg_sn,
                rng=rng,
                norm_spacing_zyx=norm_spacing_zyx,
            )
        tgt_min_val, tgt_max_val = src_min_val, src_max_val
    elif self_sr:
        # クリーン画像を1回だけアフィン変換し、targetとする
        clean = affine_transform.apply(src_img, affine_matrix, order=img_interp_order)
        if is_training and cfg.image.modality == "MR":
            clean = apply_random_contrast_augmentation(
                [clean],
                cfg.data.get("contrast_augmentation", {}),
                rng=np_rng,
                mask=img_msk,
            )[0]
        tgt_img = clean
        # sourceはスライス方向の低解像度シミュレーション
        # アフィン変換後のボクセル間隔はnorm_spacing_zyxになっている
        cfg_sr = cfg.data.self_sr
        norm_spacing_zyx = cfg.aug.affine.norm_spacing_zyx
        if is_training:
            # 学習時は劣化をランダムに適用（probe_sim2real.pyと共通の関数）
            src_img = apply_random_self_sr(
                clean, norm_spacing_zyx, cfg_sr, py_rng=py_rng, np_rng=np_rng
            )
        else:
            # 検証は固定の劣化（軸・間隔・厚みとも決定的で再現可能）
            axis_dim = SR_AXIS_TO_DIM[str(cfg_sr.val_axis).upper()]
            slice_interval_mm = float(cfg_sr.val_slice_interval_mm)
            slice_thickness_mm = float(cfg_sr.val_slice_thickness_mm)
            if cfg_sr.clamp_thickness_to_interval:
                slice_thickness_mm = min(slice_thickness_mm, slice_interval_mm)
            interpolation_order, interpolation_prefilter = _normalize_sr_interpolation(
                cfg_sr.slice_interpolation, cfg_sr.get("b_spline_order", 3)
            )
            src_img = simulate_through_plane_sr(
                clean,
                axis_dim,
                float(norm_spacing_zyx[axis_dim]),
                slice_interval_mm,
                slice_thickness_mm,
                slice_profile=cfg_sr.slice_profile,
                interpolation_order=interpolation_order,
                interpolation_prefilter=interpolation_prefilter,
                continuous_sigma_k=cfg_sr.get(
                    "continuous_sigma_k", CONTINUOUS_SR_SIGMA_FACTOR
                ),
                acquisition_order=int(cfg_sr.get("acquisition_order", 1)),
                slice_phase_px=int(cfg_sr.get("val_slice_phase_px", 0)),
                box_profile_implementation=cfg_sr.get(
                    "box_profile_implementation", "tensorflow"
                ),
                pytorch_box_interpolation_prefilter=cfg_sr.get(
                    "pytorch_box_interpolation_prefilter", True
                ),
                pytorch_box_harmonize_thickness_to_interval=cfg_sr.get(
                    "pytorch_box_harmonize_thickness_to_interval", False
                ),
            )
        tgt_min_val, tgt_max_val = src_min_val, src_max_val
    else:
        tgt_img = read_raw(
            tgt_hdr_path,
            clip_zyxzyx=img_region_zyxzyx,
            img_dtype=tgt_dtype,
            size_zyx=img_size_zyx,
            use_memmap=True,
        )
        # targetをアフィン変換
        tgt_img = affine_transform.apply(tgt_img, affine_matrix, order=img_interp_order)

        # thin->thick変換の準備（sourceのみに適用する疑似thick化）
        thin2thick_param = prepare_thin2thick(
            spacing_zyx,
            affine_transform.norm_spacing_zyx,
            crop_size_zyx,
            cfg.aug.thick2thin_rate_zyx,
            is_training=is_training,
            spacing_max_val=2,
            thickness_range=[2, 6],
            rng=py_rng,
        )

        # sourceをアフィン変換：thin->thick変換用にcrop_sizeを大きくしておく
        # なのでsourceのアフィンは一番最後にやること
        crop_size_extra = crop_size_zyx.copy()
        crop_size_extra[thin2thick_param["axis"]] += thin2thick_param["extra_slice"]
        affine_transform.crop_size_zyx = crop_size_extra  # 上書きするので注意
        src_img = affine_transform.apply(src_img, affine_matrix, order=img_interp_order)

        if thin2thick_param["apply_thin_thick"]:
            src_img = virtual_thick_generator(
                src_img,
                thin2thick_param["thickness"],
                order=1,
                axis=thin2thick_param["axis"],
            )
            src_img = crop_input(src_img, [0, 0, 0] + list(crop_size_zyx))

        if is_training and cfg.image.modality == "MR":
            src_img, tgt_img = apply_random_contrast_augmentation(
                [src_img, tgt_img],
                cfg.data.get("contrast_augmentation", {}),
                rng=np_rng,
                reference_index=1,
                mask=img_msk,
            )

        if cfg.image.share_normalization:
            # デノイズ・ぼかし修正など同一モダリティ変換ではsourceと同じ値を使う
            tgt_min_val, tgt_max_val = src_min_val, src_max_val
        else:
            tgt_min_val, tgt_max_val = get_clip_vals(tgt_hdr_path, cfg)

    # チャンネルの次元を追加
    src_img = add_channel_dim(src_img.astype(np.float32, copy=False))
    tgt_img = add_channel_dim(tgt_img.astype(np.float32, copy=False))
    img_msk = add_channel_dim(img_msk)
    return (
        src_img,
        tgt_img,
        img_msk,
        np.array(src_min_val, np.float32),
        np.array(src_max_val, np.float32),
        np.array(tgt_min_val, np.float32),
        np.array(tgt_max_val, np.float32),
        (
            str(src_hdr_path.stem)
            if is_training or val_patch_count == 1
            else f"{src_hdr_path.stem}#p{val_patch_index}"
        ).encode(),
    )


def preprocess_image(img_hdr_path_with_data_name, sample_index, is_training: bool, cfg):
    def _preprocess_image_np(img_hdr_path_with_data_name, sample_index):
        return preprocess_image_np(
            img_hdr_path_with_data_name, sample_index, is_training, cfg
        )

    (
        src_img,
        tgt_img,
        img_msk,
        src_min_clip_val,
        src_max_clip_val,
        tgt_min_clip_val,
        tgt_max_clip_val,
        img_hdr,
    ) = tf.numpy_function(
        func=_preprocess_image_np,
        inp=[img_hdr_path_with_data_name, sample_index],
        Tout=[
            tf.float32,
            tf.float32,
            tf.uint8,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.string,
        ],
    )

    # tf.numpy_functionを使ったときはset_shapeでshapeを指定する
    img_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
    src_img.set_shape(img_shape)
    tgt_img.set_shape(img_shape)
    img_msk.set_shape(img_shape)
    src_min_clip_val.set_shape(())
    src_max_clip_val.set_shape(())
    tgt_min_clip_val.set_shape(())
    tgt_max_clip_val.set_shape(())

    return (
        src_img,
        tgt_img,
        img_msk,
        src_min_clip_val,
        src_max_clip_val,
        tgt_min_clip_val,
        tgt_max_clip_val,
        img_hdr,
    )


def make_batch_dict(
    src_imgs,
    tgt_imgs,
    img_msks,
    src_min_clip_vals,
    src_max_clip_vals,
    tgt_min_clip_vals,
    tgt_max_clip_vals,
    img_hdr_list,
    cfg,
):
    """
    一般的にはモデルを GPU や TPU などのアクセラレータ上で実行している場合でも、
    tf.data パイプラインは CPU 上で実行されています。
    https://www.tensorflow.org/guide/data_performance_analysis?hl=ja#3_cpu_%E4%BD%BF%E7%94%A8%E7%8E%87%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%EF%BC%9F
    """

    cfg_repro = cfg.get("reproducibility", {})
    base_seed = int(cfg_repro.get("seed", 0))
    id_hash = tf.strings.to_hash_bucket_strong(
        img_hdr_list, num_buckets=2**31 - 1, key=[0x13579BDF, 0x2468ACE0]
    )
    sample_seeds = tf.stack(
        [tf.fill(tf.shape(id_hash), tf.cast(base_seed, tf.int64)), id_hash], axis=-1
    )

    data = dict(
        src_imgs=tf.cast(src_imgs, tf.float32),
        tgt_imgs=tf.cast(tgt_imgs, tf.float32),
        img_msks=tf.cast(img_msks, tf.float32),
        src_min_clip_vals=src_min_clip_vals,
        src_max_clip_vals=src_max_clip_vals,
        tgt_min_clip_vals=tgt_min_clip_vals,
        tgt_max_clip_vals=tgt_max_clip_vals,
        sample_seeds=tf.cast(sample_seeds, tf.int32),
    )
    if cfg.debug_dataloader:
        data["img_hdr_list"] = img_hdr_list

    return data


def create_dataloader(img_hdr_dict: dict, is_training: bool, cfg):
    """
    複数のデータセットから異なる確率で読み込むデータローダーを作成する
    ミニバッチに必ず特定のデータセットが含まれるような実装にはしていないが、それほど問題にならないはず。
    img_hdr_dict:
    e.g.
    {
       "DataSetA":
            {
                "img_hdr_list": [path1.hdr, path2.hdr, ...]  # sourceのhdrパス
                "freq": 0.8, # 80%の確率でDataSetAからサンプリング
            },
        "DataSetB":
            {
                "img_hdr_list": [path3.hdr, path4.hdr, ...]
                "freq": 0.2,
            },
    }
    """

    img_hdr_path_list = []
    for value in img_hdr_dict.values():
        img_hdr_path_list += value["img_hdr_list"]

    # 前計算が必要なものを列挙する
    with multiprocessing.Pool(cfg.num_workers) as pool:

        def _run(func, path_list, desc):
            for _ in tqdm(
                pool.imap_unordered(func, path_list), total=len(path_list), desc=desc
            ):
                pass

        # bodyクロップを使う場合のみ体表の矩形を計算しておく
        if cfg.aug.random_crop_method.body > 0:
            func = partial(save_organ_box, suffix=".body")
            _run(func, img_hdr_path_list, "saving body box")

        # MRデータはあらかじめ、min_intensityとmax_intensityを計算しておく
        if cfg.image.modality == "MR":
            func = partial(
                save_intensity,
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )
            _run(func, img_hdr_path_list, "saving source intensity")
            # denoise / self_noise / self_srではtargetはsourceと同一なので別計算は不要
            if (
                cfg.data.mode not in ("denoise", "self_noise", "self_sr")
                and not cfg.image.share_normalization
            ):
                tgt_hdr_path_list = [
                    resolve_target_hdr_path(Path(p), cfg) for p in img_hdr_path_list
                ]
                _run(func, tgt_hdr_path_list, "saving target intensity")

    cfg_repro = cfg.get("reproducibility", {})
    data_seed = int(cfg_repro.get("seed", 0))
    dataset_list = []
    frequency_list = []
    for data_name, value in img_hdr_dict.items():
        # データセットごとに処理を変えることを想定してデータセット名を付与する（処理はpreprocess_image_npで実装）
        repeat_count = 1
        if not is_training:
            cfg_eval = cfg.get("evaluation", {})
            repeat_count = max(1, int(cfg_eval.get("val_patches_per_volume", 1)))
        img_hdr_list_with_data_name = [
            (str(path), data_name)
            for path in sorted(value["img_hdr_list"])
            for _ in range(repeat_count)
        ]
        _dataset = tf.data.Dataset.from_tensor_slices(img_hdr_list_with_data_name)

        if is_training:
            # ここでrepeatしないと正しくサンプリングできない
            _dataset = _dataset.repeat()
        dataset_list.append(_dataset)
        frequency_list.append(value["freq"])
        logging.info(f"Dataset {data_name} has {len(value['img_hdr_list'])} images.")

    # データセットを結合する。学習時はここでサンプリングの重みを設定する。
    if is_training:
        dataset = tf.data.Dataset.sample_from_datasets(
            dataset_list, weights=frequency_list, seed=data_seed
        )
    else:
        dataset = dataset_list[0]
        for next_dataset in dataset_list[1:]:
            dataset = dataset.concatenate(next_dataset)

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=len(img_hdr_path_list),
            seed=data_seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.enumerate()

    def _preprocess_image(sample_index, img_hdr_path_with_data_name):
        return preprocess_image(
            img_hdr_path_with_data_name, sample_index, is_training, cfg
        )

    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=cfg.num_workers
    )  # autotuneはなんか遅かった・・・

    if not is_training:
        cfg_eval = cfg.get("evaluation", {})
        cache_setting = str(cfg_eval.get("validation_cache", ""))
        if cache_setting == "memory":
            dataset = dataset.cache()
        elif cache_setting:
            cache_path = Path(cache_setting)
            if not cache_path.is_absolute():
                cache_path = Path(cfg.exp_dir) / cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = dataset.cache(str(cache_path))

    # 学習はjit+サンプリングの都合でdrop_remainder=True。
    # 検証はdrop_remainder=Falseにして、val症例数 < batch_sizeでも
    # 0バッチにならない（余りバッチをそのまま評価する）ようにする。
    dataset = dataset.batch(cfg.batch_size, drop_remainder=is_training)

    # 他で使いやすいように辞書型で保持する
    def _make_batch_dict(*args):
        return make_batch_dict(*args, cfg)

    dataset = dataset.map(_make_batch_dict)

    # 実劣化データのDC損失用ストリームを合流させる（data.real_dc、学習時のみ）
    cfg_dc = cfg.data.get("real_dc", None)
    if is_training and cfg_dc is not None and cfg_dc.enabled:
        real_dataset = _make_real_dc_dataset(cfg)
        dataset = tf.data.Dataset.zip((dataset, real_dataset)).map(
            lambda main, real: {**main, **real}
        )

    dataset = dataset.prefetch(buffer_size=cfg.prefetch_size)

    return dataset


def preprocess_real_dc_image_np(hdr_path_bytes, sample_index, cfg):
    """
    DC損失用の実劣化画像を読み込む。
    - ランダム中心クロップ + norm_spacingへのリサンプル
    - 回転・反転・スケールなし（劣化軸をパッチ軸に揃えるため。学習側の
      幾何拡張とは独立）
    - 劣化演算子のσ[px]をヘッダ/設定から計算して返す:
        σ = sqrt(σ_profile^2 + σ_interp^2)
        σ_profile = スライス厚px / 2.3548 (FWHM→σ)
        σ_interp  = interp_sigma_factor * スライス間隔px（間引き→補間の実効平滑化の近似）
    """
    cfg_dc = cfg.data.real_dc
    hdr_path = Path(hdr_path_bytes.decode())
    cfg_repro = cfg.get("reproducibility", {})
    base_seed = int(cfg_repro.get("seed", 0))
    path_seed = zlib.crc32(str(hdr_path).encode())
    sample_seed = (base_seed + path_seed + int(sample_index) * 0x9E3779B1) & 0xFFFFFFFF
    py_rng = random.Random(sample_seed)
    img_interp_order = int(cfg.aug.image_interpolation_order)
    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, img_dtype, spacing_zyx = read_hdr(hdr_path)

    axis_dim = SR_AXIS_TO_DIM[str(cfg_dc.axis).upper()]
    interval_mm = cfg_dc.slice_interval_mm
    interval_mm = (
        float(spacing_zyx[axis_dim]) if interval_mm is None else float(interval_mm)
    )
    thickness_mm = cfg_dc.slice_thickness_mm
    thickness_mm = interval_mm if thickness_mm is None else float(thickness_mm)
    axis_spacing = float(cfg.aug.affine.norm_spacing_zyx[axis_dim])
    thickness_px = max(thickness_mm / axis_spacing, 1.0)
    interval_px = max(interval_mm / axis_spacing, 1.0)
    sigma_profile = thickness_px / 2.3548 if thickness_px > 1.0 else 0.0
    sigma_interp = (
        float(cfg_dc.interp_sigma_factor) * interval_px if interval_px > 1.0 else 0.0
    )
    sigma_px = float(np.sqrt(sigma_profile**2 + sigma_interp**2))

    # ランダム中心クロップ（回転なしの決定的アフィン）
    box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    box_zyxzyx = _apply_crop_exclude_start_slices(box_zyxzyx, img_size_zyx, cfg.aug)
    crop_center_zyx = random_crop_center_within_bb(
        box_zyxzyx,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        cfg.aug.affine.norm_spacing_zyx,
        [0, 0, 0],
        rng=py_rng,
    )
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training=False
    )
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    min_val, max_val = get_clip_vals(hdr_path, cfg)
    img = read_raw(
        hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=img_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )
    ones = np.ones(img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)
    img = affine_transform.apply(img, affine_matrix, order=img_interp_order)

    return (
        add_channel_dim(img.astype(np.float32, copy=False)),
        add_channel_dim(img_msk),
        np.array(min_val, np.float32),
        np.array(max_val, np.float32),
        np.array(sigma_px, np.float32),
    )


def _make_real_dc_dataset(cfg):
    """DC損失用の実劣化データストリーム（無限リピート・バッチ済み辞書）を作る"""
    cfg_dc = cfg.data.real_dc
    input_dir = Path(cfg_dc.input_dir)
    hdr_list = [
        p
        for p in sorted(input_dir.glob("*.hdr"))
        if ".mask" not in p.name and not p.stem.endswith(cfg.data.target_suffix)
    ]
    assert len(hdr_list) > 0, f"real_dc.input_dirに画像がありません: {input_dir}"
    logging.info(f"real_dc: {len(hdr_list)} volumes from {input_dir}")

    if cfg.image.modality == "MR":
        for hdr_path in hdr_list:
            save_intensity(
                str(hdr_path),
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )

    def _preprocess(sample_index, hdr_path_bytes):
        def _np_func(hdr_path_bytes, sample_index):
            return preprocess_real_dc_image_np(hdr_path_bytes, sample_index, cfg)

        img, msk, min_val, max_val, sigma_px = tf.numpy_function(
            func=_np_func,
            inp=[hdr_path_bytes, sample_index],
            Tout=[tf.float32, tf.uint8, tf.float32, tf.float32, tf.float32],
        )
        img_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
        img.set_shape(img_shape)
        msk.set_shape(img_shape)
        min_val.set_shape(())
        max_val.set_shape(())
        sigma_px.set_shape(())
        return img, msk, min_val, max_val, sigma_px

    cfg_repro = cfg.get("reproducibility", {})
    data_seed = int(cfg_repro.get("seed", 0))
    dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in hdr_list])
    dataset = dataset.repeat().shuffle(buffer_size=len(hdr_list), seed=data_seed)
    dataset = dataset.enumerate()
    dataset = dataset.map(_preprocess, num_parallel_calls=cfg.num_workers)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)

    def _make_dict(img, msk, min_val, max_val, sigma_px):
        return dict(
            real_imgs=tf.cast(img, tf.float32),
            real_msks=tf.cast(msk, tf.float32),
            real_min_clip_vals=min_val,
            real_max_clip_vals=max_val,
            real_sigma_px=sigma_px,
        )

    return dataset.map(_make_dict)


def preprocess_test_image_np(hdr_path: Path, cfg):
    """
    正解なしのテスト入力画像を、検証時と同じ前処理で読み込む。
    - 画像中心の固定クロップ + norm_spacing_zyxへのリサンプル（線形補間）
    - 有効領域マスクの生成、正規化用clip値の取得
    スライス方向に解像度が低い入力（thickスライス等）は、このリサンプルで
    等方グリッドへ補間されてからモデルに入る。self_srの学習時sourceは
    「劣化→間引き→元グリッドへ補間」で作られるため、テスト入力もここで
    同じグリッド・同じ補間に揃うことになる。
    """
    hdr_path = Path(hdr_path)
    img_interp_order = int(cfg.aug.image_interpolation_order)
    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, src_dtype, spacing_zyx = read_hdr(hdr_path)

    # 検証時と同じ：画像中心の固定クロップ
    crop_center_zyx = center_within_image(
        np.array(img_size_zyx) / 2,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        cfg.aug.affine.norm_spacing_zyx,
    )
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training=False
    )
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    min_val, max_val = get_clip_vals(hdr_path, cfg)

    src_img = read_raw(
        hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=src_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )
    ones = np.ones(src_img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)
    src_img = affine_transform.apply(src_img, affine_matrix, order=img_interp_order)

    return (
        add_channel_dim(src_img.astype(np.float32)),
        add_channel_dim(img_msk.astype(np.float32)),
        np.float32(min_val),
        np.float32(max_val),
        hdr_path.stem,
    )


def create_test_batch(cfg):
    """
    test.input_dirの正解なし入力をまとめて1バッチのdict（predict_step互換）にする。
    症例数は少ない想定なので学習開始前に一度だけ実行してメモリに保持する。
    戻り値: (data_dict, 症例名リスト)
    """
    test_dir = Path(cfg.test.input_dir)
    assert test_dir.exists(), f"test.input_dirが存在しません: {test_dir}"
    hdr_list = [p for p in sorted(test_dir.glob("*.hdr")) if ".mask" not in p.name]
    if cfg.test.max_items > 0:
        hdr_list = hdr_list[: cfg.test.max_items]
    assert len(hdr_list) > 0, f"テスト入力(.hdr)が見つかりません: {test_dir}"

    # MRは正規化用のintensityファイルを事前計算しておく（学習データと同じ処理）
    if cfg.image.modality == "MR":
        for hdr_path in hdr_list:
            save_intensity(
                str(hdr_path),
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )

    srcs, msks, min_vals, max_vals, names = [], [], [], [], []
    for hdr_path in hdr_list:
        src, msk, min_val, max_val, name = preprocess_test_image_np(hdr_path, cfg)
        srcs.append(src)
        msks.append(msk)
        min_vals.append(min_val)
        max_vals.append(max_val)
        names.append(name)
        logging.info(f"Test input loaded: {name} (clip=[{min_val:.1f}, {max_val:.1f}])")

    data = dict(
        src_imgs=tf.constant(np.stack(srcs), tf.float32),
        img_msks=tf.constant(np.stack(msks), tf.float32),
        src_min_clip_vals=tf.constant(np.stack(min_vals), tf.float32),
        src_max_clip_vals=tf.constant(np.stack(max_vals), tf.float32),
    )
    cfg_repro = cfg.get("reproducibility", {})
    base_seed = int(cfg_repro.get("seed", 0))
    name_tensor = tf.constant(names, tf.string)
    id_hash = tf.strings.to_hash_bucket_strong(
        name_tensor, num_buckets=2**31 - 1, key=[0x13579BDF, 0x2468ACE0]
    )
    data["sample_seeds"] = tf.cast(
        tf.stack(
            [tf.fill(tf.shape(id_hash), tf.cast(base_seed, tf.int64)), id_hash], axis=-1
        ),
        tf.int32,
    )
    return data, names
