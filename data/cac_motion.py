"""単一時相gated CAC CTからnon-gated様motion artifactを合成する。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates, rotate


HEART_MASK_SIDECAR_SUFFIXES = (".mask.hdr", ".heart.mask.hdr")


def _same_path(left: Path, right: Path) -> bool:
    """存在しないpathも含め、入力画像自身をmask候補から除外する。"""
    return left.resolve(strict=False) == right.resolve(strict=False)


def resolve_heart_mask_path(
    image_hdr: str | Path,
    *,
    image_root: str | Path | None = None,
    mask_root: str | Path | None = None,
    include_local: bool = True,
    required: bool = False,
) -> Path | None:
    """heart mask sidecarを解決する。

    ``case.hdr`` に対して、画像と同じdirectoryおよび任意のmask rootから
    ``case.mask.hdr``、``case.heart.mask.hdr`` の順に探索する。外部mask rootでは
    従来の同名 ``case.hdr`` 配置も最後に許容するが、画像自身は返さない。
    """
    image_path = Path(image_hdr)
    candidates: list[Path] = []

    def add_sidecars(parent: Path, name: str):
        stem = Path(name).stem
        candidates.extend(
            parent / f"{stem}{suffix}" for suffix in HEART_MASK_SIDECAR_SUFFIXES
        )

    if include_local:
        add_sidecars(image_path.parent, image_path.name)

    if mask_root is not None:
        mask_root = Path(mask_root)
        if image_root is None:
            relative = Path(image_path.name)
        else:
            try:
                relative = image_path.relative_to(Path(image_root))
            except ValueError as error:
                raise ValueError(
                    f"imageがimage_root配下にありません: {image_path} / {image_root}"
                ) from error
        add_sidecars(mask_root / relative.parent, relative.name)
        # 旧仕様: mask rootに画像と同じ相対path・filenameで配置。
        candidates.append(mask_root / relative)

    checked: list[Path] = []
    for candidate in candidates:
        if _same_path(candidate, image_path) or candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file():
            return candidate

    if required:
        attempted = ", ".join(map(str, checked)) or "(候補なし)"
        raise FileNotFoundError(
            f"heart maskが見つかりません: image={image_path}; tried={attempted}"
        )
    return None


def _get(config, key: str, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _sample_range(rng, value_range, scale: float = 1.0) -> float:
    lo, hi = map(float, value_range)
    return float(rng.uniform(lo, hi)) * float(scale)


def make_soft_heart_mask(
    shape_zyx, spacing_zyx, config, heart_mask: np.ndarray | None = None
) -> np.ndarray:
    """外部mask、または心臓中心の楕円体からfeather済みmotion maskを作る。"""
    shape = np.asarray(shape_zyx, dtype=np.int32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if heart_mask is not None:
        if tuple(heart_mask.shape) != tuple(shape):
            raise ValueError(
                f"heart mask shape mismatch: image={tuple(shape)}, mask={heart_mask.shape}"
            )
        binary = np.asarray(heart_mask) > 0
    else:
        center = np.asarray(
            _get(config, "fallback_center_fraction_zyx", [0.5, 0.55, 0.5]),
            dtype=np.float32,
        ) * (shape - 1)
        radius_mm = np.asarray(
            _get(config, "fallback_radius_mm_zyx", [55.0, 70.0, 70.0]), dtype=np.float32
        )
        zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
        distance = (
            ((zz - center[0]) * spacing[0] / radius_mm[0]) ** 2
            + ((yy - center[1]) * spacing[1] / radius_mm[1]) ** 2
            + ((xx - center[2]) * spacing[2] / radius_mm[2]) ** 2
        )
        binary = distance <= 1.0

    feather_mm = max(float(_get(config, "mask_feather_mm", 10.0)), 0.0)
    if feather_mm == 0.0:
        return binary.astype(np.float32)
    sigma = np.maximum(feather_mm / np.maximum(spacing, 1e-6) / 2.355, 0.01)
    soft = gaussian_filter(binary.astype(np.float32), sigma=sigma, mode="nearest")
    maximum = float(soft.max())
    if maximum > 0.0:
        soft /= maximum
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def sample_motion_parameters(config, rng, severity: float) -> dict[str, float]:
    severity = float(np.clip(severity, 0.0, 1.0))
    in_plane = _sample_range(
        rng, _get(config, "in_plane_displacement_mm_range", [1.0, 8.0]), severity
    )
    direction = float(rng.uniform(0.0, 2.0 * math.pi))
    return {
        "severity": severity,
        "heart_rate_bpm": _sample_range(
            rng, _get(config, "heart_rate_bpm_range", [55.0, 100.0])
        ),
        "rotation_time_ms": _sample_range(
            rng, _get(config, "rotation_time_ms_range", [250.0, 500.0])
        ),
        "heart_rate_jitter": _sample_range(
            rng, _get(config, "heart_rate_jitter_range", [0.0, 0.08])
        ),
        "displacement_y_mm": in_plane * math.sin(direction),
        "displacement_x_mm": in_plane * math.cos(direction),
        "displacement_z_mm": _sample_range(
            rng,
            _get(config, "through_plane_displacement_mm_range", [0.0, 3.0]),
            severity,
        )
        * float(rng.choice([-1.0, 1.0])),
        "rotation_deg": _sample_range(
            rng, _get(config, "rotation_deg_range", [0.0, 5.0]), severity
        )
        * float(rng.choice([-1.0, 1.0])),
        "contraction": _sample_range(
            rng, _get(config, "contraction_range", [0.0, 0.04]), severity
        ),
        "elastic_magnitude_mm": _sample_range(
            rng, _get(config, "elastic_magnitude_mm_range", [0.0, 3.0]), severity
        ),
        "second_harmonic": _sample_range(
            rng, _get(config, "second_harmonic_range", [-0.25, 0.25])
        ),
        "harmonic_phase": float(rng.uniform(0.0, 2.0 * math.pi)),
        "start_phase": float(rng.uniform(0.0, 1.0)),
        "start_angle_deg": float(rng.uniform(0.0, 360.0)),
    }


def _trajectory_value(phase, harmonic, harmonic_phase, phase_offset=0.0):
    angle = 2.0 * math.pi * (phase + phase_offset)
    return (math.sin(angle) + harmonic * math.sin(2.0 * angle + harmonic_phase)) / (
        1.0 + abs(harmonic)
    )


def motion_state(phase: float, target_phase: float, params) -> dict[str, float]:
    """基準target phaseで恒等変換となる相対motion stateを返す。"""
    harmonic = float(params["second_harmonic"])
    harmonic_phase = float(params["harmonic_phase"])

    def relative(offset=0.0):
        return _trajectory_value(phase, harmonic, harmonic_phase, offset) - (
            _trajectory_value(target_phase, harmonic, harmonic_phase, offset)
        )

    return {
        "shift_z_mm": float(params["displacement_z_mm"]) * relative(0.17),
        "shift_y_mm": float(params["displacement_y_mm"]) * relative(0.0),
        "shift_x_mm": float(params["displacement_x_mm"]) * relative(0.08),
        "rotation_deg": float(params["rotation_deg"]) * relative(0.12),
        "contraction": float(params["contraction"]) * relative(0.25),
        "elastic_scale": relative(0.33),
    }


def generate_smooth_elastic_field_mm(
    shape_zyx, spacing_zyx, soft_mask, config, params, rng
) -> np.ndarray:
    """heart ROI内で使う、再現可能で滑らかな3D変位場をmm単位で作る。"""
    shape = np.asarray(shape_zyx, dtype=np.int32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    weights = np.asarray(soft_mask, dtype=np.float32)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError(f"shape_zyxが不正です: {shape_zyx}")
    if spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError(f"spacing_zyxが不正です: {spacing_zyx}")
    if tuple(weights.shape) != tuple(shape):
        raise ValueError(
            f"soft_mask shape mismatch: expected={tuple(shape)}, actual={weights.shape}"
        )

    magnitude_mm = max(float(params.get("elastic_magnitude_mm", 0.0)), 0.0)
    if magnitude_mm == 0.0:
        return np.zeros((3, *shape), np.float32)

    cfg_elastic = _get(config, "elastic_parallel_fbp", {})
    control_spacing = np.asarray(
        _get(cfg_elastic, "control_point_spacing_mm_zyx", [12.0, 12.0, 12.0]),
        dtype=np.float32,
    )
    axis_scale = np.asarray(
        _get(cfg_elastic, "displacement_axis_scale_zyx", [0.35, 1.0, 1.0]),
        dtype=np.float32,
    )
    if (
        control_spacing.shape != (3,)
        or axis_scale.shape != (3,)
        or np.any(control_spacing <= 0.0)
        or np.any(axis_scale < 0.0)
        or not np.any(axis_scale > 0.0)
    ):
        raise ValueError(
            "elastic control spacing/axis scaleは正の3要素にしてください: "
            f"spacing={control_spacing}, scale={axis_scale}"
        )

    extent_mm = np.maximum((shape - 1) * spacing, spacing)
    control_shape = np.maximum(
        3, np.ceil(extent_mm / control_spacing).astype(np.int32) + 3
    )
    control = rng.normal(size=(3, *control_shape)).astype(np.float32)
    control = gaussian_filter(control, sigma=(0.0, 1.0, 1.0, 1.0), mode="reflect")

    zz, yy, xx = np.meshgrid(
        np.linspace(0.0, control_shape[0] - 1, shape[0], dtype=np.float32),
        np.linspace(0.0, control_shape[1] - 1, shape[1], dtype=np.float32),
        np.linspace(0.0, control_shape[2] - 1, shape[2], dtype=np.float32),
        indexing="ij",
    )
    coordinates = np.stack([zz, yy, xx], axis=0)
    field = np.stack(
        [
            map_coordinates(
                control[axis], coordinates, order=3, mode="reflect", prefilter=True
            )
            for axis in range(3)
        ],
        axis=0,
    ).astype(np.float32)
    field *= axis_scale[:, None, None, None]

    roi = weights >= 0.25
    if not np.any(roi):
        return np.zeros((3, *shape), np.float32)
    roi_weights = np.maximum(weights[roi].astype(np.float64), 1e-6)
    for axis in range(3):
        mean = float(np.average(field[axis][roi], weights=roi_weights))
        field[axis] -= mean

    norm = np.sqrt(np.sum(field * field, axis=0))
    reference = float(np.percentile(norm[roi], 99.0))
    if not np.isfinite(reference) or reference <= 1e-6:
        return np.zeros((3, *shape), np.float32)
    field *= magnitude_mm / reference
    return field.astype(np.float32, copy=False)


def warp_cardiac_state(
    clean_hu: np.ndarray,
    spacing_zyx,
    soft_mask: np.ndarray,
    state,
    chunk_depth: int = 8,
    elastic_field_mm: np.ndarray | None = None,
) -> np.ndarray:
    """心臓mask内だけへ局所的な3D affine/elastic motionを適用する。"""
    clean = np.asarray(clean_hu, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    shape = np.asarray(clean.shape, dtype=np.int32)
    weights = np.asarray(soft_mask, dtype=np.float32)
    if clean.ndim != 3 or weights.shape != clean.shape:
        raise ValueError("clean_huとsoft_maskは同一shapeの3D volumeにしてください")
    elastic_voxels = None
    if elastic_field_mm is not None:
        elastic = np.asarray(elastic_field_mm, dtype=np.float32)
        if elastic.shape != (3, *clean.shape):
            raise ValueError(
                "elastic_field_mm shape mismatch: "
                f"expected={(3, *clean.shape)}, actual={elastic.shape}"
            )
        elastic_voxels = elastic / spacing[:, None, None, None]

    total = float(weights.sum())
    if total > 1e-6:
        center = np.array(
            [
                float(
                    (
                        weights.sum(axis=tuple(j for j in range(3) if j != axis))
                        * np.arange(clean.shape[axis], dtype=np.float32)
                    ).sum()
                    / total
                )
                for axis in range(3)
            ],
            dtype=np.float32,
        )
    else:
        center = (shape - 1) / 2.0

    shift = (
        np.array(
            [state["shift_z_mm"], state["shift_y_mm"], state["shift_x_mm"]],
            dtype=np.float32,
        )
        / spacing
    )
    angle = math.radians(float(state["rotation_deg"]))
    cosine, sine = math.cos(angle), math.sin(angle)
    scale = float(np.clip(1.0 - state["contraction"], 0.85, 1.15))

    output = np.empty_like(clean)
    for z0 in range(0, clean.shape[0], max(1, int(chunk_depth))):
        z1 = min(z0 + max(1, int(chunk_depth)), clean.shape[0])
        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1, dtype=np.float32),
            np.arange(clean.shape[1], dtype=np.float32),
            np.arange(clean.shape[2], dtype=np.float32),
            indexing="ij",
        )
        rel_z = (zz - center[0] - shift[0]) / scale
        rel_y = yy - center[1] - shift[1]
        rel_x = xx - center[2] - shift[2]
        source_y = (cosine * rel_y + sine * rel_x) / scale + center[1]
        source_x = (-sine * rel_y + cosine * rel_x) / scale + center[2]
        source_z = rel_z + center[0]

        weight = weights[z0:z1]
        delta_z = source_z - zz
        delta_y = source_y - yy
        delta_x = source_x - xx
        if elastic_voxels is not None:
            elastic_scale = float(state.get("elastic_scale", 0.0))
            delta_z += elastic_scale * elastic_voxels[0, z0:z1]
            delta_y += elastic_scale * elastic_voxels[1, z0:z1]
            delta_x += elastic_scale * elastic_voxels[2, z0:z1]
        coords = np.stack(
            [zz + weight * delta_z, yy + weight * delta_y, xx + weight * delta_x],
            axis=0,
        )
        output[z0:z1] = map_coordinates(
            clean, coords, order=1, mode="nearest", prefilter=False
        )
    return output


def _sample_phases(
    num_samples: int, config, params, rng, projection_config=None
) -> np.ndarray:
    heart_period_ms = 60000.0 / max(float(params["heart_rate_bpm"]), 1.0)
    projection_config = (
        _get(config, "parallel_fbp", {})
        if projection_config is None
        else projection_config
    )
    angular_range = float(_get(projection_config, "angular_range_deg", 180.0))
    scan_duration_ms = float(params["rotation_time_ms"]) * angular_range / 360.0
    times = (np.arange(num_samples, dtype=np.float32) + 0.5) / num_samples
    jitter = float(params["heart_rate_jitter"])
    curve = times + jitter * (times - 0.5) ** 2 * float(rng.choice([-1.0, 1.0]))
    return np.mod(
        float(params["start_phase"]) + curve * scan_duration_ms / heart_period_ms, 1.0
    )


def _make_motion_states(clean, spacing, mask, config, params, num_states, rng):
    target_phase = float(_get(config, "target_phase", 0.75))
    phases = _sample_phases(num_states, config, params, rng)
    volumes = [
        warp_cardiac_state(
            clean, spacing, mask, motion_state(float(phase), target_phase, params)
        )
        for phase in phases
    ]
    return volumes, phases


def _make_elastic_motion_states(clean, spacing, mask, config, params, num_states, rng):
    target_phase = float(_get(config, "target_phase", 0.75))
    cfg_elastic = _get(config, "elastic_parallel_fbp", {})
    phases = _sample_phases(
        num_states, config, params, rng, projection_config=cfg_elastic
    )
    elastic_field = generate_smooth_elastic_field_mm(
        clean.shape, spacing, mask, config, params, rng
    )
    volumes = [
        warp_cardiac_state(
            clean,
            spacing,
            mask,
            motion_state(float(phase), target_phase, params),
            elastic_field_mm=elastic_field,
        )
        for phase in phases
    ]
    field_norm = np.sqrt(np.sum(elastic_field * elastic_field, axis=0))
    roi = np.asarray(mask) >= 0.25
    max_displacement = float(field_norm[roi].max()) if np.any(roi) else 0.0
    return volumes, phases, max_displacement


def simulate_image_blend(clean_hu, spacing_zyx, mask, config, params, rng):
    cfg_blend = _get(config, "image_blend", {})
    num_samples = max(2, int(_get(cfg_blend, "num_samples", 8)))
    volumes, phases = _make_motion_states(
        clean_hu, spacing_zyx, mask, config, params, num_samples, rng
    )
    output = np.mean(volumes, axis=0, dtype=np.float32)
    blur_mm = _sample_range(
        rng, _get(cfg_blend, "in_plane_blur_sigma_mm_range", [0.0, 0.6])
    ) * float(params["severity"])
    if blur_mm > 0.0:
        spacing = np.asarray(spacing_zyx, dtype=np.float32)
        output = gaussian_filter(
            output, sigma=(0.0, blur_mm / spacing[1], blur_mm / spacing[2])
        )
    noise_std = _sample_range(rng, _get(cfg_blend, "noise_std_hu_range", [0.0, 20.0]))
    if noise_std > 0.0:
        output = output + rng.normal(0.0, noise_std, output.shape).astype(np.float32)
    return output.astype(np.float32), {
        "phases": [float(value) for value in phases],
        "noise_std_hu": noise_std,
        "blur_sigma_mm": blur_mm,
    }


def _pad_axial_square(volume, cval=0.0):
    y, x = volume.shape[1:]
    side = int(math.ceil(math.sqrt(y * y + x * x)))
    if side % 2:
        side += 1
    pad_y = (side - y) // 2, side - y - (side - y) // 2
    pad_x = (side - x) // 2, side - x - (side - x) // 2
    return np.pad(volume, ((0, 0), pad_y, pad_x), constant_values=cval), (pad_y, pad_x)


def _crop_axial_padding(volume, padding):
    (y0, y1), (x0, x1) = padding
    y_stop = volume.shape[1] - y1 if y1 else volume.shape[1]
    x_stop = volume.shape[2] - x1 if x1 else volume.shape[2]
    return volume[:, y0:y_stop, x0:x_stop]


def _filter_sinogram(sinogram, detector_spacing_mm: float):
    detector_count = sinogram.shape[1]
    frequencies = np.fft.fftfreq(detector_count, d=detector_spacing_mm)
    ramp = np.abs(frequencies).astype(np.float32)
    spectrum = np.fft.fft(sinogram, axis=1)
    return np.fft.ifft(spectrum * ramp[None, :, None], axis=1).real.astype(np.float32)


def _backproject_parallel(filtered, angles_rad):
    z, detector_count, num_views = filtered.shape
    reconstruction = np.zeros((z, detector_count, detector_count), np.float32)
    for index, angle in enumerate(angles_rad):
        slab = np.repeat(filtered[:, :, index, None], detector_count, axis=2)
        reconstruction += rotate(
            slab,
            -math.degrees(float(angle)),
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    reconstruction *= math.pi / max(num_views, 1)
    return reconstruction


def _linear_intensity_scale(reconstruction_mu, clean_hu, mu_water):
    clean_mu = np.clip((clean_hu / 1000.0 + 1.0) * mu_water, 0.0, None)
    # 空気を除いた全attenuation rangeでscaleだけを推定する。単純phantomでは
    # soft tissueが一様なため、CACも含めないと分散が得られない。
    valid = clean_hu > -800.0
    x = reconstruction_mu[valid].astype(np.float64)
    y = clean_mu[valid].astype(np.float64)
    if x.size < 100 or float(x.std()) < 1e-8:
        return 1.0
    x_mean, y_mean = float(x.mean()), float(y.mean())
    slope = float(np.mean((x - x_mean) * (y - y_mean)) / np.var(x))
    if not np.isfinite(slope) or slope <= 0.0:
        return 1.0
    return slope


def _project_parallel_views(padded_states, angles, spacing_x_mm, state_indices):
    z = padded_states[0].shape[0]
    detector_count = padded_states[0].shape[1]
    sinogram = np.empty((z, detector_count, len(angles)), np.float32)
    for view_index, angle in enumerate(angles):
        rotated = rotate(
            padded_states[int(state_indices[view_index])],
            math.degrees(float(angle)),
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        sinogram[:, :, view_index] = rotated.sum(axis=2) * float(spacing_x_mm)
    return sinogram


def _import_astra():
    try:
        import astra
    except ImportError as error:
        raise ImportError(
            "projection_backend=astraにはASTRA Toolboxが必要です。"
            "`pip install astra-toolbox`で導入するか、scipyを指定してください"
        ) from error
    return astra


def _project_parallel_views_astra(padded_states, angles, spacing_x_mm, state_indices):
    """同一motion stateのviewをまとめ、ASTRA objectを都度解放して投影する。"""
    astra = _import_astra()
    z = padded_states[0].shape[0]
    detector_count = padded_states[0].shape[1]
    sinogram = np.empty((z, detector_count, len(angles)), np.float32)
    volume_geometry = astra.create_vol_geom(detector_count, detector_count)

    for state_index, state in enumerate(padded_states):
        view_indices = np.flatnonzero(state_indices == state_index)
        if view_indices.size == 0:
            continue
        projection_geometry = astra.create_proj_geom(
            "parallel", 1.0, detector_count, np.asarray(angles)[view_indices]
        )
        projector_id = astra.create_projector(
            "linear", projection_geometry, volume_geometry
        )
        try:
            for slice_index in range(z):
                sinogram_id = None
                try:
                    sinogram_id, projected = astra.create_sino(
                        np.asarray(state[slice_index], np.float32), projector_id
                    )
                    sinogram[slice_index][:, view_indices] = np.asarray(
                        projected, np.float32
                    ).T * float(spacing_x_mm)
                finally:
                    if sinogram_id is not None:
                        astra.data2d.delete(sinogram_id)
        finally:
            astra.projector.delete(projector_id)
    return sinogram


def _backproject_parallel_astra(sinogram, angles_rad):
    """全viewをまとめてASTRA FBPし、確保したobjectを必ず解放する。"""
    astra = _import_astra()
    z, detector_count, _ = sinogram.shape
    reconstruction = np.empty((z, detector_count, detector_count), np.float32)
    volume_geometry = astra.create_vol_geom(detector_count, detector_count)
    projection_geometry = astra.create_proj_geom(
        "parallel", 1.0, detector_count, np.asarray(angles_rad)
    )
    projector_id = astra.create_projector(
        "linear", projection_geometry, volume_geometry
    )
    try:
        for slice_index in range(z):
            sinogram_id = reconstruction_id = algorithm_id = None
            try:
                sinogram_id = astra.data2d.create(
                    "-sino",
                    projection_geometry,
                    np.asarray(sinogram[slice_index].T, np.float32),
                )
                reconstruction_id = astra.data2d.create(
                    "-vol", volume_geometry, data=0.0
                )
                algorithm_config = astra.astra_dict("FBP")
                algorithm_config["ProjectorId"] = projector_id
                algorithm_config["ProjectionDataId"] = sinogram_id
                algorithm_config["ReconstructionDataId"] = reconstruction_id
                algorithm_config["option"] = {"FilterType": "ram-lak"}
                algorithm_id = astra.algorithm.create(algorithm_config)
                astra.algorithm.run(algorithm_id)
                reconstruction[slice_index] = astra.data2d.get(reconstruction_id)
            finally:
                if algorithm_id is not None:
                    astra.algorithm.delete(algorithm_id)
                if reconstruction_id is not None:
                    astra.data2d.delete(reconstruction_id)
                if sinogram_id is not None:
                    astra.data2d.delete(sinogram_id)
    finally:
        astra.projector.delete(projector_id)
    return reconstruction


def _project_with_backend(states, angles, spacing_x_mm, state_indices, backend):
    if backend == "scipy":
        return _project_parallel_views(states, angles, spacing_x_mm, state_indices)
    if backend == "astra":
        return _project_parallel_views_astra(
            states, angles, spacing_x_mm, state_indices
        )
    raise ValueError(f"projection_backendはscipy/astraで指定してください: {backend}")


def _reconstruct_with_backend(sinogram, angles, backend):
    if backend == "scipy":
        return _backproject_parallel(_filter_sinogram(sinogram, 1.0), angles)
    if backend == "astra":
        return _backproject_parallel_astra(sinogram, angles)
    raise ValueError(f"projection_backendはscipy/astraで指定してください: {backend}")


def simulate_parallel_fbp(clean_hu, spacing_zyx, mask, config, params, rng):
    """AX slice-wise parallel-beamによるangle依存motion artifact近似。"""
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if not np.isclose(spacing[1], spacing[2], rtol=1e-3, atol=1e-4):
        raise ValueError("parallel_fbpは等しいin-plane pixel spacingを前提とします")
    cfg_fbp = _get(config, "parallel_fbp", {})
    num_views = max(16, int(_get(cfg_fbp, "num_views", 180)))
    num_states = max(2, int(_get(config, "num_motion_states", 8)))
    angular_range = float(_get(cfg_fbp, "angular_range_deg", 180.0))
    start_angle = math.radians(float(params["start_angle_deg"]))
    angles = start_angle + np.linspace(
        0.0, math.radians(angular_range), num_views, endpoint=False
    )
    volumes, phases = _make_motion_states(
        clean_hu, spacing, mask, config, params, num_states, rng
    )

    mu_water = float(_get(cfg_fbp, "mu_water_per_mm", 0.02))
    padded_states = []
    padding = None
    for volume in volumes:
        mu = np.clip((volume / 1000.0 + 1.0) * mu_water, 0.0, None)
        padded, padding = _pad_axial_square(mu, cval=0.0)
        padded_states.append(padded)

    state_indices = np.minimum(
        num_states - 1, np.arange(num_views) * num_states // num_views
    )
    sinogram = _project_parallel_views(
        padded_states, angles, float(spacing[2]), state_indices
    )

    # 近似projector/FBPそのものの静的誤差をsourceへ学習させないよう、同じ
    # geometryでcleanを再構成し、最終的にdynamic-staticの差だけをcleanへ加える。
    clean_mu = np.clip((clean_hu / 1000.0 + 1.0) * mu_water, 0.0, None)
    padded_clean, clean_padding = _pad_axial_square(clean_mu, cval=0.0)
    if clean_padding != padding:
        raise RuntimeError("dynamic/static projection padding mismatch")
    static_sinogram = _project_parallel_views(
        [padded_clean], angles, float(spacing[2]), np.zeros(num_views, dtype=np.int32)
    )

    photon_count = _sample_range(
        rng, _get(cfg_fbp, "photon_count_range", [30000.0, 120000.0])
    )
    if bool(_get(cfg_fbp, "add_poisson_noise", True)):
        expected = photon_count * np.exp(-np.clip(sinogram, 0.0, 20.0))
        counts = rng.poisson(np.maximum(expected, 1e-3)).astype(np.float32)
        sinogram = -np.log(np.maximum(counts, 1.0) / photon_count)

    filtered = _filter_sinogram(sinogram, float(spacing[1]))
    dynamic_reconstruction = _crop_axial_padding(
        _backproject_parallel(filtered, angles), padding
    )
    static_filtered = _filter_sinogram(static_sinogram, float(spacing[1]))
    static_reconstruction = _crop_axial_padding(
        _backproject_parallel(static_filtered, angles), padding
    )
    intensity_scale = _linear_intensity_scale(static_reconstruction, clean_hu, mu_water)
    reconstructed_mu = clean_mu + intensity_scale * (
        dynamic_reconstruction - static_reconstruction
    )
    output = (reconstructed_mu / mu_water - 1.0) * 1000.0

    blur_mm = _sample_range(
        rng, _get(cfg_fbp, "in_plane_blur_sigma_mm_range", [0.0, 0.4])
    ) * float(params["severity"])
    if blur_mm > 0.0:
        output = gaussian_filter(
            output, sigma=(0.0, blur_mm / spacing[1], blur_mm / spacing[2])
        )
    return output.astype(np.float32), {
        "phases": [float(value) for value in phases],
        "photon_count": photon_count,
        "blur_sigma_mm": blur_mm,
        "fbp_intensity_scale": float(intensity_scale),
        "projection_geometry": "axial_parallel_beam_approximation",
    }


def _simulate_elastic_parallel_fbp_core(
    clean_hu, spacing_zyx, mask, config, params, rng
):
    """crop済みvolumeへelastic parallel FBPを適用する内部実装。"""
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if not np.isclose(spacing[1], spacing[2], rtol=1e-3, atol=1e-4):
        raise ValueError(
            "elastic_parallel_fbpは等しいin-plane pixel spacingを前提とします"
        )
    cfg_fbp = _get(config, "elastic_parallel_fbp", {})
    backend = str(_get(cfg_fbp, "projection_backend", "scipy")).lower()
    if backend not in ("scipy", "astra"):
        raise ValueError(
            f"elastic_parallel_fbp.projection_backendはscipy/astraです: {backend}"
        )
    num_views = max(16, int(_get(cfg_fbp, "num_views", 180)))
    num_states = max(2, int(_get(cfg_fbp, "num_motion_states", 8)))
    angular_range = float(_get(cfg_fbp, "angular_range_deg", 180.0))
    start_angle = math.radians(float(params["start_angle_deg"]))
    angles = start_angle + np.linspace(
        0.0, math.radians(angular_range), num_views, endpoint=False
    )
    volumes, phases, max_elastic_displacement = _make_elastic_motion_states(
        clean_hu, spacing, mask, config, params, num_states, rng
    )

    mu_water = float(_get(cfg_fbp, "mu_water_per_mm", 0.02))
    if not np.isfinite(mu_water) or mu_water <= 0.0:
        raise ValueError(f"mu_water_per_mmは正にしてください: {mu_water}")
    padded_states = []
    padding = None
    for volume in volumes:
        attenuation = np.clip((volume / 1000.0 + 1.0) * mu_water, 0.0, None)
        padded, padding = _pad_axial_square(attenuation, cval=0.0)
        padded_states.append(padded.astype(np.float32, copy=False))

    state_indices = np.minimum(
        num_states - 1, np.arange(num_views) * num_states // num_views
    )
    dynamic_sinogram = _project_with_backend(
        padded_states, angles, float(spacing[2]), state_indices, backend
    )

    clean_mu = np.clip((clean_hu / 1000.0 + 1.0) * mu_water, 0.0, None)
    padded_clean, clean_padding = _pad_axial_square(clean_mu, cval=0.0)
    if clean_padding != padding:
        raise RuntimeError("dynamic/static projection padding mismatch")
    static_sinogram = _project_with_backend(
        [padded_clean.astype(np.float32, copy=False)],
        angles,
        float(spacing[2]),
        np.zeros(num_views, dtype=np.int32),
        backend,
    )

    static_fraction = float(
        np.clip(_get(cfg_fbp, "static_projection_fraction", 0.0), 0.0, 1.0)
    )
    sinogram = (
        1.0 - static_fraction
    ) * dynamic_sinogram + static_fraction * static_sinogram
    photon_count = _sample_range(
        rng, _get(cfg_fbp, "photon_count_range", [30000.0, 120000.0])
    )
    if bool(_get(cfg_fbp, "add_poisson_noise", True)):
        expected = photon_count * np.exp(-np.clip(sinogram, 0.0, 20.0))
        counts = rng.poisson(np.maximum(expected, 1e-3)).astype(np.float32)
        sinogram = -np.log(np.maximum(counts, 1.0) / photon_count)

    if backend == "scipy":
        dynamic_reconstruction = _crop_axial_padding(
            _backproject_parallel(
                _filter_sinogram(sinogram, float(spacing[1])), angles
            ),
            padding,
        )
        static_reconstruction = _crop_axial_padding(
            _backproject_parallel(
                _filter_sinogram(static_sinogram, float(spacing[1])), angles
            ),
            padding,
        )
    else:
        dynamic_reconstruction = _crop_axial_padding(
            _reconstruct_with_backend(sinogram, angles, backend), padding
        )
        static_reconstruction = _crop_axial_padding(
            _reconstruct_with_backend(static_sinogram, angles, backend), padding
        )

    intensity_scale = _linear_intensity_scale(static_reconstruction, clean_hu, mu_water)
    reconstructed_mu = clean_mu + intensity_scale * (
        dynamic_reconstruction - static_reconstruction
    )
    output = (reconstructed_mu / mu_water - 1.0) * 1000.0

    blur_mm = _sample_range(
        rng, _get(cfg_fbp, "in_plane_blur_sigma_mm_range", [0.0, 0.4])
    ) * float(params["severity"])
    if blur_mm > 0.0:
        output = gaussian_filter(
            output, sigma=(0.0, blur_mm / spacing[1], blur_mm / spacing[2])
        )
    return output.astype(np.float32), {
        "phases": [float(value) for value in phases],
        "photon_count": photon_count,
        "blur_sigma_mm": blur_mm,
        "fbp_intensity_scale": float(intensity_scale),
        "static_projection_fraction": static_fraction,
        "elastic_max_displacement_mm": max_elastic_displacement,
        "projection_backend": backend,
        "projection_geometry": "axial_parallel_beam_elastic",
        "num_motion_states": num_states,
    }


def _heart_simulation_crop(mask, spacing_zyx, config):
    shape = np.asarray(mask.shape, dtype=np.int32)
    full = tuple(slice(0, int(size)) for size in shape)
    if not bool(_get(config, "crop_to_heart", True)):
        return full
    indices = np.argwhere(np.asarray(mask) >= 0.25)
    if indices.size == 0:
        return full
    margin_mm = np.asarray(
        _get(config, "simulation_crop_margin_mm_zyx", [12.0, 20.0, 20.0]),
        dtype=np.float32,
    )
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if margin_mm.shape != (3,) or np.any(margin_mm < 0.0):
        raise ValueError(
            f"simulation_crop_margin_mm_zyxは非負の3要素にしてください: {margin_mm}"
        )
    margin = np.ceil(margin_mm / spacing).astype(np.int32)
    start = np.maximum(indices.min(axis=0) - margin, 0)
    stop = np.minimum(indices.max(axis=0) + 1 + margin, shape)
    return tuple(slice(int(start[axis]), int(stop[axis])) for axis in range(3))


def _crop_boundary_taper(
    shape_zyx, spacing_zyx, taper_mm: float, tapered_axes=(True, True, True)
):
    taper = np.ones(tuple(shape_zyx), np.float32)
    if taper_mm <= 0.0:
        return taper
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    for axis, size in enumerate(shape_zyx):
        if not tapered_axes[axis]:
            continue
        width = min(int(math.ceil(taper_mm / spacing[axis])), max(0, size // 2))
        if width == 0:
            continue
        weights = np.ones(size, np.float32)
        ramp = np.linspace(0.0, 1.0, width + 1, dtype=np.float32)[:-1]
        weights[:width] = ramp
        weights[-width:] = ramp[::-1]
        reshape = [1, 1, 1]
        reshape[axis] = size
        taper *= weights.reshape(reshape)
    return taper


def simulate_elastic_parallel_fbp(clean_hu, spacing_zyx, mask, config, params, rng):
    """heart近傍を局所3D elastic変形し、state-grouped FBPを適用する。"""
    clean = np.asarray(clean_hu, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)
    crop = _heart_simulation_crop(
        weights, spacing_zyx, _get(config, "elastic_parallel_fbp", {})
    )
    cropped_clean = clean[crop]
    cropped_mask = weights[crop]
    cropped_output, details = _simulate_elastic_parallel_fbp_core(
        cropped_clean, spacing_zyx, cropped_mask, config, params, rng
    )

    crop_start = [int(item.start) for item in crop]
    crop_stop = [int(item.stop) for item in crop]
    full_crop = all(
        crop_start[axis] == 0 and crop_stop[axis] == clean.shape[axis]
        for axis in range(3)
    )
    if full_crop:
        output = cropped_output
    else:
        cfg_fbp = _get(config, "elastic_parallel_fbp", {})
        taper = _crop_boundary_taper(
            cropped_clean.shape,
            spacing_zyx,
            float(_get(cfg_fbp, "crop_boundary_taper_mm", 5.0)),
            tapered_axes=tuple(
                crop_start[axis] > 0 or crop_stop[axis] < clean.shape[axis]
                for axis in range(3)
            ),
        )
        output = clean.copy()
        output[crop] = cropped_clean + taper * (cropped_output - cropped_clean)
    details["simulation_crop_zyxzyx"] = [*crop_start, *crop_stop]
    details["simulation_crop_fraction"] = float(cropped_clean.size / clean.size)
    return output.astype(np.float32, copy=False), details


def calculate_cac_statistics(volume_hu, spacing_zyx, heart_mask=None, threshold=130.0):
    """Simulator校正用の簡易CAC統計を返す。診断用score実装ではない。"""
    volume = np.asarray(volume_hu, dtype=np.float32)
    candidates = volume >= float(threshold)
    if heart_mask is not None:
        candidates &= np.asarray(heart_mask) > 0
    components, count = label(candidates)
    voxel_volume = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    sizes = np.bincount(components.ravel())[1:] if count else np.empty(0, np.int64)
    values = volume[candidates]
    return {
        "threshold_hu": float(threshold),
        "component_count": int(count),
        "volume_mm3": float(candidates.sum() * voxel_volume),
        "largest_component_mm3": float(sizes.max() * voxel_volume)
        if sizes.size
        else 0.0,
        "peak_hu": float(values.max()) if values.size else None,
        "mean_hu": float(values.mean()) if values.size else None,
    }


def compare_cac_pair(
    source_hu, target_hu, spacing_zyx, heart_mask=None, threshold=130.0
):
    """実/合成sourceとgated targetの校正用指標を計算する。"""
    source = np.asarray(source_hu, dtype=np.float32)
    target = np.asarray(target_hu, dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError(
            f"pair shape mismatch: source={source.shape}, target={target.shape}"
        )
    roi = np.ones(source.shape, bool)
    if heart_mask is not None:
        roi = np.asarray(heart_mask) > 0
    source_mask = (source >= float(threshold)) & roi
    target_mask = (target >= float(threshold)) & roi
    intersection = int(np.count_nonzero(source_mask & target_mask))
    denominator = int(source_mask.sum() + target_mask.sum())
    dice = 2.0 * intersection / denominator if denominator else 1.0

    def _centroid(mask):
        if not np.any(mask):
            return None
        return np.argwhere(mask).mean(axis=0)

    source_center, target_center = _centroid(source_mask), _centroid(target_mask)
    centroid_distance = None
    if source_center is not None and target_center is not None:
        displacement = (source_center - target_center) * np.asarray(
            spacing_zyx, dtype=np.float64
        )
        centroid_distance = float(np.linalg.norm(displacement))

    source_stats = calculate_cac_statistics(
        source, spacing_zyx, roi, threshold=threshold
    )
    target_stats = calculate_cac_statistics(
        target, spacing_zyx, roi, threshold=threshold
    )
    target_volume = float(target_stats["volume_mm3"])
    target_peak = target_stats["peak_hu"]
    return {
        "source": source_stats,
        "target": target_stats,
        "threshold_dice": float(dice),
        "centroid_distance_mm": centroid_distance,
        "heart_mae_hu": float(np.mean(np.abs(source[roi] - target[roi]))),
        "target_cac_mae_hu": (
            float(np.mean(np.abs(source[target_mask] - target[target_mask])))
            if np.any(target_mask)
            else None
        ),
        "volume_ratio": (
            float(source_stats["volume_mm3"] / target_volume)
            if target_volume > 0.0
            else None
        ),
        "peak_hu_ratio": (
            float(source_stats["peak_hu"] / target_peak)
            if source_stats["peak_hu"] is not None
            and target_peak is not None
            and target_peak != 0.0
            else None
        ),
    }


def simulate_cac_motion(
    clean_hu: np.ndarray,
    spacing_zyx,
    config,
    rng=None,
    heart_mask: np.ndarray | None = None,
    severity: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """設定backendでnon-gated様sourceを生成し、再現用metadataを返す。"""
    rng = np.random.default_rng() if rng is None else rng
    clean = np.asarray(clean_hu, dtype=np.float32)
    if clean.ndim != 3:
        raise ValueError(f"clean_huは3D volumeにしてください: shape={clean.shape}")

    identity_probability = float(_get(config, "identity_probability", 0.0))
    identity = severity is None and rng.random() < identity_probability
    if severity is None:
        severity = _sample_range(rng, _get(config, "severity_range", [0.15, 1.0]))
    severity = 0.0 if identity else float(np.clip(severity, 0.0, 1.0))
    params = sample_motion_parameters(config, rng, severity)
    simulator = str(_get(config, "simulator", "image_blend")).lower()
    mask = make_soft_heart_mask(clean.shape, spacing_zyx, config, heart_mask)
    statistics_mask = mask >= 0.25

    if identity or severity == 0.0:
        output = clean.copy()
        details = {"identity": True, "phases": []}
    else:
        if simulator == "image_blend":
            output, details = simulate_image_blend(
                clean, spacing_zyx, mask, config, params, rng
            )
        elif simulator == "parallel_fbp":
            output, details = simulate_parallel_fbp(
                clean, spacing_zyx, mask, config, params, rng
            )
        elif simulator == "elastic_parallel_fbp":
            output, details = simulate_elastic_parallel_fbp(
                clean, spacing_zyx, mask, config, params, rng
            )
        else:
            raise ValueError(
                "cac_motion.simulatorはimage_blend/parallel_fbp/"
                "elastic_parallel_fbpで指定してください: "
                f"{simulator}"
            )
        details["identity"] = False

    if not np.all(np.isfinite(output)):
        raise FloatingPointError("CAC motion simulationがNaN/Infを生成しました")
    metadata = {
        "simulator": simulator,
        "motion": {key: float(value) for key, value in params.items()},
        **details,
        "clean_cac": calculate_cac_statistics(clean, spacing_zyx, statistics_mask),
        "simulated_cac": calculate_cac_statistics(output, spacing_zyx, statistics_mask),
    }
    return output.astype(np.float32), metadata
