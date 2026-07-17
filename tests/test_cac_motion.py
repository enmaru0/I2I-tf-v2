import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from config_utils import load_config_with_extends
from data.cac_motion import (
    _centered_state_coefficients,
    _make_centered_heart_states,
    _recenter_elastic_motion_states,
    _sample_zero_mean_calcium_shifts,
    _soft_calcium_components,
    calculate_cac_statistics,
    compare_cac_pair,
    _project_parallel_views_astra,
    generate_smooth_elastic_field_mm,
    motion_state,
    resolve_heart_mask_path,
    simulate_cac_motion,
    simulate_elastic_parallel_fbp,
    warp_cardiac_state,
)
from data.pairing import extract_pair_key, resolve_target_hdr_path


def _small_config(simulator):
    return OmegaConf.create(
        {
            "simulator": simulator,
            "severity_range": [0.5, 0.5],
            "identity_probability": 0.0,
            "num_motion_states": 2,
            "target_phase": 0.75,
            "heart_rate_bpm_range": [70.0, 70.0],
            "rotation_time_ms_range": [300.0, 300.0],
            "heart_rate_jitter_range": [0.0, 0.0],
            "in_plane_displacement_mm_range": [2.0, 2.0],
            "through_plane_displacement_mm_range": [0.0, 0.0],
            "rotation_deg_range": [1.0, 1.0],
            "contraction_range": [0.01, 0.01],
            "elastic_magnitude_mm_range": [2.0, 2.0],
            "second_harmonic_range": [0.0, 0.0],
            "mask_feather_mm": 1.0,
            "fallback_center_fraction_zyx": [0.5, 0.5, 0.5],
            "fallback_radius_mm_zyx": [8.0, 8.0, 8.0],
            "image_blend": {
                "num_samples": 3,
                "noise_std_hu_range": [0.0, 0.0],
                "in_plane_blur_sigma_mm_range": [0.0, 0.0],
            },
            "parallel_fbp": {
                "num_views": 16,
                "angular_range_deg": 180.0,
                "mu_water_per_mm": 0.02,
                "photon_count_range": [100000.0, 100000.0],
                "add_poisson_noise": False,
                "severity_power": 1.0,
                "motion_profile_gamma": 1.0,
                "in_plane_blur_sigma_mm_range": [0.0, 0.0],
            },
            "elastic_parallel_fbp": {
                "projection_backend": "scipy",
                "projection_spacing_mm": 1.0,
                "residual_interpolation_order": 1,
                "num_views": 16,
                "num_motion_states": 2,
                "angular_range_deg": 180.0,
                "mu_water_per_mm": 0.02,
                "photon_count_range": [100000.0, 100000.0],
                "add_poisson_noise": False,
                "static_projection_fraction": 0.0,
                "in_plane_blur_sigma_mm_range": [0.0, 0.0],
                "crop_to_heart": False,
                "simulation_crop_margin_mm_zyx": [3.0, 3.0, 3.0],
                "crop_boundary_taper_mm": 1.0,
                "control_point_spacing_mm_zyx": [6.0, 6.0, 6.0],
                "displacement_axis_scale_zyx": [0.25, 1.0, 1.0],
            },
            "calcium_local_fbp": {
                "projection_backend": "scipy",
                "projection_spacing_mm": 1.0,
                "residual_interpolation_order": 1,
                "num_views": 16,
                "num_motion_states": 4,
                "angular_range_deg": 180.0,
                "mu_water_per_mm": 0.02,
                "photon_count_range": [100000.0, 100000.0],
                "add_poisson_noise": False,
                "calcium_motion_amplitude_mm_range": [1.0, 1.0],
                "through_plane_amplitude_mm_range": [0.0, 0.0],
                "motion_direction_deg_range": [0.0, 0.0],
                "second_harmonic_range": [0.0, 0.0],
                "zero_mean_trajectory": True,
                "calcium_mask_lower_hu": 100.0,
                "calcium_mask_upper_hu": 200.0,
                "calcium_mask_feather_mm": 0.0,
                "calcium_mask_dilation_mm": 0.0,
                "native_calcium_blur_sigma_mm_range": [0.4, 0.4],
                "conserve_calcium_mass": True,
                "crop_to_heart": False,
                "simulation_crop_margin_mm_zyx": [3.0, 3.0, 3.0],
                "crop_boundary_taper_mm": 1.0,
            },
            "centered_heart_fbp": {
                "projection_backend": "scipy",
                "projection_spacing_mm": 1.0,
                "residual_interpolation_order": 1,
                "num_views": 20,
                "num_motion_states": 5,
                "angular_range_deg": 180.0,
                "mu_water_per_mm": 0.02,
                "photon_count_range": [100000.0, 100000.0],
                "add_poisson_noise": False,
                "in_plane_displacement_mm_range": [1.0, 1.0],
                "through_plane_displacement_mm_range": [0.0, 0.0],
                "motion_direction_deg_range": [0.0, 0.0],
                "rotation_deg_range": [0.5, 0.5],
                "contraction_range": [0.005, 0.005],
                "elastic_magnitude_mm_range": [0.5, 0.5],
                "control_point_spacing_mm_zyx": [6.0, 6.0, 6.0],
                "displacement_axis_scale_zyx": [0.0, 1.0, 1.0],
                "calcium_extra_motion_mm_range": [0.3, 0.3],
                "calcium_extra_direction_deg_range": [90.0, 90.0],
                "calcium_mask_lower_hu": 100.0,
                "calcium_mask_upper_hu": 200.0,
                "calcium_mask_feather_mm": 0.0,
                "calcium_mask_dilation_mm": 0.0,
                "native_heart_blur_sigma_mm_range": [0.0, 0.0],
                "native_calcium_blur_sigma_mm_range": [0.2, 0.2],
                "conserve_calcium_mass": True,
                "crop_to_heart": False,
                "simulation_crop_margin_mm_zyx": [3.0, 3.0, 3.0],
                "crop_boundary_taper_mm": 1.0,
            },
        }
    )


def _phantom():
    shape = (4, 24, 24)
    _, yy, xx = np.indices(shape)
    volume = np.full(shape, -1000.0, np.float32)
    volume[(yy - 12) ** 2 + (xx - 12) ** 2 <= 9**2] = 40.0
    volume[(yy - 12) ** 2 + (xx - 16) ** 2 <= 2**2] = 600.0
    mask = (yy - 12) ** 2 + (xx - 12) ** 2 <= 10**2
    return volume, mask


class CACMotionTests(unittest.TestCase):
    def test_astra_projection_groups_views_and_releases_objects(self):
        class FakeAstra:
            def __init__(self):
                self.next_id = 1
                self.projectors = {}
                self.data_ids = set()
                self.create_sino_calls = 0
                self.projector = SimpleNamespace(delete=self.delete_projector)
                self.data2d = SimpleNamespace(delete=self.delete_data)

            @staticmethod
            def create_vol_geom(height, width):
                return height, width

            @staticmethod
            def create_proj_geom(kind, spacing, detector_count, angles):
                return {
                    "kind": kind,
                    "spacing": spacing,
                    "detector_count": detector_count,
                    "angles": np.asarray(angles),
                }

            def create_projector(self, kind, projection_geometry, volume_geometry):
                identifier = self.next_id
                self.next_id += 1
                self.projectors[identifier] = projection_geometry
                return identifier

            def create_sino(self, image, projector_id):
                self.create_sino_calls += 1
                identifier = self.next_id
                self.next_id += 1
                self.data_ids.add(identifier)
                angles = self.projectors[projector_id]["angles"]
                projection = np.repeat(image.sum(axis=1)[None], len(angles), axis=0)
                return identifier, projection

            def delete_projector(self, identifier):
                self.projectors.pop(identifier)

            def delete_data(self, identifier):
                self.data_ids.remove(identifier)

        fake_astra = FakeAstra()
        states = [np.ones((1, 4, 4), np.float32), np.full((1, 4, 4), 2.0, np.float32)]
        with patch.dict(sys.modules, {"astra": fake_astra}):
            sinogram = _project_parallel_views_astra(
                states,
                np.linspace(0.0, np.pi, 4, endpoint=False),
                0.5,
                np.array([0, 0, 1, 1]),
            )
        np.testing.assert_allclose(sinogram[0, :, :2], 2.0)
        np.testing.assert_allclose(sinogram[0, :, 2:], 4.0)
        self.assertEqual(fake_astra.create_sino_calls, 2)
        self.assertFalse(fake_astra.projectors)
        self.assertFalse(fake_astra.data_ids)

    def test_real_cac_config_matches_first_underscore_token(self):
        cfg = load_config_with_extends(Path("conf/config_cac_real.yaml"))
        self.assertEqual(cfg.data.pair_matching.method, "stem_token")
        self.assertEqual(extract_pair_key(Path("P001_non_gated.hdr"), cfg), "P001")

    def test_patient_id_pair_matching_resolves_different_filenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            target_root = root / "target"
            relative_parent = Path("train") / "cac"
            source = source_root / relative_parent / "P001_non_gated.hdr"
            target = target_root / relative_parent / "P001_gated_75pct.hdr"
            source.parent.mkdir(parents=True)
            target.parent.mkdir(parents=True)
            source.touch()
            target.touch()
            cfg = load_config_with_extends(Path("conf/config_cac_real.yaml"))
            cfg.data_dir = source_root
            cfg.data.target_data_dir = target_root
            self.assertEqual(resolve_target_hdr_path(source, cfg), target)

    def test_patient_id_pair_matching_rejects_ambiguous_targets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            target_root = root / "target"
            source = source_root / "train" / "cac" / "P001_non_gated.hdr"
            targets = [
                target_root / "train" / "cac" / "P001_gated_a.hdr",
                target_root / "train" / "cac" / "P001_gated_b.hdr",
            ]
            source.parent.mkdir(parents=True)
            targets[0].parent.mkdir(parents=True)
            source.touch()
            for target in targets:
                target.touch()
            cfg = load_config_with_extends(Path("conf/config_cac_real.yaml"))
            cfg.data_dir = source_root
            cfg.data.target_data_dir = target_root
            with self.assertRaisesRegex(ValueError, "targetが複数"):
                resolve_target_hdr_path(source, cfg)

    def test_heart_mask_sidecar_is_found_next_to_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "nested" / "case.hdr"
            mask = root / "nested" / "case.mask.hdr"
            image.parent.mkdir()
            image.touch()
            mask.touch()
            self.assertEqual(resolve_heart_mask_path(image), mask)

    def test_primary_mask_sidecar_precedes_heart_mask_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "case.hdr"
            primary = root / "case.mask.hdr"
            alias = root / "case.heart.mask.hdr"
            for path in (image, primary, alias):
                path.touch()
            self.assertEqual(resolve_heart_mask_path(image), primary)

    def test_external_mask_root_preserves_relative_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            mask_root = root / "masks"
            image = image_root / "split" / "case.hdr"
            mask = mask_root / "split" / "case.mask.hdr"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            image.touch()
            mask.touch()
            self.assertEqual(
                resolve_heart_mask_path(
                    image, image_root=image_root, mask_root=mask_root
                ),
                mask,
            )

    def test_external_mask_root_keeps_legacy_same_name_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            mask_root = root / "masks"
            image = image_root / "case.hdr"
            legacy_mask = mask_root / "case.hdr"
            image_root.mkdir()
            mask_root.mkdir()
            image.touch()
            legacy_mask.touch()
            self.assertEqual(
                resolve_heart_mask_path(
                    image, image_root=image_root, mask_root=mask_root
                ),
                legacy_mask,
            )

    def test_image_itself_is_never_resolved_as_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "case.hdr"
            image.touch()
            self.assertIsNone(
                resolve_heart_mask_path(
                    image, image_root=root, mask_root=root, include_local=False
                )
            )

    def test_cac_config_extends_default_without_modifying_it(self):
        cfg = load_config_with_extends(Path("conf/config_cac.yaml"))
        self.assertEqual(cfg.data.mode, "paired_dir")
        self.assertEqual(list(cfg.aug.affine.norm_spacing_zyx), [3.0, 0.5, 0.5])
        self.assertEqual(cfg.algorithm.name, "cac_regression")
        self.assertEqual(cfg.data.pair_matching.method, "exact")
        self.assertIn("edm_karras", cfg.algorithm)
        self.assertNotIn("extends", cfg)

    def test_reference_phase_is_identity_state(self):
        params = {
            "second_harmonic": 0.2,
            "harmonic_phase": 0.3,
            "displacement_z_mm": 2.0,
            "displacement_y_mm": 4.0,
            "displacement_x_mm": 3.0,
            "rotation_deg": 2.0,
            "contraction": 0.02,
        }
        state = motion_state(0.75, 0.75, params)
        for value in state.values():
            self.assertAlmostEqual(value, 0.0)

    def test_image_blend_is_deterministic_and_localized(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("image_blend")
        first, _ = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(12),
            heart_mask=heart_mask,
            severity=0.7,
        )
        second, _ = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(12),
            heart_mask=heart_mask,
            severity=0.7,
        )
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertGreater(float(np.mean(np.abs(first - clean))), 0.0)

    def test_parallel_fbp_is_finite_and_preserves_shape(self):
        clean, heart_mask = _phantom()
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            _small_config("parallel_fbp"),
            rng=np.random.default_rng(7),
            heart_mask=heart_mask,
            severity=0.5,
        )
        self.assertEqual(output.shape, clean.shape)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertEqual(
            metadata["projection_geometry"], "axial_parallel_beam_approximation"
        )

    def test_elastic_field_is_seeded_and_uses_mm_amplitude(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("elastic_parallel_fbp")
        params = {"elastic_magnitude_mm": 2.0}
        first = generate_smooth_elastic_field_mm(
            clean.shape,
            [3.0, 0.5, 0.5],
            heart_mask.astype(np.float32),
            cfg,
            params,
            np.random.default_rng(5),
        )
        second = generate_smooth_elastic_field_mm(
            clean.shape,
            [3.0, 0.5, 0.5],
            heart_mask.astype(np.float32),
            cfg,
            params,
            np.random.default_rng(5),
        )
        np.testing.assert_array_equal(first, second)
        norm = np.sqrt(np.sum(first * first, axis=0))
        self.assertAlmostEqual(
            float(np.percentile(norm[heart_mask], 99.0)), 2.0, places=4
        )

    def test_elastic_warp_does_not_move_hard_mask_exterior(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("elastic_parallel_fbp")
        field = generate_smooth_elastic_field_mm(
            clean.shape,
            [3.0, 0.5, 0.5],
            heart_mask.astype(np.float32),
            cfg,
            {"elastic_magnitude_mm": 2.0},
            np.random.default_rng(3),
        )
        state = {
            "shift_z_mm": 0.0,
            "shift_y_mm": 0.0,
            "shift_x_mm": 0.0,
            "rotation_deg": 0.0,
            "contraction": 0.0,
            "elastic_scale": 1.0,
        }
        warped = warp_cardiac_state(
            clean,
            [3.0, 0.5, 0.5],
            heart_mask.astype(np.float32),
            state,
            elastic_field_mm=field,
        )
        np.testing.assert_array_equal(warped[~heart_mask], clean[~heart_mask])
        self.assertGreater(
            float(np.mean(np.abs(warped[heart_mask] - clean[heart_mask]))), 0.0
        )

    def test_elastic_motion_recenter_has_zero_view_weighted_mean_and_limit(self):
        states = [
            {
                "shift_z_mm": 0.0,
                "shift_y_mm": 0.0,
                "shift_x_mm": shift,
                "rotation_deg": shift,
                "contraction": shift * 0.01,
                "elastic_scale": elastic,
            }
            for shift, elastic in ((4.0, 2.0), (0.0, 0.0), (-2.0, -1.0))
        ]
        state_indices = np.asarray([0, 0, 0, 1, 1, 2], np.int32)
        unchanged, metadata = _recenter_elastic_motion_states(
            states,
            state_indices,
            OmegaConf.create({"recenter_motion_field": False}),
            elastic_max_displacement_mm=2.0,
        )
        self.assertEqual(unchanged, states)
        self.assertFalse(metadata["motion_recentered"])
        cfg = OmegaConf.create(
            {
                "recenter_motion_field": True,
                "recenter_mode": "view_weighted_mean",
                "recenter_affine": True,
                "recenter_elastic": True,
                "recenter_spread_gain": 1.5,
                "max_centered_displacement_mm": 2.5,
                "max_centered_rotation_deg": 1.5,
                "max_centered_contraction": 0.02,
            }
        )
        centered, metadata = _recenter_elastic_motion_states(
            states, state_indices, cfg, elastic_max_displacement_mm=2.0
        )
        counts = np.bincount(state_indices, minlength=len(states))
        for key in (
            "shift_z_mm",
            "shift_y_mm",
            "shift_x_mm",
            "rotation_deg",
            "contraction",
            "elastic_scale",
        ):
            self.assertAlmostEqual(
                float(np.average([state[key] for state in centered], weights=counts)),
                0.0,
                places=6,
            )
            self.assertAlmostEqual(
                metadata["motion_post_recenter_view_weighted_mean"][key], 0.0, places=6
            )
        self.assertLess(metadata["motion_recenter_displacement_scale"], 1.0)
        self.assertEqual(metadata["motion_recenter_spread_gain"], 1.5)
        self.assertLessEqual(metadata["centered_max_nominal_displacement_mm"], 2.5)

    def test_elastic_parallel_fbp_recenter_is_opt_in_and_reported(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("elastic_parallel_fbp")
        cfg.elastic_parallel_fbp.recenter_motion_field = True
        cfg.elastic_parallel_fbp.recenter_mode = "view_weighted_mean"
        cfg.elastic_parallel_fbp.recenter_affine = True
        cfg.elastic_parallel_fbp.recenter_elastic = True
        cfg.elastic_parallel_fbp.recenter_spread_gain = 1.5
        cfg.elastic_parallel_fbp.max_centered_displacement_mm = 2.5
        cfg.elastic_parallel_fbp.max_centered_rotation_deg = 1.5
        cfg.elastic_parallel_fbp.max_centered_contraction = 0.02
        cfg.elastic_parallel_fbp.native_blur_severity_power = 1.0
        cfg.elastic_parallel_fbp.native_heart_blur_sigma_mm_range = [0.3, 0.3]
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(31),
            heart_mask=heart_mask,
            severity=0.8,
        )
        self.assertEqual(output.shape, clean.shape)
        self.assertTrue(metadata["motion_recentered"])
        self.assertEqual(metadata["motion_recenter_spread_gain"], 1.5)
        self.assertGreater(metadata["native_heart_blur_sigma_mm"], 0.0)
        for value in metadata["motion_post_recenter_view_weighted_mean"].values():
            self.assertAlmostEqual(value, 0.0, places=6)
        metrics = compare_cac_pair(
            output, clean, [3.0, 0.5, 0.5], heart_mask=heart_mask
        )
        self.assertLess(metrics["centroid_distance_mm"], 0.5)

    def test_elastic_parallel_fbp_is_deterministic(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("elastic_parallel_fbp")
        first, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(17),
            heart_mask=heart_mask,
            severity=0.6,
        )
        second, _ = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(17),
            heart_mask=heart_mask,
            severity=0.6,
        )
        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, clean.shape)
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertEqual(metadata["projection_backend"], "scipy")
        self.assertEqual(metadata["projection_geometry"], "axial_parallel_beam_elastic")
        self.assertGreater(metadata["elastic_max_displacement_mm"], 0.0)

    def test_projection_grid_resizes_only_residual_back_to_native_xy(self):
        _, yy, xx = np.indices((2, 8, 10))
        clean = ((yy + xx) % 2).astype(np.float32) * 100.0
        mask = np.ones(clean.shape, np.float32)
        cfg = _small_config("elastic_parallel_fbp")
        cfg.elastic_parallel_fbp.crop_to_heart = False
        captured = {}

        def fake_core(projection_clean, spacing, projection_mask, *args):
            captured["shape"] = projection_clean.shape
            captured["spacing"] = np.asarray(spacing)
            self.assertEqual(projection_mask.shape, projection_clean.shape)
            return projection_clean + 25.0, {"projection_backend": "scipy"}

        with patch("data.cac_motion._simulate_elastic_parallel_fbp_core", fake_core):
            output, metadata = simulate_elastic_parallel_fbp(
                clean,
                [3.0, 0.5, 0.5],
                mask,
                cfg,
                {"severity": 0.5},
                np.random.default_rng(1),
            )
        np.testing.assert_allclose(output, clean + 25.0, atol=1e-5)
        self.assertEqual(captured["shape"], (2, 5, 5))
        np.testing.assert_allclose(captured["spacing"], [3.0, 1.0, 1.0])
        self.assertEqual(metadata["projection_shape_zyx"], [2, 5, 5])
        self.assertTrue(metadata["native_residual_fusion"])

    def test_elastic_projection_crop_preserves_exterior(self):
        clean, _ = _phantom()
        _, yy, xx = np.indices(clean.shape)
        heart_mask = (yy - 12) ** 2 + (xx - 12) ** 2 <= 4**2
        cfg = _small_config("elastic_parallel_fbp")
        cfg.elastic_parallel_fbp.crop_to_heart = True
        cfg.elastic_parallel_fbp.simulation_crop_margin_mm_zyx = [0.0, 1.0, 1.0]
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(21),
            heart_mask=heart_mask,
            severity=0.6,
        )
        z0, y0, x0, z1, y1, x1 = metadata["simulation_crop_zyxzyx"]
        outside = np.ones(clean.shape, bool)
        outside[z0:z1, y0:y1, x0:x1] = False
        np.testing.assert_array_equal(output[outside], clean[outside])
        self.assertLess(metadata["simulation_crop_fraction"], 1.0)

    def test_soft_calcium_components_preserve_clean_sum(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("calcium_local_fbp").calcium_local_fbp
        background, excess, weight = _soft_calcium_components(
            clean, [3.0, 0.5, 0.5], heart_mask, cfg
        )
        np.testing.assert_allclose(background + excess, clean, atol=1e-6)
        self.assertEqual(float(excess[clean <= 100.0].max()), 0.0)
        self.assertGreater(float(excess[clean == 600.0].mean()), 0.0)
        self.assertGreater(float(weight[clean == 600.0].mean()), 0.99)

    def test_calcium_shifts_are_view_weighted_zero_mean(self):
        cfg = _small_config("calcium_local_fbp").calcium_local_fbp
        phases = np.asarray([0.1, 0.2, 0.3, 0.4], np.float32)
        state_indices = np.asarray([0, 0, 0, 1, 1, 2, 3], np.int32)
        shifts, metadata = _sample_zero_mean_calcium_shifts(
            phases, state_indices, cfg, {"severity": 1.0}, np.random.default_rng(3)
        )
        counts = np.bincount(state_indices, minlength=len(phases))
        np.testing.assert_allclose(
            np.average(shifts, axis=0, weights=counts), 0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            metadata["calcium_shift_weighted_mean_zyx_mm"], 0.0, atol=1e-6
        )
        self.assertGreater(float(np.max(np.abs(shifts[:, 2]))), 0.9)

    def test_calcium_local_fbp_changes_cac_without_global_deformation(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("calcium_local_fbp")
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(17),
            heart_mask=heart_mask,
            severity=1.0,
        )
        self.assertTrue(metadata["calcium_selective_motion"])
        self.assertEqual(
            metadata["projection_geometry"], "axial_parallel_beam_calcium_local"
        )
        self.assertGreater(metadata["calcium_weight_voxels"], 0)
        self.assertGreater(
            float(np.mean(np.abs(output[clean >= 130.0] - clean[clean >= 130.0]))), 0.0
        )
        np.testing.assert_allclose(
            metadata["calcium_shift_weighted_mean_zyx_mm"], 0.0, atol=1e-6
        )
        metrics = compare_cac_pair(
            output, clean, [3.0, 0.5, 0.5], heart_mask=heart_mask
        )
        self.assertLess(metrics["centroid_distance_mm"], 0.5)

        no_calcium = np.minimum(clean, 40.0)
        static_output, _ = simulate_cac_motion(
            no_calcium,
            [3.0, 0.5, 0.5],
            cfg,
            rng=np.random.default_rng(17),
            heart_mask=heart_mask,
            severity=1.0,
        )
        np.testing.assert_allclose(static_output, no_calcium, atol=1e-5)

    def test_centered_coefficients_include_identity_and_zero_view_mean(self):
        state_indices = np.repeat(np.arange(5, dtype=np.int32), 4)
        coefficients, counts = _centered_state_coefficients(5, state_indices)
        np.testing.assert_allclose(coefficients, [-1.0, -0.5, 0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(np.average(coefficients, weights=counts)), 0.0)
        baseline = coefficients.copy()
        uneven = np.asarray([0, 0, 1, 2, 3, 4], np.int32)
        coefficients, counts = _centered_state_coefficients(5, uneven)
        self.assertEqual(float(coefficients[2]), 0.0)
        self.assertAlmostEqual(float(np.average(coefficients, weights=counts)), 0.0)
        stronger, _ = _centered_state_coefficients(5, state_indices, profile_gamma=0.5)
        self.assertGreater(abs(float(stronger[1])), abs(float(baseline[1])))

    def test_centered_states_keep_exact_gated_middle_state(self):
        clean, heart_mask = _phantom()
        cfg = _small_config("centered_heart_fbp").centered_heart_fbp
        state_indices = np.repeat(np.arange(5, dtype=np.int32), 4)
        states, metadata = _make_centered_heart_states(
            clean,
            [3.0, 0.5, 0.5],
            heart_mask.astype(np.float32),
            cfg,
            {"severity": 1.0},
            state_indices,
            np.random.default_rng(23),
        )
        np.testing.assert_allclose(states[2], clean, atol=1e-5)
        self.assertAlmostEqual(metadata["centered_coefficient_view_weighted_mean"], 0.0)

    def test_centered_heart_fbp_blurs_whole_heart_without_large_cac_shift(self):
        clean, heart_mask = _phantom()
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            _small_config("centered_heart_fbp"),
            rng=np.random.default_rng(29),
            heart_mask=heart_mask,
            severity=1.0,
        )
        self.assertTrue(metadata["centered_whole_heart_motion"])
        self.assertEqual(
            metadata["projection_geometry"], "axial_parallel_beam_centered_heart"
        )
        self.assertAlmostEqual(
            metadata["centered_coefficient_view_weighted_mean"], 0.0, places=6
        )
        soft_tissue = (clean == 40.0) & heart_mask
        self.assertGreater(
            float(np.mean(np.abs(output[soft_tissue] - clean[soft_tissue]))), 0.0
        )
        metrics = compare_cac_pair(
            output, clean, [3.0, 0.5, 0.5], heart_mask=heart_mask
        )
        self.assertLess(metrics["centroid_distance_mm"], 0.5)

    def test_cac_statistics_are_roi_limited(self):
        clean, heart_mask = _phantom()
        stats = calculate_cac_statistics(clean, [3.0, 0.5, 0.5], heart_mask=heart_mask)
        self.assertGreater(stats["volume_mm3"], 0.0)
        self.assertEqual(stats["peak_hu"], 600.0)

    def test_pair_comparison_detects_identity(self):
        clean, heart_mask = _phantom()
        metrics = compare_cac_pair(clean, clean, [3.0, 0.5, 0.5], heart_mask=heart_mask)
        self.assertEqual(metrics["threshold_dice"], 1.0)
        self.assertEqual(metrics["centroid_distance_mm"], 0.0)
        self.assertEqual(metrics["volume_ratio"], 1.0)

    def test_zero_severity_returns_exact_identity(self):
        clean, heart_mask = _phantom()
        output, metadata = simulate_cac_motion(
            clean,
            [3.0, 0.5, 0.5],
            _small_config("parallel_fbp"),
            rng=np.random.default_rng(2),
            heart_mask=heart_mask,
            severity=0.0,
        )
        np.testing.assert_array_equal(output, clean)
        self.assertTrue(metadata["identity"])


if __name__ == "__main__":
    unittest.main()
