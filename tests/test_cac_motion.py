import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from omegaconf import OmegaConf

from config_utils import load_config_with_extends
from data.cac_motion import (
    calculate_cac_statistics,
    compare_cac_pair,
    motion_state,
    resolve_heart_mask_path,
    simulate_cac_motion,
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
                "in_plane_blur_sigma_mm_range": [0.0, 0.0],
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
