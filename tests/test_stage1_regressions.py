import unittest
from pathlib import Path
from unittest import mock

import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf

import trainers.base as base_module
from callbacks.image_logger import as_axis_list, axis_display_name, axis_section_name
from data.dataloader import (
    _reconstruct_axis0,
    _sample_axis0,
    make_batch_dict,
    resolve_target_hdr_path,
)
from data.utils import AffineTransform
from losses import masked_psnr_per_sample, ssim_per_sample
from inference_utils import resize_volume_to_shape, sliding_window_predict
from trainers.base import BaseI2ITrainer
from trainers.edm import edm_sigma_schedule
from optimizer_utils import get_optimizer_iterations


class Stage1RegressionTests(unittest.TestCase):
    def test_tensorboard_axes_accept_anatomical_names(self):
        self.assertEqual(as_axis_list(["AX", "COR", "SAG"]), ["z", "y", "x"])
        self.assertEqual(as_axis_list(["z", "COR", "sagittal"]), ["z", "y", "x"])
        self.assertEqual(
            [axis_display_name(axis) for axis in ("z", "y", "x")],
            ["AX", "COR", "SAG"],
        )
        self.assertEqual(axis_section_name("Test Prediction", "z"), "Test Prediction (AX)")
        self.assertEqual(
            axis_section_name("Test Prediction", "y"), "Test Prediction (COR)"
        )
        self.assertEqual(
            axis_section_name("Test Prediction", "x"), "Test Prediction (SAG)"
        )

    def test_resolve_target_path_accepts_full_config(self):
        cfg = OmegaConf.create(
            {"data": {"mode": "paired", "target_suffix": ".target"}}
        )
        target = resolve_target_hdr_path(Path("/tmp/case.hdr"), cfg)
        self.assertEqual(target, Path("/tmp/case.target.hdr"))

    def test_edm_one_step_schedule(self):
        self.assertEqual(edm_sigma_schedule(1, 0.002, 80.0, 7.0), [80.0, 0.0])

    def test_edm_schedule_rejects_zero_steps(self):
        with self.assertRaises(ValueError):
            edm_sigma_schedule(0, 0.002, 80.0, 7.0)

    def test_disabled_gpu_augmentations_are_not_called(self):
        cfg = OmegaConf.load("conf/config.yaml")
        imgs = tf.ones((1, 4, 4, 4, 1), tf.float32) * 0.5
        msks = tf.ones_like(imgs)
        min_vals = tf.constant([0.0], tf.float32)
        max_vals = tf.constant([1.0], tf.float32)

        disabled = (
            "random_normalize",
            "random_gamma_correction",
            "apply_random_sharpness_or_gaussian_filter",
            "apply_random_gaussian_noise",
        )
        patches = [
            mock.patch.object(
                base_module,
                name,
                side_effect=AssertionError(f"disabled augmentation called: {name}"),
            )
            for name in disabled
        ]
        for patcher in patches:
            patcher.start()
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patches)])

        output = BaseI2ITrainer.gpu_aug(
            imgs, msks, min_vals, max_vals, cfg
        ).numpy()
        self.assertTrue((output == 0.5).all())

    def test_stateless_noise_is_reproducible_per_case(self):
        reference = tf.zeros((2, 4, 4, 4, 1), tf.float32)
        seeds = tf.constant([[1, 10], [1, 20]], tf.int32)
        noise1 = BaseI2ITrainer._normal_like(reference, seeds, salt=3)
        noise2 = BaseI2ITrainer._normal_like(reference, seeds, salt=3)
        self.assertTrue((noise1.numpy() == noise2.numpy()).all())
        self.assertFalse((noise1[0].numpy() == noise1[1].numpy()).all())

    def test_batch_dict_creates_stable_case_seeds(self):
        cfg = OmegaConf.load("conf/config.yaml")
        image = tf.zeros((2, 4, 4, 4, 1), tf.float32)
        scalar = tf.zeros((2,), tf.float32)
        ids = tf.constant(["case-a", "case-b"])
        data1 = make_batch_dict(
            image,
            image,
            tf.ones_like(image),
            scalar,
            scalar + 1.0,
            scalar,
            scalar + 1.0,
            ids,
            cfg,
        )
        data2 = make_batch_dict(
            image,
            image,
            tf.ones_like(image),
            scalar,
            scalar + 1.0,
            scalar,
            scalar + 1.0,
            ids,
            cfg,
        )
        np.testing.assert_array_equal(
            data1["sample_seeds"].numpy(), data2["sample_seeds"].numpy()
        )
        self.assertEqual(tuple(data1["sample_seeds"].shape), (2, 2))
        self.assertNotIn("img_ids", data1)
        self.assertTrue(all(value.dtype != tf.string for value in data1.values()))

    def test_optimizer_iterations_unwraps_loss_scale_optimizer(self):
        iterations = object()
        inner = mock.Mock(iterations=iterations, spec=["iterations"])
        outer = mock.Mock(inner_optimizer=inner, spec=["inner_optimizer"])
        self.assertIs(get_optimizer_iterations(outer), iterations)

    def test_metrics_return_one_value_per_case(self):
        target = tf.zeros((2, 12, 12, 12, 1), tf.float32)
        pred = tf.concat([tf.zeros_like(target[:1]), tf.ones_like(target[:1])], axis=0)
        mask = tf.ones_like(target)
        psnr = masked_psnr_per_sample(target, pred, mask)
        self.assertEqual(tuple(psnr.shape), (2,))
        self.assertGreater(float(psnr[0]), float(psnr[1]))

        identical_ssim = ssim_per_sample(
            target,
            target,
            msk=mask,
            axes=("z", "y", "x"),
            min_slice_mask_coverage=0.1,
        )
        self.assertTrue((identical_ssim.numpy() > 0.999).all())

    def test_sr_sampling_and_reconstruction_are_separate(self):
        volume = np.arange(9, dtype=np.float32)[:, None, None]
        samples, positions = _sample_axis0(
            volume, interval_px=2.0, phase_px=1, order=1
        )
        np.testing.assert_allclose(positions, [1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(samples[:, 0, 0], positions)

        reconstructed = _reconstruct_axis0(
            samples, positions, out_n=9, order=1, prefilter=False
        )
        self.assertEqual(reconstructed.shape, volume.shape)
        np.testing.assert_allclose(reconstructed[0], samples[0])
        np.testing.assert_allclose(reconstructed[-1], samples[-1])

    def test_affine_image_interpolation_preserves_float_output(self):
        transform = AffineTransform(
            crop_size_zyx=(4, 4, 4),
            norm_spacing_zyx=(1.0, 1.0, 1.0),
            random_rot_deg_zyx=(0, 0, 0),
            random_flip_axis_zyx=(False, False, False),
            random_scaling_range_zyx=((0.0, 0.0),) * 3,
            random_shift_rate_zyx=(0.0, 0.0, 0.0),
        )
        image = np.arange(64, dtype=np.int16).reshape(4, 4, 4)
        output = transform.apply(image, np.eye(4), order=3)
        self.assertEqual(output.dtype, np.float32)

    def test_sliding_window_blending_preserves_identity(self):
        class IdentityModel:
            @staticmethod
            def predict_step(data):
                return data["src_imgs"] * data["img_msks"]

        rng = np.random.default_rng(7)
        image = rng.uniform(size=(13, 11, 9, 1)).astype(np.float32)
        data = {
            "img": tf.constant(image),
            "img_msk": tf.ones_like(tf.constant(image)),
            "src_min_clip_val": tf.constant(0.0, tf.float32),
            "src_max_clip_val": tf.constant(1.0, tf.float32),
        }
        pred = sliding_window_predict(
            IdentityModel(),
            data,
            patch_size_zyx=(8, 8, 8),
            overlap=0.5,
            batch_size=2,
            base_seed=0,
        )
        np.testing.assert_allclose(pred, image[..., 0], atol=1e-6)

    def test_resize_volume_matches_requested_shape(self):
        image = np.ones((5, 7, 9), np.float32)
        resized = resize_volume_to_shape(
            image, (8, 6, 11), order=3, anti_alias=True
        )
        self.assertEqual(resized.shape, (8, 6, 11))
        np.testing.assert_allclose(resized, 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
