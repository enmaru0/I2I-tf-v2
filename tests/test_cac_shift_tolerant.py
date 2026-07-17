"""微小な実ペア位置ずれを許容するCAC trainerのTensorFlow単体テスト。"""

from copy import deepcopy
from pathlib import Path
import unittest

import keras
import tensorflow as tf

from config_utils import load_config_with_extends
from trainers import CACShiftTolerantRegressionTrainer, build_trainer


class CACShiftTolerantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config_with_extends(Path("conf/config_cac_shift_tolerant.yaml"))

    def test_config_is_opt_in_and_keeps_real_pair_matching(self):
        self.assertEqual(self.cfg.algorithm.name, "cac_shift_tolerant_regression")
        self.assertEqual(self.cfg.data.pair_matching.method, "stem_token")
        self.assertEqual(self.cfg.algorithm.cac_regression.threshold_hu, 130.0)
        self.assertEqual(
            self.cfg.algorithm.cac_shift_tolerant_regression.shift_radius_xy, 2
        )

    def test_zero_padded_shift_does_not_wrap(self):
        volume = tf.reshape(tf.range(25, dtype=tf.float32), [1, 1, 5, 5, 1])
        shifted = CACShiftTolerantRegressionTrainer._shift_xy(volume, 1, -1)
        expected = tf.constant(
            [
                [0, 0, 0, 0, 0],
                [1, 2, 3, 4, 0],
                [6, 7, 8, 9, 0],
                [11, 12, 13, 14, 0],
                [16, 17, 18, 19, 0],
            ],
            tf.float32,
        )[None, None, ..., None]
        self.assertTrue(bool(tf.reduce_all(shifted == expected)))

    def test_selector_recovers_one_common_xy_shift(self):
        trainer = self._small_trainer()
        yy, xx = tf.meshgrid(
            tf.linspace(-1.0, 1.0, 32), tf.linspace(-1.0, 1.0, 32), indexing="ij"
        )
        source_slice = (
            0.15
            + 0.25 * tf.exp(-8.0 * ((yy + 0.2) ** 2 + (xx - 0.1) ** 2))
            + 0.08 * (xx + 1.0)
        )
        source = source_slice[None, None, ..., None]
        target = trainer._shift_xy(source, 1, -2)
        mask = tf.ones_like(source)
        _, indices, _, _, shifts = trainer._shift_weights_and_candidates(
            source, target, mask
        )
        selected = shifts[int(indices[0].numpy())]
        self.assertEqual(selected, (-1, 2))

    def test_selector_prefers_zero_shift_when_costs_are_tied(self):
        trainer = self._small_trainer()
        source = tf.fill([1, 1, 16, 16, 1], 0.2)
        mask = tf.ones_like(source)
        _, indices, _, _, shifts = trainer._shift_weights_and_candidates(
            source, source, mask
        )
        selected = shifts[int(indices[0].numpy())]
        self.assertEqual(selected, (0, 0))

    def test_train_step_and_gradient_accumulation_are_finite(self):
        trainer = self._small_trainer(gradient_accumulation_steps=2)
        trainer.compile(optimizer=keras.optimizers.Adam(1e-4))
        shape = (1, 2, 16, 16, 1)
        source = tf.random.stateless_uniform(
            shape, seed=(11, 17), minval=-100.0, maxval=500.0
        )
        batch = {
            "src_imgs": source,
            "tgt_imgs": source,
            "img_msks": tf.ones(shape, tf.float32),
            "src_min_clip_vals": tf.constant([-1000.0]),
            "src_max_clip_vals": tf.constant([2000.0]),
            "tgt_min_clip_vals": tf.constant([-1000.0]),
            "tgt_max_clip_vals": tf.constant([2000.0]),
        }
        first = trainer.train_step(batch)
        second = trainer.train_step(batch)
        for metrics in (first, second):
            self.assertTrue(
                all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in metrics.values())
            )
        self.assertEqual(int(trainer.optimizer.iterations.numpy()), 1)

    def _small_trainer(self, gradient_accumulation_steps=1):
        cfg = deepcopy(self.cfg)
        cfg.gradient_accumulation_steps = gradient_accumulation_steps
        cfg.model.unet.start_ch = 2
        cfg.model.unet.depth = 1
        cfg.model.unet.num_encode_blocks = 1
        cfg.model.unet.num_decode_blocks = 1
        trainer = build_trainer(cfg, (2, 16, 16, 1))
        trainer.cfg = cfg
        return trainer


if __name__ == "__main__":
    unittest.main()
