import math

import keras
import tensorflow as tf
from tensorflow import GradientTape

from losses import masked_l1_loss, masked_psnr_per_sample, ssim_per_sample

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class CACShiftTolerantRegressionTrainer(BaseI2ITrainer):
    """微小なXY位置ずれをcrop単位で許容するCAC残差回帰trainer。"""

    METRIC_NAMES = [
        "l1_loss",
        "shift_l1_loss",
        "cac_l1_loss",
        "gradient_loss",
        "source_anatomy_loss",
        "cac_mass_loss",
        "shift_mean_y_px",
        "shift_mean_x_px",
        "shift_abs_y_px",
        "shift_abs_x_px",
        "total_loss",
        "psnr",
        "ssim",
    ]

    def _normalized_hu(self, hu):
        minimum = (
            float(self.cfg.image.CT.window_level)
            - float(self.cfg.image.CT.window_width) / 2.0
        )
        width = float(self.cfg.image.CT.window_width)
        return (float(hu) - minimum) / width

    @staticmethod
    def _shift_xy(volume, dy: int, dx: int):
        """zero paddingで内容を(dy, dx)へ移動する。wrap-aroundは行わない。"""
        dy, dx = int(dy), int(dx)
        shape = tf.shape(volume)
        height, width = shape[2], shape[3]
        padded = tf.pad(
            volume,
            [
                [0, 0],
                [0, 0],
                [max(dy, 0), max(-dy, 0)],
                [max(dx, 0), max(-dx, 0)],
                [0, 0],
            ],
        )
        start_y, start_x = max(-dy, 0), max(-dx, 0)
        return tf.slice(
            padded,
            [0, 0, start_y, start_x, 0],
            [shape[0], shape[1], height, width, shape[4]],
        )

    @staticmethod
    def _gaussian_blur_xy(volume, sigma_px: float):
        sigma_px = float(sigma_px)
        if sigma_px <= 0.0:
            return volume
        channels = int(volume.shape[-1])
        radius = max(1, int(math.ceil(3.0 * sigma_px)))
        coords = tf.range(-radius, radius + 1, dtype=tf.float32)
        kernel_1d = tf.exp(-0.5 * tf.square(coords / sigma_px))
        kernel_1d /= tf.reduce_sum(kernel_1d)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = tf.reshape(kernel_2d, [2 * radius + 1, 2 * radius + 1, 1, 1])
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        kernel = tf.cast(kernel, volume.dtype)
        shape = tf.shape(volume)
        slices = tf.reshape(volume, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
        blurred = tf.nn.depthwise_conv2d(
            slices, kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        return tf.reshape(blurred, shape)

    @staticmethod
    def _masked_l1_per_sample(target, prediction, mask):
        axes = tuple(range(1, len(target.shape)))
        return tf.reduce_sum(tf.abs(target - prediction) * mask, axis=axes) / (
            tf.reduce_sum(mask, axis=axes) + 1e-6
        )

    @staticmethod
    def _high_hu_l1_per_sample(target, prediction, mask, threshold):
        high_mask = tf.cast(target >= threshold, target.dtype) * mask
        return CACShiftTolerantRegressionTrainer._masked_l1_per_sample(
            target, prediction, high_mask
        )

    @staticmethod
    def _axis_gradient_loss_per_sample(target, prediction, mask, threshold, axis):
        rank = len(target.shape)
        left = [slice(None)] * rank
        right = [slice(None)] * rank
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        target_left, target_right = target[tuple(left)], target[tuple(right)]
        pred_left, pred_right = prediction[tuple(left)], prediction[tuple(right)]
        pair_mask = mask[tuple(left)] * mask[tuple(right)]
        pair_mask *= tf.cast(
            tf.maximum(target_left, target_right) >= threshold, target.dtype
        )
        return CACShiftTolerantRegressionTrainer._masked_l1_per_sample(
            target_right - target_left, pred_right - pred_left, pair_mask
        )

    @staticmethod
    def _weighted_candidate_loss(losses, weights):
        stacked = tf.stack(losses, axis=1)
        return tf.reduce_mean(tf.reduce_sum(stacked * weights, axis=1))

    def _candidate_shifts(self):
        cfg = self.cfg.algorithm.cac_shift_tolerant_regression
        radius_xy = int(cfg.shift_radius_xy)
        radius_z = int(cfg.get("shift_radius_z", 0))
        if radius_xy < 0 or radius_xy > 4:
            raise ValueError(f"shift_radius_xyは0～4です: {radius_xy}")
        if radius_z != 0:
            raise ValueError(
                f"AX 3 mm CACではshift_radius_z=0のみ対応します: {radius_z}"
            )
        return [
            (dy, dx)
            for dy in range(-radius_xy, radius_xy + 1)
            for dx in range(-radius_xy, radius_xy + 1)
        ]

    def _shift_weights_and_candidates(self, source, target, mask):
        """低周波・低HU構造でcrop共通shiftを選択し、候補target/maskを返す。"""
        cfg = self.cfg.algorithm.cac_shift_tolerant_regression
        shifts = self._candidate_shifts()
        source = tf.cast(source, tf.float32)
        target = tf.cast(target, tf.float32)
        mask = tf.cast(mask, tf.float32)
        sigma = float(cfg.shift_selection_sigma_px)
        source_low = self._gaussian_blur_xy(source, sigma)
        target_low = self._gaussian_blur_xy(target, sigma)
        exclude_threshold = self._normalized_hu(cfg.shift_selection_max_hu)
        source_soft_mask = tf.cast(source < exclude_threshold, tf.float32)

        target_candidates = []
        mask_candidates = []
        selection_costs = []
        axes = tuple(range(1, len(source.shape)))
        shift_penalty = float(cfg.get("shift_selection_penalty", 0.0))
        if shift_penalty < 0.0:
            raise ValueError(f"shift_selection_penaltyは0以上です: {shift_penalty}")
        for dy, dx in shifts:
            shifted_target = self._shift_xy(target, dy, dx)
            shifted_target_low = self._shift_xy(target_low, dy, dx)
            shifted_mask = self._shift_xy(mask, dy, dx)
            valid_mask = mask * shifted_mask
            selection_mask = (
                valid_mask
                * source_soft_mask
                * tf.cast(shifted_target < exclude_threshold, tf.float32)
            )
            denominator = tf.reduce_sum(selection_mask, axis=axes)
            cost = tf.reduce_sum(
                tf.abs(source_low - shifted_target_low) * selection_mask, axis=axes
            ) / (denominator + 1e-6)
            fallback = tf.reduce_sum(
                tf.abs(source_low - shifted_target_low) * valid_mask, axis=axes
            ) / (tf.reduce_sum(valid_mask, axis=axes) + 1e-6)
            cost = tf.where(denominator > 1.0, cost, fallback)
            # 一様領域や同点時に探索端ではなくzero shiftを選ぶ。
            cost += shift_penalty * float(dy * dy + dx * dx)
            selection_costs.append(cost)
            target_candidates.append(shifted_target)
            mask_candidates.append(valid_mask)

        costs = tf.stop_gradient(tf.stack(selection_costs, axis=1))
        mode = str(cfg.get("shift_selection_mode", "hard")).lower()
        if mode == "hard":
            best_indices = tf.argmin(costs, axis=1, output_type=tf.int32)
            weights = tf.one_hot(best_indices, len(shifts), dtype=tf.float32)
        elif mode == "softmin":
            temperature = float(cfg.softmin_temperature)
            if temperature <= 0.0:
                raise ValueError(
                    f"softmin_temperatureは正にしてください: {temperature}"
                )
            weights = tf.nn.softmax(-costs / temperature, axis=1)
            best_indices = tf.argmin(costs, axis=1, output_type=tf.int32)
        else:
            raise ValueError(f"shift_selection_modeはhard/softminです: {mode}")
        return (
            tf.stop_gradient(weights),
            tf.stop_gradient(best_indices),
            target_candidates,
            mask_candidates,
            shifts,
        )

    @staticmethod
    def _hard_aligned_candidate(candidates, best_indices):
        stacked = tf.stack(candidates, axis=1)
        return tf.gather(stacked, best_indices, axis=1, batch_dims=1)

    def _compute_loss_and_metrics(self, logits, src_imgs, tgt_imgs, img_msks):
        cfg = self.cfg.algorithm.cac_shift_tolerant_regression
        prediction = tf.cast(self._to_image(logits, src_imgs), tf.float32)
        source = tf.cast(src_imgs, tf.float32)
        target = tf.cast(tgt_imgs, tf.float32)
        mask = tf.cast(img_msks, tf.float32)
        prediction *= mask
        prediction_clipped = tf.clip_by_value(prediction, 0.0, 1.0)
        cac_threshold = self._normalized_hu(cfg.threshold_hu)

        (shift_weights, best_indices, target_candidates, mask_candidates, shifts) = (
            self._shift_weights_and_candidates(source, target, mask)
        )

        l1 = masked_l1_loss(target, prediction, mask)
        selection_mode = str(cfg.get("shift_selection_mode", "hard")).lower()
        aligned_target = self._hard_aligned_candidate(target_candidates, best_indices)
        aligned_mask = self._hard_aligned_candidate(mask_candidates, best_indices)
        if selection_mode == "hard":
            shift_l1 = tf.reduce_mean(
                self._masked_l1_per_sample(aligned_target, prediction, aligned_mask)
            )
            cac_l1 = tf.reduce_mean(
                self._high_hu_l1_per_sample(
                    aligned_target, prediction, aligned_mask, cac_threshold
                )
            )
        else:
            shift_l1 = self._weighted_candidate_loss(
                [
                    self._masked_l1_per_sample(candidate, prediction, candidate_mask)
                    for candidate, candidate_mask in zip(
                        target_candidates, mask_candidates
                    )
                ],
                shift_weights,
            )
            cac_l1 = self._weighted_candidate_loss(
                [
                    self._high_hu_l1_per_sample(
                        candidate, prediction, candidate_mask, cac_threshold
                    )
                    for candidate, candidate_mask in zip(
                        target_candidates, mask_candidates
                    )
                ],
                shift_weights,
            )

        gradient_axes = tuple(str(axis).lower() for axis in cfg.gradient_axes)
        axis_to_dim = {"z": 1, "y": 2, "x": 3}
        invalid = [axis for axis in gradient_axes if axis not in axis_to_dim]
        if invalid:
            raise ValueError(
                f"cac_shift_tolerant_regression.gradient_axesはz/y/xです: {invalid}"
            )
        if selection_mode == "hard":
            aligned_axis_losses = [
                self._axis_gradient_loss_per_sample(
                    aligned_target,
                    prediction,
                    aligned_mask,
                    cac_threshold,
                    axis_to_dim[axis],
                )
                for axis in gradient_axes
            ]
            gradient_loss = (
                tf.reduce_mean(tf.stack(aligned_axis_losses, axis=1))
                if aligned_axis_losses
                else tf.constant(0.0, tf.float32)
            )
        else:
            candidate_gradient_losses = []
            for candidate, candidate_mask in zip(target_candidates, mask_candidates):
                axis_losses = [
                    self._axis_gradient_loss_per_sample(
                        candidate,
                        prediction,
                        candidate_mask,
                        cac_threshold,
                        axis_to_dim[axis],
                    )
                    for axis in gradient_axes
                ]
                candidate_gradient_losses.append(
                    tf.reduce_mean(tf.stack(axis_losses, axis=1), axis=1)
                    if axis_losses
                    else tf.zeros(tf.shape(source)[0], tf.float32)
                )
            gradient_loss = self._weighted_candidate_loss(
                candidate_gradient_losses, shift_weights
            )

        anatomy_threshold = self._normalized_hu(cfg.source_anatomy_max_hu)
        source_anatomy_mask = mask * tf.cast(source < anatomy_threshold, tf.float32)
        anatomy_sigma = float(cfg.source_anatomy_sigma_px)
        source_anatomy_loss = tf.reduce_mean(
            self._masked_l1_per_sample(
                self._gaussian_blur_xy(source, anatomy_sigma),
                self._gaussian_blur_xy(prediction, anatomy_sigma),
                source_anatomy_mask,
            )
        )

        axes = tuple(range(1, len(target.shape)))
        target_mass = tf.reduce_sum(
            tf.nn.relu(target - cac_threshold) * mask, axis=axes
        ) / (tf.reduce_sum(mask, axis=axes) + 1e-6)
        prediction_mass = tf.reduce_sum(
            tf.nn.relu(prediction - cac_threshold) * mask, axis=axes
        ) / (tf.reduce_sum(mask, axis=axes) + 1e-6)
        cac_mass_loss = tf.reduce_mean(tf.abs(target_mass - prediction_mass))

        total_loss = (
            float(cfg.l1_weight) * l1
            + float(cfg.shift_tolerant_l1_weight) * shift_l1
            + float(cfg.cac_shift_tolerant_weight) * cac_l1
            + float(cfg.gradient_weight) * gradient_loss
            + float(cfg.source_anatomy_weight) * source_anatomy_loss
            + float(cfg.cac_mass_weight) * cac_mass_loss
        )

        aligned_prediction = prediction_clipped * aligned_mask
        psnr = masked_psnr_per_sample(aligned_target, aligned_prediction, aligned_mask)
        ssim_values = ssim_per_sample(
            aligned_target, aligned_prediction, msk=aligned_mask
        )
        shift_values = tf.constant(shifts, dtype=tf.float32)
        selected_shifts = tf.gather(shift_values, best_indices)

        metric_values = {
            "l1_loss": l1,
            "shift_l1_loss": shift_l1,
            "cac_l1_loss": cac_l1,
            "gradient_loss": gradient_loss,
            "source_anatomy_loss": source_anatomy_loss,
            "cac_mass_loss": cac_mass_loss,
            "shift_mean_y_px": tf.reduce_mean(selected_shifts[:, 0]),
            "shift_mean_x_px": tf.reduce_mean(selected_shifts[:, 1]),
            "shift_abs_y_px": tf.reduce_mean(tf.abs(selected_shifts[:, 0])),
            "shift_abs_x_px": tf.reduce_mean(tf.abs(selected_shifts[:, 1])),
            "total_loss": total_loss,
            "psnr": psnr,
            "ssim": ssim_values,
        }
        for name, value in metric_values.items():
            self.metrics_dict[name].update_state(value)
        return total_loss

    def train_step(self, data):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)
        with GradientTape() as tape:
            logits = self([src_imgs, img_msks], training=True)
            total_loss = self._compute_loss_and_metrics(
                logits, src_imgs, tgt_imgs, img_msks
            )
            scaled_loss = self._scale_loss_for_optimizer(total_loss, self.optimizer)
        variables = self.generator.trainable_variables
        gradients = tape.gradient(scaled_loss, variables)
        self._apply_gradients(self.optimizer, gradients, variables)
        return self._get_metrics_result()
