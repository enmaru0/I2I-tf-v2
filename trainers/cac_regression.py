import keras
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_l1_loss, masked_psnr_per_sample, ssim_per_sample

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class CACRegressionTrainer(BaseI2ITrainer):
    """通常L1にCAC高HU領域と面内edgeの保存lossを加えた残差回帰。"""

    METRIC_NAMES = [
        "l1_loss",
        "cac_l1_loss",
        "gradient_loss",
        "total_loss",
        "psnr",
        "ssim",
    ]

    def _normalized_cac_threshold(self):
        cfg = self.cfg.algorithm.cac_regression
        minimum = float(self.cfg.image.CT.window_level) - float(
            self.cfg.image.CT.window_width
        ) / 2.0
        width = float(self.cfg.image.CT.window_width)
        return (float(cfg.threshold_hu) - minimum) / width

    @staticmethod
    def _high_hu_l1(target, prediction, mask, threshold):
        high_mask = ops.cast(target >= threshold, target.dtype) * mask
        return ops.sum(ops.abs(target - prediction) * high_mask) / (
            ops.sum(high_mask) + 1e-6
        )

    @staticmethod
    def _axis_gradient_loss(target, prediction, mask, threshold, axis):
        rank = len(target.shape)
        left = [slice(None)] * rank
        right = [slice(None)] * rank
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        target_left, target_right = target[tuple(left)], target[tuple(right)]
        pred_left, pred_right = prediction[tuple(left)], prediction[tuple(right)]
        pair_mask = mask[tuple(left)] * mask[tuple(right)]
        high_mask = ops.cast(
            ops.maximum(target_left, target_right) >= threshold, target.dtype
        )
        pair_mask = pair_mask * high_mask
        target_gradient = target_right - target_left
        pred_gradient = pred_right - pred_left
        return ops.sum(ops.abs(target_gradient - pred_gradient) * pair_mask) / (
            ops.sum(pair_mask) + 1e-6
        )

    def _compute_loss_and_metrics(self, logits, src_imgs, tgt_imgs, img_msks):
        cfg = self.cfg.algorithm.cac_regression
        prediction = self._to_image(logits, src_imgs) * img_msks
        prediction_clipped = ops.clip(prediction, 0.0, 1.0)
        threshold = self._normalized_cac_threshold()

        l1 = masked_l1_loss(tgt_imgs, prediction, img_msks)
        cac_l1 = self._high_hu_l1(
            tgt_imgs, prediction, img_msks, threshold=threshold
        )
        gradient_axes = tuple(str(axis).lower() for axis in cfg.gradient_axes)
        axis_to_dim = {"z": 1, "y": 2, "x": 3}
        invalid = [axis for axis in gradient_axes if axis not in axis_to_dim]
        if invalid:
            raise ValueError(f"cac_regression.gradient_axesはz/y/xです: {invalid}")
        gradient_losses = [
            self._axis_gradient_loss(
                tgt_imgs,
                prediction,
                img_msks,
                threshold,
                axis=axis_to_dim[axis],
            )
            for axis in gradient_axes
        ]
        gradient_loss = (
            ops.mean(ops.stack(gradient_losses))
            if gradient_losses
            else ops.cast(0.0, prediction.dtype)
        )
        total_loss = (
            float(cfg.l1_weight) * l1
            + float(cfg.cac_l1_weight) * cac_l1
            + float(cfg.gradient_weight) * gradient_loss
        )

        ssim_values = ssim_per_sample(tgt_imgs, prediction_clipped, msk=img_msks)
        psnr = masked_psnr_per_sample(
            tgt_imgs, prediction_clipped, img_msks
        )
        self.metrics_dict["l1_loss"].update_state(l1)
        self.metrics_dict["cac_l1_loss"].update_state(cac_l1)
        self.metrics_dict["gradient_loss"].update_state(gradient_loss)
        self.metrics_dict["total_loss"].update_state(total_loss)
        self.metrics_dict["psnr"].update_state(psnr)
        self.metrics_dict["ssim"].update_state(ssim_values)
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
