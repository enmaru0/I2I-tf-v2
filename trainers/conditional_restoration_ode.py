import math

import keras
import tensorflow as tf
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_psnr_per_sample, ssim_per_sample

from .base import BaseI2ITrainer


def conditional_restoration_sigma_schedule(
    num_steps: int, sigma_min: float, sigma_max: float
) -> list[float]:
    """log-linearなsigma scheduleを返す。末尾の0は最終Euler step用。"""
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(
            "conditional_restoration_ode.num_stepsは1以上を指定してください"
        )
    if num_steps == 1:
        return [float(sigma_max), 0.0]
    log_min = math.log(float(sigma_min))
    log_max = math.log(float(sigma_max))
    return [
        math.exp(log_max + i / (num_steps - 1) * (log_min - log_max))
        for i in range(num_steps)
    ] + [0.0]


@keras.saving.register_keras_serializable()
class ConditionalRestorationODETrainer(BaseI2ITrainer):
    """
    PyTorch版の従来 ``edm`` 目的関数を移植した条件付きrestoration ODE。

    generator入力は ``[source, x_sigma/(1+sigma), log-sigma level]``。
    generatorはODE targetを予測し、推論時はそこからclean推定とODE微分を
    再構成してsigma_maxから0までEuler/Heun積分する。
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_ode(self):
        return self.cfg.algorithm.conditional_restoration_ode

    def _sigma_level(self, sigma):
        cfg = self._cfg_ode()
        log_min = math.log(float(cfg.sigma_min))
        log_max = math.log(float(cfg.sigma_max))
        return (ops.log(sigma) - log_min) / (log_max - log_min)

    def _predict_target(self, src_x, scaled_state, sigma, img_msks, training):
        level = self._sigma_level(sigma)
        level_ch = ops.ones_like(scaled_state) * level
        net_in = ops.concatenate([src_x, scaled_state * img_msks, level_ch], axis=-1)
        return self([net_in, img_msks], training=training) * img_msks

    def _sample_log_sigma(self, batch_size):
        """任意境界のtruncated Gaussianを逆CDF法でサンプリングする。"""
        cfg = self._cfg_ode()
        mean = float(cfg.P_mean)
        std = float(cfg.P_std)
        lower = math.log(float(cfg.sigma_min))
        upper = math.log(float(cfg.sigma_max))
        sqrt2 = math.sqrt(2.0)
        z_lower = (lower - mean) / std
        z_upper = (upper - mean) / std
        cdf_lower = 0.5 * (1.0 + math.erf(z_lower / sqrt2))
        cdf_upper = 0.5 * (1.0 + math.erf(z_upper / sqrt2))
        u = tf.random.uniform(
            (batch_size, 1, 1, 1, 1), cdf_lower, cdf_upper, dtype=tf.float32
        )
        z = sqrt2 * tf.math.erfinv(2.0 * u - 1.0)
        return mean + std * z

    def _ode_loss(self, target_x, src_x, img_msks, training):
        cfg = self._cfg_ode()
        batch_size = tf.shape(target_x)[0]
        log_sigma = self._sample_log_sigma(batch_size)
        sigma = ops.exp(log_sigma)
        eps = tf.random.normal(tf.shape(target_x), dtype=target_x.dtype)

        scaled_state = (target_x + sigma * eps) / (1.0 + sigma)
        ode_target = (eps + sigma * (src_x - target_x)) / (1.0 + sigma)
        ode_pred = self._predict_target(src_x, scaled_state, sigma, img_msks, training)

        reduce_axes = (1, 2, 3, 4)
        valid_count = ops.sum(img_msks, axis=reduce_axes) + 1e-6
        target_mse = (
            ops.sum(ops.square(ode_pred - ode_target) * img_msks, axis=reduce_axes)
            / valid_count
        )
        degradation_rmse = ops.sqrt(
            ops.sum(ops.square(src_x - target_x) * img_msks, axis=reduce_axes)
            / valid_count
        )
        loss = ops.mean(
            target_mse / (degradation_rmse + float(cfg.get("degradation_eps", 1e-2)))
        )

        raw_state = scaled_state * (1.0 + sigma)
        clean_pred = (
            raw_state - ode_pred * sigma * (1.0 + sigma) + sigma**2 * src_x
        ) / (1.0 + sigma**2)
        return loss, clean_pred * img_msks

    def _derivative(self, state, sigma, src_x, img_msks, training=False):
        scaled_state = state / (1.0 + sigma)
        ode_pred = self._predict_target(src_x, scaled_state, sigma, img_msks, training)
        clean_pred = (state - ode_pred * sigma * (1.0 + sigma) + sigma**2 * src_x) / (
            1.0 + sigma**2
        )
        return (state - clean_pred) / sigma

    def _sample(self, src_x, img_msks, num_steps: int, sample_seeds=None):
        cfg = self._cfg_ode()
        solver = str(cfg.get("solver", "heun"))
        assert solver in ("euler", "heun"), solver
        sigmas = conditional_restoration_sigma_schedule(
            num_steps, cfg.sigma_min, cfg.sigma_max
        )
        state = (
            src_x
            + self._normal_like(src_x, sample_seeds=sample_seeds, salt=300) * sigmas[0]
        )

        for i in range(num_steps):
            sigma_cur, sigma_next = sigmas[i], sigmas[i + 1]
            sigma_t = ops.ones_like(state[:, :1, :1, :1, :1]) * sigma_cur
            derivative = self._derivative(
                state, sigma_t, src_x, img_msks, training=False
            )
            state_next = state + (sigma_next - sigma_cur) * derivative
            if sigma_next > 0.0 and solver == "heun":
                sigma_t2 = ops.ones_like(state[:, :1, :1, :1, :1]) * sigma_next
                derivative2 = self._derivative(
                    state_next, sigma_t2, src_x, img_msks, training=False
                )
                state_next = state + (sigma_next - sigma_cur) * 0.5 * (
                    derivative + derivative2
                )
            state = state_next
        return state * img_msks

    def train_step(self, data):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)
        src_x = self._to_x(src_imgs, img_msks)
        target_x = self._to_x(tgt_imgs, img_msks)

        with GradientTape() as tape:
            loss, clean_pred = self._ode_loss(target_x, src_x, img_msks, training=True)
            scaled_loss = self._scale_loss_for_optimizer(loss, self.optimizer)
        gradients = tape.gradient(scaled_loss, self.generator.trainable_variables)
        self._apply_gradients(
            self.optimizer, gradients, self.generator.trainable_variables
        )

        pred01 = self._to_01(clean_pred, img_msks)
        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(
            masked_psnr_per_sample(tgt_imgs, pred01, img_msks)
        )
        self.metrics_dict["ssim"].update_state(
            ssim_per_sample(tgt_imgs, pred01, msk=img_msks)
        )
        return self._get_metrics_result()

    def test_step(self, data):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)
        target_x = self._to_x(tgt_imgs, img_msks)
        loss, _ = self._ode_loss(target_x, src_x, img_msks, training=False)
        sample = self._sample(
            src_x,
            img_msks,
            self._cfg_ode().num_steps_val,
            sample_seeds=self._validation_sample_seeds(data),
        )
        preds = self._to_01(sample, img_msks)
        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(
            masked_psnr_per_sample(tgt_imgs, preds, img_msks)
        )
        self.metrics_dict["ssim"].update_state(
            ssim_per_sample(tgt_imgs, preds, msk=img_msks)
        )
        return self._get_metrics_result()

    def predict_step(self, data, return_aux=False):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)
        sample = self._sample(
            src_x,
            img_msks,
            self._cfg_ode().num_steps,
            sample_seeds=self._validation_sample_seeds(data),
        )
        preds = self._to_01(sample, img_msks)
        if return_aux:
            return sample, preds, src_imgs, tgt_imgs, img_msks
        return preds
