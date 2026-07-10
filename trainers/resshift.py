import math

import keras
import tensorflow as tf
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_psnr_per_sample, ssim_per_sample

from .base import BaseI2ITrainer


def resshift_eta_schedule(num_steps: int, kappa: float, schedule_p: float) -> list[float]:
    """
    ResShift (Yue et al. 2023) のシフトスケジュールη。edm-torch実装と同一。
    η[0]=0（便宜上）、η[1..T]はsqrt(η)が幾何級数的に増える非一様スケジュール。
    η[T]=0.999でx_Tがほぼ劣化画像+κノイズになる。
    """
    num_steps = int(num_steps)
    assert num_steps >= 2, "resshift.num_stepsは2以上を指定してください"
    kappa = float(kappa)
    assert kappa > 0, "resshift.kappaは正の値を指定してください"
    eta_1 = min((0.04 / kappa) ** 2, 0.001)
    eta_t = 0.999
    eta = [0.0] * (num_steps + 1)
    eta[1] = eta_1
    eta[num_steps] = eta_t
    if num_steps > 2:
        b0 = math.exp(0.5 * math.log(eta_t / eta_1) / float(num_steps - 1))
        for t in range(2, num_steps):
            beta_t = ((float(t - 1) / float(num_steps - 1)) ** float(schedule_p)) * float(
                num_steps - 1
            )
            sqrt_eta = math.sqrt(eta_1) * (b0**beta_t)
            eta[t] = sqrt_eta**2
    return eta


@keras.saving.register_keras_serializable()
class ResShiftTrainer(BaseI2ITrainer):
    """
    ResShift (Yue et al. 2023, https://arxiv.org/abs/2307.12348) による
    残差シフト拡散のimage-to-image translation。edm-torch実装の移植。

    - 前向き過程は純ノイズではなく「劣化画像への残差シフト」:
        x_t = target + η_t * (source - target) + κ * sqrt(η_t) * ε
      t=Tでx_T ≈ source + κノイズとなり、輸送距離が短いため少ステップで動く
    - generatorは既存U-Netを流用し、入力を3チャンネルにする:
      [x_t, source, 時刻条件(定数チャンネル)]
    - ネットワークはtarget(x0)を直接予測し、逆拡散の遷移平均
        x_{t-1} = (η_{t-1}/η_t) x_t + (α_t/η_t) x0_pred,  α_t = η_t - η_{t-1}
      を決定的（sample_noise=falseの場合）にたどる
    - ηスケジュール長は学習時cfg.num_stepsで固定。推論はnum_steps引数で
      再離散化できるが、time_condition=t（インデックス基準）は学習時と
      意味がずれるため、ステップ数を変える場合はeta/sqrt_etaを推奨
    - 学習時のpsnr/ssimはx0_pred（1回のネットワーク評価）の品質、
      検証時のpsnr/ssimはnum_steps_valステップのチェーン結果の品質を表す
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_rs(self):
        return self.cfg.algorithm.resshift

    def _time_condition_value(self, eta_t, t_index, num_steps: int):
        """時刻チャンネルに入れる値。eta（連続値）がデフォルト"""
        mode = self._cfg_rs().time_condition
        if mode == "eta":
            return eta_t
        elif mode == "sqrt_eta":
            return ops.sqrt(ops.maximum(eta_t, 0.0))
        elif mode == "t":
            return (t_index - 1.0) / max(float(num_steps - 1), 1.0)
        else:
            raise NotImplementedError(mode)

    def _predict_x0(self, x_t, time_cond, src_x, img_msks, training):
        """x_tからtarget(x0)を直接予測する。time_condは(b,1,1,1,1)"""
        cond_ch = ops.ones_like(x_t) * time_cond
        net_in = ops.concatenate([x_t * img_msks, src_x, cond_ch], axis=-1)
        x0 = self([net_in, img_msks], training=training)
        return x0 * img_msks

    def _dc_predict01(self, src01, img_msks):
        """DC損失用: t=T（x_T=source、決定的）からのx0直接予測（評価1回）"""
        cfg = self._cfg_rs()
        num_steps = int(cfg.num_steps)
        eta = resshift_eta_schedule(num_steps, cfg.kappa, cfg.schedule_p)
        src_x = self._to_x(src01, img_msks)
        mode = cfg.time_condition
        if mode == "eta":
            time_cond_val = eta[num_steps]
        elif mode == "sqrt_eta":
            time_cond_val = math.sqrt(max(eta[num_steps], 0.0))
        else:  # "t"
            time_cond_val = 1.0
        time_cond = ops.ones_like(src_x[:, :1, :1, :1, :1]) * time_cond_val
        x0_pred = self._predict_x0(src_x, time_cond, src_x, img_msks, training=True)
        return self._to_01(x0_pred, img_msks)

    def _denoise_loss(self, y, src_x, img_msks, training):
        """一様なt∈{1..T}で残差シフト拡散のx0再構成損失を計算する"""
        cfg = self._cfg_rs()
        num_steps = int(cfg.num_steps)
        eta = tf.constant(
            resshift_eta_schedule(num_steps, cfg.kappa, cfg.schedule_p), tf.float32
        )

        batch_size = tf.shape(y)[0]
        t_idx = tf.random.uniform(
            (batch_size,), minval=1, maxval=num_steps + 1, dtype=tf.int32
        )
        eta_t = tf.reshape(tf.gather(eta, t_idx), (-1, 1, 1, 1, 1))
        t_index = tf.reshape(tf.cast(t_idx, tf.float32), (-1, 1, 1, 1, 1))

        noise = tf.random.normal(tf.shape(y))
        x_t = y + eta_t * (src_x - y) + cfg.kappa * ops.sqrt(eta_t) * noise

        time_cond = self._time_condition_value(eta_t, t_index, num_steps)
        x0_pred = self._predict_x0(x_t, time_cond, src_x, img_msks, training)

        diff = (x0_pred - y) * img_msks
        if cfg.loss == "mse":
            per_voxel = ops.square(diff)
        elif cfg.loss == "l1":
            per_voxel = ops.abs(diff)
        else:
            raise NotImplementedError(cfg.loss)
        loss = ops.sum(per_voxel) / (ops.sum(img_msks) + 1e-6)
        return loss, x0_pred

    def _sample(self, src_x, img_msks, num_steps: int, sample_seeds=None):
        """
        逆拡散チェーンによるサンプリング（ネットワーク評価はnum_steps回）。
        sample_noise=falseなら決定的（メトリクス評価向き）。
        """
        cfg = self._cfg_rs()
        eta = resshift_eta_schedule(num_steps, cfg.kappa, cfg.schedule_p)

        if cfg.sample_noise:
            initial_noise = self._normal_like(src_x, sample_seeds, salt=100)
            x = src_x + cfg.kappa * math.sqrt(eta[num_steps]) * initial_noise
        else:
            x = src_x

        mode = cfg.time_condition
        for t in range(num_steps, 0, -1):
            eta_t, eta_prev = eta[t], eta[t - 1]
            alpha_t = eta_t - eta_prev
            # スケジュールはPython floatなので時刻条件もPython定数として埋め込む
            if mode == "eta":
                time_cond_val = eta_t
            elif mode == "sqrt_eta":
                time_cond_val = math.sqrt(max(eta_t, 0.0))
            elif mode == "t":
                time_cond_val = (t - 1.0) / max(float(num_steps - 1), 1.0)
            else:
                raise NotImplementedError(mode)
            time_cond = ops.ones_like(x[:, :1, :1, :1, :1]) * time_cond_val
            x0_pred = self._predict_x0(x, time_cond, src_x, img_msks, training=False)
            x = (eta_prev / eta_t) * x + (alpha_t / eta_t) * x0_pred
            if cfg.sample_noise and t > 1:
                sigma = cfg.kappa * math.sqrt((eta_prev / eta_t) * alpha_t)
                noise = self._normal_like(x, sample_seeds, salt=100 + t)
                x = x + sigma * noise
        return x

    def train_step(self, data):
        """
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)
        src_x = self._to_x(src_imgs, img_msks)
        y = self._to_x(tgt_imgs, img_msks)

        with GradientTape() as tape:
            loss, x0_pred = self._denoise_loss(y, src_x, img_msks, training=True)
            loss = self._add_real_dc_loss(loss, data)
            scaled_loss = self._scale_loss_for_optimizer(loss, self.optimizer)
        gradients = tape.gradient(scaled_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        x0_pred01 = self._to_01(x0_pred, img_msks)
        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(
            masked_psnr_per_sample(tgt_imgs, x0_pred01, img_msks)
        )
        self.metrics_dict["ssim"].update_state(
            ssim_per_sample(tgt_imgs, x0_pred01, msk=img_msks)
        )
        return self._get_metrics_result()

    def test_step(self, data):
        """
        検証ではx0再構成損失に加えて、逆拡散チェーンを実行し
        生成画像のPSNR/SSIMを計算する（val_psnrをチェックポイント選択に使う）
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)
        y = self._to_x(tgt_imgs, img_msks)

        loss, _ = self._denoise_loss(y, src_x, img_msks, training=False)

        x = self._sample(
            src_x,
            img_msks,
            self._cfg_rs().num_steps_val,
            sample_seeds=self._validation_sample_seeds(data),
        )
        preds = self._to_01(x, img_msks)

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

        x = self._sample(
            src_x,
            img_msks,
            self._cfg_rs().num_steps,
            sample_seeds=self._validation_sample_seeds(data),
        )
        preds = self._to_01(x, img_msks)

        if return_aux:
            # 最初の要素はlogits相当として作業空間の生成結果を返す
            return x, preds, src_imgs, tgt_imgs, img_msks
        else:
            return preds
