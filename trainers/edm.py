import keras
import tensorflow as tf
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_psnr, ssim

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class EDMTrainer(BaseI2ITrainer):
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models,
    Karras et al. 2022, https://arxiv.org/abs/2206.00364) による
    条件付き拡散モデルのimage-to-image translation。

    - generatorは既存U-Netを流用し、入力を3チャンネルにする:
      [ノイズ入りtarget(c_inスケール済み), source, ノイズレベル(c_noise)]
    - 画像は[0,1]正規化後に[-1,1]へ変換して扱う (sigma_data=0.5を想定)
    - サンプリングはKarrasスケジュール + 2次Heunソルバー（決定的）
    - 学習時のpsnr/ssimは「1回のdenoise出力」の品質、
      検証時のpsnr/ssimは「num_steps_valステップのサンプリング結果」の品質を表す
    注意: 本家EDMと異なりEMAは未実装。またBatchRenormを含むU-Netを
    そのまま使うため、正規化層の統計はσ全域の混合になる。
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_edm(self):
        return self.cfg.algorithm.edm

    def _denoise(self, x_noisy, sigma, src_x, img_msks, training):
        """EDMのpreconditioning付きdenoiser D(x; sigma, source)"""
        sigma_data = self._cfg_edm().sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / ops.sqrt(sigma**2 + sigma_data**2)
        c_in = 1.0 / ops.sqrt(sigma_data**2 + sigma**2)
        c_noise = ops.log(sigma) / 4.0

        # ノイズレベルは定数チャンネルとして与える（U-Net本体を変えないための簡易実装）
        noise_ch = ops.ones_like(x_noisy) * c_noise
        net_in = ops.concatenate(
            [c_in * x_noisy * img_msks, src_x, noise_ch], axis=-1
        )
        network_out = self([net_in, img_msks], training=training)
        return (c_skip * x_noisy + c_out * network_out) * img_msks

    def _denoise_loss(self, y, src_x, img_msks, training):
        """log-normalからσをサンプリングしてλ(σ)重み付きdenoise損失を計算する"""
        cfg_edm = self._cfg_edm()
        batch_size = tf.shape(y)[0]
        rnd = tf.random.normal((batch_size, 1, 1, 1, 1))
        sigma = ops.exp(cfg_edm.P_mean + cfg_edm.P_std * rnd)
        noise = tf.random.normal(tf.shape(y)) * sigma
        weight = (sigma**2 + cfg_edm.sigma_data**2) / (
            (sigma * cfg_edm.sigma_data) ** 2
        )

        denoised = self._denoise(y + noise, sigma, src_x, img_msks, training)
        loss = ops.sum(weight * ops.square(denoised - y) * img_msks) / (
            ops.sum(img_msks) + 1e-6
        )
        return loss, denoised

    def _sample(self, src_x, img_msks, num_steps: int):
        """
        Karrasスケジュール + 2次Heunソルバーによる決定的サンプリング。
        ネットワーク評価回数は 2*num_steps - 1 回。
        """
        cfg_edm = self._cfg_edm()
        rho = cfg_edm.rho
        s_min, s_max = cfg_edm.sigma_min, cfg_edm.sigma_max

        # σスケジュールはPython floatの定数としてグラフに埋め込む
        sigmas = [
            (
                s_max ** (1 / rho)
                + i / (num_steps - 1) * (s_min ** (1 / rho) - s_max ** (1 / rho))
            )
            ** rho
            for i in range(num_steps)
        ] + [0.0]

        x = tf.random.normal(tf.shape(src_x)) * sigmas[0]
        for i in range(num_steps):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            sigma_t = ops.ones_like(x[:, :1, :1, :1, :1]) * s_cur
            d = (x - self._denoise(x, sigma_t, src_x, img_msks, False)) / s_cur
            x_next = x + (s_next - s_cur) * d
            if s_next > 0:
                # 2次補正（Heun）
                sigma_t2 = ops.ones_like(x[:, :1, :1, :1, :1]) * s_next
                d2 = (
                    x_next - self._denoise(x_next, sigma_t2, src_x, img_msks, False)
                ) / s_next
                x_next = x + (s_next - s_cur) * 0.5 * (d + d2)
            x = x_next
        return x

    def train_step(self, data):
        """
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)
        src_x = self._to_x(src_imgs, img_msks)
        y = self._to_x(tgt_imgs, img_msks)

        with GradientTape() as tape:
            loss, denoised = self._denoise_loss(y, src_x, img_msks, training=True)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        # 学習中のpsnr/ssimはdenoise出力の品質（サンプリングはしない）
        denoised01 = self._to_01(denoised, img_msks)
        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(
            masked_psnr(tgt_imgs, denoised01, img_msks)
        )
        self.metrics_dict["ssim"].update_state(ssim(tgt_imgs, denoised01))
        return self._get_metrics_result()

    def test_step(self, data):
        """
        検証ではdenoise損失に加えて、少ステップのサンプリングを実行し
        生成画像のPSNR/SSIMを計算する（val_psnrをチェックポイント選択に使う）
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)
        y = self._to_x(tgt_imgs, img_msks)

        loss, _ = self._denoise_loss(y, src_x, img_msks, training=False)

        x = self._sample(src_x, img_msks, self._cfg_edm().num_steps_val)
        preds = self._to_01(x, img_msks)

        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(masked_psnr(tgt_imgs, preds, img_msks))
        self.metrics_dict["ssim"].update_state(ssim(tgt_imgs, preds))
        return self._get_metrics_result()

    def predict_step(self, data, return_aux=False):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)

        x = self._sample(src_x, img_msks, self._cfg_edm().num_steps)
        preds = self._to_01(x, img_msks)

        if return_aux:
            # 最初の要素はlogits相当としてEDM作業空間の生成結果を返す
            return x, preds, src_imgs, tgt_imgs, img_msks
        else:
            return preds
