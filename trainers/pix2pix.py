import keras
from keras import Model
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_l1_loss, masked_psnr_per_sample, ssim_per_sample

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class Pix2PixTrainer(BaseI2ITrainer):
    """
    pix2pix (条件付きGAN + L1) によるimage-to-image translation。
    https://arxiv.org/abs/1611.07004

    generatorはRegressionTrainerと同じU-Net。
    discriminatorは3D PatchGAN（models/discriminator.py）。
    discriminator用のoptimizerはd_optimizer属性に外部からセットする
    （trainers/__init__.pyのbuild_trainerで行う）。
    """

    METRIC_NAMES = ["l1_loss", "g_adv_loss", "g_total_loss", "d_loss", "psnr", "ssim"]

    def __init__(self, generator: Model, discriminator: Model, **kwargs):
        super().__init__(generator, **kwargs)
        self.discriminator = discriminator
        self._create_gradient_accumulators(
            "discriminator", self.discriminator.trainable_variables
        )
        # 復元(restore)時はd_optimizerの状態は引き継がれないので注意
        self.d_optimizer = None

    def get_config(self):
        config = super().get_config()
        config["discriminator"] = keras.saving.serialize_keras_object(
            self.discriminator
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["generator"] = keras.saving.deserialize_keras_object(config["generator"])
        config["discriminator"] = keras.saving.deserialize_keras_object(
            config["discriminator"]
        )
        return cls(**config)

    def _gan_loss(self, disc_logits, is_real: bool):
        """パッチごとのGAN損失を計算する"""
        gan_mode = self.cfg.algorithm.pix2pix.gan_mode
        target = ops.ones_like(disc_logits) if is_real else ops.zeros_like(disc_logits)
        if gan_mode == "lsgan":
            # LSGAN: 学習が安定しやすいのでデフォルト
            return ops.mean(ops.square(disc_logits - target))
        elif gan_mode == "vanilla":
            return ops.mean(
                ops.binary_crossentropy(target, disc_logits, from_logits=True)
            )
        else:
            raise NotImplementedError(gan_mode)

    def train_step(self, data):
        """
        Generator -> Discriminatorの順に1ステップずつ更新する。
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        cfg_p2p = self.cfg.algorithm.pix2pix
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)

        # --- Generatorの更新 ---
        with GradientTape() as g_tape:
            logits = self([src_imgs, img_msks], training=True)
            preds = self._to_image(logits, src_imgs) * img_msks

            d_fake_for_g = self.discriminator([src_imgs, preds], training=True)
            g_adv = self._gan_loss(d_fake_for_g, is_real=True)
            l1 = masked_l1_loss(tgt_imgs, preds, img_msks)
            g_total = cfg_p2p.l1_weight * l1 + cfg_p2p.gan_weight * g_adv
            g_total = self._add_real_dc_loss(g_total, data)
            g_scaled = self._scale_loss_for_optimizer(g_total, self.optimizer)
        g_grads = g_tape.gradient(g_scaled, self.generator.trainable_variables)
        self._apply_gradients(
            self.optimizer, g_grads, self.generator.trainable_variables
        )

        # --- Discriminatorの更新 ---
        # generatorのforwardは再利用し、勾配だけ止める
        fake_imgs = ops.stop_gradient(preds)
        with GradientTape() as d_tape:
            d_real = self.discriminator([src_imgs, tgt_imgs], training=True)
            d_fake = self.discriminator([src_imgs, fake_imgs], training=True)
            d_loss = 0.5 * (
                self._gan_loss(d_real, is_real=True)
                + self._gan_loss(d_fake, is_real=False)
            )
            d_scaled = self._scale_loss_for_optimizer(d_loss, self.d_optimizer)
        d_grads = d_tape.gradient(d_scaled, self.discriminator.trainable_variables)
        self._apply_gradients(
            self.d_optimizer,
            d_grads,
            self.discriminator.trainable_variables,
            key="discriminator",
        )

        # --- メトリクスの更新 ---
        preds_clipped = ops.clip(preds, 0, 1)
        self._update_metrics(
            l1, g_adv, g_total, d_loss, tgt_imgs, preds_clipped, img_msks
        )
        return self._get_metrics_result()

    def _compute_loss_and_metrics(self, logits, src_imgs, tgt_imgs, img_msks):
        """検証時(test_step)の損失・メトリクス計算。重みの更新は行わない"""
        cfg_p2p = self.cfg.algorithm.pix2pix
        preds = self._to_image(logits, src_imgs) * img_msks

        d_fake = self.discriminator([src_imgs, preds], training=False)
        d_real = self.discriminator([src_imgs, tgt_imgs], training=False)
        g_adv = self._gan_loss(d_fake, is_real=True)
        d_loss = 0.5 * (
            self._gan_loss(d_real, is_real=True) + self._gan_loss(d_fake, is_real=False)
        )
        l1 = masked_l1_loss(tgt_imgs, preds, img_msks)
        g_total = cfg_p2p.l1_weight * l1 + cfg_p2p.gan_weight * g_adv

        preds_clipped = ops.clip(preds, 0, 1)
        self._update_metrics(
            l1, g_adv, g_total, d_loss, tgt_imgs, preds_clipped, img_msks
        )
        return g_total

    def _update_metrics(
        self, l1, g_adv, g_total, d_loss, tgt_imgs, preds_clipped, img_msks
    ):
        psnr = masked_psnr_per_sample(tgt_imgs, preds_clipped, img_msks)
        ssim_val = ssim_per_sample(tgt_imgs, preds_clipped, msk=img_msks)

        self.metrics_dict["l1_loss"].update_state(l1)
        self.metrics_dict["g_adv_loss"].update_state(g_adv)
        self.metrics_dict["g_total_loss"].update_state(g_total)
        self.metrics_dict["d_loss"].update_state(d_loss)
        self.metrics_dict["psnr"].update_state(psnr)
        self.metrics_dict["ssim"].update_state(ssim_val)
