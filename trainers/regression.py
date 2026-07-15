import keras
from keras.src import ops
from tensorflow import GradientTape

from losses import (
    masked_l1_loss,
    masked_l2_loss,
    masked_psnr_per_sample,
    ssim_per_sample,
)

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class RegressionTrainer(BaseI2ITrainer):
    """
    U-Netによる単純な回帰ベースのimage-to-image translation。
    L1 / L2 / SSIM損失の重み付き和で学習する。
    """

    METRIC_NAMES = ["l1_loss", "l2_loss", "ssim_loss", "total_loss", "psnr", "ssim"]

    def train_step(self, data):
        """
        ここのデータ名であったりselfに渡す引数を変えた場合は、
        callbacks/image_logger.pyのpredict_stepやon_test_batch_endも変更すること
        ここはjit_compileされているのでtensorboardを含むCPUを使う処理はかけない
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=True)

        with GradientTape() as tape:
            logits = self([src_imgs, img_msks], training=True)
            total_loss = self._compute_loss_and_metrics(
                logits, src_imgs, tgt_imgs, img_msks
            )
            total_loss = self._add_real_dc_loss(total_loss, data)
            scaled_loss = self._scale_loss_for_optimizer(total_loss, self.optimizer)
        trainable_weights = self.generator.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_weights)
        self._apply_gradients(self.optimizer, gradients, trainable_weights)

        return self._get_metrics_result()

    def _compute_loss_and_metrics(self, logits, src_imgs, tgt_imgs, img_msks):
        cfg_loss = self.cfg.algorithm.regression.loss

        # 損失はクリップ前の予測で計算する（クリップすると飽和領域の勾配が消えるため）
        preds = self._to_image(logits, src_imgs) * img_msks
        preds_clipped = ops.clip(preds, 0, 1)

        l1 = masked_l1_loss(tgt_imgs, preds, img_msks)
        l2 = masked_l2_loss(tgt_imgs, preds, img_msks)
        ssim_values = ssim_per_sample(tgt_imgs, preds_clipped, msk=img_msks)
        ssim_val = ops.mean(ssim_values)
        ssim_loss = 1.0 - ssim_val

        total_loss = cfg_loss.l1 * l1 + cfg_loss.l2 * l2 + cfg_loss.ssim * ssim_loss

        # 評価指標はクリップ後の予測で計算する
        psnr = masked_psnr_per_sample(tgt_imgs, preds_clipped, img_msks)

        self.metrics_dict["l1_loss"].update_state(l1)
        self.metrics_dict["l2_loss"].update_state(l2)
        self.metrics_dict["ssim_loss"].update_state(ssim_loss)
        self.metrics_dict["total_loss"].update_state(total_loss)
        self.metrics_dict["psnr"].update_state(psnr)
        self.metrics_dict["ssim"].update_state(ssim_values)

        return total_loss
