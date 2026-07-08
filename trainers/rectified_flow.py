import keras
import tensorflow as tf
from keras.src import ops
from tensorflow import GradientTape

from losses import masked_psnr, ssim

from .base import BaseI2ITrainer


@keras.saving.register_keras_serializable()
class RectifiedFlowTrainer(BaseI2ITrainer):
    """
    Rectified Flow (Liu et al. 2022, https://arxiv.org/abs/2209.03003)
    / Flow Matching による条件付きimage-to-image translation。

    - ノイズx0とtarget x1を直線補間 x_t = (1-t)*x0 + t*x1 で結び、
      速度場 v(x_t, t, source) = x1 - x0 を回帰する
    - generatorは既存U-Netを流用し、入力を3チャンネルにする:
      [x_t, source, 時刻t(定数チャンネル)]
    - サンプリングはオイラー積分（t=0のノイズからt=1のtargetへ、num_steps回のネットワーク評価）
    - 学習時のpsnr/ssimは1ステップ推定 x1_hat = x_t + (1-t)*v の品質、
      検証時のpsnr/ssimはnum_steps_valステップのサンプリング結果の品質を表す
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_rf(self):
        return self.cfg.algorithm.rectified_flow

    def _initial_state(self, src_x):
        """
        フローの出発点x0を返す。標準のrectified flowは純粋ノイズ。
        I2I-RFRなどのサブクラスでオーバーライドする。
        学習時のx0とサンプリングの初期値の両方に使われる。
        """
        return tf.random.normal(tf.shape(src_x))

    def _velocity(self, x_t, t, src_x, img_msks, training):
        """速度場 v(x_t, t | source) を予測する。tは(b,1,1,1,1)"""
        t_ch = ops.ones_like(x_t) * t
        net_in = ops.concatenate([x_t * img_msks, src_x, t_ch], axis=-1)
        v = self([net_in, img_msks], training=training)
        return v * img_msks

    def _sample_t(self, batch_size):
        """学習用の時刻tをサンプリングする"""
        cfg_rf = self._cfg_rf()
        if cfg_rf.time_sampling == "uniform":
            t = tf.random.uniform((batch_size, 1, 1, 1, 1), 0.0, 1.0)
        elif cfg_rf.time_sampling == "logit_normal":
            # SD3で使われた中間時刻を重視するサンプリング
            rnd = tf.random.normal(
                (batch_size, 1, 1, 1, 1),
                mean=cfg_rf.logit_normal_mean,
                stddev=cfg_rf.logit_normal_std,
            )
            t = ops.sigmoid(rnd)
        else:
            raise NotImplementedError(cfg_rf.time_sampling)
        return t

    def _flow_loss(self, y, src_x, img_msks, training):
        """flow matching損失と1ステップ推定x1_hatを返す"""
        batch_size = tf.shape(y)[0]
        t = self._sample_t(batch_size)
        x0 = self._initial_state(src_x)
        x_t = (1.0 - t) * x0 + t * y
        v_target = y - x0

        v_pred = self._velocity(x_t, t, src_x, img_msks, training)
        loss = ops.sum(ops.square(v_pred - v_target) * img_msks) / (
            ops.sum(img_msks) + 1e-6
        )

        # 現在位置から直線でt=1へ飛ばした1ステップ推定（メトリクス用）
        x1_hat = x_t + (1.0 - t) * v_pred
        return loss, x1_hat

    def _sample(self, src_x, img_msks, num_steps: int):
        """
        t=0からt=1への数値積分によるサンプリング。
        solver=euler: 1次オイラー（ネットワーク評価はnum_steps回）
        solver=heun: 2次Heun（ネットワーク評価は2*num_steps回）
        """
        # 古いoutput.yamlにはsolverが無いためデフォルト(従来動作のeuler)を使う
        solver = self._cfg_rf().get("solver", "euler")
        assert solver in ("euler", "heun"), solver
        dt = 1.0 / num_steps
        x = self._initial_state(src_x)
        for i in range(num_steps):
            t = ops.ones_like(x[:, :1, :1, :1, :1]) * (i * dt)
            v = self._velocity(x, t, src_x, img_msks, training=False)
            x_next = x + dt * v
            if solver == "heun":
                # 2次補正: 仮ステップ先の速度と平均する
                t2 = ops.ones_like(x[:, :1, :1, :1, :1]) * ((i + 1) * dt)
                v2 = self._velocity(x_next, t2, src_x, img_msks, training=False)
                x_next = x + dt * 0.5 * (v + v2)
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
            loss, x1_hat = self._flow_loss(y, src_x, img_msks, training=True)
        gradients = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )

        x1_hat01 = self._to_01(x1_hat, img_msks)
        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(
            masked_psnr(tgt_imgs, x1_hat01, img_msks)
        )
        self.metrics_dict["ssim"].update_state(ssim(tgt_imgs, x1_hat01))
        return self._get_metrics_result()

    def test_step(self, data):
        """
        検証ではflow損失に加えて、少ステップのサンプリングを実行し
        生成画像のPSNR/SSIMを計算する（val_psnrをチェックポイント選択に使う）
        """
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)
        y = self._to_x(tgt_imgs, img_msks)

        loss, _ = self._flow_loss(y, src_x, img_msks, training=False)

        x = self._sample(src_x, img_msks, self._cfg_rf().num_steps_val)
        preds = self._to_01(x, img_msks)

        self.metrics_dict["total_loss"].update_state(loss)
        self.metrics_dict["psnr"].update_state(masked_psnr(tgt_imgs, preds, img_msks))
        self.metrics_dict["ssim"].update_state(ssim(tgt_imgs, preds))
        return self._get_metrics_result()

    def predict_step(self, data, return_aux=False):
        src_imgs, tgt_imgs, img_msks = self._prepare_batch(data, is_training=False)
        src_x = self._to_x(src_imgs, img_msks)

        x = self._sample(src_x, img_msks, self._cfg_rf().num_steps)
        preds = self._to_01(x, img_msks)

        if return_aux:
            # 最初の要素はlogits相当として作業空間の生成結果を返す
            return x, preds, src_imgs, tgt_imgs, img_msks
        else:
            return preds
