import keras
import tensorflow as tf
from keras.src import ops

from .rectified_flow import RectifiedFlowTrainer


@keras.saving.register_keras_serializable()
class SplitMeanFlowTrainer(RectifiedFlowTrainer):
    """
    SplitMeanFlow (Guo et al. 2025, https://arxiv.org/abs/2507.16884) による
    少ステップ生成。MeanFlow（平均速度場の学習）のJVP不要版。

    瞬間速度vではなく区間[t, s]の平均速度 u(x_t, t, s) を学習する:
        x_s = x_t + (s - t) * u(x_t, t, s)
    これが正確なら1回のネットワーク評価でノイズ(t=0)からtarget(t=1)へ
    ジャンプできる（1ステップ生成）。

    学習は2つの項の混合（fm_ratioで割合を制御）:
    - 境界条件 (s=t): u(x_t, t, t) = v = target - x0（通常のflow matching）
    - 区間分割整合性 (t < s): 中点m=(t+s)/2を使い
        u(x_t, t, s) ≈ [(m-t)*u(x_t, t, m) + (s-m)*u(x_m, m, s)] / (s-t)
        （x_m = x_t + (m-t)*u(x_t, t, m)、右辺はstop-gradientの教師値）
      JVP（前進微分）を使わないためTF/XLAでも安全に動く。

    - generatorは既存U-Netを流用し、入力を4チャンネルにする:
      [x_t, source, 開始時刻t, 終了時刻s]（時刻は定数チャンネル）
    - 教師値の計算に追加のforward2回（勾配なし・training=False）、
      メトリクス用のx1_hat計算に1回を使うため、1ステップあたりの計算量は
      rectified_flowより多い（backwardは1回分のみ）
    - サンプリングは平均速度場による等間隔ジャンプ（solver設定は使わない。
      平均速度の積分は区間ジャンプそのものなのでheun補正は不要）
    - 学習時のpsnr/ssimは1ステップ推定 x1_hat = x_t + (1-t)*u(x_t,t,1) の品質、
      検証時のpsnr/ssimはnum_steps_valステップのサンプリング結果の品質を表す
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_rf(self):
        return self.cfg.algorithm.split_mean_flow

    def _dc_predict01(self, src01, img_msks):
        """DC損失用: 平均速度場によるt=0→1の1ジャンプ予測（評価1回）"""
        src_x = self._to_x(src01, img_msks)
        x0 = self._initial_state(src_x)
        t = ops.zeros_like(x0[:, :1, :1, :1, :1])
        s = ops.ones_like(x0[:, :1, :1, :1, :1])
        u = self._avg_velocity(x0, t, s, src_x, img_msks, training=True)
        return self._to_01(x0 + u, img_msks)

    def _avg_velocity(self, x_t, t, s, src_x, img_msks, training):
        """平均速度場 u(x_t, t, s | source) を予測する。t, sは(b,1,1,1,1)"""
        t_ch = ops.ones_like(x_t) * t
        s_ch = ops.ones_like(x_t) * s
        net_in = ops.concatenate([x_t * img_msks, src_x, t_ch, s_ch], axis=-1)
        u = self([net_in, img_msks], training=training)
        return u * img_msks

    def _masked_mse(self, diff, img_msks):
        return ops.sum(ops.square(diff) * img_msks) / (ops.sum(img_msks) + 1e-6)

    def _flow_loss(self, y, src_x, img_msks, training):
        """境界条件+区間分割整合性の混合損失と、1ステップ推定x1_hatを返す"""
        cfg = self._cfg_rf()
        batch_size = tf.shape(y)[0]
        shape_b = (batch_size, 1, 1, 1, 1)

        # 区間[t, s]をサンプリング（2つの一様乱数のmin/max）。
        # fm_ratioの割合でs=tに退化させ、境界条件（flow matching）として学習する
        u1_rnd = tf.random.uniform(shape_b, 0.0, 1.0)
        u2_rnd = tf.random.uniform(shape_b, 0.0, 1.0)
        t = ops.minimum(u1_rnd, u2_rnd)
        s = ops.maximum(u1_rnd, u2_rnd)
        is_fm = tf.random.uniform(shape_b, 0.0, 1.0) < cfg.fm_ratio
        s = ops.where(is_fm, t, s)
        m = 0.5 * (t + s)

        x0 = self._initial_state(src_x)
        x_t = (1.0 - t) * x0 + t * y
        v_target = y - x0

        # 学習対象: u(x_t, t, s)
        u_pred = self._avg_velocity(x_t, t, s, src_x, img_msks, training)

        # 区間分割整合性の教師値（勾配を流さない。BatchRenormの統計も更新しない）
        u_tm = tf.stop_gradient(
            self._avg_velocity(x_t, t, m, src_x, img_msks, training=False)
        )
        x_m = x_t + (m - t) * u_tm
        u_ms = tf.stop_gradient(
            self._avg_velocity(x_m, m, s, src_x, img_msks, training=False)
        )
        split_target = ((m - t) * u_tm + (s - m) * u_ms) / ops.maximum(s - t, 1e-8)

        # fmサンプルは瞬間速度、それ以外は分割整合性を教師にする
        target = ops.where(is_fm, v_target, split_target)
        loss = self._masked_mse(u_pred - target, img_msks)

        # メトリクス用: 現在位置からt=1へ平均速度で1ジャンプした推定
        ones_t = ops.ones_like(t)
        u_full = self._avg_velocity(x_t, t, ones_t, src_x, img_msks, training=False)
        x1_hat = x_t + (1.0 - t) * u_full
        return loss, x1_hat

    def _sample(self, src_x, img_msks, num_steps: int, sample_seeds=None):
        """平均速度場による等間隔ジャンプ（ネットワーク評価はnum_steps回、1でも可）"""
        dt = 1.0 / num_steps
        x = self._initial_state(src_x, sample_seeds=sample_seeds, salt=100)
        for i in range(num_steps):
            t = ops.ones_like(x[:, :1, :1, :1, :1]) * (i * dt)
            s = ops.ones_like(x[:, :1, :1, :1, :1]) * ((i + 1) * dt)
            u = self._avg_velocity(x, t, s, src_x, img_msks, training=False)
            x = x + dt * u
        return x
