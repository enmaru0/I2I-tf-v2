import keras
import tensorflow as tf
from keras.src import ops

from .rectified_flow import RectifiedFlowTrainer


@keras.saving.register_keras_serializable()
class I2IRFRX0Trainer(RectifiedFlowTrainer):
    """
    I2I-RFR (x0予測版): edm-torch実装のi2i_rfr目的関数と同等のアルゴリズム。

    I2IRFRTrainer（フローの出発点をsourceに置く）とは別物で、こちらは
    ノイズとtargetを結ぶ直線フローをx0予測でパラメータ化する:
        x_t = (1-t) * target + t * ε   （t=0でtarget、t=1で純ノイズ）
        ネットワークはx_tからtarget(x0)を直接予測する
    sourceは条件チャンネルとしてのみ与える。

    - 時刻tは密度∝t^p（u^{1/(p+1)}, u~U(0,1)）でサンプリングし高ノイズ側を
      重視、損失はt^-pで重み付けする（速度場回帰のt重みと等価な再定式化）
    - time_condition=zero（デフォルト）では時刻をネットワークに与えない
    - サンプリングはt=1の純ノイズからオイラー積分:
        v = (x_t - x0_pred) / t,  x_{t-dt} = x_t - dt * v
    - 学習時のpsnr/ssimはx0_pred（1回のネットワーク評価）の品質、
      検証時のpsnr/ssimはnum_steps_valステップのサンプリング結果の品質を表す

    時刻の向きがRectifiedFlowTrainer（t=1がtarget）と逆である点に注意。
    """

    METRIC_NAMES = ["total_loss", "psnr", "ssim"]

    def _cfg_rf(self):
        return self.cfg.algorithm.i2i_rfr_x0

    def _time_condition_value(self, t):
        """時刻チャンネルに入れる値。zeroは時刻情報を与えない（論文準拠）"""
        mode = self._cfg_rf().time_condition
        if mode == "zero":
            return ops.zeros_like(t)
        elif mode == "t":
            return t
        elif mode == "one_minus_t":
            return 1.0 - t
        else:
            raise NotImplementedError(mode)

    def _predict_x0(self, x_t, t, src_x, img_msks, training):
        """x_tからtarget(x0)を直接予測する。tは(b,1,1,1,1)"""
        t_ch = ops.ones_like(x_t) * self._time_condition_value(t)
        net_in = ops.concatenate([x_t * img_msks, src_x, t_ch], axis=-1)
        x0 = self([net_in, img_msks], training=training)
        return x0 * img_msks

    def _sample_t(self, batch_size):
        """密度∝t^pの時刻サンプリング（[t_min, 1]にクリップ）"""
        cfg = self._cfg_rf()
        u = tf.random.uniform((batch_size, 1, 1, 1, 1), 1e-12, 1.0)
        t = u ** (1.0 / (cfg.p + 1.0))
        return ops.clip(t, cfg.t_min, 1.0)

    def _flow_loss(self, y, src_x, img_msks, training):
        """t^-p重み付きx0再構成損失と、x0_pred（メトリクス用）を返す"""
        cfg = self._cfg_rf()
        batch_size = tf.shape(y)[0]
        t = self._sample_t(batch_size)
        eps = tf.random.normal(tf.shape(y))
        x_t = (1.0 - t) * y + t * eps

        x0_pred = self._predict_x0(x_t, t, src_x, img_msks, training)

        diff = (x0_pred - y) * img_msks
        if cfg.loss == "l1":
            per_voxel = ops.abs(diff)
        elif cfg.loss == "mse":
            per_voxel = ops.square(diff)
        else:
            raise NotImplementedError(cfg.loss)

        # サンプルごとのマスク付き平均損失にt^-pの重みをかける
        axes = [1, 2, 3, 4]
        per_sample = ops.sum(per_voxel, axis=axes) / (
            ops.sum(img_msks, axis=axes) + 1e-6
        )
        weight = ops.power(ops.maximum(t[:, 0, 0, 0, 0], 1e-12), -cfg.p)
        loss = ops.mean(per_sample * weight)
        return loss, x0_pred

    def _sample(self, src_x, img_msks, num_steps: int):
        """
        t=1の純ノイズからt=0への数値積分。速度は v = (x_t - x0_pred) / t。
        solver=euler: 1次オイラー（ネットワーク評価はnum_steps回）
        solver=heun: 2次Heun（最終ステップのみオイラー。評価は2*num_steps-1回）
        """
        cfg = self._cfg_rf()
        # 古いoutput.yamlにはsolverが無いためデフォルト(従来動作のeuler)を使う
        solver = cfg.get("solver", "euler")
        assert solver in ("euler", "heun"), solver
        dt = 1.0 / num_steps
        x = tf.random.normal(tf.shape(src_x))
        for n in range(num_steps):
            t_val = max(1.0 - n / num_steps, cfg.t_min)
            t = ops.ones_like(x[:, :1, :1, :1, :1]) * t_val
            x0_pred = self._predict_x0(x, t, src_x, img_msks, training=False)
            v = (x - x0_pred) / t_val
            x_next = x - dt * v
            t_next_val = 1.0 - (n + 1) / num_steps
            if solver == "heun" and t_next_val > 0:
                # 2次補正: 仮ステップ先の速度と平均する（t=0は特異なので最終ステップは1次）
                t_next_val = max(t_next_val, cfg.t_min)
                t2 = ops.ones_like(x[:, :1, :1, :1, :1]) * t_next_val
                x0_pred2 = self._predict_x0(x_next, t2, src_x, img_msks, training=False)
                v2 = (x_next - x0_pred2) / t_next_val
                x_next = x - dt * 0.5 * (v + v2)
            x = x_next
        return x
