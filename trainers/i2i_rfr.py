import keras

from .rectified_flow import RectifiedFlowTrainer


@keras.saving.register_keras_serializable()
class I2IRFRTrainer(RectifiedFlowTrainer):
    """
    I2I-RFR: Improving Image-to-Image Translation via a Rectified Flow
    Reformulation。

    標準のrectified flow（ノイズ→target）と異なり、フローの出発点を
    source画像そのものに置き換える:
        x0 = source + noise_std * ε
        x_t = (1-t) * x0 + t * target
        v = target - x0
    デノイズ・ぼかし修正のようにsourceとtargetが近いタスクでは
    輸送距離が短くなり、少ステップ（1ステップも可）で高品質が狙える。

    noise_stdは出発点への微小ノイズ注入の強さ（作業空間[-1,1]基準）。
    0にすると完全に決定的なフローになる。
    sourceは条件チャンネルとしても入力し続ける（フロー途中でも
    劣化前の情報を常に参照できるようにするため）。
    学習ループ・サンプリングはRectifiedFlowTrainerをそのまま継承する。
    """

    def _cfg_rf(self):
        return self.cfg.algorithm.i2i_rfr

    def _initial_state(self, src_x, sample_seeds=None, salt: int = 0):
        noise_std = self._cfg_rf().noise_std
        if noise_std > 0:
            noise = self._normal_like(src_x, sample_seeds, salt=salt)
            return src_x + noise_std * noise
        return src_x
