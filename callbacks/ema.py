import keras
import tensorflow as tf
from absl import logging


class EMACallback(keras.callbacks.Callback):
    """
    generatorの重みの指数移動平均(EMA)を保持するコールバック。
    拡散モデルやフロー系（edm / rectified_flow / i2i_rfr）で特に効果が大きい。

    動作:
    - 学習バッチごとにshadow = decay*shadow + (1-decay)*weight で更新
    - 検証(on_test_begin)〜エポック終了(on_epoch_end)の間、generatorに
      EMA重みをスワップする。これにより
        * 検証メトリクス(val_psnr等)がEMA重みで計算される
        * ModelCheckpointがEMA重みを保存する
        * ImageLoggerがEMA重みで画像を記録する
      をすべて満たす。
    - エポック終了時に生の学習重みへ戻す（次エポックの学習は生の重みで継続）

    重要: このコールバックはチェックポイント保存後にスワップを戻す必要があるため、
    必ずModelCheckpointより後（callbacksリストの末尾）に置くこと。

    既知の制限: model_latest.keras はEMA重みで保存されるため、restoreで学習を
    再開するとEMA重みから継続することになる（optimizerの状態は生の重み基準）。
    厳密な再開が必要な場合はEMAを無効にすること。
    """

    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        self.shadow = None
        self.backup = None

    def _generator_weights(self):
        return self.model.generator.weights

    def on_train_begin(self, logs=None):
        if self.shadow is None:
            # 現在の重みでshadowを初期化する（restore/finetune後の重みが入る）
            self.shadow = [
                tf.Variable(w, trainable=False) for w in self._generator_weights()
            ]
            logging.info(
                f"EMA initialized (decay={self.decay}, "
                f"{len(self.shadow)} weight tensors)"
            )

    @tf.function
    def _update_shadow(self):
        for s, w in zip(self.shadow, self._generator_weights()):
            s.assign(self.decay * s + (1.0 - self.decay) * w)

    def on_train_batch_end(self, batch, logs=None):
        self._update_shadow()

    def on_test_begin(self, logs=None):
        if self.shadow is None:
            return
        # 生の重みを退避し、EMA重みをスワップイン
        self.backup = [tf.identity(w) for w in self._generator_weights()]
        for w, s in zip(self._generator_weights(), self.shadow):
            w.assign(s)

    def on_epoch_end(self, epoch, logs=None):
        # ModelCheckpointの保存後に呼ばれる（末尾配置が前提）ので生の重みに戻す
        if self.backup is None:
            return
        for w, b in zip(self._generator_weights(), self.backup):
            w.assign(b)
        self.backup = None
