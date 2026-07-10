import time

import keras
from absl import logging


class WallTimeLimit(keras.callbacks.Callback):
    """指定した学習時間を超えたバッチ終了時にfitを停止する。"""

    def __init__(self, max_minutes: float):
        super().__init__()
        self.max_seconds = float(max_minutes) * 60.0
        self.started_at = None

    def on_train_begin(self, logs=None):
        self.started_at = time.monotonic()

    def on_train_batch_end(self, batch, logs=None):
        if self.started_at is None or self.max_seconds <= 0:
            return
        elapsed = time.monotonic() - self.started_at
        if elapsed >= self.max_seconds:
            logging.info(
                f"Wall-time budget reached: {elapsed / 60.0:.2f} min; stopping training"
            )
            self.model.stop_training = True
