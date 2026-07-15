import keras


class ConciseProgbarLogger(keras.callbacks.ProgbarLogger):
    """標準出力には主要指標だけを表示する。"""

    METRIC_NAMES = ("psnr", "ssim", "total_loss")

    def set_params(self, params):
        params = dict(params)
        params["verbose"] = 1
        super().set_params(params)

    @classmethod
    def filter_logs(cls, logs):
        logs = logs or {}
        return {
            name: logs[name] for name in cls.METRIC_NAMES if name in logs
        }

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, self.filter_logs(logs))

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, self.filter_logs(logs))
