import time

import tensorflow as tf
from keras.api.callbacks import TensorBoard

from optimizer_utils import get_optimizer_iterations
from keras.src import ops


class UnifiedTensorBoardLogger(TensorBoard):
    """
    デフォルトのTensorBoardはtrainとvalに分かれたログを出力するが、
    ここではtrainとvalのログを統合して出力するように変更する。
    また、epochではなくstepで記録する
    """

    def __init__(self, log_dir, step_per_epoch, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.step_per_epoch = step_per_epoch
        self.writer = tf.summary.create_file_writer(str(log_dir))

    def _log_epoch_metrics(self, epoch, logs):
        """
        Logs all metrics (training and validation) to a single TensorBoard log.
        """
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
        train_logs = self._collect_learning_rate(train_logs)

        step = get_optimizer_iterations(self.model.optimizer).numpy()

        if self.write_steps_per_second:
            train_logs["steps_per_second"] = self._compute_steps_per_second()

        if train_logs:
            with self.writer.as_default():
                for name, value in train_logs.items():
                    self.summary.scalar(name + "/train", value, step=step)
        if val_logs:
            with self.writer.as_default():
                for name, value in val_logs.items():
                    name = name[4:]  # Remove 'val_' prefix.
                    self.summary.scalar(name + "/val", value, step=step)

    def on_epoch_begin(self, epoch, logs=None):
        # Keeps track of epoch for profiling.
        if self.write_steps_per_second:
            self._epoch_start_time = time.time()

    def _compute_steps_per_second(self):
        time_since_epoch_begin = time.time() - self._epoch_start_time
        time_since_epoch_begin = ops.convert_to_tensor(
            time_since_epoch_begin, "float32"
        )
        steps_per_second = self.step_per_epoch / time_since_epoch_begin
        return float(steps_per_second)

    def on_test_end(self, logs=None):
        self._pop_writer()

    @property
    def _train_writer(self):
        return self.writer

    @property
    def _val_writer(self):
        return self.writer
