import keras
import tensorflow as tf

from optimizer_utils import get_optimizer_iterations

from .image_logger import _take_slice, as_axis_list


class TestImageLogger(keras.callbacks.Callback):
    """
    正解なしのテスト入力画像への推論結果を学習中にTensorBoardへ記録する。
    （edm-torchのTestImageLogger相当）

    - test_dataはdata/dataloader.pyのcreate_test_batch()で作った固定バッチ
      （検証時と同じ前処理: 中心クロップ + norm_spacingへの補間リサンプル済み）
    - 入力画像は学習中変わらないので初回のみ記録し、予測はエポックごとに記録する
    - EMA有効時: EMACallbackはon_test_begin〜on_epoch_end(末尾配置)の間EMA重みを
      スワップインしているため、このコールバックをEMACallbackより前に置けば
      on_epoch_end時点の推論は検証と同じEMA重みで実行される
    """

    def __init__(
        self,
        test_data: dict,
        names: list[str],
        log_dir,
        jit_compile: bool,
        slice_axes="z",
        interval_epochs: int = 1,
    ):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.test_data = test_data
        self.names = names
        self.slice_axes = as_axis_list(slice_axes)
        self.interval_epochs = max(1, int(interval_epochs))
        self.first_log = True

        # 表示用に正規化した入力（[0,1]、マスク済み）を作っておく
        min_vals = tf.reshape(test_data["src_min_clip_vals"], (-1, 1, 1, 1, 1))
        max_vals = tf.reshape(test_data["src_max_clip_vals"], (-1, 1, 1, 1, 1))
        src01 = (test_data["src_imgs"] - min_vals) / (max_vals - min_vals + 1e-6)
        self.src_disp = tf.clip_by_value(src01, 0.0, 1.0) * test_data["img_msks"]

        def predict_step(data):
            # targetが無いのでpredsのみを返すモードで呼ぶ
            return self.model.predict_step(data)

        self.one_step = tf.function(
            predict_step, reduce_retracing=True, jit_compile=jit_compile
        )

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval_epochs != 0:
            return

        preds = self.one_step(self.test_data)

        with self.writer.as_default():
            step = get_optimizer_iterations(self.model.optimizer)
            for i, name in enumerate(self.names):
                for axis in self.slice_axes:
                    if self.first_log:
                        tf.summary.image(
                            f"Test Input/{name}/{axis}",
                            _take_slice(self.src_disp[i : i + 1], axis),
                            step=step,
                        )
                    tf.summary.image(
                        f"Test Prediction/{name}/{axis}",
                        _take_slice(preds[i : i + 1], axis),
                        step=step,
                    )
            self.first_log = False

        self.writer.flush()
