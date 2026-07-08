import keras
import tensorflow as tf


def _take_slice(vol, axis: str):
    """
    (b, z, y, x, c) のボリュームから中央断面を取り出し (b, H, W, c) を返す。
    axis: "z"（軸方向スライス, H=y,W=x） / "y"（H=z,W=x） / "x"（H=z,W=y）
    """
    if axis == "z":
        return vol[:, vol.shape[1] // 2]
    elif axis == "y":
        return vol[:, :, vol.shape[2] // 2]
    elif axis == "x":
        return vol[:, :, :, vol.shape[3] // 2]
    else:
        raise ValueError(f"Invalid slice axis: {axis} (use z/y/x)")


class ImageLogger(keras.callbacks.Callback):
    def __init__(self, val_data, log_dir, jit_compile, slice_axis: str = "z"):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.val_data = val_data
        self.first_log = True
        self.slice_axis = slice_axis

        def predict_step(data):
            return self.model.predict_step(data, return_aux=True)

        self.one_step = tf.function(
            predict_step, reduce_retracing=True, jit_compile=jit_compile
        )

    def on_test_batch_end(self, batch, logs=None):
        """
        Logs the first batch (images and predictions) during validation.
        Only triggered during the first validation step to avoid logging all validation data.
        """
        # Only log during the first validation batch (batch=0)
        if batch > 0:
            return

        _, preds, src_imgs, tgt_imgs, _ = self.one_step(self.val_data)
        axis = self.slice_axis

        with self.writer.as_default():
            step = self.model.optimizer.iterations
            if self.first_log:
                # sourceとtargetは学習中変わらないので初回のみ記録する
                tf.summary.image(
                    "Source Images", _take_slice(src_imgs, axis), step=step
                )
                tf.summary.image(
                    "Target Images", _take_slice(tgt_imgs, axis), step=step
                )
                self.first_log = False
            tf.summary.image("Predicted Images", _take_slice(preds, axis), step=step)
            tf.summary.image(
                "Absolute Error",
                _take_slice(tf.abs(preds - tgt_imgs), axis),
                step=step,
            )

        self.writer.flush()
