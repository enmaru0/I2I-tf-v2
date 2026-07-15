import keras
import tensorflow as tf

from optimizer_utils import get_optimizer_iterations


AXIS_ALIASES = {
    "z": "z",
    "ax": "z",
    "axial": "z",
    "y": "y",
    "cor": "y",
    "coronal": "y",
    "x": "x",
    "sag": "x",
    "sagittal": "x",
}
AXIS_DISPLAY_NAMES = {"z": "AX", "y": "COR", "x": "SAG"}


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


def as_axis_list(slice_axes) -> list[str]:
    """log_axis設定をz/y/xへ正規化する（AX/COR/SAG表記も受ける）。"""
    if isinstance(slice_axes, str):
        requested_axes = [slice_axes]
    else:
        requested_axes = [str(a) for a in slice_axes]

    axes = []
    for requested_axis in requested_axes:
        key = str(requested_axis).lower()
        assert key in AXIS_ALIASES, (
            f"log_axisはz/y/xまたはAX/COR/SAGで指定してください: {requested_axis}"
        )
        axis = AXIS_ALIASES[key]
        if axis not in axes:
            axes.append(axis)
    return axes


def axis_display_name(axis: str) -> str:
    return AXIS_DISPLAY_NAMES[axis]


def axis_section_name(prefix: str, axis: str) -> str:
    return f"{prefix} ({axis_display_name(axis)})"


class ImageLogger(keras.callbacks.Callback):
    def __init__(self, val_data, log_dir, jit_compile, slice_axes="z"):
        super().__init__()
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.val_data = val_data
        self.first_log = True
        self.slice_axes = as_axis_list(slice_axes)

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

        with self.writer.as_default():
            step = get_optimizer_iterations(self.model.optimizer)
            for axis in self.slice_axes:
                if self.first_log:
                    # sourceとtargetは学習中変わらないので初回のみ記録する
                    tf.summary.image(
                        axis_section_name("Source Images", axis),
                        _take_slice(src_imgs, axis),
                        step=step,
                    )
                    tf.summary.image(
                        axis_section_name("Target Images", axis),
                        _take_slice(tgt_imgs, axis),
                        step=step,
                    )
                tf.summary.image(
                    axis_section_name("Predicted Images", axis),
                    _take_slice(preds, axis),
                    step=step,
                )
                tf.summary.image(
                    axis_section_name("Absolute Error", axis),
                    _take_slice(tf.abs(preds - tgt_imgs), axis),
                    step=step,
                )
            self.first_log = False

        self.writer.flush()
