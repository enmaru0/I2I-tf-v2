from keras.api.losses import categorical_crossentropy
from keras.src import ops


def ce_loss(
    y_true,
    y_logit,
    active_msk=None,
    label_smoothing=0.0,
    epsilon=1e-6,
    num_class=None,
    to_one_hot=False,  # y_trueがone-hotでない場合はTrue
):
    """
    ignore_msk : Tensor or None, optional
        損失計算時に無視するマスク（無視したいところがFalse、計算したいところがTrueのマスク）。
        Noneの場合、全ての値が計算に含まれます。
    """
    if to_one_hot:
        assert y_true.ndim == (y_logit.ndim - 1), (y_true.ndim, y_logit.ndim)
        assert num_class is not None, (
            "num_class must be specified when to_one_hot is True"
        )
    else:
        assert y_true.ndim == y_logit.ndim, (y_true.ndim, y_logit.ndim)

    if to_one_hot:
        y_true = ops.cast(
            ops.one_hot(ops.cast(y_true, "int32"), num_class), y_true.dtype
        )

    ce_loss = categorical_crossentropy(
        y_true, y_logit, from_logits=True, label_smoothing=label_smoothing
    )
    if active_msk is not None:
        active_msk = ops.cast(active_msk, y_true.dtype)
        if active_msk.ndim == 5:
            assert active_msk.shape[-1] == 1, active_msk.shape
            active_msk = ops.squeeze(active_msk, -1)
        assert active_msk.ndim == 4, active_msk.shape
        ce_loss = ce_loss * active_msk
        return ops.sum(ce_loss) / (ops.sum(active_msk) + epsilon)
    else:
        return ops.mean(ce_loss)


def bce_loss(y_true, y_logit, active_msk=None, label_smoothing=0.0, epsilon=1e-6):
    """
    ignore_msk : Tensor or None, optional
        損失計算時に無視するマスク（無視したいところがFalse、計算したいところがTrueのマスク）。
        Noneの場合、全ての値が計算に含まれます。
    """
    if y_true.ndim == (y_logit.ndim - 1):
        y_true = ops.expand_dims(y_true, -1)
    assert y_true.ndim == y_logit.ndim, (y_true.ndim, y_logit.ndim)

    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    ce_loss = ops.binary_crossentropy(y_true, y_logit, from_logits=True)
    if active_msk is not None:
        active_msk = ops.cast(active_msk, y_true.dtype)
        ce_loss = ce_loss * active_msk
        return ops.sum(ce_loss) / (ops.sum(active_msk) + epsilon)
    else:
        return ops.mean(ce_loss)
