from keras.src import ops


def dice_loss_per_channel(
    y_true,
    y_pred,
    smooth: float = 1.0,
    epsilon: float = 1e-6,
    active_msk=None,
    flip_value=False,
):
    if flip_value:
        y_true = 1 - y_true
        y_pred = 1 - y_pred
    if active_msk is not None:
        y_pred = y_pred * active_msk
        y_true = y_true * active_msk

    y_tp = y_true * y_pred

    denom_true = ops.sum(y_true * y_true, axis=(1, 2, 3), keepdims=False)
    denom_pred = ops.sum(y_pred * y_pred, axis=(1, 2, 3), keepdims=False)

    denom = denom_true + denom_pred
    numer = 2 * ops.sum(y_tp, axis=(1, 2, 3), keepdims=False)

    dice_scores = (numer + smooth) / (denom + smooth)
    loss = 1 - dice_scores
    loss = ops.clip(loss, 0, 1)  # shape: (b, c)
    if active_msk is not None:
        active_msk_class = ops.sum(active_msk, axis=(1, 2, 3), keepdims=False) > 0
        active_msk_class = ops.cast(active_msk_class, "float32")
        loss = ops.sum(loss, axis=0) / (ops.sum(active_msk_class, axis=0) + epsilon)
    else:
        loss = ops.mean(loss, axis=0)
    return loss  # shape: (c,)


def dice_score_per_channel(y_true, y_pred, epsilon: float = 1e-6, active_msk=None):
    # (b, z, y, x, c)を想定し、zyxでDICEを計算する
    # convert to boolean
    if active_msk is not None:
        y_pred = y_pred * active_msk
        y_true = y_true * active_msk

    y_pred = ops.cast(y_pred > 0.5, "float32")
    y_true = ops.cast(y_true > 0.5, "float32")

    y_tp = y_true * y_pred

    denom_true = ops.sum(y_true, axis=(1, 2, 3), keepdims=False)
    denom_pred = ops.sum(y_pred, axis=(1, 2, 3), keepdims=False)

    denom = denom_true + denom_pred
    numer = 2 * ops.sum(y_tp, axis=(1, 2, 3), keepdims=False)

    score = numer / (denom + epsilon)
    if active_msk is not None:
        active_msk_class = ops.sum(active_msk, axis=(1, 2, 3), keepdims=False) > 0
        active_msk_class = ops.cast(active_msk_class, "float32")
        score = ops.sum(score, axis=0) / (ops.sum(active_msk_class, axis=0) + epsilon)
    else:
        score = ops.mean(score, axis=0)
    return score  # shape: (c,)
