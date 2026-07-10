import tensorflow as tf
from keras.src import ops

# image-to-image translation用の損失・評価指標
# いずれも正規化済み([0,1])の画像と有効領域マスク(img_msks)を前提とする


def masked_l1_loss(y_true, y_pred, msk, epsilon: float = 1e-6):
    diff = ops.abs(y_true - y_pred) * msk
    return ops.sum(diff) / (ops.sum(msk) + epsilon)


def masked_l2_loss(y_true, y_pred, msk, epsilon: float = 1e-6):
    diff = ops.square((y_true - y_pred) * msk)
    return ops.sum(diff) / (ops.sum(msk) + epsilon)


def masked_l1_per_sample(y_true, y_pred, msk, epsilon: float = 1e-6):
    axes = tuple(range(1, y_true.ndim))
    diff = ops.abs(y_true - y_pred) * msk
    return ops.sum(diff, axis=axes) / (ops.sum(msk, axis=axes) + epsilon)


def masked_psnr_per_sample(
    y_true, y_pred, msk, max_val: float = 1.0, epsilon: float = 1e-6
):
    axes = tuple(range(1, y_true.ndim))
    squared_error = ops.square((y_true - y_pred) * msk)
    mse = ops.sum(squared_error, axis=axes) / (
        ops.sum(msk, axis=axes) + epsilon
    )
    return 10.0 * ops.log10(max_val**2 / (mse + epsilon))


def masked_psnr(y_true, y_pred, msk, max_val: float = 1.0, epsilon: float = 1e-6):
    return ops.mean(
        masked_psnr_per_sample(y_true, y_pred, msk, max_val, epsilon)
    )


def ssim_per_sample(
    y_true,
    y_pred,
    msk=None,
    max_val: float = 1.0,
    axes=("z",),
    min_slice_mask_coverage: float = 0.0,
):
    """3D画像を指定方向の2Dスライスとして評価し、症例別SSIMを返す。"""
    if y_true.ndim == 4:
        return tf.image.ssim(y_true, y_pred, max_val=max_val)
    if y_true.ndim != 5:
        raise ValueError(f"SSIM expects rank 4 or 5, got {y_true.ndim}")

    if isinstance(axes, str):
        axes = (axes,)
    permutations = {
        "z": (0, 1, 2, 3, 4),
        "y": (0, 2, 1, 3, 4),
        "x": (0, 3, 1, 2, 4),
    }
    axis_scores = []
    for axis in axes:
        if axis not in permutations:
            raise ValueError(f"SSIM axis must be z/y/x, got {axis}")
        perm = permutations[axis]
        true_axis = tf.transpose(y_true, perm)
        pred_axis = tf.transpose(y_pred, perm)
        shape = tf.shape(true_axis)
        true_slices = tf.reshape(true_axis, [-1, shape[2], shape[3], shape[4]])
        pred_slices = tf.reshape(pred_axis, [-1, shape[2], shape[3], shape[4]])
        values = tf.reshape(
            tf.image.ssim(true_slices, pred_slices, max_val=max_val),
            [shape[0], shape[1]],
        )

        if msk is None:
            weights = tf.ones_like(values)
        else:
            mask_axis = tf.transpose(msk, perm)
            mask_slices = tf.reshape(
                mask_axis, [-1, shape[2], shape[3], shape[4]]
            )
            coverage = tf.reshape(
                tf.reduce_mean(mask_slices, axis=(1, 2, 3)),
                [shape[0], shape[1]],
            )
            weights = tf.where(
                coverage >= min_slice_mask_coverage,
                coverage,
                tf.zeros_like(coverage),
            )
        score = tf.reduce_sum(values * weights, axis=1) / (
            tf.reduce_sum(weights, axis=1) + 1e-6
        )
        axis_scores.append(score)
    return tf.reduce_mean(tf.stack(axis_scores, axis=1), axis=1)


def ssim(y_true, y_pred, max_val: float = 1.0):
    """
    SSIMを計算する。3Dボリューム(b, z, y, x, c)はzをバッチ扱いにして
    2Dスライスごとに計算した平均を返す。2D(b, y, x, c)はそのまま計算する。
    パディング領域は事前にマスクで0にしておくこと（厳密なmasked SSIMではない）。
    """
    return tf.reduce_mean(ssim_per_sample(y_true, y_pred, max_val=max_val))
