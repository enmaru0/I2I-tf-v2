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


def masked_psnr(y_true, y_pred, msk, max_val: float = 1.0, epsilon: float = 1e-6):
    mse = ops.sum(ops.square((y_true - y_pred) * msk)) / (ops.sum(msk) + epsilon)
    return 10.0 * ops.log10(max_val**2 / (mse + epsilon))


def ssim(y_true, y_pred, max_val: float = 1.0):
    """
    SSIMを計算する。3Dボリューム(b, z, y, x, c)はzをバッチ扱いにして
    2Dスライスごとに計算した平均を返す。2D(b, y, x, c)はそのまま計算する。
    パディング領域は事前にマスクで0にしておくこと（厳密なmasked SSIMではない）。
    """
    if y_true.ndim == 5:
        shape = y_true.shape
        y_true = tf.reshape(y_true, [-1, shape[2], shape[3], shape[4]])
        y_pred = tf.reshape(y_pred, [-1, shape[2], shape[3], shape[4]])
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_val))
