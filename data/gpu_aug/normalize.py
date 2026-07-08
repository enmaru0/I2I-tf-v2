import tensorflow as tf


@tf.function(jit_compile=True)
def normalize(imgs, min_val, max_val):
    if min_val.ndim == 1:
        if imgs.ndim == 4:
            min_val = min_val[:, None, None, None]
            max_val = max_val[:, None, None, None]
        elif imgs.ndim == 5:
            min_val = min_val[:, None, None, None, None]
            max_val = max_val[:, None, None, None, None]

    imgs = imgs - min_val
    imgs = imgs / (max_val - min_val)
    imgs = tf.clip_by_value(imgs, 0, 1)
    return imgs


@tf.function(jit_compile=True)
def random_normalize(
    imgs,  # (b, z, y, x, c)
    min_val,
    max_val,
    random_center_deviation: float,
    random_range_deviation: float,
    prob: float = 0.5,
):
    """
    入力画像に対してランダムな中心値と範囲の変更を加えた正規化を行います。

    Args:
        imgs : tf.Tensor
            入力画像。形状は (b, z, y, x, c)
        min_val : float または tf.Tensor
            画像の最小値として使用する値。スカラーまたは形状 (b,) のテンソル。
        max_val : float または tf.Tensor
            画像の最大値として使用する値。スカラーまたは形状 (b,) のテンソル。
        random_center_deviation : float
            値の中心に適用されるランダムな偏差の最大値。
        random_range_deviation : float
            値の範囲に適用されるランダムな偏差の最大値。
        prob : float, デフォルトは 0.5
            各バッチのサンプルにランダム正規化を適用する確率。
    Returns:
        ランダムな中心のずれと範囲の変更を適用した正規化された画像。
    """
    batch_size = tf.shape(imgs)[0]

    # Decide for each sample in the batch whether to apply random normalization
    apply_randomization = (
        tf.random.uniform(shape=(batch_size,), minval=0, maxval=1) < prob
    )

    # Ensure the deviations are applied per sample in the batch
    center_deviation = tf.random.uniform(
        shape=(batch_size,),
        minval=-random_center_deviation,
        maxval=random_center_deviation,
    )
    range_deviation = tf.random.uniform(
        shape=(batch_size,),
        minval=-random_range_deviation,
        maxval=random_range_deviation,
    )

    # Apply deviations based on probability
    center_value = (min_val + max_val) / 2
    value_range = max_val - min_val
    augmented_center = center_value + tf.where(
        apply_randomization, center_deviation * min_val, tf.zeros_like(center_deviation)
    )
    augmented_range = value_range + tf.where(
        apply_randomization, range_deviation * max_val, tf.zeros_like(range_deviation)
    )

    augmented_min_val = augmented_center - augmented_range / 2
    augmented_max_val = augmented_center + augmented_range / 2

    # Normalize the images based on modified wl and ww
    normalized_imgs = normalize(imgs, augmented_min_val, augmented_max_val)

    return normalized_imgs


@tf.function(jit_compile=True)
def random_gamma_correction(
    normalized_imgs,  # (b, z, y, x, c)
    random_gamma_range: list = [1.0, 1.0],
    prob: float = 0.5,
):
    """
    入力画像に対してランダムな中心値と範囲の変更を加えた正規化を行います。

    Args:
        imgs : tf.Tensor
            入力画像。形状は (b, z, y, x, c)
        random_gamma_range : list(float)
            値に適用されるランダムなガンマ補正のガンマ値。第一要素が明るくする方向、第二要素が暗くする方向の係数。値は(0.0, 1.0]の範囲で小さいほど変化が大きい。
        prob : float, デフォルトは 0.5
            各バッチのサンプルにランダム正規化を適用する確率。
    Returns:
        ランダムな中心のずれと範囲の変更を適用した正規化された画像。
    """
    if random_gamma_range[0] == 1 and random_gamma_range[1] == 1:
        return normalized_imgs
    batch_size = tf.shape(normalized_imgs)[0]

    # Decide for each sample in the batch whether to apply random normalization
    apply_randomization = (
        tf.random.uniform(shape=(batch_size, 1, 1, 1, 1), minval=0, maxval=1) < prob
    )

    # Ensure the gamma values are applied per sample in the batch
    gamma_value = tf.math.exp(
        tf.random.uniform(
            shape=(batch_size, 1, 1, 1, 1),
            minval=tf.math.log(random_gamma_range[0]),
            maxval=tf.math.log(1.0 / random_gamma_range[1]),
        )
    )

    # Apply gamma correction based on probability
    normalized_imgs = tf.where(
        apply_randomization, tf.math.pow(normalized_imgs, gamma_value), normalized_imgs
    )

    return normalized_imgs
