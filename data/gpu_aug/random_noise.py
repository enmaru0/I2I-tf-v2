from functools import partial

import tensorflow as tf

from .filter import gaussian_filter


@tf.function(jit_compile=True)
def random_gaussian_filter_single(x, sigma_range, sigma):
    """
    Apply a random 3D Gaussian blur with a single value of sigma.

    Args:
        x (tf.Tensor): Input images. Shape (depth, height, width, 1).
        sigma_range (list): Range of standard deviation values for random selection.
        sigma (float, optional): Standard deviation of the Gaussian distribution. If not provided, randomly selected from sigma_range.

    Returns:
        tf.Tensor: Blurred image.
    """
    if sigma is None:
        sigma = tf.random.uniform((), sigma_range[0], sigma_range[1])
    return gaussian_filter(x[None], sigma=sigma)[0]


@tf.function(jit_compile=True)
def random_gaussian_filter(imgs, sigma_range=[0.5, 1.8], sigma=None):
    """
    Apply a random 3D Gaussian blur to input images.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        sigma_range (list, optional): Range of standard deviation values for random selection.
        sigma (float, optional): Standard deviation of the Gaussian distribution. If not provided, randomly selected from sigma_range.

    Returns:
        tf.Tensor: Blurred images.
    """
    fn = partial(random_gaussian_filter_single, sigma_range=sigma_range, sigma=sigma)
    return tf.map_fn(fn=fn, elems=imgs)


@tf.function(jit_compile=True)
def random_sharpness(imgs, sigma: int = 3, alpha_range=[0, 2], alpha=None):
    """
    Apply random sharpness variation to input images.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        sigma (int, optional): Standard deviation of the Gaussian distribution for generating blur.
        alpha_range (list, optional): Range of alpha values for random selection.
        alpha (tf.Tensor, optional): Coefficient for controlling the amount of sharpness variation. If not provided, randomly generated.

    Returns:
        tf.Tensor: Transformed images.
    """
    gauss_image = gaussian_filter(imgs, sigma)
    if alpha is None:
        alpha = tf.random.uniform((tf.shape(imgs)[0],), alpha_range[0], alpha_range[1])
        alpha = alpha[:, None, None, None, None]
    imgs += alpha * (imgs - gauss_image)
    return imgs


@tf.function(jit_compile=True)
def random_gaussian_noise(imgs, stddev_range=[0, 0.05], stddev=None):
    """
    Add random Gaussian noise to input images.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        stddev_range (list, optional): Range of standard deviation values for random selection.
        stddev (tf.Tensor, optional): Standard deviation of the Gaussian noise. If not provided, randomly generated.

    Returns:
        tf.Tensor: Transformed images.
    """
    if stddev is None:
        stddev = tf.random.uniform(
            (tf.shape(imgs)[0],), stddev_range[0], stddev_range[1]
        )
        stddev = stddev[:, None, None, None, None]
    gauss = tf.random.normal(tf.shape(imgs), 0, stddev)
    imgs += gauss
    return imgs


@tf.function(jit_compile=True)
def apply_random_gaussian_noise(imgs, prob, stddev_range, stddev=None):
    """
    Apply random Gaussian noise to a subset of images in imgs with probability p.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        prob (float): Probability of applying random Gaussian noise to each image in imgs.
        stddev_range (list, optional): Range of standard deviation values for random selection.
        stddev (tf.Tensor, optional): Standard deviation of the Gaussian noise. If not provided, randomly generated.

    Returns:
        tf.Tensor: Transformed images.
    """
    num_imgs = tf.shape(imgs)[0]
    mask = tf.random.uniform((num_imgs,)) < prob
    mask = tf.cast(mask, tf.float32)[..., None, None, None, None]
    noise_imgs = random_gaussian_noise(imgs, stddev_range=stddev_range, stddev=stddev)
    imgs = mask * noise_imgs + (1 - mask) * imgs
    return imgs


@tf.function(jit_compile=True)
def apply_random_sharpness(imgs, prob, sigma, alpha_range, alpha=None):
    """
    Apply random sharpness variation to a subset of images in imgs with probability p.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        prob (float): Probability of applying random sharpness variation to each image in imgs.
        sigma (int): Standard deviation of the Gaussian distribution for generating blur.
        alpha_range (list): Range of alpha values for random selection.
        alpha (tf.Tensor, optional): Coefficient for controlling the amount of sharpness variation. If not provided, randomly generated.

    Returns:
        tf.Tensor: Transformed images.
    """
    num_imgs = tf.shape(imgs)[0]
    mask = tf.random.uniform((num_imgs,)) < prob
    mask = tf.cast(mask, tf.float32)[..., None, None, None, None]
    sharp_imgs = random_sharpness(
        imgs, sigma=sigma, alpha_range=alpha_range, alpha=alpha
    )
    imgs = mask * sharp_imgs + (1 - mask) * imgs
    return imgs


@tf.function(jit_compile=True)
def apply_random_gaussian_filter(imgs, prob, sigma_range):
    """
    Apply random 3D Gaussian blur to a subset of images in imgs with probability p.

    Args:
        imgs (tf.Tensor): Input images. Shape (batch_size, depth, height, width, 1).
        p (float): Probability of applying random Gaussian blur to each image in imgs.
        sigma_range (list): Range of standard deviation values for random selection.

    Returns:
        tf.Tensor: Transformed images.
    """
    num_imgs = tf.shape(imgs)[0]
    mask = tf.random.uniform((num_imgs,)) < prob
    mask = tf.cast(mask, tf.float32)[..., None, None, None, None]
    filtered_imgs = random_gaussian_filter(imgs, sigma_range=sigma_range)
    imgs = mask * filtered_imgs + (1 - mask) * imgs

    return imgs


@tf.function(jit_compile=True)
def apply_random_sharpness_or_gaussian_filter(
    imgs,
    sharpness_p,
    sharpness_sigma,
    sharpness_alpha_range,
    gaussian_filter_p,
    gaussian_filter_sigma_range,
):
    # randomly apply random sharpness or gaussian filter to a subset of images in imgs
    num_imgs = tf.shape(imgs)[0]

    random_values = tf.random.uniform((num_imgs,))

    mask_sharpness = random_values < sharpness_p
    mask_gaussian = (random_values >= sharpness_p) & (
        random_values < sharpness_p + gaussian_filter_p
    )

    # apply random sharpness
    mask_sharpness = tf.cast(mask_sharpness, tf.float32)[..., None, None, None, None]
    sharp_imgs = random_sharpness(
        imgs, sigma=sharpness_sigma, alpha_range=sharpness_alpha_range
    )

    # apply gaussian filter
    mask_gaussian = tf.cast(mask_gaussian, tf.float32)[..., None, None, None, None]
    gaussian_imgs = random_gaussian_filter(
        imgs, sigma_range=gaussian_filter_sigma_range
    )

    imgs = (
        mask_sharpness * sharp_imgs
        + mask_gaussian * gaussian_imgs
        + (1 - mask_sharpness - mask_gaussian) * imgs
    )

    return imgs
