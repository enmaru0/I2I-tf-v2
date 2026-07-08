import numpy as np
import tensorflow as tf


@tf.function(jit_compile=True)
def gaussian_kernel_1d(sigma, kernel_size):
    """
    Create a 1D Gaussian kernel based on sigma and truncate.

    Args:
        sigma (float): Standard deviation of the Gaussian kernel.
        kernel_size (int): Size of the kernel.

    Returns:
        1D TensorFlow tensor with the Gaussian kernel.
    """
    radius = kernel_size // 2
    x = tf.range(-radius, radius + 1, dtype=tf.float32)
    kernel = tf.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel


@tf.function(jit_compile=True)
def gaussian_filter(input, sigma, kernel_size=15):
    """
    Apply depthwise 3D Gaussian blur to a TensorFlow tensor.

    Parameters:
        input    : Input 3D tensor, shape: (batch, depth, height, width, channels)
        sigma    : Standard deviation of the Gaussian filter
        kernel_size (int): Size of the kernel.

    Returns:
        Blurred 3D tensor
    """
    assert input.shape[-1], "only supports channel=1"
    assert len(input.shape) == 5, (
        "imgs must have shape (batch_size, depth, height, width, 1)"
    )

    # Pad the input to properly handle borders
    pad = kernel_size // 2
    paddings = [[0, 0], [pad, pad], [pad, pad], [pad, pad], [0, 0]]

    input_padded = tf.pad(
        input, paddings, mode="symmetric"
    )  # equivalent to scipy reflect

    # Generate Gaussian kernel
    kernel_1d = gaussian_kernel_1d(sigma, kernel_size)
    depth_kernel = tf.reshape(kernel_1d, [-1, 1, 1, 1, 1])  # (z, 1, 1, 1, 1)
    height_kernel = tf.reshape(kernel_1d, [1, -1, 1, 1, 1])  # (1, y, 1, 1, 1)
    width_kernel = tf.reshape(kernel_1d, [1, 1, -1, 1, 1])  # (1, 1, x, 1, 1)

    # Apply depthwise separable convolutions
    # (Batch, Depth, Height, Width, Channels)
    strides = [1, 1, 1, 1, 1]
    blurred = tf.nn.conv3d(input_padded, depth_kernel, strides=strides, padding="VALID")
    blurred = tf.nn.conv3d(blurred, height_kernel, strides=strides, padding="VALID")
    blurred = tf.nn.conv3d(blurred, width_kernel, strides=strides, padding="VALID")
    return blurred


if __name__ == "__main__":
    import numpy as np
    from scipy import ndimage

    img = np.random.random((64, 61, 32))
    img_tf = tf.convert_to_tensor(img, tf.float32)[None, ..., None]
    blur_scipy = ndimage.gaussian_filter(img, sigma=2)
    blur_tf = gaussian_filter(img_tf, sigma=2)[0, ..., 0]

    assert False not in np.isclose(blur_scipy, blur_tf, rtol=1e-3)
