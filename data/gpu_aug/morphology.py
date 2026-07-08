import tensorflow as tf


def binary_erosion_3d(input, structure=None, iterations=1):
    """
    Perform 3D morphological erosion using TensorFlow, supporting arbitrary structuring elements and multiple iterations.

    Parameters
    ----------
    input : tf.Tensor
        5D tensor of shape (batch, depth, height, width, channels), binary or float.
    structure : np.ndarray or tf.Tensor, optional
        3D structuring element (like scipy.ndimage). If None, uses 6-neighborhood.
    iterations : int, optional
        Number of times erosion is applied. Default is 1.

    Returns
    -------
    tf.Tensor
        Eroded tensor of the same shape as input (binary mask).
    """
    if structure is None:
        structure = tf.convert_to_tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=tf.float32,
        )[..., None, None]
    else:
        structure = structure[..., None, None]

    x = input
    for _ in range(iterations):
        conv_result = tf.nn.conv3d(
            x, structure, strides=[1, 1, 1, 1, 1], padding="SAME"
        )
        x = tf.cast(conv_result == tf.reduce_sum(structure), input.dtype)
    return x


def binary_dilation_3d(input, structure=None, iterations=1):
    """
    Perform 3D morphological dilation using TensorFlow, supporting arbitrary structuring elements and multiple iterations.

    Parameters
    ----------
    input : tf.Tensor
        5D tensor of shape (batch, depth, height, width, channels), binary or float.
    structure : np.ndarray or tf.Tensor, optional
        3D structuring element (like scipy.ndimage). If None, uses 6-neighborhood.
    iterations : int, optional
        Number of times dilation is applied. Default is 1.

    Returns
    -------
    tf.Tensor
        Dilated tensor of the same shape as input (binary mask).
    """
    if structure is None:
        structure = tf.convert_to_tensor(
            [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=tf.float32,
        )[..., None, None]
    else:
        structure = structure[..., None, None]

    x = input
    for _ in range(iterations):
        conv_result = tf.nn.conv3d(
            x, structure, strides=[1, 1, 1, 1, 1], padding="SAME"
        )
        x = tf.cast(conv_result > 0, input.dtype)
    return x


def main():
    import numpy as np
    from scipy import ndimage

    a = np.zeros((7, 7, 7), dtype=int)
    a[1:6, 2:5, 3:8] = 1

    # 3d structure for 6 neighborhood
    structure1 = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ]
    )
    # 3d structure for 26 neighborhood
    structure2 = np.ones((3, 3, 3), dtype=bool)

    # Prepare input for TensorFlow: shape (batch, depth, height, width, channels)
    a_tf = a[None, ..., None].astype(np.float32)
    tf_input = tf.convert_to_tensor(a_tf)

    # --- Erosion tests ---
    print("Erosion tests:")
    eroded1 = ndimage.binary_erosion(a, structure=structure1).astype(a.dtype)
    tf_structure1 = structure1.astype(np.float32)
    tf_out1 = binary_erosion_3d(tf_input, structure=tf_structure1)
    tf_out_bin1 = tf_out1.numpy()[0, ..., 0].astype(a.dtype)
    print("6-neighborhood equal?", np.array_equal(eroded1, tf_out_bin1))

    eroded2 = ndimage.binary_erosion(a, structure=structure2).astype(a.dtype)
    tf_structure2 = structure2.astype(np.float32)
    tf_out2 = binary_erosion_3d(tf_input, structure=tf_structure2)
    tf_out_bin2 = tf_out2.numpy()[0, ..., 0].astype(a.dtype)
    print("26-neighborhood equal?", np.array_equal(eroded2, tf_out_bin2))

    # --- Dilation tests ---
    print("\nDilation tests:")
    dilated1 = ndimage.binary_dilation(a, structure=structure1).astype(a.dtype)
    tf_out1 = binary_dilation_3d(tf_input, structure=tf_structure1)
    tf_out_bin1 = tf_out1.numpy()[0, ..., 0].astype(a.dtype)
    print("6-neighborhood equal?", np.array_equal(dilated1, tf_out_bin1))

    dilated2 = ndimage.binary_dilation(a, structure=structure2).astype(a.dtype)
    tf_out2 = binary_dilation_3d(tf_input, structure=tf_structure2)
    tf_out_bin2 = tf_out2.numpy()[0, ..., 0].astype(a.dtype)
    print("26-neighborhood equal?", np.array_equal(dilated2, tf_out_bin2))

    # --- Erosion tests with iterations ---
    print("\nErosion tests with iterations=2:")
    eroded1_iter = ndimage.binary_erosion(a, structure=structure1, iterations=2).astype(
        a.dtype
    )
    tf_out1_iter = binary_erosion_3d(tf_input, structure=tf_structure1, iterations=2)
    tf_out_bin1_iter = tf_out1_iter.numpy()[0, ..., 0].astype(a.dtype)
    print(
        "6-neighborhood (2 iter) equal?", np.array_equal(eroded1_iter, tf_out_bin1_iter)
    )

    eroded2_iter = ndimage.binary_erosion(a, structure=structure2, iterations=2).astype(
        a.dtype
    )
    tf_out2_iter = binary_erosion_3d(tf_input, structure=tf_structure2, iterations=2)
    tf_out_bin2_iter = tf_out2_iter.numpy()[0, ..., 0].astype(a.dtype)
    print(
        "26-neighborhood (2 iter) equal?",
        np.array_equal(eroded2_iter, tf_out_bin2_iter),
    )

    # --- Dilation tests with iterations ---
    print("\nDilation tests with iterations=2:")
    dilated1_iter = ndimage.binary_dilation(
        a, structure=structure1, iterations=2
    ).astype(a.dtype)
    tf_out1_iter = binary_dilation_3d(tf_input, structure=tf_structure1, iterations=2)
    tf_out_bin1_iter = tf_out1_iter.numpy()[0, ..., 0].astype(a.dtype)
    print(
        "6-neighborhood (2 iter) equal?",
        np.array_equal(dilated1_iter, tf_out_bin1_iter),
    )

    dilated2_iter = ndimage.binary_dilation(
        a, structure=structure2, iterations=2
    ).astype(a.dtype)
    tf_out2_iter = binary_dilation_3d(tf_input, structure=tf_structure2, iterations=2)
    tf_out_bin2_iter = tf_out2_iter.numpy()[0, ..., 0].astype(a.dtype)
    print(
        "26-neighborhood (2 iter) equal?",
        np.array_equal(dilated2_iter, tf_out_bin2_iter),
    )


if __name__ == "__main__":
    main()
