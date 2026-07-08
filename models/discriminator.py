from keras import Input, Model, layers


def build_patch_discriminator(
    input_shape: tuple[int, int, int, int],  # z,y,x,c
    base_ch: int = 16,
    num_layers: int = 3,
    name: str = "discriminator",
) -> Model:
    """
    pix2pix用の3D PatchGAN discriminator。
    source画像を条件として結合し、パッチごとに真偽を判定するlogitsを返す。
    https://arxiv.org/abs/1611.07004
    """
    input_src = Input(shape=input_shape, name="src_image")
    input_img = Input(shape=input_shape, name="image")  # targetまたは生成画像

    x = layers.Concatenate()([input_src, input_img])
    for i in range(num_layers):
        x = layers.Conv3D(
            filters=base_ch << i,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=(i == 0),  # 最初の層は正規化しないのでbiasを使う
            name=f"disc_conv{i}",
            kernel_initializer="he_uniform",
        )(x)
        if i > 0:
            x = layers.BatchNormalization(name=f"disc_bn{i}")(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)

    x = layers.Conv3D(
        filters=base_ch << num_layers,
        kernel_size=4,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"disc_conv{num_layers}",
        kernel_initializer="he_uniform",
    )(x)
    x = layers.BatchNormalization(name=f"disc_bn{num_layers}")(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # パッチごとの真偽logits
    logits = layers.Conv3D(
        filters=1,
        kernel_size=4,
        strides=1,
        padding="same",
        use_bias=True,
        name="disc_logits",
        kernel_initializer="he_uniform",
    )(x)

    return Model(inputs=[input_src, input_img], outputs=logits, name=name)
