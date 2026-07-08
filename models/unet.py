from keras import Input, Model, layers
from keras.src import ops

from .layers import BatchRenormalization


def conv_block(img, img_msk, filters: int, name_conv: str, renorm: dict, kernel_size=3):
    x = layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        name=name_conv,
        kernel_initializer="he_uniform",
    )(img)
    squeeze_img_msk = ops.squeeze(img_msk)
    x = BatchRenormalization(
        **renorm, synchronized=False, name=name_conv.replace("conv", "bn")
    )(x, mask=squeeze_img_msk)
    x = x * img_msk
    x = layers.Activation("relu")(x)
    return x


def encode_pool(img, img_msk, pool_size=2):
    img_pooled = layers.MaxPooling3D(
        pool_size=pool_size, strides=pool_size, padding="same"
    )(img)
    msk_pooled = layers.MaxPooling3D(
        pool_size=pool_size, strides=pool_size, padding="same"
    )(img_msk)
    return img_pooled, msk_pooled


def decode_up(img, filters: int, name: str, kernel_size=4, strides=2):
    return layers.Conv3DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        name=name,
        kernel_initializer="he_uniform",
        use_bias=True,
    )(img)


def decoder_last(
    img,
    img_msk,
    skip_tensor,
    filters: int,
    num_channel: int,
    num_decode_blocks: int,
    renorm: dict,
    conv_kernel_size=3,
    up_kernel_size=4,
    up_strides=2,
):
    x = layers.Conv3DTranspose(
        filters=filters,
        kernel_size=up_kernel_size,
        strides=up_strides,
        padding="same",
        name="dec0_transposeconv0",
        kernel_initializer="he_uniform",
        use_bias=True,
    )(img)
    x = layers.Concatenate()([x, skip_tensor])

    for e in range(num_decode_blocks - 1):
        x = conv_block(
            x, img_msk, filters, f"dec0_conv{e}", renorm, kernel_size=conv_kernel_size
        )

    x = layers.Conv3D(
        filters=num_channel,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name=f"dec0_conv{num_decode_blocks - 1}",
        kernel_initializer="he_uniform",
    )(x)
    return x


def build_unet(
    input_shape: tuple[int, int, int, int],  # z,y,x,c
    num_channel: int,
    start_ch: int,
    depth: int,
    num_encode_blocks: int,
    num_decode_blocks: int,
    name: str = "unet",
    mode: str = "3d",
    **renorm: dict,
) -> Model:
    """
    mode（スライス方向=z方向の畳み込み量を切り替える。edm-torchのmode相当）:
      - "3d":    全レベルで3x3x3畳み込み・2x2x2プーリング（従来動作）
                 入力zは 2^depth で割り切れる必要がある
      - "mixed": 最上位レベル（最高解像度）のみ3D畳み込み+zプーリングを行い、
                 深いレベルはスライス面内(1x3x3)畳み込み・xyのみのプーリング。
                 zの畳み込み・縮小は最上位の1回だけ（入力zは2で割り切れればよい）
      - "2d":    全レベルでスライス面内畳み込み。z方向は畳み込みも縮小も行わない
                 （スライス独立処理。入力zの割り切れ制約なし）
    """
    assert mode in ("3d", "mixed", "2d"), mode

    def _use_z(level: int) -> bool:
        """このレベルの畳み込み・直後のプーリングがz方向に作用するか"""
        return mode == "3d" or (mode == "mixed" and level == 0)

    def _kernel(level: int):
        return 3 if _use_z(level) else (1, 3, 3)

    def _pool(level: int):
        return 2 if _use_z(level) else (1, 2, 2)

    def _up_kernel(level: int):
        return 4 if _use_z(level) else (1, 4, 4)

    def _up_stride(level: int):
        return 2 if _use_z(level) else (1, 2, 2)

    input_img = Input(shape=input_shape, name="image")
    input_img_msk = Input(shape=input_shape[:3] + (1,), name="mask")

    ## Encoder
    skips = []
    skips_msk = [input_img_msk]
    x = input_img
    img_msk = input_img_msk
    for d in range(depth):
        for e in range(num_encode_blocks):
            x = conv_block(
                x, img_msk, start_ch << d, f"enc{d}_conv{e}", renorm,
                kernel_size=_kernel(d),
            )
        x_pooled, msk_pooled = encode_pool(x, img_msk, pool_size=_pool(d))
        skips.append(x)
        skips_msk.append(msk_pooled)
        x, img_msk = x_pooled, msk_pooled

    # Bottleneck
    for b in range(num_encode_blocks):
        x = conv_block(
            x, img_msk, start_ch << depth, f"bottom_conv{b}", renorm,
            kernel_size=_kernel(depth),
        )

    skips_msk = skips_msk[:-1]  # delete the last one
    # Decoder
    for d in reversed(range(1, depth)):
        _skip_x = skips.pop()
        _skip_msk = skips_msk.pop()

        # decode_upはレベルdの後段プーリングを逆転させるのでstrideも対応させる
        x = decode_up(
            x, start_ch << d, f"dec{d}_transposeconv0",
            kernel_size=_up_kernel(d), strides=_up_stride(d),
        )
        x = layers.Concatenate()([x, _skip_x])
        for e in range(num_decode_blocks):
            x = conv_block(
                x, _skip_msk, start_ch << d, f"dec{d}_conv{e}", renorm,
                kernel_size=_kernel(d),
            )

    # last decode
    _skip_x = skips.pop()
    _skip_msk = skips_msk.pop()
    outputs = decoder_last(
        x,
        _skip_msk,
        _skip_x,
        start_ch,
        num_channel,
        num_decode_blocks,
        renorm,
        conv_kernel_size=_kernel(0),
        up_kernel_size=_up_kernel(0),
        up_strides=_up_stride(0),
    )

    return Model(inputs=[input_img, input_img_msk], outputs=outputs, name=name)
