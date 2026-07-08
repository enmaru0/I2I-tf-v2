from pathlib import Path

import commonlib
import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw

from .dataloader import get_clip_vals, resolve_target_hdr_path
from .utils import add_channel_dim, add_margin, get_pad_for_margin


def rescale_cpp_img_with_out_size(data_np, src_spacing, dst_spacing, out_size):
    rate_zyx = src_spacing / dst_spacing
    img_filter_z = commonlib.ImageFilter.ANTIALIASING(rate_zyx[0])
    img_filter_y = commonlib.ImageFilter.ANTIALIASING(rate_zyx[1])
    img_filter_x = commonlib.ImageFilter.ANTIALIASING(rate_zyx[2])

    rescaled_transform = commonlib.RescaleTransform3DWithFilter(
        img_filter_z,
        img_filter_y,
        img_filter_x,
        commonlib.ImageFilter.DEFAULT_OUT(np.int16),
        commonlib.ImageFilter.DEFAULT_IN,
        commonlib.ImageFilter.ZERO_OVER,
    )

    rescaled_transform.SetOrgImageSize(*data_np.shape[::-1])
    rescaled_transform.SetResultImageSize(*out_size[::-1])
    rescaled_transform.SetOrgImage(data_np)
    out = rescaled_transform.Transform()
    return out


def preprocess_image_np_test(img_hdr_list_with_data_name: list[bytes], cfg):
    src_hdr_path, dataname = img_hdr_list_with_data_name
    dataname = dataname.decode()  # noqa: F841
    src_hdr_path = Path(src_hdr_path.decode())

    img_size_zyx, _, spacing_zyx = read_hdr(src_hdr_path)

    target_spacing_zyx = cfg.aug.affine.norm_spacing_zyx

    # I2Iでは画像全体を処理対象とする。
    # モデルのプーリング回数(depth)で割り切れるサイズになるようマージンを計算する
    crop_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    margin_mm_zyx, resize_zyx = get_pad_for_margin(
        crop_zyxzyx,
        dilation_mm_zyx=(0,) * 3,
        size_zyx=img_size_zyx,
        src_spacing_zyx=spacing_zyx,
        dst_spacing=target_spacing_zyx,
        num_pool=cfg.model.unet.depth,
        return_dst_size=True,
    )

    crop_zyxzyx = add_margin(
        crop_zyxzyx,
        img_size_zyx,
        margin_mm_zyx,
        spacing_zyx,
        round_to_int=False,
        pad_remain=True,
    )
    crop_zyxzyx = crop_zyxzyx.astype(np.int32)

    img = read_raw(src_hdr_path, clip_zyxzyx=crop_zyxzyx)
    img = rescale_cpp_img_with_out_size(
        img, spacing_zyx, target_spacing_zyx, resize_zyx
    )

    src_min_val, src_max_val = get_clip_vals(src_hdr_path, cfg)
    tgt_hdr_path = resolve_target_hdr_path(src_hdr_path, cfg.data.target_suffix)
    if cfg.image.share_normalization or not tgt_hdr_path.exists():
        # 推論対象にtargetが存在しない場合もsourceの正規化値で出力を復元する
        tgt_min_val, tgt_max_val = src_min_val, src_max_val
    else:
        tgt_min_val, tgt_max_val = get_clip_vals(tgt_hdr_path, cfg)

    # チャンネルの次元を追加
    img = add_channel_dim(img)

    return (
        img,
        crop_zyxzyx,
        spacing_zyx,
        img_size_zyx,
        np.array(src_min_val, np.float32),
        np.array(src_max_val, np.float32),
        np.array(tgt_min_val, np.float32),
        np.array(tgt_max_val, np.float32),
        str(src_hdr_path.stem).encode(),
    )


def preprocess_image_test(img_hdr_path_with_data_name, cfg):
    def _preprocess_image_np(img_hdr_path_with_data_name):
        return preprocess_image_np_test(img_hdr_path_with_data_name, cfg)

    return tf.numpy_function(
        func=_preprocess_image_np,
        inp=[img_hdr_path_with_data_name],
        Tout=[
            tf.int16,  # img
            tf.int32,  # crop_zyxzyx
            tf.float32,  # spacing_zyx
            tf.uint32,  # img_size_zyx
            tf.float32,  # src_min_val
            tf.float32,  # src_max_val
            tf.float32,  # tgt_min_val
            tf.float32,  # tgt_max_val
            tf.string,  # key
        ],
    )


def make_dict_test(
    img,
    crop_zyxzyx,
    spacing_zyx,
    img_size_zyx,
    src_min_clip_val,
    src_max_clip_val,
    tgt_min_clip_val,
    tgt_max_clip_val,
    img_key,
):
    data = dict(
        img=tf.cast(img, tf.float32),
        # テスト時はパディングがないので有効領域マスクは全1とする
        img_msk=tf.ones_like(img, dtype=tf.float32),
        crop_zyxzyx=crop_zyxzyx,
        spacing_zyx=spacing_zyx,
        img_size_zyx=img_size_zyx,
        src_min_clip_val=src_min_clip_val,
        src_max_clip_val=src_max_clip_val,
        tgt_min_clip_val=tgt_min_clip_val,
        tgt_max_clip_val=tgt_max_clip_val,
        img_key=img_key,
    )

    return data


def create_dataloader_test(img_hdr_dict: dict, cfg):
    img_hdr_path_list = []
    for value in img_hdr_dict.values():
        img_hdr_path_list += value["img_hdr_list"]

    dataset_list = []
    for data_name, value in img_hdr_dict.items():
        # データセットごとに処理を変えることを想定してデータセット名を付与する（処理はpreprocess_image_npで実装）
        img_hdr_list_with_data_name = [
            (str(path), data_name) for path in sorted(value["img_hdr_list"])
        ]
        _dataset = tf.data.Dataset.from_tensor_slices(img_hdr_list_with_data_name)

        dataset_list.append(_dataset)
        logging.info(f"Dataset {data_name} has {len(value['img_hdr_list'])} images.")

    dataset = tf.data.Dataset.sample_from_datasets(dataset_list)

    def _preprocess_image(img_hdr_path_with_data_name):
        return preprocess_image_test(img_hdr_path_with_data_name, cfg)

    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=cfg.num_workers
    )  # autotuneはなんか遅かった・・・

    # 他で使いやすいように辞書型で保持する
    dataset = dataset.map(make_dict_test)
    dataset = dataset.prefetch(buffer_size=cfg.prefetch_size)

    return dataset
