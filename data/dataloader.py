import multiprocessing
import random
import zlib
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from absl import logging
from irg import read_hdr, read_raw
from scipy.ndimage import gaussian_filter, uniform_filter1d, zoom
from scipy.signal import fftconvolve
from tqdm import tqdm

from .dataloader_utils import (
    load_intensity,
    load_organ_box,
    save_intensity,
    save_organ_box,
)
from .utils import (
    AffineTransform,
    add_channel_dim,
    calc_img_crop_region,
    center_within_image,
    crop_input,
    prepare_thin2thick,
    random_crop_center_within_bb,
    virtual_thick_generator,
)


def resolve_target_hdr_path(src_hdr_path: Path, cfg) -> Path:
    """
    sourceのhdrパスから対応するtargetのhdrパスを返す。
    - paired: 同一フォルダで xxx{target_suffix}.hdr
    - paired_dir: target_data_dir配下の同名ファイル
      （data_dirからの相対パス {split}/{dataset}/{filename} を維持する）
    """
    src_hdr_path = Path(src_hdr_path)
    if cfg.data.mode == "paired_dir":
        rel = src_hdr_path.relative_to(Path(cfg.data_dir))
        return Path(cfg.data.target_data_dir) / rel
    target_suffix = cfg.data.target_suffix
    assert target_suffix.startswith("."), "target_suffix must start with ."
    return src_hdr_path.with_suffix(target_suffix + ".hdr")


def make_motion_blur_kernel(direction_zyx, length: float) -> np.ndarray:
    """
    手ブレ様の直線PSFカーネルを作る。
    direction_zyx: ブレ方向（voxel空間。正規化不要）
    length: ブレの長さ（voxel）
    直線上に等間隔にサンプル点を置き、トリリニアに堆積して正規化する。
    """
    d = np.asarray(direction_zyx, np.float32)
    d = d / (np.linalg.norm(d) + 1e-8)
    ksize = max(int(np.ceil(length)) | 1, 3)  # 奇数サイズ
    kernel = np.zeros((ksize, ksize, ksize), np.float32)
    center = (ksize - 1) / 2.0

    half = (length - 1.0) / 2.0
    num_taps = max(int(np.ceil(length) * 4), 8)
    for t in np.linspace(-half, half, num_taps):
        p = center + d * t
        i0 = np.floor(p).astype(np.int64)
        f = p - i0
        for dz in (0, 1):
            wz = f[0] if dz else 1.0 - f[0]
            for dy in (0, 1):
                wy = f[1] if dy else 1.0 - f[1]
                for dx in (0, 1):
                    wx = f[2] if dx else 1.0 - f[2]
                    z, y, x = i0[0] + dz, i0[1] + dy, i0[2] + dx
                    if 0 <= z < ksize and 0 <= y < ksize and 0 <= x < ksize:
                        kernel[z, y, x] += wz * wy * wx
    kernel /= kernel.sum() + 1e-8
    return kernel


def apply_motion_blur(img: np.ndarray, direction_zyx, length: float) -> np.ndarray:
    """直線PSFをFFT畳み込みで適用する（境界はreflectで暗化を防ぐ）"""
    kernel = make_motion_blur_kernel(direction_zyx, length)
    pad = kernel.shape[0] // 2
    padded = np.pad(img, pad, mode="reflect")
    out = fftconvolve(padded, kernel, mode="same")
    out = out[pad:-pad, pad:-pad, pad:-pad]
    return out.astype(np.float32, copy=False)


# through-plane SRの劣化軸（データはz,y,x順）: AX=z / COR=y / SAG=x
SR_AXIS_TO_DIM = {"AX": 0, "COR": 1, "SAG": 2}
CONTINUOUS_SR_SIGMA_FACTOR = 0.32


def _normalize_sr_interpolation(slice_interpolation: str) -> tuple[int, bool]:
    """self_srの補間設定をscipy.zoomのorder/prefilterへ変換する。"""
    method = str(slice_interpolation).lower().replace("_", "-")
    if method == "linear":
        return 1, False
    if method == "spline":
        return 3, True
    if method in ("b-spline", "bspline", "b-spine", "bspine"):
        return 3, False
    raise NotImplementedError(slice_interpolation)


def _resize_axis0(
    vol: np.ndarray, out_n: int, order: int, prefilter: bool = True
) -> np.ndarray:
    """軸0のスライス数をout_nに変える（scipy.zoom。端数はcrop/padで厳密に合わせる）"""
    out_n = int(out_n)
    if vol.shape[0] == out_n:
        return vol.astype(np.float32, copy=False)
    order = int(order)
    if order > 1 and vol.shape[0] < 4:
        order = 1  # スライス数が少なすぎるときはスプラインを使わない
    out = zoom(
        vol,
        (out_n / vol.shape[0], 1.0, 1.0),
        order=order,
        prefilter=bool(prefilter) and order > 1,
    )
    if out.shape[0] > out_n:
        start = (out.shape[0] - out_n) // 2
        out = out[start : start + out_n]
    elif out.shape[0] < out_n:
        pad_before = (out_n - out.shape[0]) // 2
        pad_after = out_n - out.shape[0] - pad_before
        out = np.pad(out, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return out.astype(np.float32, copy=False)


def simulate_through_plane_sr(
    img: np.ndarray,
    axis_dim: int,
    axis_spacing_mm: float,
    slice_interval_mm: float,
    slice_thickness_mm: float,
    slice_profile: str = "gaussian",
    interpolation_order: int = 1,
    interpolation_prefilter: bool = True,
) -> np.ndarray:
    """
    スライス方向の低解像度撮像をシミュレートする（edm-torchの
    ThroughPlaneSuperResolution physicalモード相当）。
    1. スライスプロファイルぼかし
       - gaussian: FWHM=スライス厚
       - box: 幅=スライス厚
       - continuous: sigma=スライス間隔*0.32
    2. スライス間隔に合わせて間引き
    3. 元グリッドへ補間で戻す（出力サイズは入力と同じ）
    """
    axis_first = np.moveaxis(img.astype(np.float32, copy=False), axis_dim, 0)
    interval_px = max(float(slice_interval_mm) / float(axis_spacing_mm), 1.0)
    thickness_px = max(float(slice_thickness_mm) / float(axis_spacing_mm), 1.0)

    slice_profile = str(slice_profile).lower()
    if slice_profile == "none":
        pass
    elif slice_profile == "gaussian" and thickness_px <= 1.0:
        pass
    elif slice_profile == "gaussian":
        sigma = thickness_px / 2.3548  # FWHM -> sigma
        axis_first = gaussian_filter(axis_first, sigma=(sigma, 0.0, 0.0))
    elif slice_profile == "box" and thickness_px <= 1.0:
        pass
    elif slice_profile == "box":
        size = max(1, int(round(thickness_px)))
        if size > 1:
            axis_first = uniform_filter1d(axis_first, size=size, axis=0, mode="nearest")
    elif slice_profile == "continuous":
        sigma = interval_px * CONTINUOUS_SR_SIGMA_FACTOR
        axis_first = gaussian_filter(axis_first, sigma=(sigma, 0.0, 0.0))
    else:
        raise NotImplementedError(slice_profile)

    n_axis = axis_first.shape[0]
    low_n = int(round((n_axis - 1) / interval_px)) + 1 if n_axis > 1 else 1
    low_n = max(1, min(n_axis, low_n))
    lowres = _resize_axis0(
        axis_first, low_n, interpolation_order, interpolation_prefilter
    )
    upsampled = _resize_axis0(
        lowres, n_axis, interpolation_order, interpolation_prefilter
    )
    return np.moveaxis(upsampled, 0, axis_dim).astype(np.float32, copy=False)


def apply_random_self_noise(clean: np.ndarray, intensity_range: float, cfg_sn) -> np.ndarray:
    """
    self_noiseの学習時ランダム劣化（モーションブラー→ぼかし→ノイズの順）。
    学習ローダとprobe_sim2real.pyの両方から使う。
    """
    degraded = clean.astype(np.float32)
    if np.random.uniform() < cfg_sn.motion_blur.prob:
        length = np.random.uniform(*cfg_sn.motion_blur.length_range)
        direction = np.random.normal(size=3)  # 一様ランダム方向
        degraded = apply_motion_blur(degraded, direction, length)
    if np.random.uniform() < cfg_sn.blur_prob:
        sigma = np.random.uniform(*cfg_sn.blur_sigma_range)
        degraded = gaussian_filter(degraded, sigma=sigma)
    std_rel = np.random.uniform(*cfg_sn.noise_std_rel_range)
    noise = np.random.normal(0.0, std_rel * intensity_range, degraded.shape)
    return degraded + noise.astype(np.float32)


def apply_random_self_sr(clean: np.ndarray, norm_spacing_zyx, cfg_sr) -> np.ndarray:
    """
    self_srの学習時ランダム劣化（軸・スライス間隔・厚みをサンプリング）。
    学習ローダとprobe_sim2real.pyの両方から使う。
    """
    interpolation_order, interpolation_prefilter = _normalize_sr_interpolation(
        cfg_sr.slice_interpolation
    )
    axis_name = str(random.choice(list(cfg_sr.axes))).upper()
    axis_dim = SR_AXIS_TO_DIM[axis_name]
    slice_interval_mm = np.random.uniform(*cfg_sr.slice_interval_mm_range)
    slice_thickness_mm = np.random.uniform(*cfg_sr.slice_thickness_mm_range)
    if cfg_sr.clamp_thickness_to_interval:
        slice_thickness_mm = min(slice_thickness_mm, slice_interval_mm)
    return simulate_through_plane_sr(
        clean,
        axis_dim,
        float(norm_spacing_zyx[axis_dim]),
        slice_interval_mm,
        slice_thickness_mm,
        slice_profile=cfg_sr.slice_profile,
        interpolation_order=interpolation_order,
        interpolation_prefilter=interpolation_prefilter,
    )


def get_clip_vals(img_hdr_path: Path, cfg) -> tuple[float, float]:
    """正規化に使うmin/max値をモダリティに応じて取得する"""
    if cfg.image.modality == "MR":
        intensity_path = img_hdr_path.with_suffix(
            f".intensity-{cfg.image.MR.min_percentile}-{cfg.image.MR.max_percentile}.txt"
        )  # save_intensityで作成
        min_val, max_val = load_intensity(intensity_path)
    else:
        window_level = float(cfg.image.CT.window_level)
        window_width = float(cfg.image.CT.window_width)
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2
    return min_val, max_val


def get_crop_center(
    img_hdr_path: Path, img_size_zyx, spacing_zyx, is_training: bool, cfg
):
    """クロップ中心を決める。検証時は画像中心、学習時はランダム"""
    crop_size_zyx = cfg.aug.crop_size_zyx
    norm_spacing_zyx = cfg.aug.affine.norm_spacing_zyx

    if not is_training:
        # 画像中心（クロップが画像内に収まるように調整）
        return center_within_image(
            np.array(img_size_zyx) / 2,
            img_size_zyx,
            crop_size_zyx,
            spacing_zyx,
            norm_spacing_zyx,
        )

    mode_list = list(cfg.aug.random_crop_method.keys())
    weight = list(cfg.aug.random_crop_method.values())
    mode = random.choices(mode_list, weight, k=1)[0]

    if mode == "body":
        # 体表マスクの矩形内でランダムクロップ（.body.box.txtはsave_organ_boxで作成）
        body_box_path = img_hdr_path.with_suffix(".body.box.txt")
        box_zyxzyx = load_organ_box(body_box_path)
    elif mode == "image":
        # 画像全体からランダムクロップ
        box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    else:
        raise NotImplementedError(mode)

    return random_crop_center_within_bb(
        box_zyxzyx,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        norm_spacing_zyx,
        [0, 0, 0],
    )


def preprocess_image_np(
    img_hdr_list_with_data_name: list[bytes], is_training: bool, cfg
):
    src_hdr_path, dataname = img_hdr_list_with_data_name
    dataname = dataname.decode()  # noqa: F841 データセットごとに処理を変える場合に使う
    src_hdr_path = Path(src_hdr_path.decode())

    # self_noiseモード: クリーン画像を1枚だけ読み込み、targetはクリーン、
    #   sourceはクリーン+合成ノイズとする（自己教師デノイジング）
    # self_srモード: クリーン画像を1枚だけ読み込み、targetはクリーン、
    #   sourceはスライス方向の低解像度シミュレーション（through-plane SR）
    # pairedモード: source(xxx.hdr)とtarget(xxx.target.hdr)を同一フォルダから読む
    # paired_dirモード: source(data_dir)とtarget(target_data_dir)を別フォルダの同名ファイルから読む
    self_noise = cfg.data.mode == "self_noise"
    self_sr = cfg.data.mode == "self_sr"

    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, src_dtype, spacing_zyx = read_hdr(src_hdr_path)

    if not (self_noise or self_sr):
        tgt_hdr_path = resolve_target_hdr_path(src_hdr_path, cfg)
        tgt_size_zyx, tgt_dtype, tgt_spacing_zyx = read_hdr(tgt_hdr_path)
        # source/targetは位置合わせ済み（同一サイズ・同一スペーシング）が前提
        assert tuple(img_size_zyx) == tuple(tgt_size_zyx), (
            f"size mismatch: {src_hdr_path} {img_size_zyx} vs {tgt_size_zyx}"
        )
        assert np.allclose(spacing_zyx, tgt_spacing_zyx, atol=1e-3), (
            f"spacing mismatch: {src_hdr_path} {spacing_zyx} vs {tgt_spacing_zyx}"
        )

    # クロップ中心を決める
    crop_center_zyx = get_crop_center(
        src_hdr_path, img_size_zyx, spacing_zyx, is_training, cfg
    )

    # アフィン変換のためのインスタンスを作成
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)

    # アフィン行列を計算（source/targetで同一の行列を使うのが最重要ポイント）
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training
    )

    # 必要な画像領域を計算
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    # 画像などは切り取って読み込むのでその分アフィン行列をシフトさせる
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    # 正規化パラメータを取得（ノイズ量の基準にも使う）
    src_min_val, src_max_val = get_clip_vals(src_hdr_path, cfg)

    # ここでは画像は読み込まずメモリマッピングをするだけ。アフィン変換で初めて画像を読む
    src_img = read_raw(
        src_hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=src_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )

    # 画像の有効領域マスク：全1配列を同じアフィン変換にかけ、回転などで
    # 生じるパディング領域(0)を検出する。BatchRenormの統計範囲や損失計算の
    # 有効領域として使う
    ones = np.ones(src_img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)

    if self_noise:
        # クリーン画像を1回だけアフィン変換し、targetとする
        clean = affine_transform.apply(src_img, affine_matrix, order=1)
        tgt_img = clean
        # sourceは劣化画像: モーションブラー→ガウシアンぼかし→ノイズの順に加える
        # ノイズ量は強度レンジ(正規化min-max)に対する相対量で指定する
        cfg_sn = cfg.data.self_noise
        intensity_range = float(src_max_val - src_min_val)
        if is_training:
            # 学習時は劣化をランダムに適用（probe_sim2real.pyと共通の関数）
            src_img = apply_random_self_noise(clean, intensity_range, cfg_sn)
        else:
            degraded = clean.astype(np.float32)
            # 検証はエポック間・実行間で再現するようファイル名でシードを固定する
            # （組み込みhashはプロセス毎に変わるためcrc32を使う）
            seed = zlib.crc32(src_hdr_path.stem.encode())
            rng = np.random.default_rng(seed)
            # 検証は固定の劣化（代表的な劣化強度。ブレ方向はファイル毎に固定）
            if cfg_sn.motion_blur.val_length > 0:
                direction = rng.normal(size=3)
                degraded = apply_motion_blur(
                    degraded, direction, float(cfg_sn.motion_blur.val_length)
                )
            if cfg_sn.val_blur_sigma > 0:
                degraded = gaussian_filter(degraded, sigma=float(cfg_sn.val_blur_sigma))
            std_rel = float(cfg_sn.val_noise_std_rel)
            noise = rng.normal(0.0, std_rel * intensity_range, degraded.shape)
            src_img = degraded + noise.astype(np.float32)
        tgt_min_val, tgt_max_val = src_min_val, src_max_val
    elif self_sr:
        # クリーン画像を1回だけアフィン変換し、targetとする
        clean = affine_transform.apply(src_img, affine_matrix, order=1)
        tgt_img = clean
        # sourceはスライス方向の低解像度シミュレーション
        # アフィン変換後のボクセル間隔はnorm_spacing_zyxになっている
        cfg_sr = cfg.data.self_sr
        norm_spacing_zyx = cfg.aug.affine.norm_spacing_zyx
        if is_training:
            # 学習時は劣化をランダムに適用（probe_sim2real.pyと共通の関数）
            src_img = apply_random_self_sr(clean, norm_spacing_zyx, cfg_sr)
        else:
            # 検証は固定の劣化（軸・間隔・厚みとも決定的で再現可能）
            axis_dim = SR_AXIS_TO_DIM[str(cfg_sr.val_axis).upper()]
            slice_interval_mm = float(cfg_sr.val_slice_interval_mm)
            slice_thickness_mm = float(cfg_sr.val_slice_thickness_mm)
            if cfg_sr.clamp_thickness_to_interval:
                slice_thickness_mm = min(slice_thickness_mm, slice_interval_mm)
            interpolation_order, interpolation_prefilter = _normalize_sr_interpolation(
                cfg_sr.slice_interpolation
            )
            src_img = simulate_through_plane_sr(
                clean,
                axis_dim,
                float(norm_spacing_zyx[axis_dim]),
                slice_interval_mm,
                slice_thickness_mm,
                slice_profile=cfg_sr.slice_profile,
                interpolation_order=interpolation_order,
                interpolation_prefilter=interpolation_prefilter,
            )
        tgt_min_val, tgt_max_val = src_min_val, src_max_val
    else:
        tgt_img = read_raw(
            tgt_hdr_path,
            clip_zyxzyx=img_region_zyxzyx,
            img_dtype=tgt_dtype,
            size_zyx=img_size_zyx,
            use_memmap=True,
        )
        # targetをアフィン変換
        tgt_img = affine_transform.apply(tgt_img, affine_matrix, order=1)

        # thin->thick変換の準備（sourceのみに適用する疑似thick化）
        thin2thick_param = prepare_thin2thick(
            spacing_zyx,
            affine_transform.norm_spacing_zyx,
            crop_size_zyx,
            cfg.aug.thick2thin_rate_zyx,
            is_training=is_training,
            spacing_max_val=2,
            thickness_range=[2, 6],
        )

        # sourceをアフィン変換：thin->thick変換用にcrop_sizeを大きくしておく
        # なのでsourceのアフィンは一番最後にやること
        crop_size_extra = crop_size_zyx.copy()
        crop_size_extra[thin2thick_param["axis"]] += thin2thick_param["extra_slice"]
        affine_transform.crop_size_zyx = crop_size_extra  # 上書きするので注意
        src_img = affine_transform.apply(src_img, affine_matrix, order=1)

        if thin2thick_param["apply_thin_thick"]:
            src_img = virtual_thick_generator(
                src_img,
                thin2thick_param["thickness"],
                order=1,
                axis=thin2thick_param["axis"],
            )
            src_img = crop_input(src_img, [0, 0, 0] + list(crop_size_zyx))

        if cfg.image.share_normalization:
            # デノイズ・ぼかし修正など同一モダリティ変換ではsourceと同じ値を使う
            tgt_min_val, tgt_max_val = src_min_val, src_max_val
        else:
            tgt_min_val, tgt_max_val = get_clip_vals(tgt_hdr_path, cfg)

    # チャンネルの次元を追加
    src_img = add_channel_dim(src_img.astype(np.int16, copy=False))
    tgt_img = add_channel_dim(tgt_img.astype(np.int16, copy=False))
    img_msk = add_channel_dim(img_msk)
    return (
        src_img,
        tgt_img,
        img_msk,
        np.array(src_min_val, np.float32),
        np.array(src_max_val, np.float32),
        np.array(tgt_min_val, np.float32),
        np.array(tgt_max_val, np.float32),
        str(src_hdr_path.stem).encode(),
    )


def preprocess_image(img_hdr_path_with_data_name, is_training: bool, cfg):
    def _preprocess_image_np(img_hdr_path_with_data_name):
        return preprocess_image_np(img_hdr_path_with_data_name, is_training, cfg)

    (
        src_img,
        tgt_img,
        img_msk,
        src_min_clip_val,
        src_max_clip_val,
        tgt_min_clip_val,
        tgt_max_clip_val,
        img_hdr,
    ) = tf.numpy_function(
        func=_preprocess_image_np,
        inp=[img_hdr_path_with_data_name],
        Tout=[
            tf.int16,
            tf.int16,
            tf.uint8,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.string,
        ],
    )

    # tf.numpy_functionを使ったときはset_shapeでshapeを指定する
    img_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
    src_img.set_shape(img_shape)
    tgt_img.set_shape(img_shape)
    img_msk.set_shape(img_shape)
    src_min_clip_val.set_shape(())
    src_max_clip_val.set_shape(())
    tgt_min_clip_val.set_shape(())
    tgt_max_clip_val.set_shape(())

    return (
        src_img,
        tgt_img,
        img_msk,
        src_min_clip_val,
        src_max_clip_val,
        tgt_min_clip_val,
        tgt_max_clip_val,
        img_hdr,
    )


def make_batch_dict(
    src_imgs,
    tgt_imgs,
    img_msks,
    src_min_clip_vals,
    src_max_clip_vals,
    tgt_min_clip_vals,
    tgt_max_clip_vals,
    img_hdr_list,
    cfg,
):
    """
    一般的にはモデルを GPU や TPU などのアクセラレータ上で実行している場合でも、
    tf.data パイプラインは CPU 上で実行されています。
    https://www.tensorflow.org/guide/data_performance_analysis?hl=ja#3_cpu_%E4%BD%BF%E7%94%A8%E7%8E%87%E3%81%8C%E9%AB%98%E3%81%8F%E3%81%AA%E3%81%A3%E3%81%A6%E3%81%84%E3%82%8B%E3%81%8B%EF%BC%9F
    """

    data = dict(
        src_imgs=tf.cast(src_imgs, tf.float32),
        tgt_imgs=tf.cast(tgt_imgs, tf.float32),
        img_msks=tf.cast(img_msks, tf.float32),
        src_min_clip_vals=src_min_clip_vals,
        src_max_clip_vals=src_max_clip_vals,
        tgt_min_clip_vals=tgt_min_clip_vals,
        tgt_max_clip_vals=tgt_max_clip_vals,
    )
    if cfg.debug_dataloader:
        data["img_hdr_list"] = img_hdr_list

    return data


def create_dataloader(img_hdr_dict: dict, is_training: bool, cfg):
    """
    複数のデータセットから異なる確率で読み込むデータローダーを作成する
    ミニバッチに必ず特定のデータセットが含まれるような実装にはしていないが、それほど問題にならないはず。
    img_hdr_dict:
    e.g.
    {
       "DataSetA":
            {
                "img_hdr_list": [path1.hdr, path2.hdr, ...]  # sourceのhdrパス
                "freq": 0.8, # 80%の確率でDataSetAからサンプリング
            },
        "DataSetB":
            {
                "img_hdr_list": [path3.hdr, path4.hdr, ...]
                "freq": 0.2,
            },
    }
    """

    img_hdr_path_list = []
    for value in img_hdr_dict.values():
        img_hdr_path_list += value["img_hdr_list"]

    # 前計算が必要なものを列挙する
    with multiprocessing.Pool(cfg.num_workers) as pool:

        def _run(func, path_list, desc):
            for _ in tqdm(
                pool.imap_unordered(func, path_list),
                total=len(path_list),
                desc=desc,
            ):
                pass

        # bodyクロップを使う場合のみ体表の矩形を計算しておく
        if cfg.aug.random_crop_method.body > 0:
            func = partial(save_organ_box, suffix=".body")
            _run(func, img_hdr_path_list, "saving body box")

        # MRデータはあらかじめ、min_intensityとmax_intensityを計算しておく
        if cfg.image.modality == "MR":
            func = partial(
                save_intensity,
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )
            _run(func, img_hdr_path_list, "saving source intensity")
            # self_noise / self_srではtargetはsourceと同一なので別計算は不要
            if cfg.data.mode not in (
                "self_noise",
                "self_sr",
            ) and not cfg.image.share_normalization:
                tgt_hdr_path_list = [
                    resolve_target_hdr_path(Path(p), cfg) for p in img_hdr_path_list
                ]
                _run(func, tgt_hdr_path_list, "saving target intensity")

    dataset_list = []
    frequency_list = []
    for data_name, value in img_hdr_dict.items():
        # データセットごとに処理を変えることを想定してデータセット名を付与する（処理はpreprocess_image_npで実装）
        img_hdr_list_with_data_name = [
            (str(path), data_name) for path in sorted(value["img_hdr_list"])
        ]
        _dataset = tf.data.Dataset.from_tensor_slices(img_hdr_list_with_data_name)

        if is_training:
            # ここでrepeatしないと正しくサンプリングできない
            _dataset = _dataset.repeat()
        dataset_list.append(_dataset)
        frequency_list.append(value["freq"])
        logging.info(f"Dataset {data_name} has {len(value['img_hdr_list'])} images.")

    # データセットを結合する。学習時はここでサンプリングの重みを設定する。
    if is_training:
        dataset = tf.data.Dataset.sample_from_datasets(
            dataset_list, weights=frequency_list
        )
    else:
        dataset = tf.data.Dataset.sample_from_datasets(dataset_list)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(img_hdr_path_list))

    def _preprocess_image(img_hdr_path_with_data_name):
        return preprocess_image(img_hdr_path_with_data_name, is_training, cfg)

    dataset = dataset.map(
        _preprocess_image, num_parallel_calls=cfg.num_workers
    )  # autotuneはなんか遅かった・・・

    # 学習はjit+サンプリングの都合でdrop_remainder=True。
    # 検証はdrop_remainder=Falseにして、val症例数 < batch_sizeでも
    # 0バッチにならない（余りバッチをそのまま評価する）ようにする。
    dataset = dataset.batch(cfg.batch_size, drop_remainder=is_training)

    # 他で使いやすいように辞書型で保持する
    def _make_batch_dict(*args):
        return make_batch_dict(*args, cfg)

    dataset = dataset.map(_make_batch_dict)

    # 実劣化データのDC損失用ストリームを合流させる（data.real_dc、学習時のみ）
    cfg_dc = cfg.data.get("real_dc", None)
    if is_training and cfg_dc is not None and cfg_dc.enabled:
        real_dataset = _make_real_dc_dataset(cfg)
        dataset = tf.data.Dataset.zip((dataset, real_dataset)).map(
            lambda main, real: {**main, **real}
        )

    dataset = dataset.prefetch(buffer_size=cfg.prefetch_size)

    return dataset


def preprocess_real_dc_image_np(hdr_path_bytes, cfg):
    """
    DC損失用の実劣化画像を読み込む。
    - ランダム中心クロップ + norm_spacingへのリサンプル
    - 回転・反転・スケールなし（劣化軸をパッチ軸に揃えるため。学習側の
      幾何拡張とは独立）
    - 劣化演算子のσ[px]をヘッダ/設定から計算して返す:
        σ = sqrt(σ_profile^2 + σ_interp^2)
        σ_profile = スライス厚px / 2.3548 (FWHM→σ)
        σ_interp  = interp_sigma_factor * スライス間隔px（間引き→補間の実効平滑化の近似）
    """
    cfg_dc = cfg.data.real_dc
    hdr_path = Path(hdr_path_bytes.decode())
    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, img_dtype, spacing_zyx = read_hdr(hdr_path)

    axis_dim = SR_AXIS_TO_DIM[str(cfg_dc.axis).upper()]
    interval_mm = cfg_dc.slice_interval_mm
    interval_mm = float(spacing_zyx[axis_dim]) if interval_mm is None else float(interval_mm)
    thickness_mm = cfg_dc.slice_thickness_mm
    thickness_mm = interval_mm if thickness_mm is None else float(thickness_mm)
    axis_spacing = float(cfg.aug.affine.norm_spacing_zyx[axis_dim])
    thickness_px = max(thickness_mm / axis_spacing, 1.0)
    interval_px = max(interval_mm / axis_spacing, 1.0)
    sigma_profile = thickness_px / 2.3548 if thickness_px > 1.0 else 0.0
    sigma_interp = (
        float(cfg_dc.interp_sigma_factor) * interval_px if interval_px > 1.0 else 0.0
    )
    sigma_px = float(np.sqrt(sigma_profile**2 + sigma_interp**2))

    # ランダム中心クロップ（回転なしの決定的アフィン）
    box_zyxzyx = np.array([0, 0, 0] + list(img_size_zyx))
    crop_center_zyx = random_crop_center_within_bb(
        box_zyxzyx,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        cfg.aug.affine.norm_spacing_zyx,
        [0, 0, 0],
    )
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training=False
    )
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    min_val, max_val = get_clip_vals(hdr_path, cfg)
    img = read_raw(
        hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=img_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )
    ones = np.ones(img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)
    img = affine_transform.apply(img, affine_matrix, order=1)

    return (
        add_channel_dim(img.astype(np.int16, copy=False)),
        add_channel_dim(img_msk),
        np.array(min_val, np.float32),
        np.array(max_val, np.float32),
        np.array(sigma_px, np.float32),
    )


def _make_real_dc_dataset(cfg):
    """DC損失用の実劣化データストリーム（無限リピート・バッチ済み辞書）を作る"""
    cfg_dc = cfg.data.real_dc
    input_dir = Path(cfg_dc.input_dir)
    hdr_list = [
        p
        for p in sorted(input_dir.glob("*.hdr"))
        if ".mask" not in p.name and not p.stem.endswith(cfg.data.target_suffix)
    ]
    assert len(hdr_list) > 0, f"real_dc.input_dirに画像がありません: {input_dir}"
    logging.info(f"real_dc: {len(hdr_list)} volumes from {input_dir}")

    if cfg.image.modality == "MR":
        for hdr_path in hdr_list:
            save_intensity(
                str(hdr_path),
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )

    def _preprocess(hdr_path_bytes):
        def _np_func(hdr_path_bytes):
            return preprocess_real_dc_image_np(hdr_path_bytes, cfg)

        img, msk, min_val, max_val, sigma_px = tf.numpy_function(
            func=_np_func,
            inp=[hdr_path_bytes],
            Tout=[tf.int16, tf.uint8, tf.float32, tf.float32, tf.float32],
        )
        img_shape = tuple(cfg.aug.crop_size_zyx) + (1,)
        img.set_shape(img_shape)
        msk.set_shape(img_shape)
        min_val.set_shape(())
        max_val.set_shape(())
        sigma_px.set_shape(())
        return img, msk, min_val, max_val, sigma_px

    dataset = tf.data.Dataset.from_tensor_slices([str(p) for p in hdr_list])
    dataset = dataset.repeat().shuffle(buffer_size=len(hdr_list))
    dataset = dataset.map(_preprocess, num_parallel_calls=cfg.num_workers)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=True)

    def _make_dict(img, msk, min_val, max_val, sigma_px):
        return dict(
            real_imgs=tf.cast(img, tf.float32),
            real_msks=tf.cast(msk, tf.float32),
            real_min_clip_vals=min_val,
            real_max_clip_vals=max_val,
            real_sigma_px=sigma_px,
        )

    return dataset.map(_make_dict)


def preprocess_test_image_np(hdr_path: Path, cfg):
    """
    正解なしのテスト入力画像を、検証時と同じ前処理で読み込む。
    - 画像中心の固定クロップ + norm_spacing_zyxへのリサンプル（線形補間）
    - 有効領域マスクの生成、正規化用clip値の取得
    スライス方向に解像度が低い入力（thickスライス等）は、このリサンプルで
    等方グリッドへ補間されてからモデルに入る。self_srの学習時sourceは
    「劣化→間引き→元グリッドへ補間」で作られるため、テスト入力もここで
    同じグリッド・同じ線形補間に揃うことになる。
    """
    hdr_path = Path(hdr_path)
    crop_size_zyx = cfg.aug.crop_size_zyx
    img_size_zyx, src_dtype, spacing_zyx = read_hdr(hdr_path)

    # 検証時と同じ：画像中心の固定クロップ
    crop_center_zyx = center_within_image(
        np.array(img_size_zyx) / 2,
        img_size_zyx,
        crop_size_zyx,
        spacing_zyx,
        cfg.aug.affine.norm_spacing_zyx,
    )
    affine_transform = AffineTransform(crop_size_zyx=crop_size_zyx, **cfg.aug.affine)
    affine_matrix = affine_transform.get_affine(
        spacing_zyx, crop_center_zyx, is_training=False
    )
    img_region_zyxzyx, shift_start = calc_img_crop_region(
        crop_size_zyx, affine_matrix, [0, 0, 0], img_size_zyx
    )
    affine_matrix = affine_transform.fix_start(affine_matrix, shift_start)

    min_val, max_val = get_clip_vals(hdr_path, cfg)

    src_img = read_raw(
        hdr_path,
        clip_zyxzyx=img_region_zyxzyx,
        img_dtype=src_dtype,
        size_zyx=img_size_zyx,
        use_memmap=True,
    )
    ones = np.ones(src_img.shape, np.uint8)
    img_msk = affine_transform.apply(ones, affine_matrix, order=0, cval=0)
    src_img = affine_transform.apply(src_img, affine_matrix, order=1)

    return (
        add_channel_dim(src_img.astype(np.float32)),
        add_channel_dim(img_msk.astype(np.float32)),
        np.float32(min_val),
        np.float32(max_val),
        hdr_path.stem,
    )


def create_test_batch(cfg):
    """
    test.input_dirの正解なし入力をまとめて1バッチのdict（predict_step互換）にする。
    症例数は少ない想定なので学習開始前に一度だけ実行してメモリに保持する。
    戻り値: (data_dict, 症例名リスト)
    """
    test_dir = Path(cfg.test.input_dir)
    assert test_dir.exists(), f"test.input_dirが存在しません: {test_dir}"
    hdr_list = [p for p in sorted(test_dir.glob("*.hdr")) if ".mask" not in p.name]
    if cfg.test.max_items > 0:
        hdr_list = hdr_list[: cfg.test.max_items]
    assert len(hdr_list) > 0, f"テスト入力(.hdr)が見つかりません: {test_dir}"

    # MRは正規化用のintensityファイルを事前計算しておく（学習データと同じ処理）
    if cfg.image.modality == "MR":
        for hdr_path in hdr_list:
            save_intensity(
                str(hdr_path),
                min_percentile=cfg.image.MR.min_percentile,
                max_percentile=cfg.image.MR.max_percentile,
            )

    srcs, msks, min_vals, max_vals, names = [], [], [], [], []
    for hdr_path in hdr_list:
        src, msk, min_val, max_val, name = preprocess_test_image_np(hdr_path, cfg)
        srcs.append(src)
        msks.append(msk)
        min_vals.append(min_val)
        max_vals.append(max_val)
        names.append(name)
        logging.info(f"Test input loaded: {name} (clip=[{min_val:.1f}, {max_val:.1f}])")

    data = dict(
        src_imgs=tf.constant(np.stack(srcs), tf.float32),
        img_msks=tf.constant(np.stack(msks), tf.float32),
        src_min_clip_vals=tf.constant(np.stack(min_vals), tf.float32),
        src_max_clip_vals=tf.constant(np.stack(max_vals), tf.float32),
    )
    return data, names
