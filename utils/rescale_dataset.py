import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path

import numpy as np
from irg import read_hdr, read_raw, read_re4, save_raw, save_re4
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_utils import resize_volume_to_shape


def rescale_cpp_img(data_np, src_spacing, dst_spacing):
    scale_zyx = np.asarray(src_spacing) / np.asarray(dst_spacing)
    out_size = np.round(np.asarray(data_np.shape) * scale_zyx).astype(int)
    return resize_volume_to_shape(
        data_np, out_size, order=3, anti_alias=True
    ).astype(np.int16)


def rescale_cpp_msk(data_np, src_spacing, dst_spacing):
    scale_zyx = np.asarray(src_spacing) / np.asarray(dst_spacing)
    out_size = np.round(np.asarray(data_np.shape) * scale_zyx).astype(int)
    return resize_volume_to_shape(
        data_np, out_size, order=0, anti_alias=False
    ).astype(np.uint16)


def rescale(img_hdr_path: Path, img_root: Path, save_root: Path, target_scale_zyx):
    spacing_zyx = read_hdr(img_hdr_path)[2]

    target_scale_zyx = np.maximum(target_scale_zyx, spacing_zyx)
    suffix = "-".join(map(str, target_scale_zyx))
    # pid_series_date_time.z-y-x.hdrのような形で保存する
    save_dir = save_root / str(img_hdr_path.parent).replace(str(img_root), "")[1:]
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path_img = save_dir / (img_hdr_path.stem + f".{suffix}.hdr")
    if save_path_img.exists():
        tqdm.write(str(save_path_img) + ": found. skipping...")
        return None

    img = read_raw(img_hdr_path)
    img = rescale_cpp_img(img, spacing_zyx, target_scale_zyx)
    save_raw(img, target_scale_zyx, save_path_img)

    msk_hdr_path_list = list(img_hdr_path.parent.glob(f"{img_hdr_path.stem}*.mask.hdr"))
    for msk_hdr_path in msk_hdr_path_list:
        save_path_msk = save_dir / msk_hdr_path.name.replace(
            img_hdr_path.stem, img_hdr_path.stem + f".{suffix}"
        )
        msk = read_re4(msk_hdr_path)
        msk = rescale_cpp_msk(msk, spacing_zyx, target_scale_zyx)
        save_re4(msk, target_scale_zyx, "mask", save_path_msk)


def read_cfg_and_parse_arg():
    # コマンドライン引数と設定ファイルを読み込む関数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="設定を上書きするフォーマット (例: 'batch_size=12 aug.crop_size_zyx=[64,64,64]')",
    )
    args = parser.parse_args()
    cmd_overrides = args.overrides

    config_path = "conf/config.yaml"
    cfg = OmegaConf.load(config_path)

    # コマンドライン引数で設定を上書きする
    override_config = OmegaConf.from_dotlist(cmd_overrides)
    for key in override_config:
        if key not in cfg:
            raise KeyError(f"設定ファイルに存在しないキー: {key}")
    cfg = OmegaConf.merge(cfg, override_config)
    return cfg


def main():
    cfg = read_cfg_and_parse_arg()
    target_scale_zyx = np.array(cfg.aug.affine.norm_spacing_zyx)
    target_scale_zyx = target_scale_zyx.astype(np.float32)
    img_root = Path(cfg.data_dir)
    save_root = img_root.parent / (
        img_root.name + "_" + "_".join(map(str, target_scale_zyx))
    )

    img_raw_path_list = list(img_root.glob("**/*.raw"))
    img_hdr_path_list = [i.with_suffix(".hdr") for i in img_raw_path_list]

    func = partial(
        rescale,
        img_root=img_root,
        save_root=save_root,
        target_scale_zyx=target_scale_zyx,
    )
    with multiprocessing.Pool(12) as pool:
        for _ in tqdm(
            pool.imap_unordered(func, img_hdr_path_list), total=len(img_hdr_path_list)
        ):
            pass


if __name__ == "__main__":
    main()
