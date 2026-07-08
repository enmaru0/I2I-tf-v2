from pathlib import Path

import numpy as np
from irg import save_raw

from data.dataloader import create_dataloader
from data.gpu_aug import normalize
from main import prepare_data_dict, read_cfg_and_parse_arg
from trainers import BaseI2ITrainer


def reverse_normalize_img(img, min_val, max_val) -> None:
    img *= max_val - min_val
    img += min_val


def save_imgs(
    key_batch, img_batch, save_root, spacing_zyx, min_val, max_val, suffix=""
):
    assert img_batch.ndim == 5
    assert save_root.exists(), save_root
    if isinstance(min_val, (int, float)):
        min_val = [min_val] * len(img_batch)
        max_val = [max_val] * len(img_batch)
    for key, img, _min, _max in zip(key_batch, img_batch, min_val, max_val):
        print(f"saving img: {key}{suffix}")
        img = img[:, :, :, 0]

        if (_min is None) or (_max is None):
            assert (_min is None) and (_max is None), (_min, max_val)
        else:
            reverse_normalize_img(img, _min, _max)

        hdr_path = save_root / (key + suffix + ".hdr")
        save_raw(img.astype(np.int16), spacing_zyx, hdr_path)


if __name__ == "__main__":
    cfg = read_cfg_and_parse_arg()
    cfg.debug_dataloader = True

    train_dict, val_dict = prepare_data_dict(cfg)

    save_root = Path(cfg.exp_dir)
    save_root.mkdir(exist_ok=True, parents=True)
    spacing_zyx = cfg.aug.affine.norm_spacing_zyx
    for is_training in [True, False]:
        if is_training:
            img_hdr_dict = train_dict
            save_dir = save_root / "sample_train"
        else:
            img_hdr_dict = val_dict
            save_dir = save_root / "sample_val"
        save_dir.mkdir(exist_ok=True)
        loader = create_dataloader(img_hdr_dict, is_training=is_training, cfg=cfg)
        for num, batch in enumerate(loader):
            if num == 2:
                # バッチサイズx2で停止
                break

            # GPUの処理を実行（学習時はsourceのみデータ拡張が入る）
            if is_training:
                batch["src_imgs"] = BaseI2ITrainer.gpu_aug(
                    batch["src_imgs"],
                    batch["img_msks"],
                    batch["src_min_clip_vals"],
                    batch["src_max_clip_vals"],
                    cfg,
                )
            else:
                batch["src_imgs"] = (
                    normalize(
                        batch["src_imgs"],
                        batch["src_min_clip_vals"],
                        batch["src_max_clip_vals"],
                    )
                    * batch["img_msks"]
                )
            batch["tgt_imgs"] = (
                normalize(
                    batch["tgt_imgs"],
                    batch["tgt_min_clip_vals"],
                    batch["tgt_max_clip_vals"],
                )
                * batch["img_msks"]
            )

            # Numpyに変換
            batch = {k: v.numpy() for k, v in batch.items()}

            img_hdr_list = batch["img_hdr_list"]
            img_hdr_list = [i.decode() for i in img_hdr_list]

            save_imgs(
                img_hdr_list,
                batch["src_imgs"],
                save_dir,
                spacing_zyx,
                batch["src_min_clip_vals"],
                batch["src_max_clip_vals"],
            )
            save_imgs(
                img_hdr_list,
                batch["tgt_imgs"],
                save_dir,
                spacing_zyx,
                batch["tgt_min_clip_vals"],
                batch["tgt_max_clip_vals"],
                suffix=".target",
            )
            # 有効領域マスクも確認用に保存する
            save_imgs(
                img_hdr_list,
                batch["img_msks"],
                save_dir,
                spacing_zyx,
                [None] * len(img_hdr_list),
                [None] * len(img_hdr_list),
                suffix=".imgmsk",
            )
