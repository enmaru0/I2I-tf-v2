import argparse
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
from irg import read_re4
from tqdm import tqdm


def calc_dice(y_true, y_pred, epsilon=1e-8, dice_type="jaccard"):
    assert y_true.shape == y_pred.shape
    if dice_type == "jaccard":
        denom_true = np.sum(y_true * y_true)
        denom_pred = np.sum(y_pred * y_pred)
    else:
        denom_true = np.sum(y_true)
        denom_pred = np.sum(y_pred)

    denom = denom_true + denom_pred
    numer = 2 * np.sum(y_true * y_pred)

    dice_scores = numer / (denom + epsilon)

    return dice_scores


def calculate_dice(pred_path, gt_dir, skip_np, organ):
    key = pred_path.name
    key = key.replace(".pred.mask.hdr", "")
    key = key.replace(".npz", "")

    msk_hdr = list(gt_dir.glob(f"**/{key}.{organ}.mask.hdr"))
    if skip_np or len(msk_hdr) == 0:
        return None, None
    assert len(msk_hdr) == 1, msk_hdr
    msk_hdr = msk_hdr[0]

    gt = read_re4(msk_hdr, src_dst_bit_dict={0: 0})
    if gt.sum() == 0:
        if skip_np:
            return None, None

    pred = read_re4(pred_path, src_dst_bit_dict={0: 0})

    dice_binary = calc_dice(pred, gt)
    return [key, dice_binary]


def main(pred_dir, gt_dir, organ, num_workers, skip_np):
    assert gt_dir.exists(), gt_dir
    pred_list = list(pred_dir.glob("*.hdr"))

    if len(pred_list) == 0:
        raise ValueError("Nothing Found")

    key_list = []
    dice_binary_list = []

    func = partial(calculate_dice, gt_dir=gt_dir, skip_np=skip_np, organ=organ)
    with multiprocessing.Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap_unordered(func, pred_list), total=len(pred_list)):
            if out[0] is not None:
                key_list.append(out[0])
                dice_binary_list.append(out[1])
                print(f"{out[0]}: {out[1]:.3f}")

    dice_binary = sum(dice_binary_list) / (len(dice_binary_list) + 1e-6)
    print(f"Dice Score (Binary): {dice_binary:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("pred_dir", type=Path, help="path to pred msk (hdr&re4 file)")
    parser.add_argument("gt_dir", type=Path, help="path to gt msk (hdr&re4 file)")
    parser.add_argument(
        "--organ", type=str, default="prostate", help="organ name (default prostate)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="num workers (default 12)"
    )
    parser.add_argument(
        "--skip_np", action="store_true", help="skip_np (default False)"
    )
    args = parser.parse_args()
    main(args.pred_dir, args.gt_dir, args.organ, args.num_workers, args.skip_np)
