import argparse
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
from irg import save_re4
from tqdm import tqdm


def convert2binary(pred_path, save_dir):
    key = pred_path.stem
    save_path = save_dir / f"{key}.pred.mask.hdr"

    np_data = np.load(pred_path)
    pred = np_data["data"]
    spacing_zyx = np_data["spacing_zyx"]

    pred_msk = (pred > 0.5).astype(np.uint16)

    src_dst_bit_dict = {i: i for i in range(pred.shape[-1])}
    save_re4(
        pred_msk, spacing_zyx, "mask", save_path, src_dst_bit_dict=src_dst_bit_dict
    )


def main(pred_dir, save_dir, num_workers):
    pred_dir = Path(pred_dir)

    pred_list = list(pred_dir.glob("*.npz"))

    save_dir.mkdir(exist_ok=True)
    func = partial(convert2binary, save_dir=save_dir)

    with multiprocessing.Pool(processes=num_workers) as pool:
        for out in tqdm(pool.imap_unordered(func, pred_list), total=len(pred_list)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("pred_dir", type=Path, help="path to pred msk (hdr&re4 file)")
    parser.add_argument(
        "--save_dir", type=Path, default=None, help="path to gt msk (hdr&re4 file)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="num workers (default 12)"
    )
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = args.pred_dir
    main(args.pred_dir, args.save_dir, args.num_workers)
