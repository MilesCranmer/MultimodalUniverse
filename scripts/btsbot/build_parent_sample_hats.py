"""Convert the BTSbot (ZTF Bright Transient Survey) dataset into a HATS catalog.

The cluster mirrors the BTSbot dataset at::

    /mnt/ceph/users/polymathic/external_data/astro/btsbot/

containing six files (train/val/test splits):

    train_triplets_v10_N100_programid1.npy  shape (N_train, 63, 63, 3)
    val_triplets_v10_N100_programid1.npy    shape (N_val, 63, 63, 3)
    test_triplets_v10_N100_programid1.npy   shape (N_test, 63, 63, 3)
    train_cand_v10_N100_programid1.csv
    val_cand_v10_N100_programid1.csv
    test_cand_v10_N100_programid1.csv

Each row has a ZTF alert with:
  - ra, dec: sky coordinates
  - candid → object_id (int64)
  - objectId → OBJECT_ID_ (string)
  - fid → band (1='g', 2='r')
  - image_triplet: (63, 63, 3) float32 array (science, reference, difference)
  - image_scale: pixel scale in arcsec/pixel (1.01)
  - plus many float, int, bool, and string alert features

Schema::

    image: struct<
        band:  list<string>,                   # ['g'] or ['r'] — single band per alert
        view:  list<string>,                   # ['science', 'reference', 'difference']
        array: list<list<list<float32>>>,      # (3, 63, 63) — triplet images
        scale: list<float32>,                  # [1.01, 1.01, 1.01]
    >
    + all float, int, bool, string features from v1
    + object_id (int64), ra (float64), dec (float64), split (string)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa


def _as_array(arr):
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


CATALOG_NAME = "btsbot"

IMAGE_SIZE = 63
PIXEL_SCALE = 1.01  # arcsec/pixel
VIEWS = ["science", "reference", "difference"]
FID_TO_BAND = {1: "g", 2: "r"}

_IMG_FILES = [
    "train_triplets_v10_N100_programid1.npy",
    "val_triplets_v10_N100_programid1.npy",
    "test_triplets_v10_N100_programid1.npy",
]
_META_FILES = [
    "train_cand_v10_N100_programid1.csv",
    "val_cand_v10_N100_programid1.csv",
    "test_cand_v10_N100_programid1.csv",
]
_SPLITS = ["train", "val", "test"]

FLOAT_FEATURES = [
    "jd", "diffmaglim", "magpsf", "sigmapsf", "chipsf", "magap", "sigmagap",
    "distnr", "magnr", "chinr", "sharpnr", "sky", "magdiff", "fwhm",
    "classtar", "mindtoedge", "seeratio", "magapbig", "sigmagapbig",
    "sgmag1", "srmag1", "simag1", "szmag1", "sgscore1", "distpsnr1",
    "jdstarthist", "scorr", "sgmag2", "srmag2", "simag2", "szmag2",
    "sgscore2", "distpsnr2", "sgmag3", "srmag3", "simag3", "szmag3",
    "sgscore3", "distpsnr3", "jdstartref", "dsnrms", "ssnrms",
    "magzpsci", "magzpsciunc", "magzpscirms", "clrcoeff", "clrcounc",
    "neargaia", "neargaiabright", "maggaia", "maggaiabright", "exptime",
    "drb", "acai_h", "acai_v", "acai_o", "acai_n", "acai_b", "new_drb",
    "peakmag", "maxmag", "peakmag_so_far", "maxmag_so_far", "age",
    "days_since_peak", "days_to_peak",
]

INT_FEATURES = [
    "label", "fid", "programid", "field", "nneg", "nbad", "ndethist",
    "ncovhist", "nmtchps", "nnotdet", "N",
]

BOOL_FEATURES = ["isdiffpos", "is_SN", "near_threshold", "is_rise"]

STRING_FEATURES = ["OBJECT_ID_", "source_set"]


def read_split(
    raw_root: str,
    split_idx: int,
    max_rows: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Read one split (train/val/test) and return a PyArrow table, or None."""
    split_name = _SPLITS[split_idx]
    meta_path = os.path.join(raw_root, _META_FILES[split_idx])
    img_path = os.path.join(raw_root, _IMG_FILES[split_idx])

    meta = pd.read_csv(meta_path)
    if max_rows is not None:
        meta = meta.iloc[:max_rows]

    ra = meta["ra"].to_numpy(dtype=np.float64)
    dec = meta["dec"].to_numpy(dtype=np.float64)

    if ra_center is not None and dec_center is not None and radius is not None:
        cone_mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
        keep_idx = np.where(cone_mask)[0]
        if keep_idx.size == 0:
            return None
    else:
        keep_idx = np.arange(len(meta))

    # Load images with mmap to avoid OOM on large files.
    img_all = np.load(img_path, mmap_mode="r")
    # meta.index holds the original row indices in the npy file.
    orig_indices = meta.index[keep_idx]
    images = np.asarray(img_all[orig_indices], dtype=np.float32)  # (M, 63, 63, 3)

    meta_sub = meta.iloc[keep_idx].reset_index(drop=True)
    ra = ra[keep_idx]
    dec = dec[keep_idx]
    n_rows = len(meta_sub)

    band_list = meta_sub["fid"].map(FID_TO_BAND).tolist()

    # Build image struct:
    # band: list<string> of length 1 (single ZTF band per alert)
    # view: list<string> of length 3 (science/reference/difference)
    # array: list<list<list<float32>>> shape (3, 63, 63)
    # scale: list<float32> of length 3
    band_arr = pa.array([[b] for b in band_list], type=pa.list_(pa.string()))
    view_arr = pa.array([VIEWS] * n_rows, type=pa.list_(pa.string()))
    # images shape: (n_rows, 63, 63, 3) → transpose to (n_rows, 3, 63, 63)
    images_chf = images.transpose(0, 3, 1, 2)
    array_nested = [[[list(row) for row in images_chf[i, v]] for v in range(3)] for i in range(n_rows)]
    array_arr = pa.array(array_nested, type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))))
    scale_arr = pa.array([[PIXEL_SCALE] * 3] * n_rows, type=pa.list_(pa.float32()))
    image_col = pa.StructArray.from_arrays(
            [_as_array(a) for a in [band_arr, view_arr, array_arr, scale_arr]],
        names=["band", "view", "array", "scale"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        "object_id": pa.array(meta_sub["candid"].to_numpy(dtype=np.int64), type=pa.int64()),
        "split": pa.array([split_name] * n_rows, type=pa.string()),
        "image": image_col,
    }

    for f in FLOAT_FEATURES:
        if f in meta_sub.columns:
            columns[f] = pa.array(meta_sub[f].to_numpy(dtype=np.float32), type=pa.float32())

    for f in INT_FEATURES:
        if f in meta_sub.columns:
            columns[f] = pa.array(meta_sub[f].to_numpy(dtype=np.int64), type=pa.int64())

    for f in BOOL_FEATURES:
        if f in meta_sub.columns:
            columns[f] = pa.array(meta_sub[f].astype(bool).tolist(), type=pa.bool_())

    string_renames = {"objectId": "OBJECT_ID_"}
    for f in STRING_FEATURES:
        src = {v: k for k, v in string_renames.items()}.get(f, f)
        if src in meta_sub.columns:
            columns[f] = pa.array(meta_sub[src].astype(str).tolist(), type=pa.string())
        elif f in meta_sub.columns:
            columns[f] = pa.array(meta_sub[f].astype(str).tolist(), type=pa.string())

    return pa.table(columns)


def read_all_splits(
    raw_root: str,
    max_rows: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> list[pa.Table]:
    """Read all three splits (train/val/test) and return as a list of tables."""
    tables = []
    for i in range(3):
        t = read_split(raw_root, i, max_rows=max_rows,
                       ra_center=ra_center, dec_center=dec_center, radius=radius)
        if t is not None and t.num_rows > 0:
            tables.append(t)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Directory containing BTSbot .npy and .csv files.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap on number of splits to process (1-3); kept for CLI consistency.",
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per split (useful for testing).")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Cone radius in degrees; requires --ra-center/--dec-center.",
    )
    args = parser.parse_args(argv)

    n_splits = args.max_files if args.max_files is not None else 3
    tables = []
    for i in range(min(n_splits, 3)):
        t = read_split(
            args.raw_root, i,
            max_rows=args.max_rows,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )
        if t is not None and t.num_rows > 0:
            tables.append(t)
            print(f"Loaded {t.num_rows} rows from split={_SPLITS[i]}", flush=True)

    if not tables:
        print("No rows after filtering; nothing to write.", file=sys.stderr)
        return 1

    catalog_dir = write_hats(
        tables,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=True,
    )
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
