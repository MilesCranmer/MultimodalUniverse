"""
Generate JPEG stamps for COSMOS-Web master catalog galaxies brighter than a
given magnitude threshold, storing one JPEG per filter under existing folders
in /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert/<FILTER>.

The script reuses the FITS mosaics described in `make_stamps.py` (via
`load_imgs` and `image_make_cutout`). Only objects satisfying the magnitude
cut (default MAG_MODEL_F277W < 25) are processed.

Example:

```
python -m moprhology.zoobot.generate_master_stamps \
    --catalog /n07data/.../COSMOSWeb_mastercatalog_v3.1.0-sersic-cgs_err-calib_LePhare.fits \
    --filters F115W F150W F277W F444W \
    --mag-column MAG_MODEL_F277W \
    --mag-limit 25.0 \
    --output-root /n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert \
    --arcsec-size 2.5 \
    --batch-size 500
```
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.table import Table
from skimage.transform import resize

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moprhology.zoobot.make_stamps import (
    load_imgs,
    image_make_cutout,
    array2img,
    zero_pix_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stamps for COSMOS master catalog.")
    parser.add_argument("--catalog", type=Path, required=True, help="Path to master FITS catalog.")
    parser.add_argument("--filters", nargs="+", default=["F150W", "F277W", "F444W"],
                        help="Filters to cut (must exist in load_imgs output).")
    parser.add_argument("--id-column", default="id", help="Column containing object IDs.")
    parser.add_argument("--tile-column", default="tile", help="Column listing mosaic tile names.")
    parser.add_argument("--ra-column", default="ra", help="Column with RA in degrees.")
    parser.add_argument("--dec-column", default="dec", help="Column with Dec in degrees.")
    parser.add_argument("--mag-column", default="MAG_MODEL_F277W", help="Column used for magnitude cuts.")
    parser.add_argument("--mag-limit", type=float, default=25.0, help="Upper magnitude limit (<= limit).")
    parser.add_argument("--arcsec-size", type=float, default=2.5, help="Half-size of the cutout in arcseconds.")
    parser.add_argument("--resize-px", type=int, default=424, help="Final JPEG size (square).")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/n03data/huertas/COSMOS-Web/zoobot/stamps_ilbert"),
                        help="Root directory containing per-filter folders.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPEGs.")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Optional limit on number of galaxies to process.")
    return parser.parse_args()


def ensure_dirs(root: Path, filters: list[str]):
    for filt in filters:
        (root / filt).mkdir(parents=True, exist_ok=True)


def save_stamp(image: np.ndarray, path: Path, resize_px: int):
    scaled = resize(image, output_shape=(resize_px, resize_px))
    array2img(scaled).save(path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ensure_dirs(args.output_root, args.filters)
    table = Table.read(args.catalog, format="fits")
    # drop multidimensional columns before converting
    flat_cols = [col for col in table.colnames if len(table[col].shape) <= 1]
    table = table[flat_cols]
    df = table.to_pandas()
    if args.mag_column not in df.columns:
        raise ValueError(f"Column {args.mag_column} missing from catalog.")

    mask = (df[args.mag_column] > 0) & (df[args.mag_column] <= args.mag_limit)
    df = df[mask]
    if args.max_objects:
        df = df.head(args.max_objects)
    logging.info("Selected %d galaxies (mag column %s <= %.2f).", len(df), args.mag_column, args.mag_limit)

    processed = 0
    skipped = 0
    for row in df.itertuples(index=False):
        obj_id = getattr(row, args.id_column)
        tile = getattr(row, args.tile_column)
        ra = float(getattr(row, args.ra_column))
        dec = float(getattr(row, args.dec_column))

        if isinstance(obj_id, bytes):
            obj_id = obj_id.decode()
        if isinstance(tile, bytes):
            tile = tile.decode()

        obj_id = str(obj_id)
        try:
            _, _, sci_imas, *_ = load_imgs(tile)
        except Exception as exc:
            logging.warning("Cannot load mosaics for tile %s (%s): %s", tile, obj_id, exc)
            skipped += 1
            continue

        for filt in args.filters:
            if filt not in sci_imas:
                continue
            out_path = args.output_root / filt / f"{filt}_{obj_id}.jpg"
            if out_path.exists() and not args.overwrite:
                continue
            try:
                stamp = image_make_cutout(sci_imas[filt], ra, dec, args.arcsec_size, get_wcs=False)
            except Exception as exc:
                logging.debug("Cutout failed for %s %s: %s", filt, obj_id, exc)
                continue
            if np.isnan(stamp).any() or zero_pix_fraction(stamp) >= 0.1:
                continue
            try:
                save_stamp(stamp, out_path, args.resize_px)
                processed += 1
            except Exception as exc:
                logging.error("Failed to save %s: %s", out_path, exc)

    logging.info("Finished: wrote %d stamps (skipped %d objects).", processed, skipped)


if __name__ == "__main__":
    main()
