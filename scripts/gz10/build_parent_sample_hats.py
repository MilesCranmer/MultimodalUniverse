"""Convert the GZ10 (Galaxy10 DECaLS) HDF5 file into a HATS catalog.

The cluster mirrors the GZ10 dataset as a single HDF5 file at::

    /mnt/ceph/users/polymathic/external_data/astro/Galaxy10_DECals.h5

The file contains 17,736 galaxy images with labels from Galaxy Zoo 2:

    - ans       shape (N,)           classification label (0-9)
    - ra        shape (N,)           right ascension (degrees)
    - dec       shape (N,)           declination (degrees)
    - redshift  shape (N,)           spectroscopic redshift
    - images    shape (N, 256, 256, 3)  uint8 RGB images in grz bands
    - pxscale   shape (N,)           pixel scale (arcsec/pixel)

Schema::

    image: struct<
        band:  list<string>,               # ['g', 'r', 'z'] per row
        array: list<list<list<uint8>>>,    # (3, 256, 256) per row (channels first)
        scale: list<float32>,              # pixel scale per band per row
    >
    + gz10_label (int32), redshift (float32)
    + object_id (string), ra (float64), dec (float64)
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pyarrow as pa


def _as_array(arr):
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


CATALOG_NAME = "gz10"

IMAGE_SIZE = 256
BANDS = ["g", "r", "z"]
N_BANDS = len(BANDS)


def read_table(
    raw_path: str,
    max_rows: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table:
    """Read the GZ10 HDF5 file and return a PyArrow table.

    If a cone cut is active, ra/dec are filtered before loading images.
    """
    with h5py.File(raw_path, "r") as f:
        n_total = f["ra"].shape[0]
        n = min(n_total, max_rows) if max_rows is not None else n_total

        ra = np.asarray(f["ra"][:n], dtype=np.float64)
        dec = np.asarray(f["dec"][:n], dtype=np.float64)

        if ra_center is not None and dec_center is not None and radius is not None:
            mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
            keep = np.where(mask)[0]
        else:
            keep = np.arange(n)

        ra = ra[keep]
        dec = dec[keep]

        ans = np.asarray(f["ans"][keep], dtype=np.int32)
        redshift = np.asarray(f["redshift"][keep], dtype=np.float32)
        pxscale = np.asarray(f["pxscale"][keep], dtype=np.float32)
        # images: (N, 256, 256, 3) uint8 — channels last
        images = np.asarray(f["images"][keep], dtype=np.uint8)

    n_rows = len(keep)
    # Transpose from (N, H, W, C) → (N, C, H, W) for channels-first storage
    images = images.transpose(0, 3, 1, 2)

    band_arr = pa.array([BANDS] * n_rows, type=pa.list_(pa.string()))
    # Nested list: list<list<list<uint8>>> with shape (N_BANDS, H, W)
    array_nested = [[[list(row) for row in images[i, c]] for c in range(N_BANDS)] for i in range(n_rows)]
    array_arr = pa.array(array_nested, type=pa.large_list(pa.large_list(pa.large_list(pa.uint8()))))
    scale_arr = pa.array(
        [[float(pxscale[i])] * N_BANDS for i in range(n_rows)],
        type=pa.list_(pa.float32()),
    )
    image_col = pa.StructArray.from_arrays(
            [_as_array(a) for a in [band_arr, array_arr, scale_arr]],
        names=["band", "array", "scale"],
    )

    object_ids = [str(i) for i in range(n_rows)]

    return pa.table({
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        "object_id": pa.array(object_ids, type=pa.string()),
        "gz10_label": pa.array(ans, type=pa.int32()),
        "redshift": pa.array(redshift, type=pa.float32()),
        "image": image_col,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-path",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Path to the Galaxy10_DECals.h5 file.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Not used for gz10 (single file); kept for CLI consistency.",
    )
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap on total rows (useful for testing).")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Cone radius in degrees; requires --ra-center/--dec-center.",
    )
    args = parser.parse_args(argv)

    table = read_table(
        args.raw_path,
        max_rows=args.max_rows,
        ra_center=args.ra_center,
        dec_center=args.dec_center,
        radius=args.radius,
    )
    print(f"Loaded {table.num_rows} rows from {args.raw_path}", flush=True)

    if table.num_rows == 0:
        print("No rows after filtering; nothing to write.", file=sys.stderr)
        return 1

    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=True,
    )
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
