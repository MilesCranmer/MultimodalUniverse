"""Convert raw GALEX GUVCat AIS catalog into a HATS catalog.

The cluster mirrors the GUVCat at:

    /mnt/ceph/users/polymathic/external_data/astro/galex/
        GUVCat_AIS_FOV055_glat*N*.fits.gz
        GUVCat_AIS_FOV055_glat*S*.fits.gz

There are 36 gzipped FITS shards split by galactic latitude. Each is a flat
photometric catalog with an ``objid`` column we use as ``object_id`` and
``ra``/``dec`` (after lowercasing).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, to_native_endian, write_hats_from_parquet_dir


CATALOG_NAME = "galex"


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "GUVCat_AIS_FOV055_glat*.fits.gz")))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_shard(path: str) -> pa.Table:
    """Read one GUVCat FITS shard, lowercase all column names, normalize ids."""
    t = Table.read(path)
    # Lowercase column names to match the legacy MMU v1 schema.
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])

    if "ra" not in t.colnames or "dec" not in t.colnames:
        raise ValueError(f"{path} missing ra/dec columns")
    if "objid" not in t.colnames:
        raise ValueError(f"{path} missing objid column")

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(np.asarray(t["ra"], dtype=np.float64))),
        "dec": pa.array(to_native_endian(np.asarray(t["dec"], dtype=np.float64))),
        "object_id": pa.array([str(int(x)) for x in t["objid"]], type=pa.string()),
    }
    for col in t.colnames:
        if col in ("ra", "dec", "objid"):
            continue
        arr = np.asarray(t[col])
        if arr.ndim != 1:
            continue
        columns[col] = pa.array(to_native_endian(arr))
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--scratch-dir", default=None,
                        help="Scratch dir for per-shard parquet files (default: ceph "
                             "MultimodalUniverse_v2_hats_scratch/galex_<pid>).")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    cone_active = (
        args.ra_center is not None
        and args.dec_center is not None
        and args.radius is not None
    )

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no GUVCat files under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} GALEX shard(s)")

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    print(f"Streaming per-shard parquet to {scratch}")

    try:
        total = 0
        n_written = 0
        for i, p in enumerate(files, 1):
            t = read_shard(p)
            if cone_active:
                ra = t.column("ra").to_numpy()
                dec = t.column("dec").to_numpy()
                mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
                t = t.filter(pa.array(mask))
                if t.num_rows == 0:
                    print(f"  [{i}/{len(files)}] {os.path.basename(p)}: 0 rows (cone filtered)")
                    continue
            out_path = os.path.join(scratch, f"part-{os.path.basename(p)}.parquet")
            pq.write_table(t, out_path)
            n_written += 1
            total += t.num_rows
            print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} rows → part-{i:04d}")
            del t

        if n_written == 0:
            print("ERROR: no rows survived the cone cut", file=sys.stderr)
            return 1

        print(f"\nIngesting {n_written} shard(s), {total} rows → HATS catalog")
        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
        )
        print(f"Done: {catalog_dir}")
    finally:
        if args.scratch_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
