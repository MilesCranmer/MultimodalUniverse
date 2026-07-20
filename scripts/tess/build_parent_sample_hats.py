"""Convert raw TESS-SPOC FFI lightcurves into a HATS catalog.

The cluster mirrors TESS SPOC FFI lightcurves at:

    /mnt/ceph/users/polymathic/external_data/astro/tess/
        hlsp_tess-spoc_tess_phot_{tic:016d}-s{sector:04d}_tess_v1_lc.fits

Each FITS file is one lightcurve for one TIC star observed in one sector.
The script reads:

    HDU[1] = LIGHTCURVE table  -> TIME, SAP_FLUX, SAP_FLUX_ERR, QUALITY
    HDU[1] header              -> ra_obj, dec_obj

and produces one row per (TIC, sector) with a ``lightcurve`` struct column
holding the time-series arrays.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import (
    default_scratch_dir,
    to_native_endian,
    write_hats,
    write_hats_from_parquet_dir,
)


CATALOG_NAME = "tess"
FILENAME_RE = re.compile(
    r"hlsp_tess-spoc_tess_phot_(?P<tic>\d+)-s(?P<sector>\d+)_tess_v1_lc\.fits$"
)


def parse_filename(path: str) -> tuple[int, int] | None:
    """Extract (TIC ID, sector) from a SPOC FFI lightcurve filename."""
    m = FILENAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    return int(m["tic"]), int(m["sector"])


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "hlsp_tess-spoc_tess_phot_*_tess_v1_lc.fits")))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_lightcurve(path: str) -> dict | None:
    """Read one TESS SPOC FFI lightcurve. Returns a per-row dict, or None on
    failure (file is unreadable / missing the expected columns / no TIC in name).

    Note: no cone-cut support. TESS filenames don't encode RA/Dec, so cone
    filtering would require opening every one of ~160k files on disk before
    even knowing which to process (~1-2 hours of serial I/O on ceph). This is
    architecturally at odds with the per-script cone filter we use elsewhere.
    For test slices, use ``--max-files=N``; for production, build the full
    set without a cone cut.
    """
    parsed = parse_filename(path)
    if parsed is None:
        return None
    tic, sector = parsed

    with fits.open(path, memmap=False) as hdul:
        if "LIGHTCURVE" not in [h.name for h in hdul]:
            return None
        hdr = hdul[1].header
        try:
            ra = float(hdr["ra_obj"])
            dec = float(hdr["dec_obj"])
        except KeyError:
            return None

        lc = hdul["LIGHTCURVE"].data

        time = np.asarray(lc["TIME"], dtype=np.float64)
        flux = np.asarray(lc["SAP_FLUX"], dtype=np.float32)
        flux_err = np.asarray(lc["SAP_FLUX_ERR"], dtype=np.float32)
        quality = np.asarray(lc["QUALITY"], dtype=np.int32)

    # Drop NaN time points (the SPOC files have ~10% NaN cadences).
    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    quality = quality[valid]

    return {
        "tic_id": tic,
        "sector": sector,
        "ra": ra,
        "dec": dec,
        "time": to_native_endian(time),
        "flux": to_native_endian(flux),
        "flux_err": to_native_endian(flux_err),
        "quality": to_native_endian(quality),
    }


def _read_lightcurve_safe(path: str) -> dict | None:
    """Pool-worker wrapper around :func:`read_lightcurve` that catches EVERY
    exception and returns ``None`` so one corrupt SPOC file can't kill the
    whole pool (see analogous safety note in legacysurvey._process_sweep...).
    """
    try:
        return read_lightcurve(path)
    except BaseException:  # noqa: BLE001
        return None


def build_table(rows: list[dict]) -> pa.Table:
    """Stack per-lightcurve dicts into a single PyArrow table.

    Each lightcurve keeps its true length: stored as PyArrow ``list_<float>``
    columns inside a ``lightcurve`` struct. No padding, no fixed-width 2D.
    Different rows can (and typically do) have different lengths.
    """
    columns = {
        "ra": pa.array([r["ra"] for r in rows], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in rows], type=pa.float64()),
        "object_id": pa.array([str(r["tic_id"]) for r in rows], type=pa.string()),
        "tic_id": pa.array([r["tic_id"] for r in rows], type=pa.int64()),
        "sector": pa.array([r["sector"] for r in rows], type=pa.int32()),
        "lightcurve": pa.StructArray.from_arrays(
            [
                pa.array([r["time"] for r in rows], type=pa.list_(pa.float64())),
                pa.array([r["flux"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["flux_err"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["quality"] for r in rows], type=pa.list_(pa.int32())),
            ],
            names=["time", "flux", "flux_err", "quality"],
        ),
    }
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--num-processes", type=int, default=96,
                        help="multiprocessing.Pool size for parallel FITS reads.")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of lightcurves per output parquet shard.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Scratch dir for per-batch parquet files.")
    parser.add_argument("--in-memory", action="store_true",
                        help="Legacy in-memory build (only safe for small slices).")
    # Accept but ignore cone args so tess rules still plug into the shared
    # Snakemake cone profile. TESS filenames don't encode RA/Dec, so the cone
    # filter is intentionally a no-op here (see read_lightcurve docstring).
    parser.add_argument("--ra-center", type=float, default=None,
                        help="(accepted but ignored for tess)")
    parser.add_argument("--dec-center", type=float, default=None,
                        help="(accepted but ignored for tess)")
    parser.add_argument("--radius", type=float, default=None,
                        help="(accepted but ignored for tess)")
    args = parser.parse_args(argv)
    if args.ra_center is not None or args.dec_center is not None or args.radius is not None:
        print("WARNING: tess does not support cone cuts "
              "(filenames don't encode RA/Dec); arguments ignored.",
              file=sys.stderr)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no SPOC lightcurves under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} TESS SPOC file(s)")

    if args.in_memory:
        # Legacy single-process in-memory path for tests and small slices.
        rows: list[dict] = []
        for i, p in enumerate(files, 1):
            row = read_lightcurve(p)
            if row is None:
                continue
            rows.append(row)
            print(f"  [{i}/{len(files)}] TIC {row['tic_id']} s{row['sector']:04d}: "
                  f"{len(row['time'])} cadences")
        if not rows:
            print("ERROR: no readable lightcurves", file=sys.stderr)
            return 1
        table = build_table(rows)
        print(f"\nWriting HATS catalog: {table.num_rows} lightcurves")
        catalog_dir = write_hats(
            [table],
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
        )
        print(f"Done: {catalog_dir}")
        return 0

    # Streaming + parallel path. Workers read one SPOC FITS file each; the
    # main process batches their return values into chunks of ``batch_size``
    # and writes one parquet shard per chunk under ``scratch``. This keeps
    # peak RAM to ~batch_size lightcurves (~20 MB) regardless of total count.
    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    print(f"Streaming per-batch parquet to {scratch} (pool={args.num_processes}, "
          f"batch={args.batch_size})")

    try:
        total = 0
        n_written = 0
        buffer: list[dict] = []

        def _flush():
            nonlocal n_written, total, buffer
            if not buffer:
                return
            table = build_table(buffer)
            out_path = os.path.join(scratch, f"part-{n_written:06d}.parquet")
            pq.write_table(table, out_path)
            total += table.num_rows
            n_written += 1
            print(f"  wrote {out_path.split('/')[-1]}: {table.num_rows} LCs "
                  f"(running total: {total})")
            buffer = []

        if args.num_processes > 1 and len(files) > 1:
            with Pool(args.num_processes) as pool:
                for i, row in enumerate(
                    pool.imap_unordered(_read_lightcurve_safe, files, chunksize=32),
                    1,
                ):
                    if row is None:
                        continue
                    buffer.append(row)
                    if len(buffer) >= args.batch_size:
                        _flush()
        else:
            for p in files:
                row = _read_lightcurve_safe(p)
                if row is None:
                    continue
                buffer.append(row)
                if len(buffer) >= args.batch_size:
                    _flush()
        _flush()

        if n_written == 0:
            print("ERROR: no readable lightcurves", file=sys.stderr)
            return 1

        print(f"\nIngesting {n_written} batch(es), {total} lightcurves → HATS catalog")
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
