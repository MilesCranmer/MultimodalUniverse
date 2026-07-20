"""Convert raw Gaia DR3 bulk GaiaSource shards into a full-DR3 HATS catalog.

This is the ``gaia`` dataset: every one of the ~1.8B Gaia DR3 sources,
with the full scalar schema (photometry, astrometry, RV, GSP-Phot,
flags, corrections). No XP filter. No spectral coefficients column.

For the ~220M subset with BP/RP continuous spectra see the sibling
``scripts/gaia_xp/build_parent_sample_hats.py``. The two catalogs share
the field lists below and the per-shard HDF5 reader; the only
difference is that ``gaia_xp`` joins against XpContinuousMeanSpectrum
and keeps a spectral coefficients column, while ``gaia`` just reads
``GaiaSource_*.hdf5`` alone.

Cluster layout::

    /mnt/ceph/users/polymathic/external_data/astro/Gaia/
        GaiaSource_{start}-{end}.hdf5   # 3386 shards, ~1.8B rows total

Each row in the output HATS catalog has:
    ``photometry``      (struct<13 scalar fields>)
    ``astrometry``      (struct<20 scalar fields>)
    ``radial_velocity`` (struct<5 scalar fields>)
    ``gspphot``         (struct<21 scalar fields>)
    ``flags``           (struct<ruwe>)
    ``corrections``     (struct<7 scalar fields>)
    ``object_id`` (int64, source_id)
    ``ra``, ``dec`` (float64, duplicated from astrometry for HATS spatial index)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from multiprocessing import Pool

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, write_hats, write_hats_from_parquet_dir


CATALOG_NAME = "gaia"


# Field lists match the gaia_xp (XP-joined) catalog except there's no
# SPECTRUM_FEATURES here — we're only reading GaiaSource shards.
PHOTOMETRY_FEATURES = [
    "phot_g_mean_mag", "phot_g_mean_flux", "phot_g_mean_flux_error",
    "phot_bp_mean_mag", "phot_bp_mean_flux", "phot_bp_mean_flux_error",
    "phot_rp_mean_mag", "phot_rp_mean_flux", "phot_rp_mean_flux_error",
    "phot_bp_rp_excess_factor", "bp_rp", "bp_g", "g_rp",
]

ASTROMETRY_FEATURES = [
    "ra", "ra_error", "dec", "dec_error",
    "parallax", "parallax_error",
    "pmra", "pmra_error", "pmdec", "pmdec_error",
    "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr", "ra_pmdec_corr",
    "dec_parallax_corr", "dec_pmra_corr", "dec_pmdec_corr",
    "parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr",
]

RV_FEATURES = [
    "radial_velocity", "radial_velocity_error",
    "rv_template_fe_h", "rv_template_logg", "rv_template_teff",
]

GSPPHOT_FEATURES = [
    "ag_gspphot", "ag_gspphot_lower", "ag_gspphot_upper",
    "azero_gspphot", "azero_gspphot_lower", "azero_gspphot_upper",
    "distance_gspphot", "distance_gspphot_lower", "distance_gspphot_upper",
    "ebpminrp_gspphot", "ebpminrp_gspphot_lower", "ebpminrp_gspphot_upper",
    "logg_gspphot", "logg_gspphot_lower", "logg_gspphot_upper",
    "mh_gspphot", "mh_gspphot_lower", "mh_gspphot_upper",
    "teff_gspphot", "teff_gspphot_lower", "teff_gspphot_upper",
]

FLAG_FEATURES = ["ruwe"]

CORRECTION_FEATURES = [
    "ecl_lat", "ecl_lon",
    "nu_eff_used_in_astrometry", "pseudocolour",
    "astrometric_params_solved",
    "rv_template_teff",  # v1 has this duplicated in both RV and corrections
    "grvs_mag",
]

ALL_SOURCE_COLUMNS = (
    ["source_id", "ra", "dec"]
    + PHOTOMETRY_FEATURES
    + ASTROMETRY_FEATURES
    + RV_FEATURES
    + GSPPHOT_FEATURES
    + FLAG_FEATURES
    + CORRECTION_FEATURES
)
ALL_SOURCE_COLUMNS = list(dict.fromkeys(ALL_SOURCE_COLUMNS))

PART_RE = re.compile(r"_(\d+)-(\d+)\.hdf5$")


def find_source_shards(raw_root: str) -> list[str]:
    """Return all GaiaSource_*.hdf5 shards under ``raw_root`` in sorted order."""
    paths = []
    for p in glob.glob(os.path.join(raw_root, "GaiaSource_*.hdf5")):
        if PART_RE.search(p):
            paths.append(p)
    return sorted(paths)


def _read_source_columns(source_path: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Load ``columns`` from a GaiaSource shard.

    Any column missing from the file is returned as an array of NaN so
    the final catalog has a uniform schema across DR3 minor-version
    differences.
    """
    with h5py.File(source_path, "r") as f:
        n = f["source_id"].shape[0]
        out: dict[str, np.ndarray] = {}
        for c in columns:
            if c in f:
                out[c] = np.asarray(f[c][:])
            else:
                out[c] = np.full(n, np.nan, dtype=np.float32)
    return out


def _struct(cols: dict[str, np.ndarray], names: list[str]) -> pa.StructArray:
    """Build a PyArrow StructArray from a dict of 1D numpy arrays.

    Every field is cast to ``float32`` to guarantee a stable schema across
    shards. Without the cast, a column that is missing in one shard (filled
    by :func:`_read_source_columns` with float32 NaN) and present as native
    int64 in another shard would produce parquet files with inconsistent
    schemas, and the gather step would reject them at concat time.
    This mirrors the dtype handling in gaia_xp's `_struct`.
    """
    arrays = []
    for name in names:
        arr = np.asarray(cols[name], dtype=np.float32)
        arrays.append(pa.array(arr, type=pa.float32()))
    return pa.StructArray.from_arrays(arrays, names=names)


def build_table(source_cols: dict[str, np.ndarray]) -> pa.Table:
    """Build a PyArrow table for one GaiaSource shard. No XP join."""
    n = source_cols["source_id"].shape[0]
    columns: dict[str, pa.Array] = {
        "object_id": pa.array(source_cols["source_id"], type=pa.int64()),
        "ra": pa.array(source_cols["ra"], type=pa.float64()),
        "dec": pa.array(source_cols["dec"], type=pa.float64()),
        "photometry": _struct(source_cols, PHOTOMETRY_FEATURES),
        "astrometry": _struct(source_cols, ASTROMETRY_FEATURES),
        "radial_velocity": _struct(source_cols, RV_FEATURES),
        "gspphot": _struct(source_cols, GSPPHOT_FEATURES),
        "flags": _struct(source_cols, FLAG_FEATURES),
        "corrections": _struct(source_cols, CORRECTION_FEATURES),
    }
    return pa.table(columns)


def process_shard(
    source_path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Read one GaiaSource shard into a PyArrow table, with optional cone cut."""
    source_cols = _read_source_columns(source_path, ALL_SOURCE_COLUMNS)
    if ra_center is not None and dec_center is not None and radius is not None:
        mask = apply_cone_filter(
            source_cols["ra"], source_cols["dec"],
            ra_center, dec_center, radius,
        )
        if not mask.any():
            return None
        source_cols = {k: v[mask] for k, v in source_cols.items()}
    if source_cols["source_id"].shape[0] == 0:
        return None
    return build_table(source_cols)


def _process_shard_to_parquet(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: read one GaiaSource shard, optionally cone-filter,
    write ``part-<basename>.parquet`` into scratch. Returns
    ``(name, n_rows, err)``.
    """
    source_path, scratch, ra_center, dec_center, radius = args
    name = os.path.basename(source_path)
    try:
        out_path = os.path.join(scratch, name.replace(".hdf5", ".parquet"))
        if os.path.exists(out_path):
            return name, 0, None  # skip-if-exists on rerun
        t = process_shard(
            source_path,
            ra_center=ra_center, dec_center=dec_center, radius=radius,
        )
        if t is None:
            return name, 0, None
        tmp_path = out_path + ".tmp"
        pq.write_table(t, tmp_path)
        os.rename(tmp_path, out_path)
        n = t.num_rows
        del t
        return name, n, None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as exc:  # noqa: BLE001
        return name, 0, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=100_000,
                        help="Max rows per HATS partition. Default 100k for gaia's "
                             "1.8B rows — keeps output to ~18k partitions instead of "
                             "485k (which caused the gather to freeze).")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory. REQUIRED in sharded mode.")
    parser.add_argument("--num-processes", type=int, default=96,
                        help="Per-shard multiprocessing.Pool size.")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="This shard's stride offset (0..num-shards-1).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-file parquet shards.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Skip build phase; run only write_hats_from_parquet_dir.")
    parser.add_argument("--ingest-workers", type=int, default=8,
                        help="Dask workers for the ingest step.")
    args = parser.parse_args(argv)

    if args.skip_ingest and args.only_ingest:
        print("ERROR: --skip-ingest and --only-ingest are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print(f"ERROR: shard-idx {args.shard_idx} outside [0, {args.num_shards})",
              file=sys.stderr)
        return 2

    sharded_mode = args.num_shards > 1 or args.skip_ingest or args.only_ingest
    if sharded_mode and args.scratch_dir is None:
        print("ERROR: --scratch-dir is required in sharded mode", file=sys.stderr)
        return 2

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)

    try:
        if not args.only_ingest:
            sources = find_source_shards(args.raw_root)
            if args.max_files is not None:
                sources = sources[: args.max_files]
            if not sources:
                print(f"ERROR: no GaiaSource_*.hdf5 under {args.raw_root}",
                      file=sys.stderr)
                return 1
            print(f"[shard {args.shard_idx}/{args.num_shards}] found "
                  f"{len(sources)} GaiaSource shard(s)", flush=True)

            if args.num_shards > 1:
                sources = sources[args.shard_idx::args.num_shards]
                print(f"[shard {args.shard_idx}] my stride: {len(sources)} shard(s)",
                      flush=True)

            work = [
                (p, scratch, args.ra_center, args.dec_center, args.radius)
                for p in sources
            ]
            n_written = 0
            total = 0
            if args.num_processes > 1 and len(work) > 1:
                with Pool(args.num_processes) as pool:
                    for i, (name, n, err) in enumerate(
                        pool.imap_unordered(_process_shard_to_parquet, work), 1
                    ):
                        if err:
                            print(f"[shard {args.shard_idx}] [{i}/{len(sources)}] "
                                  f"{name}: SKIPPED {err}", file=sys.stderr, flush=True)
                            continue
                        if n == 0:
                            continue
                        n_written += 1
                        total += n
                        if n_written % 20 == 0:
                            print(f"[shard {args.shard_idx}] [{i}/{len(sources)}] "
                                  f"{name}: {n} rows (running total {total})",
                                  flush=True)
            else:
                for i, item in enumerate(work, 1):
                    name, n, err = _process_shard_to_parquet(item)
                    if err:
                        print(f"[shard {args.shard_idx}] {name}: SKIPPED {err}",
                              file=sys.stderr, flush=True)
                        continue
                    if n == 0:
                        continue
                    n_written += 1
                    total += n

            print(f"[shard {args.shard_idx}] BUILD DONE: {n_written} shards, "
                  f"{total} rows", flush=True)

        if args.skip_ingest:
            return 0

        print(f"Ingesting {scratch} → HATS catalog (workers={args.ingest_workers})",
              flush=True)
        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
            n_workers=args.ingest_workers,
            debug=False,
        )
        print(f"Done: {catalog_dir}")
    finally:
        if args.scratch_dir is None and not sharded_mode:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
