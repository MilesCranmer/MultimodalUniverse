"""Convert raw Gaia DR3 bulk download files into the XP-joined HATS catalog.

This is the ``gaia_xp`` dataset: the ~220M sources that have BP/RP continuous
spectra, joined to their full GaiaSource row. For the full ~1.8B DR3 source
catalog with no XP filter, see ``scripts/gaia/build_parent_sample_hats.py``.

Matches v1 MMU's ``scripts/gaia/gaia.py`` HuggingFace schema 1:1, joining the
raw Gaia Archive bulk tables on ``source_id`` directly (duplicating what v1's
``merge_parts.py`` does) instead of consuming a pre-built intermediate.

Cluster layout::

    /mnt/ceph/users/polymathic/external_data/astro/Gaia/
        GaiaSource_{start}-{end}.hdf5              # main DR3 source catalog
        XpContinuousMeanSpectrum_{start}-{end}.hdf5 # BP/RP coefficients
        AstrophysicalParameters_{start}-{end}.hdf5  # (not used — all needed
                                                     # gspphot fields are
                                                     # mirrored in GaiaSource)
        RvsMeanSpectrum_{start}-{end}.hdf5          # (not used — v1 only
                                                     # exposes the scalar RV
                                                     # fields from GaiaSource)

The GaiaSource and XpContinuousMeanSpectrum shards share the same ``{start}-{end}``
suffix scheme (the "parts" numbering from the Gaia Archive's bulk split). For
every part, every XP source_id is a subset of the corresponding GaiaSource
source_id set — v1 ``merge_parts.py`` asserts this.

This script, for each part:

1. Opens the XP file and loads source_ids + bp/rp coefficients. This is the
   smaller of the two (~16% of GaiaSource rows). It's the defining set —
   only sources with XP spectra end up in the Gaia HATS catalog.
2. Opens the GaiaSource file and loads source_ids + all required columns.
3. Inner-joins on source_id (np.intersect1d, matching v1's approach).
4. Applies the cone cut (if one is active) on the joined ra/dec BEFORE
   building the struct columns.
5. Builds a PyArrow table where each row has:
     ``spectral_coefficients`` (struct<coeff: list<f32>, coeff_error: list<f32>>),
     ``photometry``            (struct<13 scalar fields>),
     ``astrometry``            (struct<20 scalar fields>),
     ``radial_velocity``       (struct<5 scalar fields>),
     ``gspphot``               (struct<21 scalar fields>),
     ``flags``                 (struct<1 scalar field: ruwe>),
     ``corrections``           (struct<7 scalar fields>),
     ``object_id`` (int64, source_id),
     ``ra``, ``dec`` (float64, duplicated from astrometry for HATS spatial indexing)

The struct-of-parallel-scalar-fields layout survives hats-import's
``nested_pandas`` round-trip (no extension types anywhere). See
``project_image_storage`` memory for details on the nested_pandas pitfalls.
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


CATALOG_NAME = "gaia_xp"


# Field lists copied verbatim from v1 scripts/gaia/gaia.py. These must stay
# in sync with v1; if v1 ever adds/removes a column, this script must too.
SPECTRUM_FEATURES = ["coeff", "coeff_error"]

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
# ra and dec appear in both ASTROMETRY_FEATURES and the top-level ra/dec, and
# are only read once from the HDF5 file; dedupe to avoid h5py errors.
ALL_SOURCE_COLUMNS = list(dict.fromkeys(ALL_SOURCE_COLUMNS))


PART_RE = re.compile(r"_(\d+)-(\d+)\.hdf5$")


def find_shard_pairs(raw_root: str) -> list[tuple[str, str]]:
    """Pair up GaiaSource_*.hdf5 with XpContinuousMeanSpectrum_*.hdf5 shards
    by their ``{start}-{end}`` suffix.

    Returns a list of ``(source_path, xp_path)`` tuples in sorted order.
    """
    sources = {}
    xps = {}
    for p in glob.glob(os.path.join(raw_root, "GaiaSource_*.hdf5")):
        m = PART_RE.search(p)
        if m:
            sources[m.group(0)] = p
    for p in glob.glob(os.path.join(raw_root, "XpContinuousMeanSpectrum_*.hdf5")):
        m = PART_RE.search(p)
        if m:
            xps[m.group(0)] = p
    common = sorted(set(sources) & set(xps))
    return [(sources[k], xps[k]) for k in common]


def _read_xp_coords(xp_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only ``source_id``, ``ra``, ``dec`` from one XP shard.

    This is the cheap pre-filter read: ~2 MB per shard vs ~77 MB for the
    full coefficient arrays. Used by :func:`process_shard` to apply the
    cone cut BEFORE touching the expensive coefficient columns or the
    even-larger GaiaSource file.
    """
    with h5py.File(xp_path, "r") as f:
        source_id = np.asarray(f["source_id"][:], dtype=np.int64)
        ra = np.asarray(f["ra"][:], dtype=np.float64)
        dec = np.asarray(f["dec"][:], dtype=np.float64)
    return source_id, ra, dec


def _read_xp_coeffs(xp_path: str) -> dict[str, np.ndarray]:
    """Load the full coefficient arrays from one XP shard.

    Returns a dict with:
        coeff:     (N, 110) float32  (55 bp + 55 rp)
        coeff_error: (N, 110) float32

    Only called for shards that have at least one source surviving the
    cone cut (or for full-catalog builds where no cone is set).
    """
    with h5py.File(xp_path, "r") as f:
        bp_c = np.asarray(f["bp_coefficients"][:], dtype=np.float32)
        rp_c = np.asarray(f["rp_coefficients"][:], dtype=np.float32)
        bp_e = np.asarray(f["bp_coefficient_errors"][:], dtype=np.float32)
        rp_e = np.asarray(f["rp_coefficient_errors"][:], dtype=np.float32)
    coeff = np.concatenate([bp_c, rp_c], axis=-1).astype(np.float32)
    coeff_error = np.concatenate([bp_e, rp_e], axis=-1).astype(np.float32)
    return {"coeff": coeff, "coeff_error": coeff_error}


def _read_source_columns(source_path: str, columns: list[str]) -> dict[str, np.ndarray]:
    """Load ``columns`` from a GaiaSource shard.

    Any column missing from the file is returned as an array of NaN (for float
    cols) or zeros (for int cols) so that v1's schema stays uniform across
    Gaia DR3 data releases where column availability can vary slightly.
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


def process_shard(
    source_path: str,
    xp_path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Process one (source, xp) shard pair into a PyArrow table.

    Returns ``None`` if the cone cut leaves zero rows.

    Fast-path for cone cuts: reads only the cheap XP coord columns first,
    applies the cone filter, and skips the shard entirely if no XP source
    falls in the cone. Since XP sources are a strict subset of GaiaSource
    sources (v1 ``merge_parts.py`` asserts this) and XP already carries
    ra/dec, this avoids both the expensive coefficient read and the even
    larger GaiaSource read for ~99% of shards in a 1° cone.
    """
    cone_active = (
        ra_center is not None
        and dec_center is not None
        and radius is not None
    )

    # Cheap pre-filter: read only source_id/ra/dec from the XP file.
    xp_source_id_all, xp_ra_all, xp_dec_all = _read_xp_coords(xp_path)

    if cone_active:
        xp_mask = apply_cone_filter(
            xp_ra_all, xp_dec_all, ra_center, dec_center, radius
        )
        if not xp_mask.any():
            return None
        xp_source_id = xp_source_id_all[xp_mask]
    else:
        xp_mask = None
        xp_source_id = xp_source_id_all

    # At least one source is interesting — pay the cost of the full XP
    # coefficient read and the GaiaSource read now.
    xp_coeffs = _read_xp_coeffs(xp_path)
    coeff_all = xp_coeffs["coeff"]
    coeff_error_all = xp_coeffs["coeff_error"]
    if xp_mask is not None:
        coeff_all = coeff_all[xp_mask]
        coeff_error_all = coeff_error_all[xp_mask]

    src = _read_source_columns(source_path, ALL_SOURCE_COLUMNS)

    # Inner-join XP survivors with GaiaSource on source_id.
    _, ix_xp, ix_src = np.intersect1d(
        xp_source_id, src["source_id"], return_indices=True, assume_unique=True
    )
    n = ix_xp.size
    if n == 0:
        return None

    coeff = coeff_all[ix_xp]
    coeff_error = coeff_error_all[ix_xp]
    source_id = xp_source_id[ix_xp]
    src_joined = {c: src[c][ix_src] for c in ALL_SOURCE_COLUMNS if c != "source_id"}

    ra = np.asarray(src_joined["ra"], dtype=np.float64)
    dec = np.asarray(src_joined["dec"], dtype=np.float64)

    def _struct(feature_list: list[str]) -> pa.StructArray:
        """Build a struct array from a list of float32 scalar columns."""
        arrays = [
            pa.array(np.asarray(src_joined[f], dtype=np.float32), type=pa.float32())
            for f in feature_list
        ]
        return pa.StructArray.from_arrays(arrays, names=feature_list)

    spectral_struct = pa.StructArray.from_arrays(
        [
            pa.array(
                [list(row) for row in coeff],
                type=pa.list_(pa.float32()),
            ),
            pa.array(
                [list(row) for row in coeff_error],
                type=pa.list_(pa.float32()),
            ),
        ],
        names=["coeff", "coeff_error"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        "object_id": pa.array(source_id.astype(np.int64), type=pa.int64()),
        "spectral_coefficients": spectral_struct,
        "photometry": _struct(PHOTOMETRY_FEATURES),
        "astrometry": _struct(ASTROMETRY_FEATURES),
        "radial_velocity": _struct(RV_FEATURES),
        "gspphot": _struct(GSPPHOT_FEATURES),
        "flags": _struct(FLAG_FEATURES),
        "corrections": _struct(CORRECTION_FEATURES),
    }
    return pa.table(columns)


def _process_shard_pair_to_parquet(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: read one (GaiaSource, XP) shard pair, optionally cone-filter,
    write out ``part-<xp basename>.parquet`` under ``scratch``.

    Input tuple: ``(source_path, xp_path, scratch, ra_center, dec_center, radius)``
    Returns ``(xp_basename, n_rows, err)``. All exceptions converted to string
    errors so one bad shard can't kill the pool.
    """
    source_path, xp_path, scratch, ra_center, dec_center, radius = args
    name = os.path.basename(xp_path)
    try:
        t = process_shard(
            source_path, xp_path,
            ra_center=ra_center, dec_center=dec_center, radius=radius,
        )
        if t is None:
            return name, 0, None
        part_name = name.replace(".hdf5", ".parquet")
        out_path = os.path.join(scratch, part_name)
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
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of (source, xp) shard pairs to process.")
    parser.add_argument("--pixel-threshold", type=int, default=100_000)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    parser.add_argument("--in-memory", action="store_true",
                        help="Use the legacy in-memory build (only safe for the cosmos test "
                             "slice; full Gaia DR3 builds OOM single nodes — default streams "
                             "via per-shard parquet files in --scratch-dir).")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory for per-shard-pair parquet "
                             "files. REQUIRED in sharded mode.")
    parser.add_argument("--num-processes", type=int, default=96,
                        help="Per-shard multiprocessing.Pool size.")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="This shard's stride offset (0..num-shards-1).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-shard-pair parquet shards.")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Skip build phase; run only write_hats_from_parquet_dir.")
    parser.add_argument("--ingest-workers", type=int, default=96,
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

    # In --only-ingest mode we don't touch raw files at all — the gather job
    # just reads the already-written per-shard parquets out of scratch. Skip
    # find_shard_pairs so the raw Gaia mirror doesn't need to be reachable.
    pairs: list[tuple[str, str]] = []
    if not args.only_ingest:
        pairs = find_shard_pairs(args.raw_root)
        if args.max_files is not None:
            pairs = pairs[:args.max_files]
        if not pairs:
            print(f"ERROR: no GaiaSource/XP shard pairs under {args.raw_root}", file=sys.stderr)
            return 1
        print(f"[shard {args.shard_idx}/{args.num_shards}] Found {len(pairs)} "
              f"(source, xp) shard pair(s)", flush=True)

        # Apply stride slicing for sharded mode.
        if args.num_shards > 1:
            pairs = pairs[args.shard_idx::args.num_shards]
            print(f"[shard {args.shard_idx}] my stride: {len(pairs)} shard pair(s)",
                  flush=True)

    if args.in_memory:
        tables: list[pa.Table] = []
        total = 0
        for i, (source_path, xp_path) in enumerate(pairs, 1):
            t = process_shard(
                source_path, xp_path,
                ra_center=args.ra_center,
                dec_center=args.dec_center,
                radius=args.radius,
            )
            if t is None:
                continue
            tables.append(t)
            total += t.num_rows
            print(f"  [{i}/{len(pairs)}] {os.path.basename(xp_path)}: {t.num_rows} rows")
        if not tables:
            print("ERROR: no rows survived the cone cut", file=sys.stderr)
            return 1
        print(f"\nWriting HATS catalog (in-memory): {total} rows from {len(tables)} shard(s)")
        catalog_dir = write_hats(
            tables,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
        )
        print(f"Done: {catalog_dir}")
        return 0

    # Streaming + parallel path: per-shard-pair parquet → write_hats_from_parquet_dir.
    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    print(f"[shard {args.shard_idx}] Streaming per-shard-pair parquet to {scratch} "
          f"(pool={args.num_processes})", flush=True)
    work = [
        (source_path, xp_path, scratch, args.ra_center, args.dec_center, args.radius)
        for source_path, xp_path in pairs
    ]
    n_written = 0
    total = 0
    try:
        if not args.only_ingest:
            if args.num_processes > 1 and len(work) > 1:
                with Pool(args.num_processes) as pool:
                    for i, (name, n, err) in enumerate(
                        pool.imap_unordered(_process_shard_pair_to_parquet, work), 1
                    ):
                        if err:
                            print(f"[shard {args.shard_idx}] [{i}/{len(pairs)}] "
                                  f"{name}: SKIPPED {err}", file=sys.stderr, flush=True)
                            continue
                        if n == 0:
                            continue
                        n_written += 1
                        total += n
                        if n_written % 20 == 0:
                            print(f"[shard {args.shard_idx}] [{i}/{len(pairs)}] "
                                  f"{name}: {n} rows (running total {total})",
                                  flush=True)
            else:
                for i, item in enumerate(work, 1):
                    name, n, err = _process_shard_pair_to_parquet(item)
                    if err:
                        print(f"[shard {args.shard_idx}] [{i}/{len(pairs)}] "
                              f"{name}: SKIPPED {err}", file=sys.stderr, flush=True)
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
            n_workers=args.ingest_workers,
            debug=False,
            pixel_threshold=args.pixel_threshold,
        )
    finally:
        if args.scratch_dir is None and not sharded_mode:
            shutil.rmtree(scratch, ignore_errors=True)
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
