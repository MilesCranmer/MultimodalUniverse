"""Convert raw DESI DR1 iron spectra into a HATS catalog.

Matches v1 MMU's ``scripts/desi/build_parent_sample.py`` pipeline exactly,
with HATS output instead of HDF5:

    1. Load the full redshift catalog ``zall-pix-iron.fits`` (28M rows).
    2. Apply the v1 ``selection_fn``: SURVEY=='main', MAIN_PRIMARY,
       OBJTYPE=='TGT', COADD_FIBERSTATUS==0, and exclude BAD_HANDLES
       (missing-on-server files).
    3. Group surviving rows by (SURVEY, PROGRAM, HEALPIX) so all fibers from
       the same coadd file are processed together.
    4. For each coadd file:
         - ``desispec.io.read_spectra(...).select(targets=target_ids)``
         - ``desispec.coaddition.coadd_cameras(spectra)`` to merge B/R/Z
           onto a common wavelength grid (with interpolation in the overlap
           regions — this is the thing a naive simple-stack cannot do).
         - Extract ``wave["brz"]``, ``flux["brz"]``, ``ivar["brz"]``,
           ``mask["brz"]``, and ``resolution_data["brz"]``.
         - Estimate a Gaussian LSF sigma via ``scipy.optimize.curve_fit`` on
           the mean resolution profile, matching v1.
    5. Join the per-fiber spectra with the curated scalar columns from the
       redshift catalog (Z, ZERR, ZWARN, EBV, FLUX_G/R/Z, ...).
    6. Build a PyArrow table with one row per fiber, where the ``spectrum``
       column is a struct-of-parallel-lists matching v1's HuggingFace
       schema exactly::

          spectrum: struct<
              flux:      list<float32>,
              ivar:      list<float32>,
              lsf_sigma: list<float32>,
              lambda:    list<float32>,
              mask:      list<bool>,
          >

       No extension types anywhere in the tree (see ``project_image_storage``
       memory for why: nested_pandas crashes on extension types inside
       list/struct).
    7. Write the HATS catalog via ``mmu.hats_import.write_hats``.

Cluster layout:

    /mnt/ceph/users/polymathic/external_data/astro/DESI_DR1/
        zall-pix-iron.fits                          # full redshift catalog
        coadd-main-{program}-{healpix}.fits         # ~32k coadd files, flat

The v1 raw-data layout on DESI's servers is nested by survey/program/
pix_group/healpix, but the cluster mirror is flattened to a single directory.
We find coadd files by their filename pattern, not by directory walk.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
from astropy.table import Table
from scipy.optimize import curve_fit

os.environ.setdefault("DESI_LOGLEVEL", "WARNING")

import desispec.io
from desispec import coaddition

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import (
    default_scratch_dir,
    np_to_pyarrow_list,
    to_native_endian,
    write_hats,
    write_hats_from_parquet_dir,
)
import pyarrow.parquet as pq


CATALOG_NAME = "desi"


# Curated scalar columns matching v1's HuggingFace schema exactly.
# See scripts/desi/desi.py _FLOAT_FEATURES and _BOOL_FEATURES.
FLOAT_FEATURES = [
    "Z",
    "ZERR",
    "EBV",
    "FLUX_G",
    "FLUX_R",
    "FLUX_Z",
    "FLUX_IVAR_G",
    "FLUX_IVAR_R",
    "FLUX_IVAR_Z",
    "FIBERFLUX_G",
    "FIBERFLUX_R",
    "FIBERFLUX_Z",
    "FIBERTOTFLUX_G",
    "FIBERTOTFLUX_R",
    "FIBERTOTFLUX_Z",
]
BOOL_FEATURES = ["ZWARN"]


# BAD_HANDLES lifted directly from v1 MMU scripts/desi/build_parent_sample.py.
# These healpix values appear in tilepix.fits but the corresponding coadd
# files don't exist on DESI's servers — see the v1 docstring for the issue
# references. Same applies to the cluster mirror.
BAD_HANDLES: dict[str, list[int]] = {
    "bright": [9836, 4802, 4561, 4730],
    "dark": [26535, 15051, 10844, 9913],
    "backup": [10786, 10810],
}


def _as_str_array(col) -> np.ndarray:
    """Decode a FITS byte/str column into a stripped-string numpy array."""
    out = np.empty(len(col), dtype=object)
    for i, v in enumerate(col):
        if isinstance(v, (bytes, np.bytes_)):
            out[i] = v.decode().strip()
        else:
            out[i] = str(v).strip()
    return out.astype(str)


def selection_fn(catalog: Table) -> np.ndarray:
    """Return a boolean mask selecting v1's standard science fibers.

    Matches scripts/desi/build_parent_sample.py:selection_fn exactly:
    SURVEY=='main' & MAIN_PRIMARY & OBJTYPE=='TGT' & COADD_FIBERSTATUS==0,
    minus the BAD_HANDLES per program.
    """
    survey = _as_str_array(catalog["SURVEY"])
    objtype = _as_str_array(catalog["OBJTYPE"])
    mask = (survey == "main")
    mask &= np.asarray(catalog["MAIN_PRIMARY"]).astype(bool)
    mask &= (objtype == "TGT")
    mask &= np.asarray(catalog["COADD_FIBERSTATUS"]) == 0
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    for bad_program, bad_hp in BAD_HANDLES.items():
        mask &= ~((program == bad_program) & np.isin(healpix, bad_hp))
    return mask


def find_matching_indices(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Return indices into arr2 that reorder arr2 to match arr1. Lifted
    verbatim from v1 MMU."""
    sort_idx_arr1 = np.argsort(arr1)
    sort_idx_arr2 = np.argsort(arr2)
    inverse_sort_idx_arr1 = np.argsort(sort_idx_arr1)
    return sort_idx_arr2[inverse_sort_idx_arr1]


def _gauss(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def estimate_lsf_sigma(resolution_data: np.ndarray) -> float:
    """Estimate a single Gaussian LSF sigma (in pixel units) from the mean
    resolution profile across fibers and wavelengths. Matches v1 exactly.
    """
    # resolution_data shape: (n_fibers, ndiag, nwave)
    lsf = resolution_data.mean(axis=-1).mean(axis=0)
    popt, _ = curve_fit(_gauss, np.arange(len(lsf)), lsf, p0=[1.0, 5.0, 1.0])
    return float(popt[2])


def coadd_file_path(raw_root: str, survey: str, program: str, healpix: int) -> str:
    """Return the flat-layout coadd file path on the cluster mirror."""
    return os.path.join(raw_root, f"coadd-{survey}-{program}-{healpix}.fits")


def process_coadd(
    filename: str,
    target_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read a coadd file, select the requested targets, run coadd_cameras,
    and return per-fiber arrays. Matches v1's processing_fn exactly, minus
    the SNR columns which aren't in the v1 HF schema.
    """
    spectra = desispec.io.read_spectra(filename).select(targets=target_ids)
    combined = coaddition.coadd_cameras(spectra)

    reordering = find_matching_indices(target_ids, np.array(combined.target_ids()))

    wavelength = np.asarray(combined.wave["brz"], dtype=np.float32)
    flux = np.asarray(combined.flux["brz"])[reordering].astype(np.float32)
    ivar = np.asarray(combined.ivar["brz"])[reordering].astype(np.float32)
    mask = np.asarray(combined.mask["brz"])[reordering].astype(np.uint32)
    resolution = np.asarray(combined.resolution_data["brz"])[reordering].astype(np.float32)
    tgt_ids = np.asarray(combined.target_ids())[reordering]

    if not np.array_equal(tgt_ids, target_ids):
        raise RuntimeError(
            f"Target ID mismatch after reordering in {filename}: "
            f"got {tgt_ids[:5]}, expected {target_ids[:5]}"
        )

    lsf_sigma_scalar = estimate_lsf_sigma(resolution)
    n = len(target_ids)
    n_wave = len(wavelength)
    lsf_sigma = np.full((n, n_wave), lsf_sigma_scalar, dtype=np.float32)
    lam = np.tile(wavelength.reshape(1, -1), (n, 1))
    bad_mask = (mask > 0) | (ivar <= 1e-6)

    return {
        "TARGETID": tgt_ids,
        "flux": flux,
        "ivar": ivar,
        "lambda": lam,
        "lsf_sigma": lsf_sigma,
        "mask": bad_mask,
    }


def _group_catalog(catalog: Table) -> list[tuple[str, str, int, np.ndarray, np.ndarray]]:
    """Group the catalog by (SURVEY, PROGRAM, HEALPIX). Returns a list of
    tuples ``(survey, program, healpix, target_ids, row_indices)`` where
    ``row_indices`` are the positions of the group's rows in the full
    catalog, so we can later stitch the scalar columns back together.
    """
    survey = _as_str_array(catalog["SURVEY"])
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    target_ids = np.asarray(catalog["TARGETID"])

    # Sort by (survey, program, healpix) so group boundaries are contiguous.
    order = np.lexsort((healpix, program, survey))
    groups: list[tuple[str, str, int, np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(order):
        j = i
        s0, p0, h0 = survey[order[i]], program[order[i]], healpix[order[i]]
        while (
            j < len(order)
            and survey[order[j]] == s0
            and program[order[j]] == p0
            and healpix[order[j]] == h0
        ):
            j += 1
        idxs = order[i:j]
        groups.append((s0, p0, int(h0), target_ids[idxs], idxs))
        i = j
    return groups


def _build_group_table(
    catalog_subset: Table,
    spectrum_arrays: dict[str, np.ndarray],
) -> pa.Table:
    """Build the per-group PyArrow table from one coadd-file's spectra +
    matching catalog rows. Used by both ``build_table`` (in-memory path)
    and ``build_to_parquet_dir`` (streaming path).
    """
    target_ids_ordered = spectrum_arrays["TARGETID"]
    cat_tids = np.asarray(catalog_subset["TARGETID"])
    if not np.array_equal(cat_tids, target_ids_ordered):
        raise RuntimeError(
            "TARGETID ordering mismatch between catalog subset and processed spectra. "
            f"catalog[:5]={cat_tids[:5]}, spectra[:5]={target_ids_ordered[:5]}"
        )
    ra = np.asarray(catalog_subset["TARGET_RA"], dtype=np.float64)
    dec = np.asarray(catalog_subset["TARGET_DEC"], dtype=np.float64)

    # Build the spectrum struct using fast offset-based ListArrays (avoid
    # Python list-of-list materialization which doubles memory).
    flux = spectrum_arrays["flux"]
    ivar = spectrum_arrays["ivar"]
    lam = spectrum_arrays["lambda"]
    lsf = spectrum_arrays["lsf_sigma"]
    bad_mask = spectrum_arrays["mask"]

    flux_arr = np_to_pyarrow_list(flux.astype(np.float32))
    ivar_arr = np_to_pyarrow_list(ivar.astype(np.float32))
    lsf_arr = np_to_pyarrow_list(lsf.astype(np.float32))
    lam_arr = np_to_pyarrow_list(lam.astype(np.float32))
    # ListArray.from_arrays requires a values array; build via offsets too.
    nrows, length = bad_mask.shape
    mask_values = pa.array(bad_mask.reshape(-1).astype(bool))
    mask_offsets = np.arange(0, (nrows + 1) * length, length, dtype=np.int32)
    mask_arr = pa.ListArray.from_arrays(values=mask_values, offsets=mask_offsets)

    spectrum_struct = pa.StructArray.from_arrays(
        [flux_arr, ivar_arr, lsf_arr, lam_arr, mask_arr],
        names=["flux", "ivar", "lsf_sigma", "lambda", "mask"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(ra)),
        "dec": pa.array(to_native_endian(dec)),
        "object_id": pa.array([str(int(t)) for t in target_ids_ordered], type=pa.string()),
        "spectrum": spectrum_struct,
    }
    for f in FLOAT_FEATURES:
        arr = np.asarray(catalog_subset[f], dtype=np.float32)
        columns[f] = pa.array(to_native_endian(arr))
    for f in BOOL_FEATURES:
        arr = np.asarray(catalog_subset[f]).astype(bool)
        columns[f] = pa.array(arr)
    return pa.table(columns)


def build_table(
    catalog: Table,
    raw_root: str,
    max_groups: int | None = None,
) -> pa.Table:
    """Small/test path: process all groups and concatenate into ONE PyArrow
    table held in memory. Used for the cosmos test slice and for legacy
    in-memory builds. Don't use for full production — see
    :func:`build_to_parquet_dir` for the streaming alternative.
    """
    groups = _group_catalog(catalog)
    if max_groups is not None:
        groups = groups[:max_groups]

    per_group_tables: list[pa.Table] = []
    for i, (survey, program, healpix, target_ids, row_idx) in enumerate(groups, 1):
        path = coadd_file_path(raw_root, survey, program, healpix)
        if not os.path.exists(path):
            print(
                f"  [{i}/{len(groups)}] MISSING {os.path.basename(path)}"
                f" ({len(target_ids)} fibers skipped)",
                file=sys.stderr,
            )
            continue
        result = process_coadd(path, target_ids)
        catalog_subset = catalog[row_idx]
        per_group_tables.append(_build_group_table(catalog_subset, result))
        print(
            f"  [{i}/{len(groups)}] {os.path.basename(path)}: {len(target_ids)} fibers"
        )

    if not per_group_tables:
        raise RuntimeError("No spectra produced — all groups had missing coadd files.")
    return pa.concat_tables(per_group_tables, promote_options="default")


# Module-level globals populated by :func:`_worker_init` in each pool child.
# Under Linux fork semantics, the full catalog and config are inherited via
# copy-on-write from the parent process with zero pickling cost, so workers
# can index into a 20M-row Table without the main process having to dehydrate
# per-group dicts (which is O(n) Python-loop slow and blew up memory during
# an earlier run).
_WORKER_CATALOG: Table | None = None
_WORKER_RAW_ROOT: str | None = None
_WORKER_PARQUET_DIR: str | None = None


def _worker_init(catalog: Table, raw_root: str, parquet_dir: str) -> None:
    global _WORKER_CATALOG, _WORKER_RAW_ROOT, _WORKER_PARQUET_DIR
    _WORKER_CATALOG = catalog
    _WORKER_RAW_ROOT = raw_root
    _WORKER_PARQUET_DIR = parquet_dir


def _process_group_to_parquet(args: tuple) -> tuple[str, int, str | None]:
    """Pool worker: process ONE (survey, program, healpix) group.

    Input tuple: ``(survey, program, healpix, target_ids, row_idx)``. The
    catalog, raw_root, and parquet_dir come from the module-level globals
    populated by :func:`_worker_init`.

    Returns ``(out_basename, n_fibers, err)``. All exceptions are
    converted to string errors so one bad coadd file can't kill the pool.
    """
    survey, program, healpix, target_ids, row_idx = args
    assert _WORKER_CATALOG is not None  # initializer runs before first task
    assert _WORKER_RAW_ROOT is not None
    assert _WORKER_PARQUET_DIR is not None
    path = coadd_file_path(_WORKER_RAW_ROOT, survey, program, healpix)
    try:
        if not os.path.exists(path):
            return (
                os.path.basename(path),
                0,
                f"MISSING {os.path.basename(path)} ({len(target_ids)} fibers skipped)",
            )
        result = process_coadd(path, target_ids)
        catalog_subset = _WORKER_CATALOG[row_idx]
        table = _build_group_table(catalog_subset, result)
        out_name = f"part-{survey}-{program}-{healpix:08d}.parquet"
        pq.write_table(table, os.path.join(_WORKER_PARQUET_DIR, out_name))
        n = table.num_rows
        del result, catalog_subset, table
        return out_name, n, None
    except BaseException as exc:  # noqa: BLE001
        return os.path.basename(path), 0, f"{type(exc).__name__}: {exc}"


def build_to_parquet_dir(
    catalog: Table,
    raw_root: str,
    parquet_dir: str,
    max_groups: int | None = None,
    num_processes: int = 1,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> int:
    """Streaming path: process each (survey, program, healpix) group, build
    its per-group PyArrow table, write it as one parquet file under
    ``parquet_dir``, and free the memory before moving on. Returns the
    number of parquet files written.

    Peak RAM is bounded by ``num_processes`` simultaneously-in-flight
    coadd-file groups (~80 MB each) rather than the full DR1 (~3 TB).
    ``num_processes > 1`` fans out the per-group work across a
    ``multiprocessing.Pool`` — essential for full-scale builds where
    serial single-core processing would take >24 h.
    """
    os.makedirs(parquet_dir, exist_ok=True)
    groups = _group_catalog(catalog)
    if max_groups is not None:
        groups = groups[:max_groups]

    # Stride-slice for sharded mode.
    if num_shards > 1:
        groups = groups[shard_idx::num_shards]
        print(f"[shard {shard_idx}] my stride: {len(groups)} groups", flush=True)

    n_written = 0
    if num_processes > 1 and len(groups) > 1:
        # Use a Pool initializer to share the catalog with workers via fork
        # COW instead of pickling per-group dicts (the latter is O(n)
        # Python-loop slow and blew memory on an earlier run). For this to
        # be meaningful the start method must be 'fork' — the default on
        # Linux.
        with Pool(
            num_processes,
            initializer=_worker_init,
            initargs=(catalog, raw_root, parquet_dir),
        ) as pool:
            for i, (out_name, n, err) in enumerate(
                pool.imap_unordered(_process_group_to_parquet, groups, chunksize=16),
                1,
            ):
                if err:
                    print(f"  [{i}/{len(groups)}] SKIPPED: {err}", file=sys.stderr)
                    continue
                n_written += 1
                print(f"  [{i}/{len(groups)}] {out_name}: {n} fibers")
    else:
        # Serial fallback — still need the worker globals set.
        _worker_init(catalog, raw_root, parquet_dir)
        for i, item in enumerate(groups, 1):
            out_name, n, err = _process_group_to_parquet(item)
            if err:
                print(f"  [{i}/{len(groups)}] SKIPPED: {err}", file=sys.stderr)
                continue
            n_written += 1
            print(f"  [{i}/{len(groups)}] {out_name}: {n} fibers")

    if n_written == 0:
        raise RuntimeError("No spectra produced — all groups had missing coadd files.")
    return n_written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument(
        "--zcatalog",
        default=None,
        help="Path to zall-pix-iron.fits; defaults to {raw_root}/zall-pix-iron.fits",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Cap on number of (survey, program, healpix) groups to process.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None,
                        help="Cone-cut center RA in degrees; requires --dec-center/--radius.")
    parser.add_argument("--dec-center", type=float, default=None,
                        help="Cone-cut center Dec in degrees; requires --ra-center/--radius.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone-cut radius in degrees; requires --ra-center/--dec-center.")
    parser.add_argument("--in-memory", action="store_true",
                        help="Use the legacy in-memory build (only safe for the cosmos test "
                             "slice; full-DR1 builds OOM single nodes — default streams via "
                             "per-group parquet files in --scratch-dir).")
    parser.add_argument("--scratch-dir", default=None,
                        help="Shared ceph scratch directory. REQUIRED in sharded mode.")
    parser.add_argument("--num-processes", type=int, default=96,
                        help="Per-shard multiprocessing.Pool size.")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="This shard's stride offset (0..num-shards-1).")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Only write per-group parquet shards.")
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

    # --only-ingest should skip ALL build-phase setup, including the
    # expensive 28M-row zall-pix-iron.fits load. The raw mirror may not
    # even be accessible from the gather node, so accessing it here breaks
    # pure-ingest reruns unnecessarily.
    catalog = None
    if not args.only_ingest:
        zcat_path = args.zcatalog or os.path.join(args.raw_root, "zall-pix-iron.fits")
        if not os.path.exists(zcat_path):
            print(f"ERROR: missing redshift catalog {zcat_path}", file=sys.stderr)
            return 1

        print(f"Loading {zcat_path}...")
        catalog = Table.read(zcat_path)
        print(f"  {len(catalog)} total rows")
        mask = selection_fn(catalog)
        catalog = catalog[mask]
        print(f"  {len(catalog)} rows after selection_fn")

        if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
            cone_mask = apply_cone_filter(
                np.asarray(catalog["TARGET_RA"]),
                np.asarray(catalog["TARGET_DEC"]),
                args.ra_center,
                args.dec_center,
                args.radius,
            )
            catalog = catalog[cone_mask]
            print(
                f"  {len(catalog)} rows after cone cut "
                f"(ra={args.ra_center}, dec={args.dec_center}, radius={args.radius})"
            )
            if len(catalog) == 0:
                print("  WARNING: cone cut left zero rows; no catalog to build.", file=sys.stderr)
                return 1

    if args.in_memory:
        table = build_table(catalog, args.raw_root, max_groups=args.max_groups)
        print(f"\nWriting HATS catalog (in-memory): {table.num_rows} rows")
        catalog_dir = write_hats(
            [table],
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
        )
    else:
        scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
        os.makedirs(scratch, exist_ok=True)
        print(f"[shard {args.shard_idx}/{args.num_shards}] streaming per-group "
              f"parquet to {scratch} (pool={args.num_processes})", flush=True)
        try:
            if not args.only_ingest:
                n = build_to_parquet_dir(
                    catalog, args.raw_root, scratch,
                    max_groups=args.max_groups,
                    num_processes=args.num_processes,
                    shard_idx=args.shard_idx,
                    num_shards=args.num_shards,
                )
                print(f"[shard {args.shard_idx}] BUILD DONE: {n} groups", flush=True)
            if args.skip_ingest:
                return 0
            print(f"Ingesting {scratch} → HATS catalog "
                  f"(workers={args.ingest_workers})", flush=True)
            catalog_dir = write_hats_from_parquet_dir(
                scratch,
                output_path=args.output_root,
                catalog_name=CATALOG_NAME,
                pixel_threshold=args.pixel_threshold,
                n_workers=args.ingest_workers,
                debug=False,
            )
        finally:
            # Clean up auto-generated single-job scratch dirs. Never touch a
            # caller-provided shared scratch (other shard jobs / ingest job
            # may still need it).
            if args.scratch_dir is None and not sharded_mode:
                shutil.rmtree(scratch, ignore_errors=True)
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
