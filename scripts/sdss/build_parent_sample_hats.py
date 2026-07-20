"""Convert raw SDSS spectroscopic data into a single HATS catalog.

The cluster mirrors SDSS DR17 at:

    /mnt/ceph/users/polymathic/external_data/astro/SDSS/
        specObj-dr17.fits                  # master spectroscopic catalog
        {sdss,boss,eboss,segue1,segue2}/   # plate FITS files per sub-survey
            {plate:04d}/spPlate-{plate:04d}-{mjd}.fits

This script:
1. Reads specObj-dr17.fits and applies the standard selection cuts.
2. Groups by (sub-survey, plate) so each plate FITS file is opened once.
3. For each plate, extracts the spectra (flux/ivar/lambda/lsf_sigma/mask) for
   the requested fibers.
4. Joins spectra back to the catalog rows and builds one PyArrow table.
5. Hands the table to ``write_hats`` to produce a single HATS catalog under
   ``{output_root}/sdss/sdss/``.

A ``survey`` column distinguishes the sub-surveys in the unified catalog.

Usage:
    python -m scripts.sdss.build_parent_sample_hats \\
        --raw-root /mnt/ceph/users/polymathic/external_data/astro/SDSS \\
        --output-root /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/sdss \\
        --max-files 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.table import Table, join
from tqdm import tqdm

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, np_to_pyarrow_list, write_hats_from_parquet_dir


CATALOG_NAME = "sdss"

# Per-object scalar features kept from specObj-dr17.fits.
FLOAT_FEATURES = ["VDISP", "VDISP_ERR", "Z", "Z_ERR"]
BOOL_FEATURES = ["ZWARNING"]
# 2D features (one row per object, one column per filter): unrolled at write time.
FLUX_FEATURES = ["SPECTROFLUX", "SPECTROFLUX_IVAR", "SPECTROSYNFLUX", "SPECTROSYNFLUX_IVAR"]
FLUX_FILTERS = ["U", "G", "R", "I", "Z"]


def _as_str(val):
    """astropy reads FITS string columns as numpy bytes; normalize to Python str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode()
    return val


def _str_eq_stripped(col, target: str) -> np.ndarray:
    """Element-wise compare an astropy string column (str or bytes) to a target,
    stripping trailing whitespace. astropy's FITS reader strips trailing spaces
    so the original MMU v1 code's space-padded comparisons (e.g. ``'SCIENCE '``)
    no longer match — strip on both sides.
    """
    target = target.strip()
    return np.array([_as_str(v).strip() == target for v in col])


def selection_fn(catalog: Table) -> np.ndarray:
    """Standard MMU v1 cuts: primary spectra, science targets, good plates."""
    mask = np.asarray(catalog["SPECPRIMARY"]) == 1
    mask &= _str_eq_stripped(catalog["TARGETTYPE"], "SCIENCE")
    mask &= _str_eq_stripped(catalog["PLATEQUALITY"], "good")
    return mask


def _read_plate(args):
    """Worker: read one plate FITS file and return spectra for the requested fibers."""
    plate_path, fiber_ids, object_ids = args
    fiber_ids = np.asarray(fiber_ids) - 1  # SDSS fibers are 1-indexed

    with fits.open(plate_path) as hdus:
        flux = hdus[0].data[fiber_ids]
        ivar = hdus[1].data[fiber_ids]
        and_mask = hdus[2].data[fiber_ids]
        lsf_sigma = hdus[4].data[fiber_ids]
        # Wavelength solution is in the HDU 0 header (log-linear).
        h = hdus[0].header
        loglam = h["CRVAL1"] + h["CD1_1"] * (np.arange(flux.shape[1]) + 1 - h["CRPIX1"])

    mask = and_mask.astype(bool) | (ivar <= 1e-6)
    lam = np.tile(10**loglam, (len(fiber_ids), 1)).astype(np.float32)
    return {
        "object_id": np.asarray(object_ids),
        "spectrum_flux": flux.astype(np.float32),
        "spectrum_ivar": ivar.astype(np.float32),
        "spectrum_lambda": lam,
        "spectrum_lsf_sigma": lsf_sigma.astype(np.float32),
        "spectrum_mask": mask,
    }


def _pad_to_max_length(results: list[dict]) -> dict:
    """Pad per-plate spectra to a common length and concatenate.

    NOTE: this matches the MMU v1 behavior to keep schemas comparable. Padding
    extends ``flux`` / ``lsf_sigma`` with the edge value (visually weird but
    spectroscopically harmless because the corresponding ``mask`` row is True),
    pads ``lambda`` with -1 (so plotting against lambda still gives a clear
    boundary), and pads ``mask`` with True so all padding pixels are masked.
    """
    max_len = max(r["spectrum_flux"].shape[1] for r in results)
    for r in results:
        n = max_len - r["spectrum_flux"].shape[1]
        if n == 0:
            continue
        r["spectrum_flux"]      = np.pad(r["spectrum_flux"],      ((0, 0), (0, n)), mode="edge")
        r["spectrum_ivar"]      = np.pad(r["spectrum_ivar"],      ((0, 0), (0, n)), mode="constant")
        r["spectrum_lambda"]    = np.pad(r["spectrum_lambda"],    ((0, 0), (0, n)), mode="constant", constant_values=-1)
        r["spectrum_lsf_sigma"] = np.pad(r["spectrum_lsf_sigma"], ((0, 0), (0, n)), mode="edge")
        r["spectrum_mask"]      = np.pad(r["spectrum_mask"],      ((0, 0), (0, n)), mode="constant", constant_values=True)

    return {k: np.concatenate([r[k] for r in results], axis=0) for k in results[0]}


def _build_arrow_table(catalog: Table) -> pa.Table:
    """Convert the joined catalog+spectra astropy table into a PyArrow table."""
    columns: dict[str, pa.Array] = {}

    columns["spectrum"] = pa.StructArray.from_arrays(
        [
            np_to_pyarrow_list(np.asarray(catalog["spectrum_flux"], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(catalog["spectrum_ivar"], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(catalog["spectrum_lsf_sigma"], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(catalog["spectrum_lambda"], dtype=np.float32)),
            np_to_pyarrow_list(np.asarray(catalog["spectrum_mask"], dtype=bool)),
        ],
        names=["flux", "ivar", "lsf_sigma", "lambda", "mask"],
    )

    columns["ra"] = pa.array(np.asarray(catalog["ra"], dtype=np.float64))
    columns["dec"] = pa.array(np.asarray(catalog["dec"], dtype=np.float64))
    columns["object_id"] = pa.array([_as_str(o) for o in catalog["object_id"]], type=pa.string())
    columns["survey"] = pa.array(
        [_as_str(s).strip() for s in catalog["SURVEY"]], type=pa.string()
    )

    for f in FLOAT_FEATURES:
        columns[f] = pa.array(np.asarray(catalog[f], dtype=np.float32))

    for f in BOOL_FEATURES:
        columns[f] = pa.array(np.asarray(catalog[f], dtype=bool))

    for f in FLUX_FEATURES:
        flux_data = np.asarray(catalog[f])  # shape (N, 5)
        for i, band in enumerate(FLUX_FILTERS):
            columns[f"{f}_{band}"] = pa.array(flux_data[:, i].astype(np.float32))

    return pa.table(columns)


def find_plate_groups(
    raw_root: str,
    max_files: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> list[tuple[str, Table]]:
    """Read specObj, apply cuts, and return a list of (plate_path, sub_catalog) groups.

    Each entry is one plate FITS file with the catalog rows that belong to it.
    """
    spec_obj_path = os.path.join(raw_root, "specObj-dr17.fits")
    if not os.path.exists(spec_obj_path):
        raise FileNotFoundError(f"Missing {spec_obj_path}")

    catalog = Table.read(spec_obj_path)
    catalog = catalog[selection_fn(catalog)]
    catalog["ra"] = catalog["PLUG_RA"]
    catalog["dec"] = catalog["PLUG_DEC"]
    catalog["object_id"] = catalog["SPECOBJID"]

    if ra_center is not None and dec_center is not None and radius is not None:
        cone_mask = apply_cone_filter(
            np.asarray(catalog["ra"]),
            np.asarray(catalog["dec"]),
            ra_center, dec_center, radius,
        )
        catalog = catalog[cone_mask]
        if len(catalog) == 0:
            return []

    grouped = catalog.group_by(["SURVEY", "PLATE"])
    groups: list[tuple[str, Table]] = []
    for sub in grouped.groups:
        survey_str = _as_str(sub["SURVEY"][0]).strip()
        plate = int(sub["PLATE"][0])
        mjd = int(sub["MJD"][0])
        plate_path = os.path.join(
            raw_root, survey_str, f"{plate:04d}", f"spPlate-{plate:04d}-{mjd}.fits"
        )
        if not os.path.exists(plate_path):
            continue  # plate not mirrored on cluster, skip
        groups.append((plate_path, sub))
        if max_files is not None and len(groups) >= max_files:
            break
    return groups


def process_plate(args) -> Table:
    """Process one plate group: read spectra, join with catalog, return astropy Table."""
    plate_path, sub_catalog = args
    spectra_dict = _read_plate((plate_path, np.asarray(sub_catalog["FIBERID"]),
                                np.asarray(sub_catalog["object_id"])))
    spectra_table = Table(spectra_dict)
    joined = join(sub_catalog, spectra_table, keys="object_id", join_type="inner")
    return joined


def _process_plate_to_table(group):
    """Module-level pool worker: process one plate group into a (plate_path, pa.Table)
    tuple, or None if the plate produced no spectra. Module-level (not nested)
    so it's picklable for multiprocessing.Pool.
    """
    plate_path, _ = group
    joined = process_plate(group)
    if len(joined) == 0:
        return None
    return plate_path, _build_arrow_table(joined)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root of the raw SDSS data on disk (must contain specObj-dr17.fits and per-survey plate dirs).",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
        help="Where to write the HATS catalog (collection root).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit to the first N plate files (for fast smoke tests).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Worker processes for plate I/O.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8192,
        help="Max rows per HATS partition.",
    )
    parser.add_argument("--scratch-dir", default=None,
                        help="Scratch dir for per-plate parquet files (default: ceph "
                             "MultimodalUniverse_v2_hats_scratch/sdss_<pid>).")
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    print(f"Reading specObj from {args.raw_root}")
    groups = find_plate_groups(
        args.raw_root,
        max_files=args.max_files,
        ra_center=args.ra_center,
        dec_center=args.dec_center,
        radius=args.radius,
    )
    if not groups:
        print(f"ERROR: no plate FITS files found under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Processing {len(groups)} plate(s)")

    scratch = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch, exist_ok=True)
    print(f"Streaming per-plate parquet to {scratch}")

    # Stream: process each plate, write a single-plate parquet shard, free.
    # Unlike the v1 HDF5 path, we do NOT pad spectra to a global max length —
    # each shard's spectrum column is a variable-length pa.list_<float32> and
    # parquet/HATS merges variable lengths across shards natively. Peak RAM
    # is bounded by one plate (~500 fibers × 4612 wavelengths × 5 arrays ~
    # ~45 MB) rather than the full DR17 (~250 GB after padding).
    try:
        n_written = 0
        total = 0

        if args.num_processes > 1 and len(groups) > 1:
            with Pool(args.num_processes) as pool:
                iter_results = pool.imap(_process_plate_to_table, groups)
                for i, result in enumerate(tqdm(iter_results, total=len(groups)), 1):
                    if result is None:
                        continue
                    plate_path, table = result
                    out_name = os.path.basename(plate_path).replace(".fits", ".parquet")
                    pq.write_table(table, os.path.join(scratch, f"part-{out_name}"))
                    n_written += 1
                    total += table.num_rows
                    del table
        else:
            for i, group in enumerate(tqdm(groups), 1):
                result = _process_plate_to_table(group)
                if result is None:
                    continue
                plate_path, table = result
                out_name = os.path.basename(plate_path).replace(".fits", ".parquet")
                pq.write_table(table, os.path.join(scratch, f"part-{out_name}"))
                n_written += 1
                total += table.num_rows
                del table

        if n_written == 0:
            print("ERROR: no plates produced spectra", file=sys.stderr)
            return 1

        print(f"\nIngesting {n_written} plate(s), {total} spectra → HATS catalog")
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
