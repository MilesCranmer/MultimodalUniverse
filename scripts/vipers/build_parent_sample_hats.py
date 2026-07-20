"""Convert raw VIPERS 1D spectra FITS files into a single HATS catalog.

The cluster mirrors VIPERS PDR2 spectra at:

    /mnt/ceph/users/polymathic/external_data/astro/VIPERS/
        VIPERS_W1_SPECTRA_1D_PDR2/   # *.fits per-object spectra
        VIPERS_W4_SPECTRA_1D_PDR2/   # *.fits per-object spectra

Each per-object FITS file carries header keys (ID, RA, DEC, REDSHIFT, REDFLAG,
EXPTIME, NORM, MAG) and a binary table extension with columns FLUXES, WAVES,
NOISE, MASK.

This script:
1. Globs all *.fits files under the W1 and W4 subdirectories.
2. Reads each file, extracts the header scalars and spectrum arrays.
3. Builds one PyArrow table with a struct-of-lists spectrum column.
4. Hands the table to ``write_hats`` to produce a HATS catalog under
   ``{output_root}/vipers/vipers/``.

Usage:
    python -m scripts.vipers.build_parent_sample_hats \\
        --raw-root /mnt/ceph/users/polymathic/external_data/astro/VIPERS \\
        --output-root /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/vipers \\
        --max-files 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pyarrow as pa


def _as_array(arr):
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr

from astropy.io import fits
from tqdm import tqdm

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import np_to_pyarrow_list, write_hats


CATALOG_NAME = "vipers"

SURVEY_SUBDIRS = [
    "VIPERS_W1_SPECTRA_1D_PDR2",
    "VIPERS_W4_SPECTRA_1D_PDR2",
]

HEADER_KEYS = ["ID", "RA", "DEC", "REDSHIFT", "REDFLAG", "EXPTIME", "NORM", "MAG"]


def extract_spectrum(path: str) -> dict | None:
    """Read one VIPERS per-object FITS file.

    Returns a dict with scalar header values and 1D spectrum arrays, or None
    if the file cannot be read.
    """
    try:
        with fits.open(path) as hdu:
            header = hdu[1].header
            data = hdu[1].data
            result: dict = {}
            for key in HEADER_KEYS:
                result[key] = float(header[key])
            result["spectrum_flux"] = np.asarray(data["FLUXES"]).reshape(-1).astype(np.float32)
            result["spectrum_wave"] = np.asarray(data["WAVES"]).reshape(-1).astype(np.float32)
            result["spectrum_noise"] = np.asarray(data["NOISE"]).reshape(-1).astype(np.float32)
            result["spectrum_mask"] = np.asarray(data["MASK"]).reshape(-1).astype(np.float32)
        return result
    except Exception as exc:
        print(f"WARNING: skipping {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def find_fits_files(
    raw_root: str,
    max_files: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> list[str]:
    """Return sorted list of VIPERS FITS file paths under ``raw_root``.

    If a cone cut is specified, files are still all returned here; the cone
    cut is applied after reading (since RA/DEC are inside each file).
    The ``max_files`` limit is applied to the file list before reading.
    """
    files: list[str] = []
    for subdir in SURVEY_SUBDIRS:
        pattern = os.path.join(raw_root, subdir, "*.fits")
        files.extend(sorted(glob.glob(pattern)))
    if max_files is not None:
        files = files[:max_files]
    return files


def build_arrow_table(records: list[dict]) -> pa.Table:
    """Convert a list of per-object record dicts into a PyArrow table."""
    n = len(records)
    ra = np.array([r["RA"] for r in records], dtype=np.float64)
    dec = np.array([r["DEC"] for r in records], dtype=np.float64)

    flux_arrays = [r["spectrum_flux"] for r in records]
    wave_arrays = [r["spectrum_wave"] for r in records]
    noise_arrays = [r["spectrum_noise"] for r in records]
    mask_arrays = [r["spectrum_mask"] for r in records]

    spectrum = pa.StructArray.from_arrays(
            [_as_array(a) for a in [
            pa.array(flux_arrays, type=pa.list_(pa.float32())),
            pa.array(wave_arrays, type=pa.list_(pa.float32())),
            pa.array(noise_arrays, type=pa.list_(pa.float32())),
            pa.array(mask_arrays, type=pa.list_(pa.float32())),
        ]],
        names=["flux", "wave", "noise", "mask"],
    )

    columns: dict[str, pa.Array] = {
        "object_id": pa.array([str(int(r["ID"])) for r in records], type=pa.string()),
        "ra": pa.array(ra),
        "dec": pa.array(dec),
        "spectrum": spectrum,
        "REDSHIFT": pa.array(np.array([r["REDSHIFT"] for r in records], dtype=np.float32)),
        "REDFLAG": pa.array(np.array([r["REDFLAG"] for r in records], dtype=np.float32)),
        "EXPTIME": pa.array(np.array([r["EXPTIME"] for r in records], dtype=np.float32)),
        "NORM": pa.array(np.array([r["NORM"] for r in records], dtype=np.float32)),
        "MAG": pa.array(np.array([r["MAG"] for r in records], dtype=np.float32)),
    }

    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root of the raw VIPERS data on disk.",
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
        help="Limit to the first N FITS files (for fast smoke tests).",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Worker processes for FITS I/O.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8192,
        help="Max rows per HATS partition.",
    )
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    files = find_fits_files(
        args.raw_root,
        max_files=args.max_files,
    )
    if not files:
        print(f"ERROR: no FITS files found under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} FITS file(s)")

    if args.num_processes > 1 and len(files) > 1:
        with Pool(args.num_processes) as pool:
            records_raw = list(tqdm(pool.imap(extract_spectrum, files), total=len(files)))
    else:
        records_raw = [extract_spectrum(f) for f in tqdm(files)]

    records = [r for r in records_raw if r is not None]
    if not records:
        print("ERROR: no valid spectra extracted", file=sys.stderr)
        return 1

    if args.ra_center is not None and args.dec_center is not None and args.radius is not None:
        ra = np.array([r["RA"] for r in records])
        dec = np.array([r["DEC"] for r in records])
        mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
        records = [r for r, m in zip(records, mask) if m]
        if not records:
            print("ERROR: no spectra survived the cone cut", file=sys.stderr)
            return 1
        print(f"{len(records)} spectra survive cone cut")

    print(f"Building Arrow table from {len(records)} spectra")
    table = build_arrow_table(records)

    print(f"Writing HATS catalog to {args.output_root}")
    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
