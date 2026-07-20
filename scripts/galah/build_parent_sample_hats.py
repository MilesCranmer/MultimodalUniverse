"""Convert raw GALAH DR3 spectra into a HATS catalog.

This builder reads the mirrored GALAH DR3 raw inputs under::

    /mnt/ceph/users/polymathic/external_data/astro/galah/dr3/

using:
    - GALAH_DR3_main_allspec_v2.fits
    - GALAH_DR3_VAC_ages_v2.fits
    - resolution_maps/ccd{1..4}_piv.fits
    - the spectra tarball (contains galah/dr3/spectra/hermes/{sobject_id}{1..4}.fits)

It preserves the existing HF-facing GALAH schema while adding top-level
``ra``/``dec`` for HATS partitioning.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tarfile
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from scipy.optimize import curve_fit

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, to_native_endian, write_hats_from_parquet_dir


def _load_dataset_contract():
    path = os.path.join(os.path.dirname(__file__), "galah.py")
    spec = importlib.util.spec_from_file_location("_galah_dataset_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._FLOAT_FEATURES, module._INT_FEATURES


_FLOAT_FEATURES, _INT_FEATURES = _load_dataset_contract()


CATALOG_NAME = "galah"
BANDS = ["BLUE", "GREEN", "RED", "NIR"]
SPECTRUM_KEYS = [
    "flux",
    "ivar",
    "lsf",
    "lsf_sigma",
    "lambda",
    "norm_flux",
    "norm_ivar",
    "norm_lambda",
]

VAC_KEY_MAP = {
    "log_lum": "log_lum_bstep",
    "m_act": "m_act_bstep",
    "age": "age_bstep",
    "distance": "distance_bstep",
    "radius": "radius_bstep",
    "e_log_lum": "e_log_lum_bstep",
    "e_m_act": "e_m_act_bstep",
    "e_age": "e_age_bstep",
    "e_distance": "e_distance_bstep",
    "e_radius": "e_radius_bstep",
}

CATALOG_KEY_MAP = {
    **{k: k for k in _FLOAT_FEATURES if k not in VAC_KEY_MAP},
    **{k: k for k in _INT_FEATURES},
    "ra": "ra_dr2",
    "dec": "dec_dr2",
    "rv": "rv_galah",
    "e_rv": "e_rv_galah",
}


def _dr3_root(raw_root: str) -> str:
    if os.path.basename(os.path.normpath(raw_root)) == "dr3":
        return raw_root
    return os.path.join(raw_root, "dr3")


def _allspec_path(raw_root: str) -> str:
    return os.path.join(_dr3_root(raw_root), "GALAH_DR3_main_allspec_v2.fits")


def _vac_path(raw_root: str) -> str:
    return os.path.join(_dr3_root(raw_root), "GALAH_DR3_VAC_ages_v2.fits")


def _missing_ids_path(raw_root: str) -> str:
    return os.path.join(_dr3_root(raw_root), "GALAH_DR3_list_missing_reduced_spectra_v2.csv")


def _resolution_map_dir(raw_root: str) -> str:
    return os.path.join(_dr3_root(raw_root), "resolution_maps")


def _tar_path(raw_root: str) -> str:
    matches = sorted(
        p for p in os.listdir(_dr3_root(raw_root)) if p.endswith(".tar.gz")
    )
    if not matches:
        raise FileNotFoundError(f"No GALAH spectra tarball under {_dr3_root(raw_root)}")
    return os.path.join(_dr3_root(raw_root), matches[0])


def ang2pix(ra: np.ndarray, dec: np.ndarray, nside: int) -> np.ndarray:
    import healpy as hp  # noqa: PLC0415

    return hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True, nest=True)


def get_resolution(ccd_resolution_map_filename: str) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(ccd_resolution_map_filename) as hdul:
        resolution = to_native_endian(np.asarray(hdul[0].data, dtype=np.float32))

    mean_resolution = np.mean(resolution, axis=0)

    def _gauss(x, a, x0, sigma):
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    try:
        popt, _ = curve_fit(
            _gauss, np.arange(len(mean_resolution)), mean_resolution, p0=[1, 5, 1]
        )
        sigma = np.float32(popt[2])
    except Exception:
        sigma = np.float32(-1)

    return mean_resolution.astype(np.float32), np.full(len(mean_resolution), sigma, dtype=np.float32)


def _process_band_fits(fileobj) -> dict[str, np.ndarray]:
    with fits.open(fileobj) as hdu:
        flux = to_native_endian(np.asarray(hdu[0].data, dtype=np.float32))
        sigma = to_native_endian(np.asarray(hdu[1].data, dtype=np.float32))
        norm_flux = to_native_endian(np.asarray(hdu[4].data, dtype=np.float32))

        start_wavelength = np.float32(hdu[0].header["CRVAL1"])
        dispersion = np.float32(hdu[0].header["CDELT1"])
        nr_pixels = int(hdu[0].header["NAXIS1"])
        reference_pixel = int(hdu[0].header["CRPIX1"] or 1)

        norm_start_wavelength = np.float32(hdu[4].header["CRVAL1"])
        norm_dispersion = np.float32(hdu[4].header["CDELT1"])
        norm_nr_pixels = int(hdu[4].header["NAXIS1"])
        norm_reference_pixel = int(hdu[4].header["CRPIX1"] or 1)
        timestamp = np.float32(hdu[0].header["UTMJD"])

    ivar = np.zeros_like(sigma, dtype=np.float32)
    valid_sigma = sigma != 0
    ivar[valid_sigma] = 1.0 / np.square(flux[valid_sigma] * sigma[valid_sigma])

    norm_ivar = np.zeros_like(sigma, dtype=np.float32)
    valid_norm = sigma != 0
    norm_ivar[valid_norm] = 1.0 / np.square(norm_flux[valid_norm] * sigma[valid_norm])

    return {
        "flux": flux,
        "lambda": (np.arange(nr_pixels, dtype=np.float32) + reference_pixel - 1) * dispersion + start_wavelength,
        "ivar": ivar,
        "norm_flux": norm_flux,
        "norm_lambda": (np.arange(norm_nr_pixels, dtype=np.float32) + norm_reference_pixel - 1) * norm_dispersion + norm_start_wavelength,
        "norm_ivar": norm_ivar,
        "timestamp": timestamp,
    }


def load_catalogs(raw_root: str) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(_allspec_path(raw_root)) as hdul:
        catalog = np.array(hdul[1].data.copy())
    with fits.open(_vac_path(raw_root)) as hdul:
        vac = np.array(hdul[1].data.copy())
    return catalog, vac


def selection_mask(catalog: np.ndarray, raw_root: str) -> np.ndarray:
    if os.path.exists(_missing_ids_path(raw_root)):
        missing_ids = np.loadtxt(_missing_ids_path(raw_root), skiprows=1, delimiter=",")[:, 0].astype(int)
        return ~np.isin(catalog["sobject_id"], missing_ids)
    return np.ones(len(catalog), dtype=bool)


def prepare_rows(raw_root: str, nside: int, max_files: int | None, ra_center: float | None, dec_center: float | None, radius: float | None) -> list[dict]:
    catalog, vac = load_catalogs(raw_root)
    catalog = catalog[selection_mask(catalog, raw_root)]

    if max_files is not None:
        catalog = catalog[:max_files]

    _, idx_catalog, idx_vac = np.intersect1d(
        catalog["sobject_id"],
        vac["sobject_id"],
        assume_unique=True,
        return_indices=True,
    )
    catalog = catalog[idx_catalog]
    vac = vac[idx_vac]

    ra = np.asarray(catalog["ra_dr2"], dtype=np.float64)
    dec = np.asarray(catalog["dec_dr2"], dtype=np.float64)
    if ra_center is not None and dec_center is not None and radius is not None:
        cone_mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
        catalog = catalog[cone_mask]
        vac = vac[cone_mask]
        ra = ra[cone_mask]
        dec = dec[cone_mask]

    healpix = ang2pix(ra, dec, nside=nside)

    resolution_maps = {
        band: get_resolution(os.path.join(_resolution_map_dir(raw_root), f"ccd{ccd}_piv.fits"))
        for band, ccd in zip(BANDS, [1, 2, 3, 4])
    }
    tar_path = _tar_path(raw_root)

    rows = []
    for i in range(len(catalog)):
        rows.append({
            "catalog": catalog[i],
            "vac": vac[i],
            "healpix": int(healpix[i]),
            "tar_path": tar_path,
            "extract_dir": None,  # set by main() after extraction
            "resolution_maps": resolution_maps,
        })
    return rows


def _process_object(row: dict) -> dict | None:
    sobject_id = int(row["catalog"]["sobject_id"])
    extract_dir = row["extract_dir"]
    spectra = {}

    try:
        for band, suffix in zip(BANDS, [1, 2, 3, 4]):
            fits_path = os.path.join(
                extract_dir, "galah", "dr3", "spectra", "hermes",
                f"{sobject_id}{suffix}.fits",
            )
            with open(fits_path, "rb") as fh:
                spectra[band] = _process_band_fits(fh)
            spectra[band]["lsf"], spectra[band]["lsf_sigma"] = row["resolution_maps"][band]
    except (KeyError, OSError, FileNotFoundError, ValueError) as exc:
        print(f"WARNING: skipping GALAH {sobject_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None

    lengths = {band: len(spectra[band]["lambda"]) for band in BANDS}
    offsets = {
        "B_start": 0,
        "B_end": lengths["BLUE"] - 1,
        "G_start": lengths["BLUE"],
        "G_end": lengths["BLUE"] + lengths["GREEN"] - 1,
        "R_start": lengths["BLUE"] + lengths["GREEN"],
        "R_end": lengths["BLUE"] + lengths["GREEN"] + lengths["RED"] - 1,
        "I_start": lengths["BLUE"] + lengths["GREEN"] + lengths["RED"],
        "I_end": lengths["BLUE"] + lengths["GREEN"] + lengths["RED"] + lengths["NIR"] - 1,
    }

    record = {
        "object_id": str(sobject_id),
        "flux_unit": "erg/cm^2/A/s",
        "filter_indices": {k: np.int32(v) for k, v in offsets.items()},
        "ra": np.float64(row["catalog"]["ra_dr2"]),
        "dec": np.float64(row["catalog"]["dec_dr2"]),
    }

    for key in SPECTRUM_KEYS:
        record[key] = np.concatenate([to_native_endian(np.asarray(spectra[band][key], dtype=np.float32)) for band in BANDS]).astype(np.float32)

    record["timestamp"] = np.float32(np.mean([spectra[band]["timestamp"] for band in BANDS]))

    for out_name, src_name in VAC_KEY_MAP.items():
        record[out_name] = np.float32(row["vac"][src_name])
    for out_name, src_name in CATALOG_KEY_MAP.items():
        if out_name in ("timestamp", "healpix"):
            continue
        value = row["catalog"][src_name]
        if out_name in _INT_FEATURES:
            record[out_name] = np.int32(value)
        else:
            record[out_name] = np.float32(value)
    record["healpix"] = np.int32(row["healpix"])
    return record


def build_arrow_table(records: list[dict]) -> pa.Table:
    spectrum = pa.StructArray.from_arrays(
        arrays=[pa.array([r[k].tolist() for r in records], type=pa.list_(pa.float32())) for k in SPECTRUM_KEYS],
        names=SPECTRUM_KEYS,
    )
    filter_indices = pa.StructArray.from_arrays(
        arrays=[
            pa.array(np.array([r["filter_indices"][name] for r in records], dtype=np.int32), type=pa.int32())
            for name in ["B_start", "B_end", "G_start", "G_end", "R_start", "R_end", "I_start", "I_end"]
        ],
        names=["B_start", "B_end", "G_start", "G_end", "R_start", "R_end", "I_start", "I_end"],
    )

    columns: dict[str, pa.Array] = {
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "flux_unit": pa.array([r["flux_unit"] for r in records], type=pa.string()),
        "ra": pa.array(np.array([r["ra"] for r in records], dtype=np.float64), type=pa.float64()),
        "dec": pa.array(np.array([r["dec"] for r in records], dtype=np.float64), type=pa.float64()),
        "spectrum": spectrum,
        "filter_indices": filter_indices,
    }
    for name in _FLOAT_FEATURES:
        if name in {"ra", "dec"}:
            continue
        columns[name] = pa.array(np.array([r[name] for r in records], dtype=np.float32), type=pa.float32())
    for name in _INT_FEATURES:
        columns[name] = pa.array(np.array([r[name] for r in records], dtype=np.int32), type=pa.int32())
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pixel-threshold", type=int, default=100_000)
    parser.add_argument("--extract-dir", default="/dev/shm/galah_spectra",
                        help="Directory to extract the spectra tarball into. "
                             "Defaults to /dev/shm for fast RAM-backed I/O. "
                             "Cleaned up on exit.")
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args(argv)

    rows = prepare_rows(
        args.raw_root,
        nside=args.nside,
        max_files=args.max_files,
        ra_center=args.ra_center,
        dec_center=args.dec_center,
        radius=args.radius,
    )
    if not rows:
        print("ERROR: no GALAH rows selected", file=sys.stderr)
        return 1

    # Extract the spectra tarball to a fast local directory (default /dev/shm).
    # The tarball is 229 GB compressed with ~1.3M individual FITS files.
    # Opening it per-star via tarfile.open("r:gz") decompresses the whole
    # archive each time — completely unusable. Extracting once to RAM-backed
    # /dev/shm makes individual file reads instant.
    tar_path = rows[0]["tar_path"]
    extract_dir = args.extract_dir
    os.makedirs(extract_dir, exist_ok=True)
    spectra_check = os.path.join(extract_dir, "galah", "dr3", "spectra", "hermes")
    if not os.path.isdir(spectra_check):
        print(f"Extracting {tar_path} → {extract_dir} ...", flush=True)
        import tarfile as _tf
        with _tf.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        if not os.path.isdir(spectra_check):
            print(f"ERROR: extraction produced no files at {spectra_check}",
                  file=sys.stderr, flush=True)
            return 1
        n_extracted = len(os.listdir(spectra_check))
        print(f"Extracted {n_extracted} spectra files.", flush=True)
        print(f"Extraction complete.", flush=True)
    else:
        print(f"Using existing extraction at {extract_dir}", flush=True)

    # Patch all rows with the extract dir
    for row in rows:
        row["extract_dir"] = extract_dir

    scratch_dir = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch_dir, exist_ok=True)

    work = rows
    shard_idx = 0
    pending: list[dict] = []

    n_total = len(work)
    n_done = 0

    def _process(record):
        nonlocal pending, shard_idx, n_done
        n_done += 1
        if n_done % 500 == 0 or n_done == n_total:
            sys.stderr.write(f"  [{n_done}/{n_total}] {len(pending)} pending, "
                           f"{shard_idx} shards written\n")
            sys.stderr.flush()
        if record is None:
            return
        pending.append(record)
        if len(pending) >= args.batch_size:
            out = os.path.join(scratch_dir, f"part-{shard_idx:04d}.parquet")
            tmp = out + ".tmp"
            pq.write_table(build_arrow_table(pending), tmp)
            os.rename(tmp, out)
            shard_idx += 1
            pending = []

    if args.num_processes > 1:
        with Pool(args.num_processes) as pool:
            for record in pool.imap_unordered(_process_object, work):
                _process(record)
    else:
        for row in work:
            _process(_process_object(row))

    if pending:
        pq.write_table(build_arrow_table(pending), os.path.join(scratch_dir, f"part-{shard_idx:04d}.parquet"))
        shard_idx += 1

    if shard_idx == 0:
        print("ERROR: no GALAH spectra processed successfully", file=sys.stderr)
        return 1

    catalog_dir = write_hats_from_parquet_dir(
        scratch_dir,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=False,
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    # Clean up the extracted spectra from /dev/shm (or wherever --extract-dir pointed)
    if os.path.isdir(extract_dir):
        print(f"Cleaning up {extract_dir}...", flush=True)
        shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
