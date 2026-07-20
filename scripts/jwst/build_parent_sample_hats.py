"""Convert raw JWST DJA mosaics into a single HATS catalog.

This builder uses the mirrored original-data layout under::

    /mnt/ceph/users/polymathic/external_data/astro/JWST/
        <mosaic-name>/
            <mosaic>-fix_phot_apcorr.fits
            <mosaic>-<filter>-clear_drc_sci.fits.gz
            <mosaic>-<filter>-clear_drc_wht_full.fits.gz  # preferred
            <mosaic>-<filter>-clear_drc_wht.fits.gz       # fallback
            <mosaic>-<filter>-clear_drc_exp.fits.gz       # fallback

It preserves the current JWST image schema:
    image: struct<
        band: list<string>,
        flux: list<list<list<float32>>>,
        ivar: list<list<list<float32>>>,
        mask: list<list<list<bool>>>,
        psf_fwhm: list<float32>,
        scale: list<float32>,
    >
plus the scalar morphology/photometry features and top-level ra/dec.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import default_scratch_dir, to_native_endian, write_hats_from_parquet_dir


CATALOG_NAME = "jwst"

IMAGE_SIZE = 96
MOSAIC_FILTERS = ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
FLOAT_FEATURES = [
    "mag_auto",
    "flux_radius",
    "flux_auto",
    "fluxerr_auto",
    "cxx_image",
    "cyy_image",
    "cxy_image",
]

MOSAIC_MAG_AUTO_CUT = {
    "primer": 27.0,
    "ceers": 27.0,
    "ngdeep": 27.5,
    "gds": 27.5,
    "gdn": 27.5,
}
MIN_FILTERS_CUT = 4

EMPIRICAL_PSF_FWHM = {
    "f090w": 0.033,
    "f115w": 0.040,
    "f150w": 0.050,
    "f200w": 0.066,
    "f277w": 0.092,
    "f356w": 0.116,
    "f444w": 0.145,
}


def _mosaic_key(mosaic_name: str) -> str:
    return mosaic_name.split("-")[0]


def _survey_id(mosaic_name: str) -> str:
    if "primer-cosmos" in mosaic_name:
        return "primer-cosmos"
    if "primer-uds" in mosaic_name:
        return "primer-uds"
    return _mosaic_key(mosaic_name)


def _catalog_path(mosaic_dir: str, mosaic_name: str) -> str:
    return os.path.join(mosaic_dir, f"{mosaic_name}-fix_phot_apcorr.fits")


def _sci_path(mosaic_dir: str, mosaic_name: str, filt: str) -> str:
    return os.path.join(mosaic_dir, f"{mosaic_name}-{filt}-clear_drc_sci.fits.gz")


def _wht_full_path(mosaic_dir: str, mosaic_name: str, filt: str) -> str:
    return os.path.join(mosaic_dir, f"{mosaic_name}-{filt}-clear_drc_wht_full.fits.gz")


def _wht_path(mosaic_dir: str, mosaic_name: str, filt: str) -> str:
    return os.path.join(mosaic_dir, f"{mosaic_name}-{filt}-clear_drc_wht.fits.gz")


def _exp_path(mosaic_dir: str, mosaic_name: str, filt: str) -> str:
    return os.path.join(mosaic_dir, f"{mosaic_name}-{filt}-clear_drc_exp.fits.gz")


def find_mosaic_dirs(raw_root: str, max_files: int | None = None) -> list[tuple[str, str]]:
    pattern = re.compile(r"(.+)-fix_phot_apcorr\.fits$")
    mosaics = []
    for path in sorted(glob.glob(os.path.join(raw_root, "*", "*-fix_phot_apcorr.fits"))):
        name = os.path.basename(path)
        m = pattern.match(name)
        if not m:
            continue
        mosaic_name = m.group(1)
        mosaics.append((mosaic_name, os.path.dirname(path)))
    if max_files is not None:
        mosaics = mosaics[:max_files]
    return mosaics


def selection_function(catalog: Table, mag_cut: float, min_filters: int = MIN_FILTERS_CUT) -> np.ndarray:
    filters = [f for f in MOSAIC_FILTERS if f"{f}_flux_aper_0" in catalog.colnames]
    if not filters:
        return np.zeros(len(catalog), dtype=bool)
    non_zero = np.zeros(len(catalog), dtype=np.int32)
    for filt in filters:
        non_zero += (np.asarray(catalog[f"{filt}_flux_aper_0"]) > 0).astype(np.int32)
    return (non_zero >= min_filters) & (np.asarray(catalog["mag_auto"]) < mag_cut)


def build_total_inverse_variance(mosaic_name: str, filt: str, mosaic_dir: str) -> str:
    out_path = _wht_full_path(mosaic_dir, mosaic_name, filt)
    if os.path.exists(out_path):
        return out_path

    with fits.open(_sci_path(mosaic_dir, mosaic_name, filt)) as sci_hdu, \
         fits.open(_wht_path(mosaic_dir, mosaic_name, filt)) as wht_hdu, \
         fits.open(_exp_path(mosaic_dir, mosaic_name, filt)) as exp_hdu:
        sci = to_native_endian(np.asarray(sci_hdu[0].data, dtype=np.float32))
        wht = to_native_endian(np.asarray(wht_hdu[0].data, dtype=np.float32))
        exp = to_native_endian(np.asarray(exp_hdu[0].data, dtype=np.float32))
        header = exp_hdu[0].header

        full_exp = np.zeros_like(sci, dtype=np.float32)
        full_exp[2::4, 2::4] += exp
        full_exp = np.maximum(full_exp, np.roll(full_exp, 1, axis=0))
        full_exp = np.maximum(full_exp, np.roll(full_exp, -1, axis=0))
        full_exp = np.maximum(full_exp, np.roll(full_exp, 1, axis=1))
        full_exp = np.maximum(full_exp, np.roll(full_exp, -1, axis=1))

        phot_scale = 1.0
        for key in ["PHOTMJSR", "PHOTSCAL"]:
            if key in header and header[key] != 0:
                phot_scale /= header[key]
        if "OPHOTFNU" in header and "PHOTFNU" in header and header["OPHOTFNU"] != 0:
            phot_scale *= header["PHOTFNU"] / header["OPHOTFNU"]

        effective_gain = phot_scale * full_exp
        var_poisson_dn = np.zeros_like(sci, dtype=np.float32)
        positive_gain = effective_gain > 0
        var_poisson_dn[positive_gain] = np.maximum(sci[positive_gain], 0) / effective_gain[positive_gain]

        var_wht = np.zeros_like(wht, dtype=np.float32)
        positive_wht = wht > 0
        var_wht[positive_wht] = 1.0 / wht[positive_wht]
        var_total = var_wht + var_poisson_dn

        full_wht = np.zeros_like(var_total, dtype=np.float32)
        positive_total = var_total > 0
        full_wht[positive_total] = 1.0 / var_total[positive_total]

        fits.PrimaryHDU(data=full_wht, header=wht_hdu[0].header).writeto(out_path, overwrite=True)
    return out_path


def process_mosaic(mosaic_name: str, mosaic_dir: str, pixel_threshold: int, scratch_dir: str, ra_center: float | None, dec_center: float | None, radius: float | None) -> int:
    os.makedirs(scratch_dir, exist_ok=True)
    catalog = Table.read(_catalog_path(mosaic_dir, mosaic_name))
    mag_cut = MOSAIC_MAG_AUTO_CUT[_mosaic_key(mosaic_name)]
    sel = selection_function(catalog, mag_cut=mag_cut, min_filters=MIN_FILTERS_CUT)
    catalog = catalog[sel]
    if len(catalog) == 0:
        return 0

    if ra_center is not None and dec_center is not None and radius is not None:
        cone = apply_cone_filter(
            np.asarray(catalog["ra"], dtype=np.float64),
            np.asarray(catalog["dec"], dtype=np.float64),
            ra_center,
            dec_center,
            radius,
        )
        catalog = catalog[cone]
        if len(catalog) == 0:
            return 0

    available_filters = [
        filt for filt in MOSAIC_FILTERS
        if os.path.exists(_sci_path(mosaic_dir, mosaic_name, filt))
    ]
    if not available_filters:
        return 0

    images = {}
    for filt in available_filters:
        sci_hdu = fits.open(_sci_path(mosaic_dir, mosaic_name, filt))
        ivar_path = _wht_full_path(mosaic_dir, mosaic_name, filt)
        ivar_mode = "full"
        if not os.path.exists(ivar_path):
            if os.path.exists(_wht_path(mosaic_dir, mosaic_name, filt)) and os.path.exists(_exp_path(mosaic_dir, mosaic_name, filt)):
                ivar_path = build_total_inverse_variance(mosaic_name, filt, mosaic_dir)
                ivar_mode = "full"
            else:
                ivar_mode = "synthetic"
        ivar_hdu = fits.open(ivar_path) if ivar_mode == "full" else None
        wcs = WCS(sci_hdu[0].header)
        pix_scale = float(round(np.sqrt(np.linalg.det(np.abs(wcs.pixel_scale_matrix))) * 3600, 4))
        images[filt] = {
            "sci": sci_hdu[0],
            "ivar": ivar_hdu[0] if ivar_hdu is not None else None,
            "ivar_mode": ivar_mode,
            "wcs": wcs,
            "pix_scale": pix_scale,
        }

    if not images:
        return 0

    survey_hash = zlib.crc32(mosaic_name.encode())
    records = []
    n_cat = len(catalog)
    for ri, row in enumerate(catalog):
        ra = float(row["ra"])
        dec = float(row["dec"])
        flux_stack = []
        ivar_stack = []
        mask_stack = []
        bands = []
        psf = []
        scales = []

        for filt, img in images.items():
            x, y = img["wcs"].all_world2pix(ra, dec, 0)
            cutout_flux = Cutout2D(
                to_native_endian(np.asarray(img["sci"].data, dtype=np.float32)),
                (x, y),
                (IMAGE_SIZE, IMAGE_SIZE),
                wcs=img["wcs"],
                mode="partial",
                fill_value=0,
            ).data
            if img["ivar"] is not None:
                cutout_ivar = Cutout2D(
                    to_native_endian(np.asarray(img["ivar"].data, dtype=np.float32)),
                    (x, y),
                    (IMAGE_SIZE, IMAGE_SIZE),
                    wcs=img["wcs"],
                    mode="partial",
                    fill_value=0,
                ).data
            else:
                cutout_ivar = np.ones_like(cutout_flux, dtype=np.float32)
            cutout_flux = np.nan_to_num(cutout_flux).astype(np.float32)
            cutout_ivar = np.nan_to_num(cutout_ivar).astype(np.float32)
            flux_stack.append(cutout_flux)
            ivar_stack.append(cutout_ivar)
            mask_stack.append((cutout_ivar > 0).astype(bool))
            bands.append(filt)
            psf.append(np.float32(EMPIRICAL_PSF_FWHM[filt]))
            scales.append(np.float32(img["pix_scale"]))

        if not flux_stack:
            continue

        record = {
            "object_id": str(int(row["id"]) + survey_hash),
            "ra": np.float64(ra),
            "dec": np.float64(dec),
            "image_band": bands,
            "image_flux": np.stack(flux_stack, axis=0).astype(np.float32),
            "image_ivar": np.stack(ivar_stack, axis=0).astype(np.float32),
            "image_mask": np.stack(mask_stack, axis=0).astype(bool),
            "image_psf_fwhm": np.array(psf, dtype=np.float32),
            "image_scale": np.array(scales, dtype=np.float32),
        }
        for feat in FLOAT_FEATURES:
            record[feat] = np.float32(row[feat])
        records.append(record)
        if (ri + 1) % 1000 == 0 or ri == n_cat - 1:
            print(f"  {mosaic_name}: [{ri+1}/{n_cat}] {len(records)} cutouts",
                  flush=True)

    for img in images.values():
        img["sci"]._file.close()
        if img["ivar"] is not None:
            img["ivar"]._file.close()

    if not records:
        return 0

    band_arr = pa.array([r["image_band"] for r in records], type=pa.list_(pa.string()))
    flux_arr = pa.array(
        [[[list(row) for row in band] for band in r["image_flux"]] for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    ivar_arr = pa.array(
        [[[list(row) for row in band] for band in r["image_ivar"]] for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.float32()))),
    )
    mask_arr = pa.array(
        [[[list(map(bool, row)) for row in band] for band in r["image_mask"]] for r in records],
        type=pa.large_list(pa.large_list(pa.large_list(pa.bool_()))),
    )
    psf_arr = pa.array([r["image_psf_fwhm"].tolist() for r in records], type=pa.list_(pa.float32()))
    scale_arr = pa.array([r["image_scale"].tolist() for r in records], type=pa.list_(pa.float32()))
    def _as_array(a):
        return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a

    image_col = pa.StructArray.from_arrays(
        [_as_array(a) for a in [band_arr, flux_arr, ivar_arr, mask_arr, psf_arr, scale_arr]],
        names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"],
    )
    columns = {
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "ra": pa.array(np.array([r["ra"] for r in records], dtype=np.float64), type=pa.float64()),
        "dec": pa.array(np.array([r["dec"] for r in records], dtype=np.float64), type=pa.float64()),
        "image": image_col,
    }
    for feat in FLOAT_FEATURES:
        columns[feat] = pa.array(np.array([r[feat] for r in records], dtype=np.float32), type=pa.float32())
    table = pa.table(columns)

    shard_path = os.path.join(scratch_dir, f"{mosaic_name}.parquet")
    pq.write_table(table, shard_path)
    return table.num_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--scratch-dir", default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Cap number of mosaics.")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    args = parser.parse_args(argv)

    mosaics = find_mosaic_dirs(args.raw_root, max_files=args.max_files)
    if not mosaics:
        print(f"ERROR: no JWST raw mosaics found under {args.raw_root}", file=sys.stderr)
        return 1

    scratch_dir = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch_dir, exist_ok=True)

    total_rows = 0
    for mosaic_name, mosaic_dir in mosaics:
        total_rows += process_mosaic(
            mosaic_name,
            mosaic_dir,
            pixel_threshold=args.pixel_threshold,
            scratch_dir=scratch_dir,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )

    if total_rows == 0:
        print("ERROR: no JWST rows processed", file=sys.stderr)
        return 1

    catalog_dir = write_hats_from_parquet_dir(
        scratch_dir,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=True,
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
