# Raw survey data layout on the Flatiron cluster

All raw inputs for MMU surveys are mirrored under
`/mnt/ceph/users/polymathic/external_data/astro/{survey}/`.
**No downloads are needed** for any of the surveys we plan to port to HATS.

This is the input that the new Snakefile per-survey rules should consume,
replacing the old `MultimodalUniverse/{survey}/{config}/healpix=N/*.hdf5` v1 inputs.

## Per-survey raw layout

| Survey | Path under `external_data/astro/` | Format | Notes |
|---|---|---|---|
| **AllWISE** | `allwise/healpix_k0=N/healpix_k5=M/part0.snappy.parquet` | Parquet (snappy) | Already healpix-partitioned (k=0 inside k=5). 341 columns. **Easiest port — pyarrow native.** |
| AllWISE (alt) | `allwise_parquet/healpix=N/...` | Parquet | Different layout, also pre-partitioned. |
| **2MASS** | `twomass/psc/...` | (need to inspect) | |
| **GALEX** | `galex/GUVCat_AIS_FOV055_glat*.fits.gz` | FITS gzipped | Sliced by galactic latitude. |
| **SAGES** | `sages/dr1s-gri.fits`, `dr1-uv.fits` | FITS | Two big tables. |
| **SDSS** | `SDSS/{sdss,boss,eboss,segue1,segue2}/...` | (FITS plate files) | Already partitioned by sub-survey. |
| **DESI** | `DESI/coadd-sv3-*.fits`, `DESI_DR1/...` | FITS coadd files | One file per (survey, program, healpix). |
| **Gaia** | `Gaia/AstrophysicalParameters_*.hdf5`, `XpContinuousMeanSpectrum_*.hdf5` | HDF5 | Pre-staged Gaia DR3 hdf5s; not the raw VOtable. |
| **GALAH** | `galah/dr3/...` | (need to inspect) | |
| **APOGEE** | `APOGEE/{apogee,apogee_v2}/...` | (FITS) | DR17. |
| **Chandra** | `chandra/cdapackage.0.*/...` | tarballs / FITS | One sub-dir per CDA package release. |
| **HSC** | `hsc/{pdr3_dud,pdr3_wide}/...` | (FITS image cutouts) | |
| **LegacySurvey** | `legacysurvey/{dr10,dr9}/...` | brick FITS | |
| **JWST** | `JWST/{primer-cosmos,...}/...` | grizli reductions | |
| **MaNGA** | `manga/dr17/`, `drpall-v3_1_1.fits`, `dapall-*.fits` | FITS | DR17 release files. |
| **PLAsTiCC** | `PLAsTiCC/` | (empty as of 2026-04-07) | Need to download. |
| **TESS** | `tess/hlsp_tess-spoc_tess_phot_*.fits` | FITS lightcurves | One per (TIC, sector). |
| **Foundation** | `foundation/Foundation_DR1_*.txt` | snana .txt files | |
| **2MASS / SAGES / GALEX** | (small flat files) | FITS | All small enough to load whole. |

## What this means for the Snakefile

The new pipeline replaces the old `(dataset, config, healpix)` triple with
`(dataset, config)` only. Per-dataset rules read directly from the paths above
and let `hats-import` do the healpix partitioning.

For the simplest cases (AllWISE, 2MASS, Galex), the source is already a flat
parquet/FITS table and conversion is essentially:

```python
# 1. Read the raw table(s)
table = pa.parquet.read_table(raw_path)  # or astropy.table.Table.read(...)

# 2. Normalize ra/dec/object_id column names
# 3. Hand to write_hats() — hats-import does the healpix partitioning
write_hats([arrow_table], output_path, catalog_name)
```

For more complex cases (SDSS, DESI, MaNGA, etc.) the existing
`scripts/{dataset}/build_parent_sample.py` already does the per-row catalog
joins and FITS munging. We port those to call `write_hats` instead of `h5py`.
