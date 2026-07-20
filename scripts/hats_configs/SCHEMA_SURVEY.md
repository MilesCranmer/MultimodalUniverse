# MMU Dataset Schema Survey

Survey of all dataset HF builders in `scripts/` (skipping `manga`). Each entry summarizes
modality, HDF5 location pattern on the cluster, key features, and quirks.

Cluster root: `/mnt/ceph/users/polymathic/MultimodalUniverse/{dataset}/{config}/healpix=N/001-of-001.hdf5`

## Datasets by modality

### Spectra (flux/ivar/lambda/lsf_sigma/mask struct)
- **sdss** — configs: all, sdss, segue1, segue2, boss, eboss; multi-band flux features; ZWARNING bool
- **desi** — config: dr1_main; ZWARN bool needs inversion (0=good)
- **vipers** — configs: vipers_w1, vipers_w4, all
- **lamost** — 8 sub-configs (lrs_catalogue, lrs_stellar, ...); huge feature lists for QSO config
- **galah** — config: dr3; spectrum has extra norm_flux/norm_ivar/norm_lambda; ~150 float features
- **apogee** — config: apogee; spectrum has extra pseudo_continuum; ~24 float features

### Spectra-like (special)
- **chandra** — X-ray spectra (ene_low/center/high_bin, flux, flux_error)
- **gaia** — spectral_coefficients (50 floats) + nested structs (photometry, astrometry, radial_velocity, gspphot, flags, corrections)
- **desi_provabgs** — MCMC posteriors: PROVABGS_MCMC (100x13 array), PROVABGS_THETA_BF

### Images (image_array or image struct with bands)
- **legacysurvey** — config: dr10_south_21; image_array (N, 4, 160, 160) DES-G/R/I/Z + blobmodel/rgb/object_mask
- **ssl_legacysurvey** — configs: stein_et_al, stein_et_al_north; 152x152, 3 bands G/R/Z
- **hsc** — config: pdr3_dud_22.5; 160x160, 5 bands G/R/I/Z/Y
- **jwst** — 6 configs (primer-cosmos, primer-uds, ceers, ngdeep, gds, gdn); variable image sizes per config
- **btsbot** — splits: train/val/test; 63x63 images, 3 views per band (sci/ref/diff)
- **gz10** — configs: gz10, gz10_rgb_images; 256x256 RGB

### Time series (band/time/flux/flux_err)
- **plasticc** — configs: plasticc, train_only, test_only
- **tess** — 6 configs (spoc, qlp, tglc, plus -tiny); **uses RA/DEC uppercase**; pipeline-specific extra columns
- **kepler** — config: data; lightcurve has time/pdcsap_flux/pdcsap_flux_err/sap_flux/sap_flux_err
- **cfa** — configs: cfa3, cfa4, cfa_SECCSN, cfa_snII; mag/mag_err format
- **foundation** — config: foundation_dr1
- **snls** — config: data
- **ps1_sne_ia**, **des_y3_sne_ia**, **swift_sne_ia**, **csp** — supernova lightcurves
- **yse** — config: yse_dr1

### Tabular (catalogs, no spectrum/image/lightcurve)
- **allwise** — config: allwise; ~341 photometric fields
- **twomass** — config: psc; **uses `decl` not `dec`**; 103 fields
- **galex** — config: ais; ~153 fields
- **sages** — config: dr1; ~11 fields

## Quirks to handle

| Quirk | Datasets |
|---|---|
| RA/Dec uppercase | `tess` |
| `decl` instead of `dec` | `twomass` |
| Bool inverted (0=good) | `desi` (`ZWARN`) |
| Special groupings (struct of structs) | `gaia` (photometry/astrometry/...) |
| 4D arrays (image cubes) | `legacysurvey`, `hsc`, `jwst` (image_array shape varies) |
| MCMC posteriors / 2D arrays | `desi_provabgs` |
| Multi-band flux features | `sdss` (flux × filter) |
| Multi-pipeline configs | `tess`, `lamost` |
