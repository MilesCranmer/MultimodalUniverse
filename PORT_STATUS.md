# MMU v2 HATS — Port Status

Live tracker for the raw → HATS port of v1 MMU. Every non-skipped dataset in `mmu/hats_configs.py` gets a `scripts/<dataset>/build_parent_sample_hats.py` that matches v1's HuggingFace `Features(...)` schema 1:1, with HATS output instead of HDF5.

**Test slice:** COSMOS field — RA=150°, Dec=+2°, radius=0.5° (1° cone). Every build script accepts `--ra-center/--dec-center/--radius` to run the test slice in seconds instead of the hours a full build takes. The `cosmos` Snakemake profile sets these automatically.

**Schema rule:** Struct-of-parallel-lists with plain pyarrow types. NO HF `datasets` extension types inside `list<>` or `struct<>` (see `project_image_storage` memory for why — nested_pandas crashes on them in hats-import's finishing stage).

**Legend:** `[x]` done · `[ ]` pending · `[~]` in progress · `[skip]` out of scope

---

## Ported baseline

- [x] **sdss** (spectra) — `scripts/sdss/build_parent_sample_hats.py`
  - [x] Spectrum struct<flux, ivar, lsf_sigma, lambda, mask> + photometry scalars
  - [ ] Retrofit: `--ra-center/--dec-center/--radius` args
  - [ ] COSMOS slice validation on cluster
- [x] **desi** (spectra) — uses `desispec.coadd_cameras`, matches v1 exactly
  - [x] Spectrum struct<flux, ivar, lsf_sigma, lambda, mask> + Z/ZERR/ZWARN + photometry
  - [ ] Retrofit: cone-cut args (filter `zall-pix-iron.fits` before grouping)
  - [ ] COSMOS slice validation
- [x] **ssl_legacysurvey** (image) — struct-of-parallel-lists (reference for all image datasets)
  - [x] `image: struct<band, flux (3,152,152), psf_fwhm, scale>` + photometry
  - [ ] Retrofit: cone-cut at h5 chunk level
  - [ ] COSMOS slice validation
- [x] **tess** (timeseries) — variable-length lightcurves
  - [x] `lightcurve: struct<time, flux, flux_err>` per (TIC, sector)
  - [ ] Retrofit: filter TIC catalog by RA/Dec
  - [ ] COSMOS slice validation
- [x] **allwise** (tabular) — 298 cols from IRSA parquet
  - [x] Retrofit: cone filter + healpix-k5 prefilter (skips 99.98% of shards)
  - [x] COSMOS slice validation — 11,277 rows across 2 k5 pixels ✓
- [x] **twomass** (tabular) — 2MASS PSC pipe-delimited CSV
  - [ ] Retrofit: cone filter
  - [ ] COSMOS slice validation
- [x] **galex** (tabular) — GUVCat FITS shards
  - [ ] Retrofit: cone filter
  - [ ] COSMOS slice validation
- [x] **sages** (tabular) — u/v photometry FITS table
  - [ ] Retrofit: cone filter
  - [ ] COSMOS slice validation

## Phase 0 — Infrastructure

- [x] `mmu/cone.py` — shared cone-cut helper (haversine + bbox)
- [x] `tests/test_cone.py` — 17 tests covering wraparound, poles, edge precision, vectorization
- [x] Retrofit 8 ported scripts with cone args
- [x] `snakemake_config.yaml` — `cosmos` profile (RA=150, Dec=+2, radius=0.5)
- [x] Snakefile — threaded `ra_center`/`dec_center`/`radius` through `build_command()`
- [x] `PORT_STATUS.md` — this file, created
- [~] Run all 8 ported datasets against COSMOS profile on cluster (in progress)
  - [x] allwise ✓ 11,277 rows
  - [~] twomass (in progress)
  - [ ] galex
  - [ ] sages
  - [ ] sdss
  - [ ] tess
  - [ ] desi
  - [ ] ssl_legacysurvey

## Phase 1 — Flagship datasets

- [ ] **gaia** — BUNDLED: XP spectra + photometry + astrometry + RV + stellar params
  - [ ] Raw layout inspection
  - [ ] v1 schema review
  - [ ] `build_parent_sample_hats.py`
  - [ ] Unit tests + local smoke
  - [ ] COSMOS cluster validation
- [ ] **legacysurvey** — BUNDLED: multi-band images + RGB + masks + catalog
  - [ ] Raw layout inspection
  - [ ] v1 schema review
  - [ ] `build_parent_sample_hats.py`
  - [ ] Unit tests + local smoke
  - [ ] COSMOS cluster validation
- [ ] **hsc** — BUNDLED: 5-band images + 50+ scalars
  - [ ] Raw layout inspection
  - [ ] v1 schema review
  - [ ] `build_parent_sample_hats.py`
  - [ ] Unit tests + local smoke
  - [ ] COSMOS cluster validation
- [skip] **kepler** — raw Kepler FITS not on cluster mirror (only v1 HDF5 exists)
  - Moved to skip list; requires raw MAST data mirror before port is possible
- [ ] **manga** — BUNDLED, MOST COMPLEX: IFU cubes + spaxel coords + griz reconstructions
  - [ ] Raw layout inspection
  - [ ] v1 schema review
  - [ ] `build_parent_sample_hats.py`
  - [ ] Unit tests + local smoke
  - [ ] COSMOS cluster validation

## Phase 2 — Remaining datasets

### Spectra (reuse sdss/desi pattern)
- [ ] **vipers** — VIMOS Public Extragalactic Redshift Survey
- [ ] **galah** — GALactic Archaeology with HERMES
- [ ] **apogee** — Near-IR stellar spectra with pseudo_continuum
- [ ] **chandra** — X-ray spectra (ene_low/high/center bins)

### Images (reuse ssl_legacysurvey pattern)
- [ ] **jwst** — BUNDLED: NIRCam multi-band + metadata
- [ ] **btsbot** — ZTF BTS image triplets
- [ ] **gz10** — Galaxy Zoo 10 single RGB + classification label

### SN time series (reuse tess variable-length pattern)
- [ ] **foundation** — Foundation DR1 SNe Ia
- [ ] **snls** — Supernova Legacy Survey
- [ ] **ps1_sne_ia** — Pan-STARRS1 SNe Ia
- [ ] **des_y3_sne_ia** — DES Y3 SNe Ia
- [ ] **swift_sne_ia** — Swift UV/optical SNe

### Tabular
- [ ] **desi_provabgs** — Stellar parameters + MCMC posteriors (100×13)

## Skipped (out of scope)

- [skip] **plasticc** — raw dir empty on cluster (`hats_configs.skip=True`)
- [skip] **kepler** — raw FITS not mirrored on cluster; only v1-processed HDF5 exists at `spoc/SPOC/`
- [skip] **cfa** — in v1 but deliberately excluded
- [skip] **csp** — in v1 but deliberately excluded
- [skip] **lamost** — in v1 but deliberately excluded
- [skip] **yse** — in v1 but deliberately excluded

---

## Totals

| Category | Count |
|---|---|
| Ported baseline (cone args added) | 8 |
| Phase 1 flagship (to port) | 4 |
| Phase 2 fill-in (to port) | 13 |
| Skipped | 6 |
| **v1 datasets total** | **31** |
| **To ship in MMU v2 HATS** | **25** |

## Verification criteria (per dataset)

1. `uv run pytest tests/test_<dataset>_build.py -v` → all pass
2. `uv run pytest tests/` → full suite green
3. `ssh worker5018 '...build_parent_sample_hats --ra-center=150 --dec-center=2 --radius=0.5'` → exits 0
4. Read-back parquet from COSMOS output matches v1 HF schema field-for-field
5. `lsdb.read_hats(<cosmos_output>).compute().head()` → returns rows without exceptions
6. This file's checkboxes ticked
7. Commit pushed to `feat/hats`
