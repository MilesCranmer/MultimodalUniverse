# How HATS stores MMU data

*Status: 18 datasets ported and tested as of 2026-04-08. See `PORT_STATUS.md` at the repo root for the live tracker.*

## 1. What HATS actually is (on disk)

HATS = **HEALPix-partitioned Parquet, with margin caches and metadata sidecars**. That's it. No custom binary format. Any Parquet reader can read a HATS catalog; the "HATS-ness" is the *directory layout* and the *metadata files* around the Parquet.

### 1.1 Directory layout

![Directory layout](diagrams/directory_layout.svg)

### 1.2 Parquet in 30 seconds

![Row-oriented vs columnar](diagrams/parquet_columnar.svg)

Each column is stored contiguously, compressed independently (snappy/zstd), split into **row groups** (typically ~1 M rows each). This is why `pq.read_table(f, columns=["ra", "dec"])` only reads a handful of bytes even when the file has 300 columns.

### 1.3 What HATS adds on top of Parquet

| Feature | How |
|---|---|
| **Spatial partitioning** | Each `.parquet` file = one HEALPix pixel. Filename encodes `Norder` (refinement level) and `Npix` (pixel index). |
| **Adaptive mesh** | Dense sky regions get subdivided to higher `Norder` (smaller pixels, more files); sparse regions stay coarse. Visible in `Norder=4/` through `Norder=10/` in the tree above. |
| **Fast spatial joins** | A hidden `_healpix_29` column on every row holds the pixel index at the deepest level. Sorts cross-matches to near-linear time. |
| **Margin cache** | A sibling "10arcs" catalog duplicates every object within 10″ of a pixel boundary into the neighbouring pixel's file. Makes cross-match radius searches correct across edges. |
| **Metadata sidecars** | `_metadata` lets a reader plan queries (min/max per row group, counts) without opening any data files. `partition_info.csv` is the same info in human-readable form. |

### 1.4 Cross-match flow (why margins exist)

![Margin cache](diagrams/margin_cache.svg)

---

## 2. The one schema rule

> **Plain pyarrow types only. No HF `datasets` extension types — anywhere.**

We tried three image-storage strategies before settling on this rule:

| Attempt | Schema | Outcome |
|---|---|---|
| 1 | `pa.fixed_shape_tensor(f32, (n_bands, H, W))` (PyArrow native) | hats-import's finishing stage crashed inside `nested_pandas` on the extension type. |
| 2 | `struct< flux: list<Array2DExtensionType> >` (HF nested ext, matched Mike's HSC transformer) | Same crash. nested_pandas can't traverse extension types nested inside `list<>`/`struct<>`. |
| 3 | `struct< flux: list<list<list<f32>>> >` (struct-of-parallel-lists, plain pyarrow) | **Works.** This is what every MMU v2 image dataset uses now. |

The cost is that shape information lives in code, not in the Arrow schema — readers reshape on load. Worth it: every modality from spectra through IFU cubes uses the same primitive (`pa.list_(pa.list_(...))`), and hats-import never crashes.

---

## 3. How each modality lands in the Parquet schema

All rows in a HATS parquet are flat in terms of **columns**, but the columns hold **structs of parallel lists**. That's how spectra, lightcurves, image cubes, and IFU cubes all sit inside the same "tabular" file format.

### 3.1 Modality → schema table

| Modality | Datasets (ported) | Top-level column(s) | PyArrow type | Notes |
|---|---|---|---|---|
| **Tabular flat** | allwise, twomass, galex, sages | one column per source field | `float32` / `int64` / `string` / `bool` | Plain columns. Hundreds of them is fine — Parquet only reads the ones you ask for. allwise has 298. |
| **Scalar photometry (per band)** | sdss, twomass, desi, gaia | `MAG_J`, `H_m`, `FLUX_G`, `phot_g_mean_mag`, … | `float32` | Multi-band fluxes are **unrolled into one column per band** so filtering `WHERE mag_r < 18` hits one column only. |
| **1D spectrum (per object)** | sdss, desi | single struct `spectrum` | `struct< flux: list<f32>, ivar: list<f32>, lambda: list<f32>, lsf_sigma: list<f32>, mask: list<bool> >` | Each row holds five parallel variable-length lists. Lengths can differ per row (Parquet supports it natively). SDSS uses this even though v1 padded to fixed width — variable-length is the v2 default. |
| **Spectral basis coefficients** | gaia | struct `spectral_coefficients` | `struct< coeff: list<f32>, coeff_error: list<f32> >` | Gaia XP continuous-mean spectra are 110 Hermite-style coefficients per object, not a sampled flux array. Same struct-of-lists pattern, just shorter. |
| **Photometric time series** | tess | single struct `lightcurve` | `struct< time: list<f64>, flux: list<f32>, flux_err: list<f32>, quality: list<i32> >` | One row per (TIC, sector). Variable-length lists — true cadence count, no padding. |
| **SN-Ia photometric time series** | foundation, snls, ps1_sne_ia, des_y3_sne_ia, swift_sne_ia | struct `lightcurve` with per-band parallel lists | `struct< time: list<f32>, band: list<string>, flux: list<f32>, flux_err: list<f32>, … >` | Built by the shared `mmu/sn_ia_snana.py` helper. SNANA ASCII inputs across all 5 surveys hit the same Arrow shape. |
| **Multi-band image cube (per object)** | ssl_legacysurvey, legacysurvey | struct `image` (+ `rgb`, `blobmodel`, `object_mask` for legacysurvey) | `struct< band: list<string>, flux: list<list<list<f32>>>, psf_fwhm: list<f32>, scale: list<f32> >` | Per row, `flux` is `(n_bands, H, W)` as nested plain lists. ssl is grz at 152². legacysurvey is grz at 96² with `rgb` added as `list<list<list<u8>>>` and `object_mask` as `list<list<u8>>`. |
| **IFU cubes + reconstructed images + DAP maps** | manga | three sibling struct columns: `spaxels`, `images`, `maps` | see §3.3 below | The most complex modality in the project. ~9216 spaxels × 4563 wavelength samples per cube, plus four griz reconstructions and ~50 DAP analysis maps, all in one row. |
| **Bundled catalog as nested struct** | legacysurvey | struct `catalog` | `struct< field_1: list<f32>, field_2: list<f32>, … >` | Nearby-object sweep table is folded in as a single struct of parallel lists per row, so each row's "neighbours within radius" travel with the image. |

**Key patterns:**
- **Spectra and lightcurves** are always a single `struct` with several parallel `list<T>` fields.
- **Image cubes** are always nested plain float lists — no extension types, no fixed-shape tensors. Shape is implicit and reapplied on read.
- **Bundled datasets** (gaia, legacysurvey, manga) glue several modalities together in one row using sibling top-level structs, so a single pixel read returns the complete observation per object.

### 3.2 Concrete schema fragment from a real SDSS row

![SDSS schema](diagrams/sdss_schema.svg)

34 columns total. All partitions in the catalog share this schema via `_common_metadata`.

### 3.3 The MaNGA IFU schema (the hardest modality)

MaNGA is the IFU dataset and ended up being the most complex port. Each plate-IFU contributes one row with three sibling struct columns:

**`spaxels: struct<…>`** — variable-length per row (each IFU has a different number of fibers, typically 1900–9216). Inner 2D arrays are spectra:

```
spaxels: struct<
  flux:   list<list<f32>>     # (n_spaxels, n_wavelengths)
  ivar:   list<list<f32>>
  mask:   list<list<i64>>
  lsf:    list<list<f32>>
  lambda: list<list<f32>>
  x, y:           list<i8>     # spaxel grid coordinates
  spaxel_idx:     list<i16>
  skycoo_x/y:     list<f32>    # WCS coords
  ellcoo_r/rre/rkpc/theta: list<f32>
  *_units: list<string>
>
```

**`images: struct<…>`** — the four griz reconstructed broad-band images at 96×96:

```
images: struct<
  filter:     list<string>          # ["g", "r", "i", "z"]
  flux:       list<list<list<f32>>> # (4, 96, 96)
  flux_units: list<string>
  psf:        list<list<list<f32>>> # (4, 96, 96)
  psf_units:  list<string>
  scale:      list<f32>
  scale_units: list<string>
>
```

**`maps: struct<…>`** — the ~50 MaNGA DAP analysis maps (Hα flux, velocity, σ, Dn4000, …) packed as parallel lists:

```
maps: struct<
  group:       list<string>
  label:       list<string>
  flux:        list<list<list<f32>>>  # (n_maps, H, W)
  ivar:        list<list<list<f32>>>
  mask:        list<list<list<f32>>>
  array_units: list<string>
>
```

This is a strict translation of v1 MMU's MaNGA HDF5 schema into pyarrow with no information loss, and it never touches extension types — so hats-import processes it without complaint.

### 3.4 Why nested types instead of separate tables

The obvious alternative would be a separate "spectra" table keyed on `object_id`. HATS could support that in principle, but:

- **One file per pixel stays intact.** Reading a pixel gives you the complete observation of every object in it (metadata + spectrum + photometry + image cube) with one I/O. A separate spectra table would mean a second join per query.
- **Column projection still works.** Parquet can skip the `spectrum` struct or `image` struct entirely when a query only needs `ra`/`dec`/`Z`. The cost of carrying them in the same file is near zero until you ask for them.
- **hats-import's partitioning key is ra/dec.** Anything keyed on `object_id` alone wouldn't get spatial partitioning — it'd be one flat table, exactly what LSDB/hats is trying to replace.

### 3.5 Design rules

Applied consistently to every `scripts/{dataset}/build_parent_sample_hats.py`:

1. `ra` / `dec` → top-level `float64`.
2. `object_id` → top-level `string` (or `int64` if the source designation is genuinely an integer, e.g. Gaia `source_id`).
3. Scalar metadata → top-level columns with native dtypes.
4. Multi-band scalars → **unrolled** into `FIELD_BAND` columns (enables per-band Parquet pushdown filters).
5. Per-object array data (spectrum, lightcurve) → a single `struct` column containing parallel `list<T>` fields. Variable-length allowed.
6. Per-object image cubes → a single `struct` column containing `list<list<list<f32>>>` for the cube and parallel scalar lists (`band`, `psf_fwhm`, `scale`) keyed by the same band index. **Plain pyarrow types only — no extension types anywhere.**
7. Bundled datasets (gaia, legacysurvey, manga) → sibling top-level struct columns, one per modality, all on the same row.
8. Bytes from FITS → decoded to Python `str` before going to Arrow. No `b'…'` in the catalog.
9. v1 HF `Features(...)` schemas are matched **field for field**, except where the v1 schema padded variable-length data into fixed-width 2D arrays — in v2 we use the natural variable-length list (recorded in memory as the SDSS/spectrum deviation, user-approved).

---

## 4. Build infrastructure

The schema is only half the story. Each dataset has a `build_parent_sample_hats.py` script that reads raw survey data from `/mnt/ceph/users/polymathic/external_data/astro/<survey>/`, applies an optional COSMOS cone cut, and writes a HATS catalog. Shared helpers live under `mmu/`:

| Module | Purpose |
|---|---|
| `mmu/cone.py` | Cone-cut filter (haversine + bbox prefilter) used by every script's `--ra-center/--dec-center/--radius` args. |
| `mmu/safety.py` | Output-path guardrails. Allowlist of write roots, denylist for `external_data/`, refuses bare polymathic root. 30 unit tests. |
| `mmu/hats_import.py` | `write_hats(table, …)` and `write_hats_from_parquet_dir(…)` wrappers. Defaults flipped to `debug=False, n_workers=32` (the hats-import single-core default was a 32× ingest bottleneck). `default_scratch_dir()` points at per-catalog ceph scratch. |
| `mmu/hats_configs.py` | Dataset registry (`DatasetConfig` per survey, raw paths, expected columns). |
| `mmu/sn_ia_snana.py` | Shared SNANA-ASCII build helper backing all 5 SN-Ia datasets. |

### 4.1 Streaming, sharding, scatter-gather

The interesting datasets don't fit in memory. Three patterns evolved:

1. **Streaming per-shard parquet.** The build script writes one parquet file per input shard (per plate, per FITS file, per HEALPix shard, …) into a scratch dir, then calls `write_hats_from_parquet_dir()` to convert the dir into a HATS catalog. Used by sdss, desi, twomass, galex, allwise, tess, manga. Avoids ever holding the full catalog in memory.
2. **`multiprocessing.Pool` scatter inside one job.** Each shard is processed by a worker; on Linux the catalog table is shared via fork-COW so workers don't re-read it. Pool sizes between 32 and 96 depending on the dataset.
3. **Scatter-gather across sbatch jobs.** For the largest datasets, N independent sbatch jobs each process a stride slice of the inputs and write per-unit parquet shards into a shared ceph scratch dir; one final gather job runs `write_hats_from_parquet_dir`. Controlled by `SHARDED_DATASETS` in the `Snakefile`. Currently active for legacysurvey (8 shards × `Pool(32)`); manga is wired and eligible.

### 4.2 Snakemake DAG

`Snakefile` exposes one `build_<dataset>` rule per port. Each rule lists its build script as an `input:` so editing the script auto-reruns the build, and a top-level `validate_hats_root` rule enforces the safety guardrails before any write. Profiles in `snakemake_config.yaml`:

- `cluster` — full all-sky production builds
- `cosmos` — 1° cone at (RA, Dec) = (150°, +2°) for fast end-to-end validation
- `test` — local pytest fixtures

`scripts/slurm/run_cosmos_slurm.sh` and `scripts/slurm/run_production_slurm.sh` invoke Snakemake with the slurm executor; per-rule resource budgets live in the rule definitions. Bespoke `/tmp/<dataset>_sbatch.sh` scripts on the cluster from earlier crash-recovery runs are being phased out in favor of pure Snakemake DAG execution via `SHARDED_DATASETS`.

---

## 5. Where we are (2026-04-08)

**18 datasets ported, tested, and matching v1 HF schemas.** Counts are from `PORT_STATUS.md`.

| Modality bucket | Ported | Datasets |
|---|---:|---|
| Tabular flat | 4 | allwise, twomass, galex, sages |
| 1D spectra | 2 | sdss, desi |
| Spectral coefficients | 1 | gaia (XP) |
| Photometric time series | 1 | tess |
| SN-Ia time series | 5 | foundation, snls, ps1_sne_ia, des_y3_sne_ia, swift_sne_ia |
| Multi-band image cubes | 2 | ssl_legacysurvey, legacysurvey |
| IFU + reconstructions + maps | 1 | manga |
| Bundled (image+catalog or spectra+photometry) | (3) | gaia, legacysurvey, manga (counted in their primary modality above) |

**Phase 2 remaining (8):**

- Spectra: vipers, galah, apogee, chandra
- Images: jwst (bundled), btsbot, gz10
- Tabular: desi_provabgs

**Skipped (out of scope):** plasticc (raw dir empty on cluster), kepler (raw FITS not mirrored), hsc (catalog not on cluster), cfa, csp, lamost, yse.

**v1 dataset total:** 31. **Coverage so far:** 18 / 24 in-scope = 75%.

### 5.1 Per-dataset verification criteria

Each ported dataset has to clear all of:

1. `uv run pytest tests/test_<dataset>_build.py -v` → green
2. `uv run pytest tests/` → full suite green
3. Cluster COSMOS slice (1° cone) produces a valid HATS catalog with `hats.properties`
4. Read-back parquet matches v1 HF schema field-for-field
5. `lsdb.read_hats(<output>).compute().head()` → returns rows without exceptions
6. Checkbox ticked in `PORT_STATUS.md`

---

**TL;DR**: HATS is a directory of Parquet files, one per HEALPix pixel, plus a sibling margin-cache directory. Parquet is columnar so you only pay for the columns you read. Every MMU v2 modality — tabular, spectra, lightcurves, image cubes, IFU cubes — uses the same primitive: a top-level `struct` of parallel plain-pyarrow `list<T>` fields, with no extension types anywhere. Bundled datasets stack several such structs as siblings on one row. 18 of 24 in-scope v1 datasets are ported, tested, and read back through `lsdb.read_hats()`.
