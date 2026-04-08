# How HATS stores MMU data

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

## 2. How each modality lands in the Parquet schema

All rows in a HATS parquet are flat in terms of **columns**, but the columns can hold **nested types** (list, struct, list-of-list, list-of-extension-type). That's how we get spectra and image cubes into a "tabular" format without losing shape.

### 2.1 Modality → schema table

| Modality | Dataset(s) | Column name(s) | PyArrow type | Notes |
|---|---|---|---|---|
| **Tabular flat** | allwise, 2mass, galex, sages, desi_provabgs | one column per field | `float32` / `int64` / `string` / `bool` | Plain columns. Hundreds of them is fine — Parquet only reads the ones you ask for. |
| **Scalar photometry (per band)** | sdss, 2mass, desi | `SPECTROFLUX_U`, `SPECTROFLUX_G`, `J_m`, `H_m`, `FLUX_G`, … | `float32` | Multi-band fluxes are **unrolled into one column per band** so filtering `WHERE mag_r < 18` hits one column only. |
| **Spectrum (1D per object)** | sdss, desi, apogee, galah, vipers, chandra | single struct `spectrum` | `struct< flux: list<f32>, ivar: list<f32>, lambda: list<f32>, lsf_sigma: list<f32>, mask: list<bool> >` | Each row has five parallel variable-length lists. Lengths can differ per row (Parquet supports it natively). |
| **Time series** | tess, kepler (todo) | single struct `lightcurve` | `struct< time: list<f64>, flux: list<f32>, flux_err: list<f32>, quality: list<i32> >` | Variable-length lists — each TIC/sector has its true cadence count. No padding, no fake zeros. |
| **Image cube (multi-band per object)** | ssl_legacysurvey | `image: fixed_shape_tensor(f32, (n_bands, H, W))` + `psf_fwhm: fixed_shape_tensor(f32, (n_bands,))` | PyArrow `FixedShapeTensorType` | Each row holds one `(n_bands, H, W)` tensor directly. Shape lives in the parquet schema metadata; `chunk.to_numpy_ndarray()` gives back a real `(N, n_bands, H, W)` numpy array. This is the **native PyArrow** extension type and is hats-import-safe. |
| **Image cube (alternative tried)** | (abandoned) | `image: struct< flux: list<Array2DExtensionType> >` | nested HF `Array2DExtensionType` | Attempted first (matches Mike's HSC transformer). **Crashes hats-import's finishing step on nested extension types** — don't use for new ports. |
| **Image cube (fallback)** | (not currently used) | `image_array: list<f32>` + sibling `image_array_shape: list<i32>` | flat list + shape | What the deleted auto-converter used. Loses schema-level shape info; reader has to reshape on read. |
| **IFU / complex grouped** | manga (stub only) | scalars + compound columns skipped | mixed | MaNGA raw is grouped-by-object HDF5 with compound spaxel/image/map structs. Currently we only extract the per-object metadata (ra, dec, z, spaxel_size). Full spaxel cube support is todo. |

### 2.2 Concrete schema fragment from a real SDSS row

![SDSS schema](diagrams/sdss_schema.svg)

34 columns total. All partitions in the catalog share this schema via `_common_metadata`.

### 2.3 Why nested types instead of separate tables

The obvious alternative would be a separate "spectra" table keyed on `object_id`. HATS could support that in principle, but:

- **One file per pixel stays intact.** Reading a pixel gives you the complete observation of every object in it (metadata + spectrum + photometry) with one I/O. A separate spectra table would mean a second join for every query.
- **Column projection still works.** Parquet can skip the `spectrum` struct entirely when a query only needs `ra`/`dec`/`Z`. The cost of having it in the same file is near zero until you ask for it.
- **hats-import's partitioning key is ra/dec.** Anything keyed on `object_id` alone wouldn't get spatial partitioning — it'd be one flat table, which is the format LSDB/hats is trying to replace.

### 2.4 Design rules

Applied consistently to every `scripts/{dataset}/build_parent_sample_hats.py`:

1. `ra` / `dec` → top-level `float64`.
2. `object_id` → top-level `string`, whatever the source integer/designation looks like.
3. Scalar metadata → top-level columns with native dtypes.
4. Multi-band scalars → **unrolled** into `FIELD_BAND` columns (enables per-band Parquet pushdown filters).
5. Per-object array data (spectrum, lightcurve) → a single `struct` column containing parallel `list<T>` fields. Variable-length allowed.
6. Per-object image cubes → `pa.fixed_shape_tensor(float32, (n_bands, H, W))` as a top-level `image` column. Per-band scalars that logically belong to the cube (e.g. `psf_fwhm`) go in a matching `fixed_shape_tensor(float32, (n_bands,))` column. **Do not** nest `Array2DExtensionType` inside a list/struct — it crashes hats-import's finishing stage.
7. Everything else that happens to be 1-D numeric per row → pass through as a top-level list column.
8. Bytes from FITS → decoded to Python `str` before going to Arrow. No `b'…'` in the catalog.

---

**TL;DR**: HATS is a directory of Parquet files, one per HEALPix pixel, plus a sibling margin-cache directory. Parquet is columnar so you only pay for the columns you read. Modalities fit by using nested PyArrow types — `struct` for grouped-related fields, variable-length `list` for spectra and lightcurves, `list< Array2DExtensionType >` for per-band image cutouts.
