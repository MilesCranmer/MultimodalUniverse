# Direct MaNGA -> HATS validation

This note summarizes the current status of the direct raw MaNGA -> HATS path in
`scripts/manga/build_parent_sample_hats.py`.

## Goal

Build MaNGA directly from the native DR17 products into HATS, without going
through the MMU v1 grouped HDF5 intermediate, while staying aligned with MMU's
logical MaNGA schema.

## Design choice

The direct builder keeps the MMU MaNGA field semantics:

- `object_id`, `ra`, `dec`, `healpix`, `z`
- `spaxel_size`, `spaxel_size_units`
- `spaxels`
- `images`
- `maps`

The on-disk HATS encoding is intentionally different from the MMU v1 HDF5
physical layout:

- native MaNGA spatial footprints are preserved per object
- `spaxels.flux/ivar/mask/lsf` are stored as `(y, x, lambda)` cubes
- `images.flux/psf` are stored as `(band, y, x)`
- `maps.flux/ivar/mask` are stored as `(map, y, x)`
- small parquet shards are written incrementally before `hats-import`

This keeps the MMU science content while avoiding the memory blow-up from
materializing the entire catalog or forcing every target onto the padded
`96x96` v1 canvas.

## Robustness fixes added during validation

- FITS header lookup now accepts both `C1`/`C01` and `U1`/`U01` numbering
  conventions.
- Shared `mmu.hats_import` now provides a parquet-shard path for large surveys.
- Shared `mmu.hats_import` patches `np.unique(..., sorted=...)` compatibility so
  the current `hats` stack works under NumPy `1.26`.
- Debug HATS writes now use an in-process Dask client so local smoke tests and
  monkeypatch-based compatibility fixes behave the same way.

## Validation fixture

The strongest local validation currently available is a 4-object fixture derived
from the existing local MMU v1 MaNGA sample:

- `8726-1901` (`32x32`)
- `11945-1902` (`32x32`)
- `11945-6104` (`54x54`)
- `8323-12704` (`72x72`)

The fixture contains:

- synthetic `drpall-v3_1_1.fits`
- synthetic `dapall-v3_1_1-3.1.0.fits`
- 4 `LOGCUBE` FITS files
- 4 `MAPS-HYB10-MILESHC-MASTARSSP` FITS files

This is a semantic validation fixture, not a substitute for a future run on a
genuine raw DR17 subset.

## Exact parity result

The validator at `scripts/manga/validate_parent_sample_hats.py` adapts the
native-shape HATS rows back onto the padded MMU v1 `96x96` view and compares all
major payloads field-by-field:

- top-level metadata
- `spaxels`
- `images`
- `maps`

Validated result:

```text
11945-1902: parity OK
11945-6104: parity OK
8323-12704: parity OK
8726-1901: parity OK
```

That means the direct HATS builder reproduces the existing MMU v1 MaNGA science
content exactly on this sample after applying the compatibility adapter.

## Storage result on the 4-object sample

Measured local sizes:

| Representation | Size |
|---|---:|
| synthetic raw FITS fixture | `429M` |
| direct native-shape HATS output | `400M` |
| existing padded MMU v1 HDF5 sample | `3.5G` |

The HATS output is slightly smaller than the raw fixture and dramatically
smaller than the padded MMU v1 HDF5 representation for the same 4 galaxies.

## End-to-end benchmark

Two direct-build runs were benchmarked end-to-end, including the final
`hats-import` phase:

| Config | Real time | Max RSS | Peak memory footprint |
|---|---:|---:|---:|
| `rows-per-shard=1` | `76.29s` | `6.13 GB` | `4.50 GB` |
| `rows-per-shard=2` | `71.62s` | `6.17 GB` | `4.56 GB` |

On this sample, `rows-per-shard=2` is slightly faster with essentially the same
memory footprint. Both variants passed exact parity validation.

## Reproduction commands

Build:

```bash
PYTHONPATH=/Users/vassig/research/polymathic/forks/MultimodalUniverse \
  /Users/vassig/research/polymathic/mangahats/.venv/bin/python \
  /Users/vassig/research/polymathic/forks/MultimodalUniverse/scripts/manga/build_parent_sample_hats.py \
  --raw-root /Users/vassig/research/polymathic/mangahats/artifacts/manga_raw_fixture_from_v1b \
  --output-root /Users/vassig/research/polymathic/mangahats/artifacts/manga_hats_from_v1_fixture_validated \
  --rows-per-shard 1 \
  --debug
```

Validate:

```bash
PYTHONPATH=/Users/vassig/research/polymathic/forks/MultimodalUniverse \
  /Users/vassig/research/polymathic/mangahats/.venv/bin/python \
  /Users/vassig/research/polymathic/forks/MultimodalUniverse/scripts/manga/validate_parent_sample_hats.py \
  --v1-root /Users/vassig/research/polymathic/mangahats/data/MultimodalUniverse/v1/manga/manga \
  --hats-root /Users/vassig/research/polymathic/mangahats/artifacts/manga_hats_from_v1_fixture_validated/manga/manga
```

## Recommended next step

The direct builder is locally de-risked enough to continue. The next real
checkpoint should be a run on a genuine raw MaNGA DR17 subset, not a synthetic
fixture, followed by the same parity and performance checks.
