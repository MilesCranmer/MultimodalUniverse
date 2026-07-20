# MMU v2 HATS — Port Status

Live tracker for the raw → HATS port of v1 MMU. Every non-skipped dataset in `mmu/hats_configs.py` gets a `scripts/<dataset>/build_parent_sample_hats.py` that matches v1's HuggingFace `Features(...)` schema 1:1, with HATS output instead of HDF5.

**Test slice:** COSMOS field — RA=150°, Dec=+2°, radius=0.5° (1° cone). Every build script accepts `--ra-center/--dec-center/--radius`. The `cosmos` Snakemake profile sets these automatically.

**Schema rule:** Struct-of-parallel-lists with plain pyarrow types. NO HF `datasets` extension types inside `list<>` or `struct<>` (see `project_image_storage` memory — nested_pandas crashes on them in hats-import's finishing stage).

**No normalization:** Every catalog keeps its survey-native column names, units, and values untouched. No canonical aliases, no unit conversions, no cross-survey unification. Users who want a unified view build it on top.

**Native shape (no padding):** Image / IFU datasets store the survey's native spatial shape (e.g. MaNGA's per-IFU `(ny, nx, nwave)`) plus `spatial_shape_y/x` columns, NOT v1's 96×96 zero-padded fixed-size cubes. Padding is a destructive transformation that wastes 60–70% of cells; the native-shape choice came after we caught the v1 padding decision and recognized it as the same kind of synthetic normalization we ban elsewhere.

**Scatter/gather sharding:** Large datasets are built via scatter-gather — N independent sbatch jobs each process a stride slice of the inputs and write per-unit parquet shards to a shared ceph scratch dir, then one gather job runs `write_hats_from_parquet_dir`. Controlled by `SHARDED_DATASETS` in the Snakefile. All dependency tracking goes through Snakemake's DAG, no bespoke shell launchers. Scatter jobs run on `preempt` (bursts into idle rome nodes cluster-wide); gather jobs run on `ccm` (guaranteed, non-preemptible so a half-written HATS catalog is impossible). For one-off `--only-ingest` reruns of an already-scattered scratch dir, `scripts/gather_one_off.py` is a thin wrapper around `write_hats_from_parquet_dir` that we sbatch directly.

**Legend:** `[x]` done · `[ ]` pending · `[~]` in progress · `[skip]` out of scope

---

## Production catalogs landed on disk (`hats.properties` exists)

| # | Dataset | Modality | Notes |
|---|---|---|---|
| 1 | **sdss** | spectra | DR17 |
| 2 | **gaia_xp** | spectra | GaiaSource × XP join, ~220M rows, full astrometry + BP/RP + RV + gspphot |
| 3 | **galex** | tabular | GUVCat AIS |
| 4 | **sages** | tabular | DR1 u/v photometry |
| 5 | **apogee** | spectra | Near-IR stellar with pseudo_continuum |
| 6 | **chandra** | spectra | X-ray, ene_low/high/center bins |
| 7 | **galah** | spectra | GALactic Archaeology with HERMES |
| 8 | **tess** | timeseries | SPOC FFI lightcurves |
| 9 | **foundation** | timeseries | SN-Ia, SNANA ASCII via `mmu.sn_ia_snana` |
| 10 | **snls** | timeseries | SN-Ia, SNANA ASCII |
| 11 | **ps1_sne_ia** | timeseries | SN-Ia, SNANA ASCII |
| 12 | **des_y3_sne_ia** | timeseries | SN-Ia, SNANA ASCII |
| 13 | **swift_sne_ia** | timeseries | SN-Ia, SNANA ASCII |

**Total landed: 13 datasets.**

---

## In flight right now

| Dataset | Phase | Notes |
|---|---|---|
| **legacysurvey** | scatter (snakemake `legacy` tmux) | 128-shard preempt scatter via `build_legacysurvey`. Leak fix in place — was OOMing at 939 GB MaxRSS because per-worker `sweep_records` accumulated cutouts for an entire ~1000-brick sweep before flushing. New code flushes per-256-cutouts to `part-{sweep}-{batch:05d}.parquet` keeping per-worker peak ~100 MB. ~5200 batch parquets / 1.5 TB in scratch. |
| **ssl_legacysurvey** | scatter (snakemake `ssl` tmux) | Newly added sharded scatter/gather mode, 8 preempt shards, 77 raw chunks total. Was a single in-memory build before the leak forced sharding. |
| **manga** | gather (one-off sbatch) | Native-shape scatter complete (10735 plate-ifus × per-IFU `(ny, nx, nwave)` parquets, 1.7 TB). Gather running from `manga_sharded_v2` scratch. |
| **gaia (full DR3)** | gather (one-off sbatch) | 16-shard scatter complete (3386 GaiaSource HDF5 → 379 GB, all 16 done markers). Gather submitted as `6213559` after the snia parallel master was killed and migrated. |
| **desi** | gather (`6213303`) | Spectra scatter complete (32155 group parquets, 1.5 TB). Gather currently at ~24% Catalog: Splitting. |
| **twomass** | gather (`6213525`) | Tabular scatter complete (92 chunk parquets, 41 GB). Gather just resubmitted after fork-bug fix. |
| **allwise** | gather (`6213526`) | Tabular scatter complete (12288 healpix-k5 prefilter parquets, 342 GB). Gather just resubmitted. |

---

## Ported + tests passing, but no production build yet

(none right now — every script with green tests has either landed or is in flight above.)

---

## Phase 2 — datasets still to port

These have no `scripts/<dataset>/build_parent_sample_hats.py` yet. Six left.

### Spectra (reuse sdss/desi pattern)
- [ ] **vipers** — VIMOS Public Extragalactic Redshift Survey

### Images (reuse ssl_legacysurvey native-shape pattern)
- [ ] **hsc** — BUNDLED: 5-band images + 50+ photometric scalars
- [ ] **jwst** — BUNDLED: NIRCam multi-band + metadata
- [ ] **btsbot** — ZTF Bright Transient Survey image triplets
- [ ] **gz10** — Galaxy Zoo 10 single RGB + classification label

### Tabular
- [ ] **desi_provabgs** — Stellar parameters + MCMC posteriors (100×13 samples per object)

---

## Skipped (out of scope)

- [skip] **plasticc** — raw dir empty on cluster
- [skip] **kepler** — raw FITS not mirrored; only v1-processed HDF5 exists
- [skip] **cfa**, **csp**, **lamost**, **yse** — in v1 but deliberately excluded per user decision

---

## Infrastructure shipped

- `mmu/cone.py` — shared cone-cut helper (haversine + bbox prefilter, with full-RA fallback for wide equatorial cones)
- `mmu/safety.py` — output-path guardrails (allowlist + `external_data/` denylist) using `os.path.realpath` + path-segment comparison so `/foo/bar_extra` doesn't false-positive against `/foo/bar` and a symlink to `external_data/` can't bypass the check
- `mmu/hats_import.py` — `write_hats` and `write_hats_from_parquet_dir` helpers
  - Production defaults: `debug=False, n_workers=32` (was silently bottlenecked at 1 worker by `debug=True` in an earlier round)
  - `default_scratch_dir()` for per-catalog ceph scratch paths
- `mmu/sn_ia_snana.py` — shared inline SNANA ASCII parser for the 5 SN-Ia datasets (replaces sncosmo, which has no Linux wheel and won't build against the cluster's Python.h-less compute nodes). Hardened against OBS-before-VARLIST and duplicate metadata keys.
- `mmu/hats_configs.py` — registry with `DatasetConfig` + raw paths. `gaia` is the full-DR3 build, `gaia_xp` is the XP-joined subset.
- `Snakefile`
  - Per-dataset `build_<name>` rules, each declaring its build script as an `input:` (auto-rerun on edit)
  - `SHARDED_DATASETS` dict + generic `build_sharded_shard` wildcard rule + loop-generated gather rules
  - Currently sharded: legacysurvey (128), desi (16), gaia (16), gaia_xp (8), manga (8), ssl_legacysurvey (8)
  - `SHARDED_SLURM_PARTITION` for scatter (preempt), gather always runs on `SLURM_PARTITION` (ccm, guaranteed)
  - `qos="preempt"` resource for preempt scatter, `--slurm-requeue` auto-requeue on preemption
  - Cone args + `--max-files` threaded through `build_sharded_shard` so `profile=cosmos` actually produces a cosmos slice
- `scripts/gather_one_off.py` — standalone `write_hats_from_parquet_dir` wrapper for manual gather sbatch jobs (used when we need to re-ingest an existing scratch without restarting the snakemake DAG). Has the `if __name__ == "__main__":` guard required by dask's spawn workers.
- `snakemake_config.yaml` — `cluster`, `cosmos`, `test` profiles

---

## Notable incidents this session

- **legacysurvey scatter OOM at 939 GB MaxRSS.** Per-worker `sweep_records` list accumulated all cutouts for a whole sweep (~1000 bricks × ~200 objects × ~370 KB ≈ 70 GB per worker × 30 active workers ≈ 1 TB combined). Fixed by per-256-cutouts batch flush in `_process_sweep_to_parquet`.
- **manga schema overhauled.** The original v1 schema padded everything to 96×96 — destructive normalization that wasted 60–70% of cells. New schema stores native `(ny, nx, nwave)` IFU shape with `spatial_shape_y/x` companion columns and a recursive `_ndarray_to_nested_array` helper. Tests rewritten to use realistic native shapes.
- **gather_one_off.py multiprocessing fork bug.** First version was a flat script with no `if __name__ == "__main__":` guard; dask's spawn workers re-imported the module as `__main__` and recursed into another `write_hats_from_parquet_dir` call, looping forever. Fixed by wrapping in `def main()` + `if __name__ == "__main__":`.
- **desi/twomass partial output dirs.** Earlier hats-import attempts wrote `Norder=4..7` partition dirs to the output path but never finalized `hats.properties`. Wiped via instant-rename trick (`mv twomass .twomass.deleting && rm -rf .twomass.deleting &`) since recursive `rm -rf` on ceph is painfully slow at scale.
- **gaia full-DR3 master migration.** Was running in a separate `mmu_hats_snia` parallel workdir with its own snakemake master and own tmux session. Killed the master, killed the dependent gather attempt (which had been at 24% Catalog: Splitting), submitted the gather as a `gather_one_off.py` sbatch since snakemake's `--touch + --cleanup-metadata + --rerun-incomplete` dance still wanted to re-run the scatter shards.

---

## SLURM launchers

- `scripts/slurm/run_cosmos_slurm.sh` — COSMOS 1° validation via Snakemake/slurm executor
- `scripts/slurm/run_production_slurm.sh` — all-sky production via Snakemake/slurm executor

No bespoke `/tmp/*.sh` launchers — everything flows through Snakemake (or, for one-off ingest reruns, a single direct `sbatch --wrap=...` calling `gather_one_off.py`).

---

## Totals

| Category | Count |
|---|---|
| Landed in production (`hats.properties` on disk) | 13 |
| In flight right now | 7 (legacysurvey, ssl_legacysurvey, manga, gaia, desi, twomass, allwise) |
| Phase 2 still to port | 6 (vipers, hsc, jwst, btsbot, gz10, desi_provabgs) |
| Skipped | 6 (plasticc, kepler, cfa, csp, lamost, yse) |

**Target: 26 production catalogs (13 done + 7 in flight + 6 to port).**

---

## Verification criteria (per dataset)

1. `uv run pytest tests/test_<dataset>_build.py -v` → all pass
2. `uv run pytest tests/` → full suite green
3. Cluster build produces `hats.properties` at `MultimodalUniverse_v2_hats/<dataset>/<dataset>/<dataset>/`
4. Read-back parquet matches the v1 HF schema field-for-field (modulo deliberate native-shape changes that we've documented)
5. `lsdb.read_hats(<output>).compute().head()` → returns rows without exceptions
6. Checkbox ticked here
