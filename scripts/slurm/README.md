# SLURM launchers for the MMU v2 HATS port

Two shell wrappers around the snakemake-slurm executor for running the MMU
v2 HATS pipeline on Flatiron's Rusty cluster (or any SLURM cluster with the
`ccm` partition pattern). They translate the Snakefile rules into per-rule
sbatch jobs.

## Files

- **`run_cosmos_slurm.sh`** — submits the per-dataset rules under the
  `cosmos` profile. Builds a 1° cone slice of each dataset against the
  COSMOS field (ra=150°, dec=+2°). Used as a fast end-to-end test that
  every rule runs cleanly through snakemake → slurm → hats-import.

- **`run_production_slurm.sh`** — submits the per-dataset rules under the
  `cluster` profile. Builds the full all-sky catalog for each dataset to
  `/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/{dataset}/`.

Both scripts:

- Locate the repo root from their own location, so you can run them from
  anywhere.
- Set `SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+hats` (workaround for the
  `07-hackathon` git tag that confuses setuptools-scm).
- Run `uv sync` first to make sure deps are up-to-date.
- `tee` snakemake's stdout/stderr to a `/tmp/*_slurm.log` file so the
  log survives ssh disconnects.
- End the log with a `DONE_EXIT_<N>` sentinel for unambiguous success/fail
  detection.

## Recommended usage

Always run inside a `tmux` session so the parent snakemake process
survives ssh drops:

```bash
# cosmos test slice (~minutes per dataset)
tmux new-session -d -s cosmos_slurm "bash scripts/slurm/run_cosmos_slurm.sh"
tmux attach -t cosmos_slurm   # to watch interactively
tail -f /tmp/cosmos_slurm.log # or just tail the log

# full production (~hours to days per dataset)
tmux new-session -d -s production "bash scripts/slurm/run_production_slurm.sh"
```

## Subsetting which datasets to run

Both scripts honor an env var override:

```bash
# only validate the 3 refactored datasets
COSMOS_RULES="build_allwise build_gaia build_desi" \
    bash scripts/slurm/run_cosmos_slurm.sh

# launch only one production dataset
PRODUCTION_RULES="build_sdss" \
    bash scripts/slurm/run_production_slurm.sh
```

## ssl_legacysurvey is intentionally excluded

The default production rule list does NOT include `build_ssl_legacysurvey`
because its full output is ~42 TB, which crosses the Flatiron ceph wiki's
"please notify us if you plan to generate >10 TB in a short period"
threshold. To include it, override `PRODUCTION_RULES` and email
`scicomp@flatironinstitute.org` first.

## Cancelling jobs — IMPORTANT

If you ever need to cancel jobs that snakemake-slurm submitted, **only
cancel by specific numeric job IDs**:

```bash
scancel 6209117 6209118 6209119   # GOOD: explicit IDs
```

snakemake's stdout (and the `tee`d log) prints a line per submitted job:

```
Job 0 has been submitted with SLURM jobid 6209118 (log: ...)
```

Read those IDs out, then cancel the specific ones you need.

**NEVER do this:**

```bash
scancel --user=$USER                      # BAD: kills interactive job too
scancel --partition=ccm                   # BAD: kills interactive job too
scancel --user=$USER --partition=ccm      # BAD: still kills interactive
```

These filter-based forms will cancel your interactive shell job alongside
the snakemake-submitted ones, which is how the user got disconnected from
their working session at least once during the v2 port hackathon. Just
read the job IDs out of the snakemake log.

## Per-rule resources

The actual sbatch resources (mem, cores, walltime, partition) come from
the `resources:` blocks in the top-level `Snakefile`. To change resources
for a specific rule, edit the Snakefile, not these wrappers.

The `cluster` profile uses `cpus_per_task = 96` for every rule because
Flatiron's CCM partition is whole-node-exclusive — asking for fewer
cores wastes the rest of the node anyway.
