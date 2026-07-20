#!/bin/bash
# All-sky production HATS build via snakemake-slurm executor.
#
# Submits per-rule sbatch jobs that build the FULL HATS catalogs for the
# datasets named below. Output goes to the production tree at
# /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/{dataset}/
# (the path is set by the `cluster` profile in snakemake_config.yaml and
# is verified by mmu.safety.validate_hats_root before any rule runs).
#
# By default we exclude `build_ssl_legacysurvey` from the production set
# because its expected output (~42 TB) crosses the Flatiron ceph "let us
# know if you'll generate >10 TB" courtesy threshold and the user must
# email scicomp before launching it. Override with PRODUCTION_RULES env
# var if you want a different set.
#
# Usage:
#     bash scripts/slurm/run_production_slurm.sh
#
# Or detached in tmux so it survives an ssh disconnect:
#     tmux new-session -d -s production "bash scripts/slurm/run_production_slurm.sh"
#
# To monitor: `tail -f /tmp/production_slurm.log` or
#             `tmux attach -t production`.
#
# Snakemake-slurm prints a "Job N has been submitted with SLURM jobid M"
# line for each rule. RECORD THOSE JOB IDS — if you ever need to cancel,
# cancel by specific numeric job ID only:
#     scancel <jobid> [<jobid> ...]
# NEVER use `scancel --user=...` or `scancel --partition=...` because that
# will also kill any interactive shell job you have running on the cluster.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+hats

# 10 production rules. ssl_legacysurvey is intentionally NOT in this list
# (see header). To include it, pass: PRODUCTION_RULES="... build_ssl_legacysurvey"
RULES="${PRODUCTION_RULES:-build_sdss build_allwise build_twomass build_sages build_galex build_desi build_tess build_gaia build_legacysurvey build_manga}"

LOG=/tmp/production_slurm.log

{
    echo "=== production launch via snakemake-slurm $(date) ==="
    echo "rules: $RULES"
    git -C "$REPO_ROOT" log --oneline -1 || true
    uv sync --quiet 2>&1 | tail -3 || true
    echo
    # --jobs 20 = up to 20 sbatch jobs in flight at once. Each rule = 1 sbatch
    # job per dataset; with only 10 datasets in $RULES we'll never submit more
    # than 10, but the headroom matches what the CCM partition can absorb
    # without hitting per-user core limits.
    uv run snakemake \
        --executor slurm \
        --jobs 20 \
        --config profile=cluster \
        --keep-going \
        -- $RULES
    echo "DONE_EXIT_$?"
    echo "=== production complete $(date) ==="
} 2>&1 | tee "$LOG"
