#!/bin/bash
# Cosmos validation via snakemake-slurm executor.
#
# Submits per-rule sbatch jobs that build the COSMOS 1° test slice for the
# datasets named below. Used as a fast end-to-end test of the orchestration
# (snakemake → slurm → hats-import) before committing to a full production
# build. Each rule's resources come from the Snakefile.
#
# Usage (from anywhere on a cluster login or compute node with sbatch):
#     bash scripts/slurm/run_cosmos_slurm.sh
#
# Output:
#     - Per-rule sbatch jobs in the ccm partition (one job per dataset).
#     - Snakemake submission log echoed to stdout AND tee'd to
#       /tmp/cosmos_slurm.log so it survives shell disconnects.
#     - DONE_EXIT_<N> sentinel at the end of the log indicates the
#       snakemake exit code.
#
# To monitor: `tail -f /tmp/cosmos_slurm.log` or
#             `tmux attach -t cosmos_slurm` if launched inside a tmux session.
#
# To launch detached so it survives ssh drop:
#     tmux new-session -d -s cosmos_slurm "bash scripts/slurm/run_cosmos_slurm.sh"

set -e

# Locate the repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+hats

# Datasets to validate. Override with COSMOS_RULES env var if you only want
# a subset, e.g.:  COSMOS_RULES="build_allwise build_gaia" bash run_cosmos_slurm.sh
RULES="${COSMOS_RULES:-build_sdss build_allwise build_twomass build_sages build_galex build_desi build_tess build_gaia build_legacysurvey build_manga}"

LOG=/tmp/cosmos_slurm.log

{
    echo "=== cosmos validation via snakemake-slurm $(date) ==="
    echo "rules: $RULES"
    git -C "$REPO_ROOT" log --oneline -1 || true
    uv sync --quiet 2>&1 | tail -3 || true
    echo
    uv run snakemake \
        --executor slurm \
        --jobs 10 \
        --config profile=cosmos \
        --keep-going \
        -- $RULES
    echo "DONE_EXIT_$?"
    echo "=== cosmos validation complete $(date) ==="
} 2>&1 | tee "$LOG"
