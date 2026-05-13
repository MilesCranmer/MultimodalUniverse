#!/bin/bash
# Build the COSMOS-Web HATS catalog on the Candide cluster.
#
# Runs build_parent_sample_hats.py directly (no Snakemake), one sbatch job,
# processing 20 tiles with a Pool of workers.
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   1. Clone the repo on Candide if you haven't already:
#        git clone <repo-url> /n03data/huertas/python/MultimodalUniverse-1
#
#   2. Install mmu dependencies into your conda env (one-time):
#        source /n03data/huertas/python/miniconda3/etc/profile.d/conda.sh
#        conda activate /n03data/huertas/python/miniconda3/envs/cosmos_visual
#        pip install -e /n03data/huertas/python/MultimodalUniverse-1[dev]
#      (or create a fresh env:  conda create -n mmu python=3.11 && pip install -e .[dev])
#
#   3. Edit the variables in the "── Configuration ──" block below.
#
#   4. Submit:
#        sbatch scripts/slurm/run_cosmos_candide.sh
#
#   5. Monitor:
#        squeue -u $USER
#        tail -f <OUTPUT_DIR>/cosmos_hats_<jobid>.out
#
# ── Memory / time guidance ───────────────────────────────────────────────────
#   Each Pool worker loads 5 FITS mosaics for one tile.
#   NIRCam at 30 mas → ~20k×20k px → ~1.6 GB per filter in RAM.
#   MIRI at 60 mas   → ~10k×10k px → ~0.4 GB.
#   Peak per worker  ≈ 4×1.6 + 0.4 ≈ 7 GB.
#   Pool(4) workers  → ~28 GB + Python/PyArrow overhead → 64 GB safe.
#   Increase --mem and --num-processes together if you want faster throughput.

#SBATCH --job-name=cosmos_hats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# No --nodelist or --partition: SLURM picks any available node.
# Add e.g.  #SBATCH --partition=htc  if your cluster requires a partition name.

# ── Configuration ─────────────────────────────────────────────────────────────
# Edit these paths to match your setup on Candide.

# Where the MultimodalUniverse repo lives on the cluster.
REPO_DIR=/n03data/huertas/python/MultimodalUniverse

# Conda environment that has mmu + its dependencies installed (see Quick start).
CONDA_ROOT=/n03data/huertas/python/miniconda3
CONDA_ENV=${CONDA_ROOT}/envs/cosmos_visual

# Where to write the finished HATS catalog.
OUTPUT_ROOT=/n03data/huertas/mmu/cosmos

# Scratch directory for intermediate per-tile parquet shards (~500 MB total).
# Must be writable and on a shared filesystem (survives between steps).
SCRATCH_DIR=/n03data/huertas/mmu/cosmos_scratch

# Input data paths (defaults already match the Candide/Flatiron server layout;
# change only if your mounts differ).
CATALOG_PATH=/n03data/huertas/COSMOS-Web/cats/COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits
NIRCAM_ROOT=/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8
MIRI_ROOT=/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7

# Pool workers. Keep ≤ cpus-per-task; each needs ~7 GB RAM (see header).
NUM_PROCESSES=4

# Dask workers for the HATS ingest step (CPU-bound, low memory).
INGEST_WORKERS=8

# ── SLURM output files ────────────────────────────────────────────────────────
# Written to OUTPUT_ROOT so they sit next to the catalog.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# (overridden below once OUTPUT_ROOT is known at submission time)

# ── Environment ──────────────────────────────────────────────────────────────
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Make the repo importable as a package (needed for `from mmu.xxx import ...`).
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}

# Suppress SETUPTOOLS_SCM noise when running without a proper install.
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+cosmos

cd ${REPO_DIR}

# ── Dependency check ─────────────────────────────────────────────────────────
python - <<'EOF'
import importlib, sys
missing = [pkg for pkg in ("hats", "lsdb", "pyarrow", "astropy", "dask")
           if importlib.util.find_spec(pkg) is None]
if missing:
    print("ERROR: missing Python packages:", missing)
    print("Run:  pip install -e .[dev]  inside your conda env and resubmit.")
    sys.exit(1)
print("Dependency check passed.")
EOF
if [ $? -ne 0 ]; then exit 1; fi

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_ROOT}" "${SCRATCH_DIR}"

# Redirect SLURM output now that OUTPUT_ROOT exists.
# (The #SBATCH directives above write to the submission directory by default;
#  to write into OUTPUT_ROOT instead, pass --output when calling sbatch:
#    sbatch --output="${OUTPUT_ROOT}/%x_%j.out" \
#           --error="${OUTPUT_ROOT}/%x_%j.err"  \
#           scripts/slurm/run_cosmos_candide.sh )

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== cosmos HATS build started $(date) ==="
echo "  repo:        ${REPO_DIR}"
echo "  conda env:   ${CONDA_ENV}"
echo "  catalog:     ${CATALOG_PATH}"
echo "  nircam root: ${NIRCAM_ROOT}"
echo "  miri root:   ${MIRI_ROOT}"
echo "  output:      ${OUTPUT_ROOT}"
echo "  scratch:     ${SCRATCH_DIR}"
echo "  workers:     ${NUM_PROCESSES}"
echo

python -u -m scripts.cosmos.build_parent_sample_hats \
    --catalog-path   "${CATALOG_PATH}"  \
    --nircam-root    "${NIRCAM_ROOT}"   \
    --miri-root      "${MIRI_ROOT}"     \
    --output-root    "${OUTPUT_ROOT}"   \
    --scratch-dir    "${SCRATCH_DIR}"   \
    --num-processes  ${NUM_PROCESSES}   \
    --ingest-workers ${INGEST_WORKERS}

EXIT_CODE=$?
echo
echo "=== cosmos HATS build finished $(date) — exit ${EXIT_CODE} ==="
exit ${EXIT_CODE}
