#!/bin/bash
# Validate the COSMOS-Web HATS catalog on the Candide cluster.
#
# Runs validate_hats_dataset.py to produce:
#   schema_columns.csv   — full column list with types
#   null_rates.csv       — missing-value rates per column
#   hist_*.png/csv       — magnitude / redshift histograms
#   f277w_grid.png       — random F277W grayscale cutouts
#   rgb_grid.png         — random F444W/F277W/F115W RGB cutouts
#   bin_plots/           — per-magnitude-bin image grids
#   coverage_moc.fits/png — sky footprint
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   sbatch --output="${OUT_DIR}/cosmos_validate_%j.out" \
#          --error="${OUT_DIR}/cosmos_validate_%j.err"  \
#          scripts/slurm/run_cosmos_validate_candide.sh
#
# ── Memory / time guidance ───────────────────────────────────────────────────
#   Validation streams parquet files row-by-row (no full-catalog load).
#   Plot grids load ~25 image rows ≈ 50 MB.
#   16 GB RAM and 2 h wall-time are generous upper bounds.

#SBATCH --job-name=cosmos_validate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cosmos_validate_%j.out
#SBATCH --error=cosmos_validate_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_DIR=/n03data/huertas/python/MultimodalUniverse
CONDA_ROOT=/n03data/huertas/python/miniconda3
CONDA_ENV=${CONDA_ROOT}/envs/mmu

# Root of the finished HATS catalog (inner catalog dir produced by build script).
HATS_CATALOG=/n03data/huertas/mmu/cosmos/cosmos/cosmos

# Where validation artifacts are written.
OUT_DIR=/n03data/huertas/mmu/cosmos/validation

# Histograms to generate (COLUMN or COLUMN:TRANSFORM).
HIST_SPECS=(
    "MAG_MODEL_F277W"
    "MAG_MODEL_F444W"
    "MAG_MODEL_F115W"
    "ZPHOT"
    "RADIUS:log10"
    "AXRATIO"
)

# Binned image grids: COLUMN:BIN_WIDTH:MIN:MAX.
BIN_SPECS=(
    "MAG_MODEL_F277W:0.5:17:27"
)

# Columns shown in plot titles.
LABEL_COLS=(
    "object_id"
    "MAG_MODEL_F277W"
    "ZPHOT"
)

# Random cutouts per grid.
SAMPLE_SIZE=25

# ── Environment ──────────────────────────────────────────────────────────────
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+cosmos

cd ${REPO_DIR}

# ── Dependency check ─────────────────────────────────────────────────────────
python - <<'EOF'
import importlib.util, sys
missing = [pkg for pkg in ("pyarrow", "astropy", "matplotlib", "numpy")
           if importlib.util.find_spec(pkg) is None]
if missing:
    print("ERROR: missing Python packages:", missing)
    sys.exit(1)
print("Dependency check passed.")
EOF
if [ $? -ne 0 ]; then exit 1; fi

# ── Create output dir ─────────────────────────────────────────────────────────
mkdir -p "${OUT_DIR}"

# ── Build argument lists ──────────────────────────────────────────────────────
HIST_ARGS=""
for spec in "${HIST_SPECS[@]}"; do
    HIST_ARGS="${HIST_ARGS} --hist ${spec}"
done

BIN_ARGS=""
for spec in "${BIN_SPECS[@]}"; do
    BIN_ARGS="${BIN_ARGS} --bin-plot ${spec}"
done

LABEL_ARGS=""
for col in "${LABEL_COLS[@]}"; do
    LABEL_ARGS="${LABEL_ARGS} --label-column ${col}"
done

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== cosmos validation started $(date) ==="
echo "  catalog: ${HATS_CATALOG}"
echo "  out dir: ${OUT_DIR}"
echo

python -u -m scripts.cosmos.validation.validate_hats_dataset \
    --dataset     "${HATS_CATALOG}" \
    --out-dir     "${OUT_DIR}"      \
    --sample-size ${SAMPLE_SIZE}    \
    ${HIST_ARGS}                    \
    ${BIN_ARGS}                     \
    ${LABEL_ARGS}

EXIT_CODE=$?
echo
echo "=== cosmos validation finished $(date) — exit ${EXIT_CODE} ==="
exit ${EXIT_CODE}
