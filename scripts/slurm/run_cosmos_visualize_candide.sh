#!/bin/bash
# Visualise a random sample from the COSMOS-Web HATS catalog on Candide.
#
# Reads N_SAMPLE objects directly from parquet partitions (no lsdb/dask),
# flattens the image struct, and produces:
#   f277w_cutouts.png     — grayscale F277W grid
#   rgb_cutouts.png       — F444W/F277W/F115W RGB grid
#   multiband_strip.png   — one row per object × 5 band columns
#
# ── Quick start ──────────────────────────────────────────────────────────────
#   sbatch --output="${OUT_DIR}/cosmos_visualize_%j.out" \
#          --error="${OUT_DIR}/cosmos_visualize_%j.err"  \
#          scripts/slurm/run_cosmos_visualize_candide.sh
#
# ── Memory / time guidance ───────────────────────────────────────────────────
#   64 objects × 5 bands × 160×160 × float32 ≈ 130 MB.
#   8 GB RAM and 30 min wall-time are more than enough.

#SBATCH --job-name=cosmos_visualize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cosmos_visualize_%j.out
#SBATCH --error=cosmos_visualize_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_DIR=/n03data/huertas/python/MultimodalUniverse
CONDA_ROOT=/n03data/huertas/python/miniconda3
CONDA_ENV=${CONDA_ROOT}/envs/mmu

# Inner catalog directory produced by the build script.
HATS_CATALOG=/n03data/huertas/mmu/cosmos/cosmos/cosmos

# Where output PNGs are written.
OUT_DIR=/n03data/huertas/mmu/cosmos/validation

# Number of random objects to visualise.
N_SAMPLE=64

# Number of objects in the multi-band strip.
N_STRIP=10

# Random seed for reproducibility.
SEED=42

# ── Environment ──────────────────────────────────────────────────────────────
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

export PYTHONPATH=${REPO_DIR}:${PYTHONPATH}
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+cosmos
# Use non-interactive matplotlib backend.
export MPLBACKEND=Agg

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

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=== cosmos visualisation started $(date) ==="
echo "  catalog: ${HATS_CATALOG}"
echo "  out dir: ${OUT_DIR}"
echo "  n_sample: ${N_SAMPLE}   n_strip: ${N_STRIP}   seed: ${SEED}"
echo

python -u - <<PYEOF
import glob, os, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "${REPO_DIR}")
import matplotlib
matplotlib.use("Agg")
from scripts.cosmos.validation.viusalize import plot_all_modalities, plot_multiband_strip

hats_catalog = "${HATS_CATALOG}"
out_dir      = "${OUT_DIR}"
n_sample     = int("${N_SAMPLE}")
n_strip      = int("${N_STRIP}")
seed         = int("${SEED}")

# ── Collect random rows ────────────────────────────────────────────────────
parquet_files = sorted(
    glob.glob(os.path.join(hats_catalog, "**", "Npix=*.parquet"), recursive=True)
)
if not parquet_files:
    print(f"ERROR: no parquet files found under {hats_catalog}", file=sys.stderr)
    sys.exit(1)
print(f"Found {len(parquet_files)} parquet partitions.")

rng = np.random.default_rng(seed)
rng.shuffle(parquet_files)  # randomise partition order

rows = []
for path in parquet_files:
    if len(rows) >= n_sample:
        break
    try:
        table = pq.read_table(
            path,
            columns=["image", "obj_id", "MAG_MODEL_F277W", "ZPHOT"],
        )
    except Exception as exc:
        print(f"  skipping {path}: {exc}")
        continue

    flux_col  = table.column("image").field("flux")
    obj_col   = table.column("obj_id")
    mag_col   = table.column("MAG_MODEL_F277W")
    zphot_col = table.column("ZPHOT")

    n_take = min(len(table), n_sample - len(rows))
    indices = rng.choice(len(table), size=n_take, replace=False)
    for i in indices:
        flux_val = flux_col[int(i)].as_py()
        if flux_val is None:
            continue
        rows.append({
            "image_flux":       np.asarray(flux_val, dtype=np.float32),
            "obj_id":           obj_col[int(i)].as_py(),
            "MAG_MODEL_F277W":  mag_col[int(i)].as_py(),
            "ZPHOT":            zphot_col[int(i)].as_py(),
        })

print(f"Loaded {len(rows)} objects.")
if not rows:
    print("ERROR: no valid rows loaded.", file=sys.stderr)
    sys.exit(1)

# ── Plot all-modalities grids (grayscale + RGB) ────────────────────────────
print("Plotting grayscale and RGB grids …")
plot_all_modalities(
    rows,
    max_plots=n_sample,
    object_id_col="obj_id",
    show=False,
    save=True,
    save_dir=out_dir,
    dpi=180,
)

# ── Plot multi-band strip ──────────────────────────────────────────────────
print("Plotting multi-band strip …")
plot_multiband_strip(
    rows,
    n_objects=min(n_strip, len(rows)),
    object_id_col="obj_id",
    show=False,
    save=True,
    save_dir=out_dir,
    dpi=180,
)

print(f"Done. Outputs in {out_dir}")
PYEOF

EXIT_CODE=$?
echo
echo "=== cosmos visualisation finished $(date) — exit ${EXIT_CODE} ==="
exit ${EXIT_CODE}
