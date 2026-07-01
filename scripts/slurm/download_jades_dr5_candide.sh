#!/bin/bash
# Download JADES DR5 (GOODS-S + GOODS-N) to Candide.
#
# Output layout:
#   /n03data/huertas/JADES/DR5/
#     mosaics/    — science + weight FITS mosaics (GOODS-S and GOODS-N)
#     psfs/       — PSF FITS files (GOODS-S and GOODS-N)
#     cats/       — photometric catalogs (GOODS-S and GOODS-N)
#
# ── Sources ───────────────────────────────────────────────────────────────────
#   GOODS-S mosaics : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/mosaics/
#   GOODS-N mosaics : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/mosaics/
#   GOODS-S PSFs    : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/submosaics/ (*mpsf.fits)
#   GOODS-N PSFs    : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/submosaics/ (*mpsf.fits)
#   GOODS-S cats    : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/
#   GOODS-N cats    : https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/catalogs/
#
# ── Submit ────────────────────────────────────────────────────────────────────
#   sbatch scripts/slurm/download_jades_dr5_candide.sh
#
# ── Resume ────────────────────────────────────────────────────────────────────
#   wget uses -nc (no-clobber): already-downloaded files are skipped.
#   Simply resubmit to resume a partial download.

#SBATCH --job-name=jades_dr5_download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=/n23data1/huertas/jades/DR5/jades_dr5_download_%j.out
#SBATCH --error=/n23data1/huertas/jades/DR5/jades_dr5_download_%j.err

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR=/n23data1/huertas/jades/DR5
MOSAIC_DIR=${BASE_DIR}/mosaics
PSF_DIR=${BASE_DIR}/psfs
CAT_DIR=${BASE_DIR}/cats

# lftp is used instead of wget because wget -r loads the entire URL tree into
# RAM before downloading, which exhausts memory on large JADES directory trees.
# lftp mirror streams the listing and downloads concurrently with low overhead.
#
# lftp mirror options:
#   --only-newer   : skip files already present (resume-safe)
#   --parallel=4   : 4 concurrent transfers per mirror call
#   --verbose      : print each downloaded file
#   --log=FILE     : per-section transfer log

# ── Helper: mirror one remote URL into a local directory ──────────────────────

lftp_mirror() {
    local remote_url="$1"
    local local_dir="$2"
    local include_glob="${3:-}"   # optional glob, e.g. "*mpsf.fits"
    local log_file="${BASE_DIR}/lftp_$(basename ${local_dir}).log"

    local include_opt=""
    if [ -n "${include_glob}" ]; then
        include_opt="--include-glob=${include_glob}"
    fi

    lftp -c "
set net:max-retries 5
set net:reconnect-interval-base 10
set ftp:passive-mode yes
open ${remote_url%/*}/
mirror --only-newer --parallel=4 --verbose ${include_opt} \
       --log=${log_file} \
       $(basename ${remote_url})/ ${local_dir}/
bye
"
}

# ── Create output directories ─────────────────────────────────────────────────

mkdir -p "${MOSAIC_DIR}/GOODS-S" \
         "${MOSAIC_DIR}/GOODS-N" \
         "${PSF_DIR}/GOODS-S"    \
         "${PSF_DIR}/GOODS-N"    \
         "${CAT_DIR}/GOODS-S"    \
         "${CAT_DIR}/GOODS-N"

echo "=== JADES DR5 download started $(date) ==="
echo "  output base: ${BASE_DIR}"
echo

# ── Mosaics ───────────────────────────────────────────────────────────────────

echo "--- GOODS-S mosaics ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/mosaics/" \
    "${MOSAIC_DIR}/GOODS-S"

echo "--- GOODS-N mosaics ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/mosaics/" \
    "${MOSAIC_DIR}/GOODS-N"

# ── PSFs ──────────────────────────────────────────────────────────────────────
# PSFs live under .../submosaics/ in subregion subdirectories.
# Only *mpsf.fits files are downloaded; subregion structure is preserved.

echo "--- GOODS-S PSFs (*mpsf.fits) ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/submosaics/" \
    "${PSF_DIR}/GOODS-S" \
    "*mpsf.fits"

echo "--- GOODS-N PSFs (*mpsf.fits) ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/submosaics/" \
    "${PSF_DIR}/GOODS-N" \
    "*mpsf.fits"

# ── Catalogs ──────────────────────────────────────────────────────────────────

echo "--- GOODS-S catalogs ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/" \
    "${CAT_DIR}/GOODS-S"

echo "--- GOODS-N catalogs ---"
lftp_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/catalogs/" \
    "${CAT_DIR}/GOODS-N"

# ── Summary ───────────────────────────────────────────────────────────────────

echo
echo "=== JADES DR5 download finished $(date) ==="
echo
echo "File counts:"
echo "  mosaics/GOODS-S : $(find ${MOSAIC_DIR}/GOODS-S -type f | wc -l) files"
echo "  mosaics/GOODS-N : $(find ${MOSAIC_DIR}/GOODS-N -type f | wc -l) files"
echo "  psfs/GOODS-S    : $(find ${PSF_DIR}/GOODS-S    -type f | wc -l) files"
echo "  psfs/GOODS-N    : $(find ${PSF_DIR}/GOODS-N    -type f | wc -l) files"
echo "  cats/GOODS-S    : $(find ${CAT_DIR}/GOODS-S    -type f | wc -l) files"
echo "  cats/GOODS-N    : $(find ${CAT_DIR}/GOODS-N    -type f | wc -l) files"
echo
echo "Disk usage: $(du -sh ${BASE_DIR})"
