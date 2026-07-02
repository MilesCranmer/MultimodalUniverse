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

# Python urllib is used for downloading: it needs no external tools (no lftp,
# no wget -r) and streams file listings from the HTTP index page, so it never
# loads the whole directory tree into memory.
#
# Behaviour:
#   - Skips files that already exist on disk (resume-safe).
#   - Recurses into subdirectories (needed for PSF submosaics).
#   - Optional fnmatch glob filters which files are downloaded.

# ── Helper: mirror one remote HTTP index URL into a local directory ───────────

http_mirror() {
    local remote_url="$1"
    local local_dir="$2"
    local include_glob="${3:-}"   # optional glob, e.g. "*mpsf.fits"

    python3 - "${remote_url}" "${local_dir}" "${include_glob}" <<'PYEOF'
import sys, re, urllib.request
from pathlib import Path
from fnmatch import fnmatch

def mirror(url, local_dir, include_glob):
    url = url.rstrip('/') + '/'
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"Scanning {url}", flush=True)
    try:
        with urllib.request.urlopen(url) as r:
            html = r.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  ERROR fetching {url}: {e}", flush=True)
        return

    hrefs = re.findall(r'href="([^"?#][^"]*)"', html)
    for href in sorted(set(hrefs)):
        if href.startswith(('/', '..', '?', 'http')):
            continue
        is_dir = href.endswith('/')
        name = href.rstrip('/')
        if not name:
            continue
        if is_dir:
            mirror(url + name + '/', str(Path(local_dir) / name), include_glob)
        else:
            if include_glob and not fnmatch(name, include_glob):
                continue
            dest = Path(local_dir) / name
            if dest.exists():
                print(f"  skip  {name}", flush=True)
                continue
            print(f"  get   {name}", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                urllib.request.urlretrieve(url + name, str(tmp))
                tmp.rename(dest)
            except Exception as e:
                print(f"  ERROR {name}: {e}", flush=True)
                tmp.unlink(missing_ok=True)

mirror(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
PYEOF
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
http_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/mosaics/" \
    "${MOSAIC_DIR}/GOODS-S"

echo "--- GOODS-N mosaics ---"
http_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/mosaics/" \
    "${MOSAIC_DIR}/GOODS-N"

# ── PSFs ──────────────────────────────────────────────────────────────────────
# PSFs live under .../submosaics/ in subregion subdirectories.
# Only *mpsf.fits files are downloaded; subregion structure is preserved.

echo "--- GOODS-S PSFs (*mpsf.fits) ---"
http_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/images/submosaics/" \
    "${PSF_DIR}/GOODS-S" \
    "*mpsf.fits"

echo "--- GOODS-N PSFs (*mpsf.fits) ---"
http_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-N/hlsp/images/submosaics/" \
    "${PSF_DIR}/GOODS-N" \
    "*mpsf.fits"

# ── Catalogs ──────────────────────────────────────────────────────────────────

echo "--- GOODS-S catalogs ---"
http_mirror \
    "https://slate.ucsc.edu/~brant/jades-dr5/GOODS-S/hlsp/catalogs/" \
    "${CAT_DIR}/GOODS-S"

echo "--- GOODS-N catalogs ---"
http_mirror \
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
