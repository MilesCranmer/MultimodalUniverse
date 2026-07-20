"""Snakemake pipeline: raw survey data -> HATS catalogs.

Each dataset has a `build_parent_sample_hats.py` under `scripts/{dataset}/`
that reads the survey-native raw inputs (parquet, FITS, etc.) and writes a
HATS catalog under `HATS_ROOT/{dataset}/{dataset}/{dataset}/`.

The Snakefile is intentionally a thin wrapper: it just declares one rule per
ported dataset and a top-level `all` target. Adding a new dataset is two lines
in `PORTED` plus the build script.

REQUIREMENTS: this pipeline only runs on the Flatiron cluster. Raw inputs live
under `/mnt/ceph/users/polymathic/external_data/astro/` and are NOT downloaded
by anything here. For local development use the unit tests in `tests/`.

Usage (run from a cluster login/compute node):

    # Build everything that's been ported, full size:
    uv run snakemake --cores 4 all

    # Build just one dataset:
    uv run snakemake --cores 1 build_allwise

    # Smoke-test against one raw shard per dataset:
    uv run snakemake --cores 1 build_allwise --config profile=test
"""

import os

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT


configfile: "snakemake_config.yaml"

# Optional per-machine override (gitignored). See snakemake_config.local.yaml.example.
if os.path.exists("snakemake_config.local.yaml"):
    configfile: "snakemake_config.local.yaml"


def _resolve(key, default=None):
    """Look up `key` in the active profile, falling back to top-level config."""
    profile_name = config.get("profile", "cluster")
    profile = config.get("profiles", {}).get(profile_name, {})
    if key in config:
        return config[key]  # explicit --config wins
    return profile.get(key, default)


HATS_ROOT = _resolve("hats_root", MMU_V2_HATS_ROOT)
MAX_FILES = _resolve("max_files", None)  # None = build everything
RA_CENTER = _resolve("ra_center", None)
DEC_CENTER = _resolve("dec_center", None)
RADIUS = _resolve("radius", None)


# --------------------------------------------------------------------------- #
#                               OUTPUT SAFETY                                 #
# --------------------------------------------------------------------------- #
# /mnt/ceph/users/polymathic/ contains datasets that took MONTHS to produce,
# INCLUDING the READ-ONLY raw data mirror under external_data/ that every
# build script reads from. A Snakefile misconfiguration (typo in hats_root,
# wrong profile, stale envvar) could in principle ask Snakemake to write into
# these paths directly. Before ANY rule is parsed we validate HATS_ROOT so a
# bad value aborts the pipeline before a single directory is created.
#
# The validation is in `mmu.safety.validate_hats_root` so we can unit-test it
# rigorously. If you're adding a new allowed output prefix, update the
# whitelist there (and the tests).
from mmu.safety import validate_hats_root

validate_hats_root(HATS_ROOT)

# Datasets that have a working raw -> HATS build script under scripts/{name}/
PORTED = [
    "sdss",      # raw FITS plate files
    "apogee",    # raw APOGEE DR17 allStar + apStar/aspcapStar FITS
    "allwise",   # raw IRSA parquet, healpix-prefiltered for cone cuts
    "btsbot",    # raw ZTF triplets + alert metadata
    "desi_provabgs",  # raw PROVABGS VAC HDF5
    "twomass",   # raw IRSA gzipped CSV
    "sages",     # single FITS table (DR1 u/v photometry)
    "galex",     # multi-FITS GUVCat shards (latitude-partitioned)
    "desi",      # raw DESI coadd FITS files, merged via desispec.coadd_cameras
    "gz10",      # raw Galaxy10 DECals HDF5
    "tess",      # raw TESS-SPOC FFI lightcurves (one file per TIC, sector)
    "gaia",      # raw Gaia DR3 GaiaSource, full ~1.8B source catalog
    "gaia_xp",   # raw Gaia DR3, GaiaSource ∩ XpContinuousMeanSpectrum (~220M)
    "galah",     # raw GALAH DR3 catalogs + spectra tarball
    "legacysurvey",  # raw DECaLS DR10 south sweeps + brick coadds (image cutouts + nearby catalog)
    "manga",     # raw SDSS-IV MaNGA IFU LOGCUBE + DAP MAPS files (spaxels + griz images + analysis maps)
    "hsc",         # HSC SSP PDR3 Deep/UltraDeep, 5-band 160×160 cutouts + 65 scalars
    "foundation",  # Foundation DR1 SNe Ia (SNANA ASCII lightcurves, ~180 SNe)
    "snls",        # JLA2014 SNLS SNe Ia (~239 SNe, shared mmu.sn_ia_snana helper)
    "ps1_sne_ia",  # Pan-STARRS1 SNe Ia (~369 SNe)
    "des_y3_sne_ia",  # DES Y3 SNe Ia (~251 SNe)
    "swift_sne_ia",   # Swift UV/optical SNe Ia (~117 SNe)
    "cosmos",         # COSMOS-Web NIRCam+MIRI, 5 JWST bands, F277W<27
]


def catalog_marker(name: str) -> str:
    """Path to the inner-catalog `hats.properties` file (the per-dataset target)."""
    return f"{HATS_ROOT}/{name}/{name}/{name}/hats.properties"


def build_script(name: str) -> str:
    """Path to the per-dataset build script. Declaring this as a rule input
    means any edit to the script bumps its mtime and Snakemake automatically
    re-runs the rule on the next invocation — no manual --forceall or
    cleanup-metadata needed."""
    return f"scripts/{name}/build_parent_sample_hats.py"


# Datasets where the script's filenames don't encode RA/Dec, so a cone cut
# would require opening every one of O(100k) files just to read headers.
# For these we fall back to a ``--max-files`` cap when a cone is requested,
# which gives a usable test slice without ~hours of header I/O. A proper
# production build of these datasets simply doesn't set cone args.
DATASETS_WITHOUT_CONE_SUPPORT = {"tess"}
CONE_FALLBACK_MAX_FILES = 3


def build_command(dataset: str) -> str:
    parts = [
        "python", "-m", f"scripts.{dataset}.build_parent_sample_hats",
        "--output-root", f"{HATS_ROOT}/{dataset}",
    ]

    cone_active = (
        RA_CENTER is not None and DEC_CENTER is not None and RADIUS is not None
    )

    if dataset in DATASETS_WITHOUT_CONE_SUPPORT and cone_active:
        # Cap the file count so test-slice builds of cone-incompatible
        # datasets finish quickly. The script will still write a valid HATS
        # catalog, just against the first few input files rather than a
        # sky-region slice.
        effective_max = MAX_FILES if MAX_FILES is not None else CONE_FALLBACK_MAX_FILES
        parts += ["--max-files", str(effective_max)]
        return " ".join(parts)

    if MAX_FILES is not None:
        parts += ["--max-files", str(MAX_FILES)]
    if cone_active:
        parts += [
            "--ra-center", str(RA_CENTER),
            "--dec-center", str(DEC_CENTER),
            "--radius", str(RADIUS),
        ]
    return " ".join(parts)


# --------------------------------------------------------------------------- #
#                          PER-RULE SLURM RESOURCES                            #
# --------------------------------------------------------------------------- #
# When invoked with `snakemake --executor slurm`, each rule below is submitted
# as one sbatch job with the resources declared in its `resources:` block.
# Memory budgets are deliberately fat — we trust the CCM partition to have
# the headroom — and reflect rough scale of the dataset:
#
#   tabular (single-pass) :  ~200 GB RAM, 1 core, 8h walltime
#   spectra (desi, sdss)  :  ~500 GB RAM, 8 cores (desispec coadd_cameras),
#                            24h walltime
#   gaia (XP join)        :  ~750 GB RAM (228-col GaiaSource shards), 24h
#   image  (ssl, legacy)  :  ~750 GB RAM, 24h, lots of I/O
#   ifu    (manga)        :  ~500 GB RAM, 24h
#
# Adjust if rules OOM. The slurm partition is set to ``ccm`` because that's
# where the user has allocation headroom.

SLURM_PARTITION = _resolve("slurm_partition", "ccm")
SHARDED_SLURM_PARTITION = _resolve("sharded_slurm_partition", SLURM_PARTITION)


def slurm_qos(partition: str) -> str | None:
    """Return the required QoS string for a partition, or None."""
    return "preempt" if partition == "preempt" else None


# --------------------------------------------------------------------------- #
#                         SCATTER/GATHER SHARDED BUILDS                        #
# --------------------------------------------------------------------------- #
# For datasets where a single-node build would run past its walltime, we
# scatter the work across N independent sbatch jobs ("shards"), each of
# which writes per-unit parquet files into a shared ceph scratch dir. A
# single "gather" job then runs ``write_hats_from_parquet_dir`` against
# that scratch to produce the final HATS catalog.
#
# Snakemake natively manages this via two rules per sharded dataset:
#   1. ``build_<name>_shard`` — one sbatch job per shard_idx, writes
#      ``{scratch}/.shard_<idx>.done`` markers as Snakemake-visible outputs.
#   2. ``build_<name>`` — gather rule whose inputs are all shard markers.
#      When all scatter tasks have their markers, Snakemake submits this
#      gather job; it runs the --only-ingest path.
#
# Adding a new sharded dataset = one entry in SHARDED_DATASETS below.
# No bespoke shell launchers, no /tmp/*.sh files, no manual dependency
# chaining — pure Snakemake DAG-driven execution.
#
# Per-dataset build scripts need to honor these args:
#   --scratch-dir <shared path>    (shared across all shards; REQUIRED)
#   --num-shards N  --shard-idx I  (stride assignment)
#   --skip-ingest                  (shard tasks: write parquets only)
#   --only-ingest                  (gather task: run ingest only)
#   --num-processes P              (Pool size inside one shard)
#   --ingest-workers W             (dask workers for the gather step)

SHARDED_SCRATCH_ROOT = f"{HATS_ROOT}_scratch"

SHARDED_DATASETS = {
    "legacysurvey": dict(
        num_shards=128,
        num_processes=1,    # Pool(1) — one sweep at a time. Big sweeps peak at 588 GB
                            # during build_table (196k cutouts × 1 MB × 3x copy).
                            # Pool(2)+ OOMs when multiple workers hit build_table.
        build_mem_mb=900_000,
        # Sweeps are sequential: ~11 sweeps/shard × ~20 min/sweep = ~220 min nominal.
        # 480 min gives 2× headroom for slow sweeps (some have 200k+ objects).
        build_runtime_min=480,
        ingest_mem_mb=900_000,
        ingest_runtime_min=240,
        ingest_workers=96,
    ),
    "desi": dict(
        num_shards=16,
        num_processes=96,   # desispec.coadd_cameras workers, per-group ~1 GB
        build_mem_mb=750_000,
        build_runtime_min=360,    # 6h per shard
        ingest_mem_mb=900_000,
        ingest_runtime_min=240,
        # desi rows are big (7781-wavelength spectrum struct per row), so
        # dask ingest workers need to be few-and-fat rather than many-and-
        # small. With 96 workers the hats-import splitting stage crashed
        # on `30605 split stages did not complete successfully` from dask
        # OOM thrashing on the per-worker 5 GB default.
        ingest_workers=8,
    ),
    # gaia = full Gaia DR3 source catalog (~1.8B rows from 3386 GaiaSource
    # shards). 16 shards × ~211 shards each × Pool(96) keeps the scatter
    # wall under ~1h on preempt. ingest_workers=8 because the per-row
    # struct schema is big enough that 32+ dask workers OOM (same as the
    # desi/manga lesson).
    "gaia": dict(
        num_shards=16,
        num_processes=96,
        build_mem_mb=900_000,
        build_runtime_min=240,
        ingest_mem_mb=900_000,
        ingest_runtime_min=360,
        ingest_workers=8,
    ),
    # gaia_xp = XP-joined subset (~220M rows), same raw Gaia/ dir.
    "gaia_xp": dict(
        num_shards=8,
        num_processes=96,   # per-worker reads one (source, xp) HDF5 pair
        build_mem_mb=900_000,
        build_runtime_min=240,
        ingest_mem_mb=900_000,
        ingest_runtime_min=240,
        ingest_workers=8,
    ),
    "manga": dict(
        num_shards=8,
        num_processes=32,   # per-worker holds one IFU cube ~500 MB
        build_mem_mb=500_000,
        build_runtime_min=240,
        ingest_mem_mb=900_000,
        ingest_runtime_min=2880,  # 48h — fat IFU rows are slow to split/reduce
        # manga rows are ~500 MB each (native-shape IFU cubes + griz images
        # + DAP maps). With only ~10k objects, 2 workers × 450 GB each is
        # the safest config. 8 workers OOMed repeatedly — each reduce task
        # that touches a dense healpix pixel with multiple IFU cubes can
        # easily exceed 100 GB.
        ingest_workers=2,
    ),
    # hsc = HSC SSP PDR3 Deep/UltraDeep, 477k objects across ~600 (tract,
    # patch) groups. Each pool worker opens 5 calexp FITS HDUs (~150 MB
    # combined per band) for one patch and cuts out 160×160 stamps. With
    # 32 workers per shard the per-worker peak is ~5 GB. 8 shards × 75
    # patches each.
    "hsc": dict(
        num_shards=8,
        num_processes=32,
        build_mem_mb=400_000,
        build_runtime_min=240,
        ingest_mem_mb=400_000,
        ingest_runtime_min=180,
        ingest_workers=8,
    ),
}


def shard_done_file(name: str, idx: int) -> str:
    return f"{SHARDED_SCRATCH_ROOT}/{name}_sharded/.shard_{int(idx):03d}.done"


def sharded_scratch_dir(name: str) -> str:
    return f"{SHARDED_SCRATCH_ROOT}/{name}_sharded"


# Wildcard constraints keep the generic shard rule from accidentally
# matching paths for non-sharded datasets. Updated automatically when a
# dataset is added to SHARDED_DATASETS.
wildcard_constraints:
    shard_name = "|".join(SHARDED_DATASETS.keys()) if SHARDED_DATASETS else "_never_",
    shard_idx = r"\d{3}",


rule build_sharded_shard:
    """Generic scatter task: process ONE stride slice of a sharded dataset.

    Matches for any dataset listed in :data:`SHARDED_DATASETS`. Each sbatch
    job runs the per-dataset build script with ``--skip-ingest``, writes
    per-unit parquets to the shared scratch dir, and `touch`es the
    marker file that the gather rule waits on.
    """
    output:
        done = SHARDED_SCRATCH_ROOT + "/{shard_name}_sharded/.shard_{shard_idx}.done",
    input:
        script = lambda w: build_script(w.shard_name),
    params:
        scratch = lambda w: sharded_scratch_dir(w.shard_name),
        output_root = lambda w: f"{HATS_ROOT}/{w.shard_name}",
        num_shards = lambda w: SHARDED_DATASETS[w.shard_name]["num_shards"],
        num_processes = lambda w: SHARDED_DATASETS[w.shard_name]["num_processes"],
        # Cone + max-files args from the active config profile. Honoured by
        # every sharded build script so `profile=cosmos` actually produces a
        # cosmos slice instead of the full-sky scatter.
        cone_args = (
            f"--ra-center {RA_CENTER} --dec-center {DEC_CENTER} --radius {RADIUS}"
            if RA_CENTER is not None and DEC_CENTER is not None and RADIUS is not None
            else ""
        ),
        max_files_arg = f"--max-files {MAX_FILES}" if MAX_FILES is not None else "",
    resources:
        mem_mb = lambda w: SHARDED_DATASETS[w.shard_name]["build_mem_mb"],
        runtime = lambda w: SHARDED_DATASETS[w.shard_name]["build_runtime_min"],
        cpus_per_task = 96,
        slurm_partition = SHARDED_SLURM_PARTITION,
        qos = slurm_qos(SHARDED_SLURM_PARTITION),
    shell:
        "mkdir -p {params.scratch} && "
        "python -u -m scripts.{wildcards.shard_name}.build_parent_sample_hats "
        "--scratch-dir {params.scratch} "
        "{params.cone_args} {params.max_files_arg} "
        "--output-root {params.output_root} "
        "--num-shards {params.num_shards} "
        "--shard-idx {wildcards.shard_idx} "
        "--num-processes {params.num_processes} "
        "--skip-ingest && "
        "touch {output.done}"


# For each sharded dataset we generate one gather rule. We have to
# generate these individually (rather than a single wildcard gather rule)
# because the gather's ``input:`` list needs to know N shards at DAG-build
# time. The loop captures the dataset name + cfg via default args so all
# generated rules don't share a stale loop variable.
for _name, _cfg in SHARDED_DATASETS.items():

    rule:
        name: f"build_{_name}"
        input:
            shards = lambda w, _n=_name, _c=_cfg: [
                shard_done_file(_n, i) for i in range(_c["num_shards"])
            ],
            script = build_script(_name),
        output:
            marker = catalog_marker(_name),
        params:
            scratch = sharded_scratch_dir(_name),
            output_root = f"{HATS_ROOT}/{_name}",
            ingest_workers = _cfg["ingest_workers"],
            num_shards = _cfg["num_shards"],
            name = _name,
        resources:
            mem_mb = _cfg["ingest_mem_mb"],
            runtime = _cfg["ingest_runtime_min"],
            cpus_per_task = 96,
            # Gather jobs always run on the guaranteed (non-preempt) partition
            # so they can't get killed mid-HATS-write. A preempted gather leaves
            # a half-written catalog dir that the next run would have to clean
            # up. Scatter shards are preemptible because they're idempotent
            # and cheap to requeue.
            slurm_partition = SLURM_PARTITION,
            qos = slurm_qos(SLURM_PARTITION),
        shell:
            "python -u -m scripts.{params.name}.build_parent_sample_hats "
            "--scratch-dir {params.scratch} "
            "--output-root {params.output_root} "
            "--num-shards {params.num_shards} "
            "--shard-idx 0 "
            "--only-ingest "
            "--ingest-workers {params.ingest_workers}"


rule all:
    input:
        [catalog_marker(name) for name in PORTED],


rule build_allwise:
    output:
        marker = catalog_marker("allwise"),
    input:
        script = build_script("allwise"),
    params:
        cmd = build_command("allwise"),
    resources:
        mem_mb = 200_000,
        runtime = 480,            # minutes (8h)
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_apogee:
    output:
        marker = catalog_marker("apogee"),
    input:
        script = build_script("apogee"),
    params:
        cmd = build_command("apogee"),
    resources:
        mem_mb = 500_000,
        runtime = 1440,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_sdss:
    output:
        marker = catalog_marker("sdss"),
    input:
        script = build_script("sdss"),
    params:
        cmd = build_command("sdss"),
    resources:
        mem_mb = 500_000,
        runtime = 1440,           # 24h
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_twomass:
    output:
        marker = catalog_marker("twomass"),
    input:
        script = build_script("twomass"),
    params:
        cmd = build_command("twomass"),
    resources:
        mem_mb = 200_000,
        runtime = 480,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_sages:
    output:
        marker = catalog_marker("sages"),
    input:
        script = build_script("sages"),
    params:
        cmd = build_command("sages"),
    resources:
        mem_mb = 100_000,
        runtime = 240,            # 4h
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_galex:
    output:
        marker = catalog_marker("galex"),
    input:
        script = build_script("galex"),
    params:
        cmd = build_command("galex"),
    resources:
        mem_mb = 200_000,
        runtime = 480,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_galah:
    output:
        marker = catalog_marker("galah"),
    input:
        script = build_script("galah"),
    params:
        cmd = build_command("galah"),
    resources:
        mem_mb = 500_000,
        runtime = 1440,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_jwst:
    output:
        marker = catalog_marker("jwst"),
    input:
        script = build_script("jwst"),
    params:
        cmd = build_command("jwst"),
    resources:
        mem_mb = 500_000,
        runtime = 1440,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_gz10:
    output:
        marker = catalog_marker("gz10"),
    input:
        script = build_script("gz10"),
    params:
        cmd = build_command("gz10"),
    resources:
        mem_mb = 250_000,
        runtime = 480,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_btsbot:
    output:
        marker = catalog_marker("btsbot"),
    input:
        script = build_script("btsbot"),
    params:
        cmd = build_command("btsbot"),
    resources:
        mem_mb = 250_000,
        runtime = 480,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_desi_provabgs:
    output:
        marker = catalog_marker("desi_provabgs"),
    input:
        script = build_script("desi_provabgs"),
    params:
        # Cluster currently lacks the provabgs runtime dependency, so the
        # production path skips the optional best-fit augmentation until that
        # environment is restored.
        cmd = build_command("desi_provabgs") + " --skip-best-fit",
    resources:
        mem_mb = 250_000,
        runtime = 480,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_tess:
    output:
        marker = catalog_marker("tess"),
    input:
        script = build_script("tess"),
    params:
        cmd = build_command("tess"),
    resources:
        mem_mb = 100_000,
        runtime = 1440,           # 24h — opens 160k single-LC FITS serially
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_foundation:
    output:
        marker = catalog_marker("foundation"),
    input:
        script = build_script("foundation"),
    params:
        cmd = build_command("foundation"),
    resources:
        mem_mb = 50_000,
        runtime = 60,             # 1h — only ~180 SNANA ASCII files
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


# Remaining SN-Ia datasets share the SNANA ASCII parser in mmu.sn_ia_snana.
# Each is a thin wrapper script with the same shape as build_foundation;
# per-dataset size caps are all small (100-400 SNe) so they fit in a single
# node with minimal resources.
rule build_snls:
    output:
        marker = catalog_marker("snls"),
    input:
        script = build_script("snls"),
    params:
        cmd = build_command("snls"),
    resources:
        mem_mb = 50_000,
        runtime = 60,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_ps1_sne_ia:
    output:
        marker = catalog_marker("ps1_sne_ia"),
    input:
        script = build_script("ps1_sne_ia"),
    params:
        cmd = build_command("ps1_sne_ia"),
    resources:
        mem_mb = 50_000,
        runtime = 60,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_des_y3_sne_ia:
    output:
        marker = catalog_marker("des_y3_sne_ia"),
    input:
        script = build_script("des_y3_sne_ia"),
    params:
        cmd = build_command("des_y3_sne_ia"),
    resources:
        mem_mb = 50_000,
        runtime = 60,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_swift_sne_ia:
    output:
        marker = catalog_marker("swift_sne_ia"),
    input:
        script = build_script("swift_sne_ia"),
    params:
        cmd = build_command("swift_sne_ia"),
    resources:
        mem_mb = 50_000,
        runtime = 60,
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"


rule build_cosmos:
    output:
        marker = catalog_marker("cosmos"),
    input:
        script = build_script("cosmos"),
    params:
        cmd = build_command("cosmos"),
    resources:
        # 20 tiles × 5 FITS mosaics; Pool(4) workers each hold ~2–10 GB
        # of memory-mapped image data + cutout buffers.
        mem_mb   = 400_000,
        runtime  = 480,           # 8 h; large tiles may take ~20 min each
        cpus_per_task = 96,
        slurm_partition = SLURM_PARTITION,
        qos = slurm_qos(SLURM_PARTITION),
    shell:
        "{params.cmd}"
