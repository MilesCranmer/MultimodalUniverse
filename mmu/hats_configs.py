"""Per-dataset configuration for the raw -> HATS pipeline.

This is a thin registry that records:

- which datasets MMU v2 produces (by name),
- their modality (spectra/image/timeseries/tabular/ifu),
- the path to their raw inputs on the Flatiron cluster,
- which datasets are intentionally skipped.

The actual conversion logic lives in ``scripts/{dataset}/build_parent_sample_hats.py``.
This file just exists so the Snakefile, tests, and tooling have a single
source of truth for the dataset list.
"""

from dataclasses import dataclass


# Cluster filesystem locations.
RAW_DATA_ROOT = "/mnt/ceph/users/polymathic/external_data/astro"
MMU_V2_HATS_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats"


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata for one MMU dataset."""

    name: str
    modality: str  # spectra | image | timeseries | tabular | ifu
    raw_subdir: str  # relative to RAW_DATA_ROOT
    skip: bool = False
    skip_reason: str = ""

    @property
    def raw_path(self) -> str:
        return f"{RAW_DATA_ROOT}/{self.raw_subdir}"


# All datasets MMU v2 currently knows about.
# Verified against /mnt/ceph/users/polymathic/external_data/astro on 2026-04-07.
DATASETS: dict[str, DatasetConfig] = {
    # --- Spectra ---
    "sdss":         DatasetConfig("sdss", "spectra", "SDSS"),
    "desi":         DatasetConfig("desi", "spectra", "DESI_DR1"),
    "vipers":       DatasetConfig("vipers", "spectra", "VIPERS"),
    "galah":        DatasetConfig("galah", "spectra", "galah/dr3"),
    "apogee":       DatasetConfig("apogee", "spectra", "APOGEE/apogee_v2"),
    "chandra":      DatasetConfig("chandra", "spectra", "chandra"),
    # gaia = full Gaia DR3 source catalog, ~1.8B rows, no XP filter.
    # gaia_xp = the ~220M subset joined with XpContinuousMeanSpectrum
    # (same raw HDF5 dir; the two scripts read different files from it).
    "gaia":         DatasetConfig("gaia", "tabular", "Gaia"),
    "gaia_xp":      DatasetConfig("gaia_xp", "spectra", "Gaia"),
    "desi_provabgs": DatasetConfig("desi_provabgs", "tabular", "DESI_PROVABGS"),

    # --- Images ---
    "legacysurvey":     DatasetConfig("legacysurvey", "image", "legacysurvey"),
    "hsc":              DatasetConfig("hsc", "image", "hsc/pdr3_dud"),
    "jwst":             DatasetConfig("jwst", "image", "JWST"),
    "btsbot":           DatasetConfig("btsbot", "image", "btsbot"),
    "gz10":             DatasetConfig("gz10", "image", "Galaxy10_DECals.h5"),

    # --- Time series ---
    "plasticc":     DatasetConfig("plasticc", "timeseries", "PLAsTiCC",
                                  skip=True, skip_reason="raw dir empty on cluster"),
    "tess":         DatasetConfig("tess", "timeseries", "tess"),
    "kepler":       DatasetConfig("kepler", "timeseries", "Kepler",
                                  skip=True,
                                  skip_reason="raw Kepler FITS not mirrored on cluster; "
                                              "only v1-processed HDF5 exists at spoc/SPOC/"),
    "foundation":   DatasetConfig("foundation", "timeseries", "foundation"),
    "snls":         DatasetConfig("snls", "timeseries", "snls"),
    "ps1_sne_ia":   DatasetConfig("ps1_sne_ia", "timeseries", "ps1_sne_ia"),
    "des_y3_sne_ia": DatasetConfig("des_y3_sne_ia", "timeseries", "des_y3_sne_ia"),
    "swift_sne_ia": DatasetConfig("swift_sne_ia", "timeseries", "swift_sne_ia"),

    # --- Tabular ---
    "allwise":      DatasetConfig("allwise", "tabular", "allwise"),
    "twomass":      DatasetConfig("twomass", "tabular", "twomass/psc"),
    "galex":        DatasetConfig("galex", "tabular", "galex"),
    "sages":        DatasetConfig("sages", "tabular", "sages"),

    # --- IFU ---
    "manga":        DatasetConfig("manga", "ifu", "manga"),

    # --- COSMOS-Web ---
    # Raw data lives on the Flatiron n17/n03 servers, not under RAW_DATA_ROOT.
    # The build script uses its own --catalog-path / --nircam-root / --miri-root
    # args; raw_path here is a placeholder only.
    "cosmos": DatasetConfig("cosmos", "image", "COSMOS-Web"),
}


def get_dataset(name: str) -> DatasetConfig:
    """Look up a dataset by name. Raises KeyError if unknown, ValueError if skipped."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name}. Known: {sorted(DATASETS)}")
    cfg = DATASETS[name]
    if cfg.skip:
        raise ValueError(f"Dataset {name} is skipped: {cfg.skip_reason}")
    return cfg


def list_datasets(modality: str | None = None) -> list[str]:
    """List all non-skipped dataset names, optionally filtered by modality."""
    return sorted(
        name for name, cfg in DATASETS.items()
        if not cfg.skip and (modality is None or cfg.modality == modality)
    )
