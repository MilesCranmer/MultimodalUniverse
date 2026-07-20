"""Tests for the per-dataset HATS config registry."""

import pytest

from mmu.hats_configs import DATASETS, get_dataset, list_datasets


EXPECTED_NON_SKIPPED = {
    "sdss", "desi", "vipers", "galah", "apogee", "chandra",
    "gaia", "gaia_xp", "desi_provabgs",
    "legacysurvey", "hsc", "jwst", "btsbot", "gz10",
    "tess", "foundation", "snls",
    "ps1_sne_ia", "des_y3_sne_ia", "swift_sne_ia",
    "allwise", "twomass", "galex", "sages",
    "manga",
}
EXPECTED_SKIPPED = {
    "plasticc",  # raw dir empty on cluster
    "kepler",    # raw FITS not on cluster; only v1 HDF5 exists
}
VALID_MODALITIES = {"spectra", "image", "timeseries", "tabular", "ifu"}


class TestRegistry:
    def test_non_skipped_datasets_present(self):
        non_skipped = {n for n, c in DATASETS.items() if not c.skip}
        assert non_skipped == EXPECTED_NON_SKIPPED

    def test_skipped_datasets_present(self):
        skipped = {n for n, c in DATASETS.items() if c.skip}
        assert skipped == EXPECTED_SKIPPED

    def test_all_modalities_known(self):
        for name, cfg in DATASETS.items():
            assert cfg.modality in VALID_MODALITIES, (
                f"{name} has unknown modality {cfg.modality}"
            )

    def test_skipped_have_reason(self):
        for name, cfg in DATASETS.items():
            if cfg.skip:
                assert cfg.skip_reason, f"{name} skipped without reason"

    def test_raw_subdir_set(self):
        for name, cfg in DATASETS.items():
            assert cfg.raw_subdir, f"{name} missing raw_subdir"

    def test_raw_path_format(self):
        sdss = DATASETS["sdss"]
        assert sdss.raw_path.startswith("/mnt/ceph/users/polymathic/external_data/astro/")
        assert sdss.raw_path.endswith("/SDSS")


class TestHelpers:
    def test_get_dataset_known(self):
        cfg = get_dataset("sdss")
        assert cfg.modality == "spectra"
        assert cfg.name == "sdss"

    def test_get_dataset_unknown(self):
        with pytest.raises(KeyError):
            get_dataset("not_a_dataset")

    def test_get_dataset_skipped(self):
        with pytest.raises(ValueError, match="skipped"):
            get_dataset("plasticc")


class TestListDatasets:
    def test_list_all(self):
        all_ds = list_datasets()
        assert set(all_ds) == EXPECTED_NON_SKIPPED

    def test_list_by_modality(self):
        spectra = set(list_datasets("spectra"))
        assert "sdss" in spectra
        assert "desi" in spectra
        assert "legacysurvey" not in spectra

        images = set(list_datasets("image"))
        assert "legacysurvey" in images
        assert "hsc" in images
        assert "sdss" not in images

        tabular = set(list_datasets("tabular"))
        assert "allwise" in tabular
        assert "twomass" in tabular
