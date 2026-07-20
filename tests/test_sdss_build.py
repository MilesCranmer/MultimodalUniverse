"""Fast unit tests for scripts/sdss/build_parent_sample_hats.py.

These do not touch the cluster — they use synthetic FITS plate files written
under a tmp dir, so the parsing/joining/Arrow-conversion code is exercised
without any I/O against real SDSS data.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits
from astropy.table import Table


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "sdss", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_sdss_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()


def _make_fake_specobj(path: str, n_objects: int = 6) -> None:
    """Write a tiny specObj-dr17.fits with the columns the script depends on.

    Two plates with three fibers each, all from the 'sdss' sub-survey.
    """
    rng = np.random.default_rng(0)
    cols = [
        fits.Column(name="SPECPRIMARY", format="B", array=np.ones(n_objects, dtype=np.uint8)),
        fits.Column(name="TARGETTYPE", format="8A", array=np.array(["SCIENCE "] * n_objects)),
        fits.Column(name="PLATEQUALITY", format="8A", array=np.array(["good    "] * n_objects)),
        fits.Column(name="SURVEY", format="6A", array=np.array(["sdss  "] * n_objects)),
        fits.Column(name="PLATE", format="J", array=np.array([266, 266, 266, 267, 267, 267])),
        fits.Column(name="MJD", format="J", array=np.array([51630, 51630, 51630, 51608, 51608, 51608])),
        fits.Column(name="FIBERID", format="J", array=np.array([1, 2, 3, 1, 2, 3])),
        fits.Column(name="SPECOBJID", format="22A", array=np.array([f"obj_{i:04d}" for i in range(n_objects)])),
        fits.Column(name="PLUG_RA", format="D", array=rng.uniform(0, 360, n_objects)),
        fits.Column(name="PLUG_DEC", format="D", array=rng.uniform(-90, 90, n_objects)),
        fits.Column(name="VDISP", format="E", array=rng.uniform(50, 300, n_objects).astype(np.float32)),
        fits.Column(name="VDISP_ERR", format="E", array=rng.uniform(1, 10, n_objects).astype(np.float32)),
        fits.Column(name="Z", format="E", array=rng.uniform(0, 1, n_objects).astype(np.float32)),
        fits.Column(name="Z_ERR", format="E", array=rng.uniform(0, 0.01, n_objects).astype(np.float32)),
        fits.Column(name="ZWARNING", format="J", array=np.zeros(n_objects, dtype=np.int32)),
        fits.Column(name="SPECTROFLUX", format="5E", array=rng.uniform(0, 100, (n_objects, 5)).astype(np.float32)),
        fits.Column(name="SPECTROFLUX_IVAR", format="5E", array=rng.uniform(0, 1, (n_objects, 5)).astype(np.float32)),
        fits.Column(name="SPECTROSYNFLUX", format="5E", array=rng.uniform(0, 100, (n_objects, 5)).astype(np.float32)),
        fits.Column(name="SPECTROSYNFLUX_IVAR", format="5E", array=rng.uniform(0, 1, (n_objects, 5)).astype(np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)


def _make_fake_plate(path: str, n_fibers: int = 4, n_pixels: int = 32) -> None:
    """Write a fake spPlate FITS file with the 5 HDUs the script reads.

    HDU 0: flux + log-lambda WCS in header
    HDU 1: ivar
    HDU 2: and_mask
    HDU 3: (unused) or_mask
    HDU 4: lsf_sigma
    """
    rng = np.random.default_rng(int(os.path.basename(path).split("-")[1]))
    flux = rng.normal(0, 1, (n_fibers, n_pixels)).astype(np.float32)
    ivar = rng.uniform(0.1, 1, (n_fibers, n_pixels)).astype(np.float32)
    and_mask = np.zeros((n_fibers, n_pixels), dtype=np.int32)
    or_mask = np.zeros((n_fibers, n_pixels), dtype=np.int32)
    lsf_sigma = np.ones((n_fibers, n_pixels), dtype=np.float32)

    primary = fits.PrimaryHDU(flux)
    primary.header["CRVAL1"] = 3.5
    primary.header["CD1_1"] = 0.0001
    primary.header["CRPIX1"] = 1
    fits.HDUList([
        primary,
        fits.ImageHDU(ivar),
        fits.ImageHDU(and_mask),
        fits.ImageHDU(or_mask),
        fits.ImageHDU(lsf_sigma),
    ]).writeto(path, overwrite=True)


@pytest.fixture
def fake_sdss_root(tmp_path):
    root = tmp_path / "SDSS"
    root.mkdir()
    _make_fake_specobj(str(root / "specObj-dr17.fits"))

    # Two plates: 0266/spPlate-0266-51630.fits and 0267/spPlate-0267-51608.fits
    for plate, mjd in [(266, 51630), (267, 51608)]:
        plate_dir = root / "sdss" / f"{plate:04d}"
        plate_dir.mkdir(parents=True)
        _make_fake_plate(str(plate_dir / f"spPlate-{plate:04d}-{mjd}.fits"))

    return str(root)


class TestSelectionFn:
    def test_keeps_clean_rows(self):
        cat = Table({
            "SPECPRIMARY": [1, 1, 0],
            "TARGETTYPE": ["SCIENCE ", "SCIENCE ", "SCIENCE "],
            "PLATEQUALITY": ["good    ", "good    ", "good    "],
        })
        mask = build.selection_fn(cat)
        assert mask.tolist() == [True, True, False]

    def test_drops_bad_quality(self):
        cat = Table({
            "SPECPRIMARY": [1, 1],
            "TARGETTYPE": ["SCIENCE ", "SCIENCE "],
            "PLATEQUALITY": ["good    ", "marginal"],
        })
        mask = build.selection_fn(cat)
        assert mask.tolist() == [True, False]


class TestFindPlateGroups:
    def test_returns_existing_plates(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root)
        assert len(groups) == 2
        for path, sub in groups:
            assert os.path.exists(path)
            assert "spPlate" in os.path.basename(path)

    def test_max_files(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root, max_files=1)
        assert len(groups) == 1

    def test_missing_specobj_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="specObj"):
            build.find_plate_groups(str(tmp_path))


class TestProcessPlate:
    def test_returns_table_with_spectra(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root)
        joined = build.process_plate(groups[0])
        assert "spectrum_flux" in joined.colnames
        assert "spectrum_lambda" in joined.colnames
        assert "spectrum_mask" in joined.colnames
        assert len(joined) > 0

    def test_spectrum_array_shape(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root)
        joined = build.process_plate(groups[0])
        flux = np.asarray(joined["spectrum_flux"])
        assert flux.ndim == 2
        assert flux.shape[0] == len(joined)


class TestPadToMaxLength:
    def test_no_padding_needed(self):
        a = {"spectrum_flux": np.zeros((2, 5), dtype=np.float32),
             "spectrum_ivar": np.zeros((2, 5), dtype=np.float32),
             "spectrum_lambda": np.zeros((2, 5), dtype=np.float32),
             "spectrum_lsf_sigma": np.zeros((2, 5), dtype=np.float32),
             "spectrum_mask": np.zeros((2, 5), dtype=bool)}
        b = {k: v.copy() for k, v in a.items()}
        out = build._pad_to_max_length([a, b])
        assert out["spectrum_flux"].shape == (4, 5)

    def test_pad_to_longer(self):
        a = {"spectrum_flux": np.ones((1, 3), dtype=np.float32),
             "spectrum_ivar": np.ones((1, 3), dtype=np.float32),
             "spectrum_lambda": np.array([[100.0, 200.0, 300.0]], dtype=np.float32),
             "spectrum_lsf_sigma": np.ones((1, 3), dtype=np.float32),
             "spectrum_mask": np.zeros((1, 3), dtype=bool)}
        b = {"spectrum_flux": np.ones((1, 5), dtype=np.float32),
             "spectrum_ivar": np.ones((1, 5), dtype=np.float32),
             "spectrum_lambda": np.array([[100.0, 200.0, 300.0, 400.0, 500.0]], dtype=np.float32),
             "spectrum_lsf_sigma": np.ones((1, 5), dtype=np.float32),
             "spectrum_mask": np.zeros((1, 5), dtype=bool)}
        out = build._pad_to_max_length([a, b])
        assert out["spectrum_flux"].shape == (2, 5)
        # First row's lambda padding should be -1 (the marker for padded pixels)
        assert (out["spectrum_lambda"][0, 3:] == -1).all()
        # First row's mask padding should be True (so padding is masked out)
        assert out["spectrum_mask"][0, 3:].all()


class TestBuildArrowTable:
    def test_minimal_schema(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root)
        joined_tables = [build.process_plate(g) for g in groups]
        padded = build._pad_to_max_length([
            {k: np.asarray(t[k]) for k in (
                "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
                "spectrum_lsf_sigma", "spectrum_mask")}
            for t in joined_tables
        ])
        combined = Table()
        for col in joined_tables[0].colnames:
            if col.startswith("spectrum_"):
                combined[col] = padded[col]
            else:
                combined[col] = np.concatenate([np.asarray(t[col]) for t in joined_tables])

        table = build._build_arrow_table(combined)
        names = table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "spectrum" in names
        assert "survey" in names
        assert "Z" in names
        assert "SPECTROFLUX_G" in names

    def test_object_id_string(self, fake_sdss_root):
        groups = build.find_plate_groups(fake_sdss_root)
        t = build.process_plate(groups[0])
        # build a one-plate combined table
        padded = build._pad_to_max_length([{
            k: np.asarray(t[k]) for k in (
                "spectrum_flux", "spectrum_ivar", "spectrum_lambda",
                "spectrum_lsf_sigma", "spectrum_mask")
        }])
        combined = Table()
        for col in t.colnames:
            if col.startswith("spectrum_"):
                combined[col] = padded[col]
            else:
                combined[col] = np.asarray(t[col])
        table = build._build_arrow_table(combined)
        assert table.schema.field("object_id").type == pa.string()
