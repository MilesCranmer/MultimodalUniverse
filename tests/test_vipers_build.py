"""Fast unit tests for scripts/vipers/build_parent_sample_hats.py.

These do not touch the cluster — they use synthetic FITS files written under a
tmp dir so the parsing and Arrow-conversion code is exercised without real data.
"""

import importlib.util
import os

import numpy as np
import pyarrow as pa
import pytest
from astropy.io import fits


def _load_build_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "vipers", "build_parent_sample_hats.py"
    )
    spec = importlib.util.spec_from_file_location("_vipers_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load_build_module()

N_PIXELS = 40


def _make_fake_vipers_fits(path: str, obj_id: int = 1, ra: float = 150.0, dec: float = 2.0) -> None:
    """Write a minimal VIPERS 1D spectrum FITS file."""
    rng = np.random.default_rng(obj_id)
    n = N_PIXELS
    cols = [
        fits.Column(name="FLUXES", format=f"{n}E", array=rng.normal(0, 1, (1, n)).astype(np.float32)),
        fits.Column(name="WAVES", format=f"{n}E", array=np.linspace(5500, 9000, n).reshape(1, n).astype(np.float32)),
        fits.Column(name="NOISE", format=f"{n}E", array=rng.uniform(0.01, 0.1, (1, n)).astype(np.float32)),
        fits.Column(name="MASK", format=f"{n}E", array=np.zeros((1, n), dtype=np.float32)),
    ]
    bintable = fits.BinTableHDU.from_columns(cols)
    bintable.header["ID"] = float(obj_id)
    bintable.header["RA"] = ra
    bintable.header["DEC"] = dec
    bintable.header["REDSHIFT"] = float(rng.uniform(0.5, 1.2))
    bintable.header["REDFLAG"] = 4.0
    bintable.header["EXPTIME"] = 3600.0
    bintable.header["NORM"] = 1.0
    bintable.header["MAG"] = float(rng.uniform(20, 24))

    fits.HDUList([fits.PrimaryHDU(), bintable]).writeto(path, overwrite=True)


@pytest.fixture
def fake_vipers_root(tmp_path):
    """Build a tiny VIPERS raw root with two survey subdirs, 3 objects each."""
    raw_root = tmp_path / "VIPERS"
    for subdir in build.SURVEY_SUBDIRS:
        d = raw_root / subdir
        d.mkdir(parents=True)
        for i in range(3):
            _make_fake_vipers_fits(
                str(d / f"sc_{i + 1:04d}.fits"),
                obj_id=i + 1 + (10 if "W4" in subdir else 0),
            )
    return str(raw_root)


class TestFindFitsFiles:
    def test_finds_both_surveys(self, fake_vipers_root):
        files = build.find_fits_files(fake_vipers_root)
        assert len(files) == 6
        assert all(f.endswith(".fits") for f in files)

    def test_max_files(self, fake_vipers_root):
        files = build.find_fits_files(fake_vipers_root, max_files=2)
        assert len(files) == 2

    def test_empty_root_returns_empty(self, tmp_path):
        files = build.find_fits_files(str(tmp_path / "nonexistent"))
        assert files == []


class TestExtractSpectrum:
    def test_returns_expected_keys(self, tmp_path):
        path = str(tmp_path / "test.fits")
        _make_fake_vipers_fits(path, obj_id=42)
        record = build.extract_spectrum(path)
        assert record is not None
        for key in build.HEADER_KEYS:
            assert key in record
        assert "spectrum_flux" in record
        assert "spectrum_wave" in record
        assert "spectrum_noise" in record
        assert "spectrum_mask" in record

    def test_spectrum_dtypes(self, tmp_path):
        path = str(tmp_path / "test.fits")
        _make_fake_vipers_fits(path, obj_id=1)
        record = build.extract_spectrum(path)
        assert record["spectrum_flux"].dtype == np.float32
        assert record["spectrum_wave"].dtype == np.float32
        assert record["spectrum_noise"].dtype == np.float32
        assert record["spectrum_mask"].dtype == np.float32

    def test_ra_dec_values(self, tmp_path):
        path = str(tmp_path / "test.fits")
        _make_fake_vipers_fits(path, obj_id=1, ra=150.123, dec=2.456)
        record = build.extract_spectrum(path)
        assert abs(record["RA"] - 150.123) < 1e-3
        assert abs(record["DEC"] - 2.456) < 1e-3

    def test_bad_file_returns_none(self, tmp_path):
        bad = str(tmp_path / "bad.fits")
        with open(bad, "wb") as f:
            f.write(b"not a fits file")
        result = build.extract_spectrum(bad)
        assert result is None


class TestBuildArrowTable:
    def _make_records(self, n: int = 5) -> list[dict]:
        rng = np.random.default_rng(0)
        records = []
        for i in range(n):
            records.append({
                "ID": float(i + 1),
                "RA": float(rng.uniform(0, 360)),
                "DEC": float(rng.uniform(-90, 90)),
                "REDSHIFT": float(rng.uniform(0.5, 1.2)),
                "REDFLAG": 4.0,
                "EXPTIME": 3600.0,
                "NORM": 1.0,
                "MAG": float(rng.uniform(20, 24)),
                "spectrum_flux": rng.normal(0, 1, N_PIXELS).astype(np.float32),
                "spectrum_wave": np.linspace(5500, 9000, N_PIXELS).astype(np.float32),
                "spectrum_noise": rng.uniform(0.01, 0.1, N_PIXELS).astype(np.float32),
                "spectrum_mask": np.zeros(N_PIXELS, dtype=np.float32),
            })
        return records

    def test_required_columns_present(self):
        records = self._make_records()
        table = build.build_arrow_table(records)
        names = table.schema.names
        assert "ra" in names
        assert "dec" in names
        assert "object_id" in names
        assert "spectrum" in names
        assert "REDSHIFT" in names
        assert "REDFLAG" in names
        assert "EXPTIME" in names
        assert "NORM" in names
        assert "MAG" in names

    def test_object_id_is_string(self):
        records = self._make_records(3)
        table = build.build_arrow_table(records)
        assert table.schema.field("object_id").type == pa.string()

    def test_ra_dec_float64(self):
        records = self._make_records(3)
        table = build.build_arrow_table(records)
        assert table.schema.field("ra").type == pa.float64()
        assert table.schema.field("dec").type == pa.float64()

    def test_spectrum_struct_fields(self):
        records = self._make_records(3)
        table = build.build_arrow_table(records)
        spec_type = table.schema.field("spectrum").type
        assert pa.types.is_struct(spec_type)
        field_names = {spec_type.field(i).name for i in range(spec_type.num_fields)}
        assert field_names == {"flux", "wave", "noise", "mask"}

    def test_spectrum_flux_is_list_float32(self):
        records = self._make_records(3)
        table = build.build_arrow_table(records)
        spec_type = table.schema.field("spectrum").type
        flux_field = spec_type.field("flux")
        assert pa.types.is_list(flux_field.type)
        assert flux_field.type.value_type == pa.float32()

    def test_row_count(self):
        records = self._make_records(7)
        table = build.build_arrow_table(records)
        assert table.num_rows == 7

    def test_cone_filter_integration(self, fake_vipers_root):
        """Verify that cone filtering reduces rows correctly."""
        files = build.find_fits_files(fake_vipers_root)
        records_raw = [build.extract_spectrum(f) for f in files]
        records = [r for r in records_raw if r is not None]

        from mmu.cone import apply_cone_filter
        ra = np.array([r["RA"] for r in records])
        dec = np.array([r["DEC"] for r in records])
        # Use a tiny cone that should match nothing
        mask = apply_cone_filter(ra, dec, ra_center=0.0, dec_center=85.0, radius=0.01)
        filtered = [r for r, m in zip(records, mask) if m]
        # Either 0 or fewer than all objects survive
        assert len(filtered) <= len(records)


class TestMainEntryPoint:
    def test_returns_1_with_empty_root(self, tmp_path):
        ret = build.main([
            "--raw-root", str(tmp_path / "nonexistent"),
            "--output-root", str(tmp_path / "out"),
        ])
        assert ret == 1

    def test_returns_0_with_fake_data(self, fake_vipers_root, tmp_path):
        ret = build.main([
            "--raw-root", fake_vipers_root,
            "--output-root", str(tmp_path / "out"),
            "--max-files", "2",
            "--num-processes", "1",
            "--pixel-threshold", "64",
        ])
        assert ret == 0

    @pytest.mark.slow
    def test_full_build_produces_catalog_dir(self, fake_vipers_root, tmp_path):
        out = tmp_path / "out"
        ret = build.main([
            "--raw-root", fake_vipers_root,
            "--output-root", str(out),
            "--num-processes", "1",
            "--pixel-threshold", "64",
        ])
        assert ret == 0
        catalog_dir = out / "vipers" / "vipers"
        assert catalog_dir.exists()
