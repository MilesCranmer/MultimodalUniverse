"""Unit tests for mmu.cone (cone-cut helper used by every build script)."""

import numpy as np
import pytest

from mmu.cone import ConeCut, apply_cone_filter, cone_bounding_box


COSMOS_RA = 150.0
COSMOS_DEC = 2.0
COSMOS_RADIUS = 0.5


class TestConeCut:
    def test_all_none_is_inactive(self):
        c = ConeCut(None, None, None)
        assert not c.is_active

    def test_all_set_is_active(self):
        c = ConeCut(COSMOS_RA, COSMOS_DEC, COSMOS_RADIUS)
        assert c.is_active

    def test_partial_raises(self):
        with pytest.raises(ValueError):
            ConeCut(COSMOS_RA, COSMOS_DEC, None)
        with pytest.raises(ValueError):
            ConeCut(COSMOS_RA, None, COSMOS_RADIUS)
        with pytest.raises(ValueError):
            ConeCut(None, COSMOS_DEC, COSMOS_RADIUS)

    def test_zero_radius_raises(self):
        with pytest.raises(ValueError):
            ConeCut(COSMOS_RA, COSMOS_DEC, 0.0)

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError):
            ConeCut(COSMOS_RA, COSMOS_DEC, -1.0)

    def test_huge_radius_raises(self):
        with pytest.raises(ValueError):
            ConeCut(COSMOS_RA, COSMOS_DEC, 181.0)


class TestApplyConeFilter:
    def test_center_is_inside(self):
        mask = apply_cone_filter(
            np.array([COSMOS_RA]),
            np.array([COSMOS_DEC]),
            COSMOS_RA,
            COSMOS_DEC,
            COSMOS_RADIUS,
        )
        assert mask.tolist() == [True]

    def test_far_point_is_outside(self):
        mask = apply_cone_filter(
            np.array([200.0]),
            np.array([-30.0]),
            COSMOS_RA,
            COSMOS_DEC,
            COSMOS_RADIUS,
        )
        assert mask.tolist() == [False]

    def test_inside_and_outside_mix(self):
        ra = np.array([150.0, 150.1, 155.0, 150.0])
        dec = np.array([2.0, 2.0, 2.0, 5.0])
        mask = apply_cone_filter(ra, dec, COSMOS_RA, COSMOS_DEC, COSMOS_RADIUS)
        # 150/2 is the center → inside
        # 150.1/2 is ~0.1 deg from center → inside
        # 155/2 is ~5 deg from center → outside
        # 150/5 is ~3 deg from center → outside
        assert mask.tolist() == [True, True, False, False]

    def test_ra_wraparound_at_zero(self):
        """Points on either side of the RA=0 meridian should both be inside
        a cone centered at RA=0."""
        ra = np.array([359.5, 0.0, 0.5, 90.0])
        dec = np.array([0.0, 0.0, 0.0, 0.0])
        mask = apply_cone_filter(ra, dec, 0.0, 0.0, 1.0)
        assert mask.tolist() == [True, True, True, False]

    def test_polar_region(self):
        """Near the pole, a cone at dec=89 should include the pole itself."""
        ra = np.array([0.0, 90.0, 180.0, 270.0])
        dec = np.array([90.0, 90.0, 90.0, 90.0])
        mask = apply_cone_filter(ra, dec, 0.0, 89.0, 1.5)
        # All 4 "points" are actually the same point (the north pole),
        # which is 1° from the center (89°). Should all be inside.
        assert mask.all()

    def test_edge_precision(self):
        """A point exactly at the cone edge should be included (<=, not <)."""
        ra = np.array([150.0])
        dec = np.array([2.5])  # exactly 0.5° from (150, 2)
        mask = apply_cone_filter(ra, dec, COSMOS_RA, COSMOS_DEC, COSMOS_RADIUS)
        assert mask.tolist() == [True]

    def test_vectorized_input(self):
        """Should handle large arrays efficiently."""
        rng = np.random.default_rng(0)
        n = 10_000
        ra = rng.uniform(0, 360, n)
        dec = rng.uniform(-90, 90, n)
        mask = apply_cone_filter(ra, dec, COSMOS_RA, COSMOS_DEC, 10.0)
        assert mask.dtype == bool
        assert mask.shape == (n,)
        # Sanity: a 10° cone covers ~pi*(10 deg)^2 / (4*pi*sr*(180/pi)^2) ≈ 0.0076
        # of the sky. For uniform random, expect ~76 points. Loose bound.
        assert 10 < mask.sum() < 300

    def test_list_input(self):
        """Python list input should work (will be converted to ndarray)."""
        mask = apply_cone_filter(
            [150.0, 0.0],
            [2.0, 0.0],
            COSMOS_RA,
            COSMOS_DEC,
            COSMOS_RADIUS,
        )
        assert mask.tolist() == [True, False]


class TestConeBoundingBox:
    def test_cosmos_bbox(self):
        ra_min, ra_max, dec_min, dec_max = cone_bounding_box(
            COSMOS_RA, COSMOS_DEC, COSMOS_RADIUS
        )
        # Dec bounds should be [1.5, 2.5]
        assert np.isclose(dec_min, 1.5)
        assert np.isclose(dec_max, 2.5)
        # RA bounds should be roughly [149.5/cos(2.5), 150.5/cos(2.5)] which is
        # a bit wider than [149.5, 150.5]
        assert ra_min < 149.5
        assert ra_max > 150.5
        assert ra_max - ra_min < 1.1  # but not hugely wider

    def test_ra_wraparound(self):
        """A cone centered at RA=0 should produce ra_min > ra_max (indicating
        the box wraps around the meridian)."""
        ra_min, ra_max, _, _ = cone_bounding_box(0.0, 0.0, 1.0)
        assert ra_min > ra_max  # wraparound: [ra_min, 360) ∪ [0, ra_max]
        assert ra_min > 358.0
        assert ra_max < 2.0

    def test_polar_bbox_spans_all_ra(self):
        """A cone reaching the pole should span all RA."""
        ra_min, ra_max, dec_min, dec_max = cone_bounding_box(0.0, 89.5, 1.0)
        assert ra_min == 0.0
        assert ra_max == 360.0
        assert dec_max == 90.0

    def test_wide_equatorial_cone_returns_full_ra(self):
        """A cone wide enough that Δra = radius/cos(|dec|) exceeds 180°
        should return the full RA range instead of a bogus wraparound
        interval that would silently drop real data."""
        ra_min, ra_max, dec_min, dec_max = cone_bounding_box(0.0, 0.0, 80.0)
        assert ra_min == 0.0
        assert ra_max == 360.0
        assert dec_min == -80.0
        assert dec_max == 80.0
