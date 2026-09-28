"""Hand-rolled statistics: known-value checks against textbook chi-square."""

from __future__ import annotations

import math

import pytest

from forensiq._stats import chi2_sf, mean_std, percentile, zscore


class TestChiSquare:
    def test_known_critical_values_df1(self):
        # chi2 sf at the classic critical points: 3.841 -> 0.05, 6.635 -> 0.01
        assert chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-4)
        assert chi2_sf(6.634897, 1) == pytest.approx(0.01, abs=1e-4)

    def test_known_critical_values_df2(self):
        assert chi2_sf(9.210340, 2) == pytest.approx(0.01, abs=1e-4)
        assert chi2_sf(4.605170, 2) == pytest.approx(0.10, abs=1e-4)

    def test_zero_is_one_high_is_small(self):
        assert chi2_sf(0.0, 5) == 1.0
        assert chi2_sf(50.0, 5) < 0.001
        assert chi2_sf(50.0, 5) > 0.0

    def test_monotone_decreasing(self):
        vals = [chi2_sf(x, 3) for x in (0.5, 2.0, 7.8, 20.0)]
        assert vals == sorted(vals, reverse=True)

    def test_matches_closed_form_df2(self):
        # df=2 has closed form sf = exp(-x/2)
        for x in (1.0, 4.605, 9.0):
            assert chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-6)

    def test_invalid_df(self):
        with pytest.raises(ValueError):
            chi2_sf(1.0, 0)


class TestPrimitives:
    def test_percentile_interpolation(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3.0
        assert percentile([10, 20], 50) == 15.0
        assert percentile([7], 99) == 7.0
        assert percentile([], 50) == 0.0

    def test_mean_std(self):
        m, s = mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert m == 5.0
        assert s == pytest.approx(2.0)

    def test_mean_std_empty(self):
        assert mean_std([]) == (0.0, 0.0)

    def test_zscore_degenerate_spread(self):
        # all baseline values equal: identical value -> 0, any deviation -> huge
        assert zscore(5.0, 5.0, 0.0) == 0.0
        assert zscore(0.0, 5.0, 0.0) == -10.0
        assert zscore(9.0, 5.0, 0.0) == 10.0

    def test_zscore_normal(self):
        assert zscore(10.0, 5.0, 5.0) == pytest.approx(1.0)
