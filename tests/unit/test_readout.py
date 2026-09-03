"""Unit tests for the shared leave-one-out ridge readout. No Isaac Sim required.

The readout decides which probe design the project adopts, so the properties that matter are
the ones whose absence produced a wrong conclusion before: that it cannot reward a fit for
having more columns, that it reports RMSE next to R-squared, and that it drops rather than
imputes what a probe failed to measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.readout import MIN_ROWS, RIDGE_PENALTY, leave_one_out


def linear_problem(points: int = 40, features: int = 4, noise: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(points, features))
    weights = np.linspace(1.0, -1.0, features)
    return design, design @ weights + rng.normal(scale=noise, size=points)


class TestItMeasuresInformation:
    def test_a_linear_target_is_recovered(self) -> None:
        report = leave_one_out(*linear_problem())
        assert report["r2"] > 0.95
        assert report["rmse"] < 0.3

    def test_pure_noise_features_score_at_or_below_zero(self) -> None:
        """Leave-one-out, so uninformative columns cannot buy fit. In-sample they would."""
        design, target = linear_problem()
        rng = np.random.default_rng(99)
        report = leave_one_out(rng.normal(size=design.shape), target)
        assert report["r2"] < 0.1

    def test_more_useless_columns_do_not_improve_the_score(self) -> None:
        """The failure the ridge penalty exists for: 18 columns once scored R-squared = -82."""
        design, target = linear_problem()
        rng = np.random.default_rng(7)
        padded = np.hstack([design, rng.normal(size=(len(design), 14))])
        narrow = leave_one_out(design, target)
        wide = leave_one_out(padded, target)
        assert wide["r2"] <= narrow["r2"]
        assert wide["r2"] > -1.0, "a penalised fit must degrade gracefully, not explode"


class TestReportedQuantities:
    def test_r2_and_rmse_and_target_sd_all_come_back(self) -> None:
        """R-squared alone is normalised by the target's spread and misleads across subsets."""
        report = leave_one_out(*linear_problem())
        assert set(report) == {"r2", "rmse", "n", "target_sd"}
        assert report["target_sd"] > 0

    def test_r2_is_consistent_with_rmse_and_target_sd(self) -> None:
        report = leave_one_out(*linear_problem())
        assert report["r2"] == pytest.approx(1.0 - (report["rmse"] / report["target_sd"]) ** 2, abs=1e-9)

    def test_a_constant_target_has_no_variance_to_explain(self) -> None:
        design, _ = linear_problem()
        report = leave_one_out(design, np.full(len(design), 2.0))
        assert np.isnan(report["r2"])
        assert report["rmse"] == pytest.approx(0.0, abs=1e-6)


class TestMissingMeasurements:
    def test_non_finite_rows_are_dropped_not_imputed(self) -> None:
        design, target = linear_problem()
        broken = design.copy()
        broken[3, 1] = np.nan
        broken[7, 0] = np.inf
        assert leave_one_out(broken, target)["n"] == len(target) - 2

    def test_a_non_finite_target_is_dropped_too(self) -> None:
        design, target = linear_problem()
        target = target.copy()
        target[5] = np.nan
        assert leave_one_out(design, target)["n"] == len(target) - 1

    def test_too_few_surviving_rows_report_nan_rather_than_a_quotable_number(self) -> None:
        design, target = linear_problem(points=MIN_ROWS - 1)
        report = leave_one_out(design, target)
        assert np.isnan(report["r2"]) and np.isnan(report["rmse"])
        assert report["n"] == MIN_ROWS - 1


class TestArguments:
    def test_mismatched_lengths_are_rejected(self) -> None:
        design, target = linear_problem()
        with pytest.raises(ValueError, match="rows and target has"):
            leave_one_out(design, target[:-1])

    def test_a_negative_penalty_is_rejected(self) -> None:
        design, target = linear_problem()
        with pytest.raises(ValueError, match="penalty must be >= 0"):
            leave_one_out(design, target, penalty=-1.0)

    def test_the_default_penalty_is_the_shared_constant(self) -> None:
        """One value for every candidate: a per-candidate penalty would fit the comparison."""
        design, target = linear_problem()
        assert leave_one_out(design, target) == leave_one_out(design, target, penalty=RIDGE_PENALTY)

    def test_columns_need_not_share_units(self) -> None:
        """Standardised internally, so a feature in metres and one in newtons compare fairly."""
        design, target = linear_problem()
        scaled = design * np.array([1.0, 1000.0, 0.001, 1.0])
        assert leave_one_out(scaled, target)["r2"] == pytest.approx(leave_one_out(design, target)["r2"], abs=1e-6)
