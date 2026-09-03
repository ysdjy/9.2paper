"""Unit tests for the affine-in-goal baseline. No Isaac Sim required.

The pilot's conclusion is that a global slope suffices, so the tests that matter are the ones
that would catch it concluding that wrongly: the split must not leak, the baseline must be
given the correct reference force (that generosity is the point, and if it silently were not
the result would be pessimistic), and a drawer whose own slope is far from the global one must
actually fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.affine_goal_baseline import evaluate_affine_goal_baseline, per_state_slopes

STEP = 0.10
TOL = 0.0075
RANGE = (0.5, 6.5)
GOALS = [0.08, 0.10, 0.12]
GRID = np.arange(0.25, 9.05, STEP)


def row(centre_at_100: float, slope: float) -> dict:
    r"""A drawer whose achieved displacement is affine: ``d = 0.10 + (F - c)/slope``.

    ``slope`` is ``dF/dd`` in N/m, so it is exactly the per-drawer quantity the baseline is
    trying to replace with one global number.
    """
    displacement = 0.10 + (GRID - centre_at_100) / slope
    return {
        "hidden_state": {"mass": 8.0, "static_friction": 1.5, "dynamic_friction": 0.9, "damping": 6.0},
        "forces": [float(f) for f in GRID],
        "displacement": [float(d) for d in displacement],
        "valid": [True] * len(GRID),
    }


def evaluate(rows, calibration, held_out, goals=GOALS):
    return evaluate_affine_goal_baseline(
        rows, calibration, held_out, goals, 0.10, TOL, STEP, RANGE
    )


class TestSlopeRecovery:
    def test_a_drawers_own_slope_is_recovered(self) -> None:
        states = per_state_slopes([row(3.0, 25.0)], GOALS, TOL, STEP, RANGE)
        assert states[0]["slope"] == pytest.approx(25.0, rel=0.15)
        assert states[0]["goals_solved"] == 3

    def test_a_drawer_solvable_at_one_goal_has_no_slope(self) -> None:
        """A slope through one point is not a slope."""
        states = per_state_slopes([row(3.0, 25.0)], [0.10], TOL, STEP, RANGE)
        assert states[0]["slope"] is None

    def test_the_global_slope_is_the_calibration_mean(self) -> None:
        rows = [row(3.0, 20.0), row(2.5, 30.0), row(4.0, 25.0), row(3.5, 25.0)]
        result = evaluate(rows, [0, 1], [2, 3])
        assert result["k_global"] == pytest.approx(25.0, rel=0.15)
        assert result["calibration"]["states_with_a_slope"] == 2


class TestTheSplitIsHonest:
    def test_overlapping_subsets_are_refused(self) -> None:
        """Fitting the slope on part of its own test set would inflate the result."""
        rows = [row(3.0, 25.0), row(3.5, 25.0)]
        with pytest.raises(ValueError, match="subsets overlap"):
            evaluate(rows, [0, 1], [1])

    def test_only_calibration_drawers_inform_the_slope(self) -> None:
        """A wildly different held-out slope must not move k_global."""
        shared = [row(3.0, 25.0), row(3.2, 25.0)]
        tame = evaluate(shared + [row(3.4, 25.0)], [0, 1], [2])
        wild = evaluate(shared + [row(3.4, 80.0)], [0, 1], [2])
        assert tame["k_global"] == pytest.approx(wild["k_global"])

    def test_a_missing_reference_force_puts_a_drawer_outside_the_domain(self) -> None:
        """The baseline is a correction to F_100; with no F_100 there is nothing to correct,
        so the drawer is excluded rather than counted as a failure of the correction."""
        rows = [row(3.0, 25.0), row(3.2, 25.0), row(50.0, 25.0)]
        result = evaluate(rows, [0, 1], [2])
        assert result["held_out"][0.10]["states_with_a_reference"] == 0


class TestItIsGenerousOnPurpose:
    def test_the_reference_goal_is_reached_by_construction(self) -> None:
        """The baseline is handed the correct F_100, so the reference goal must be perfect --
        if it is not, the reference is not being passed through and every other number is
        pessimistic for the wrong reason."""
        rows = [row(3.0, 25.0), row(2.0, 22.0), row(4.0, 28.0), row(3.5, 24.0)]
        result = evaluate(rows, [0, 1], [2, 3])
        assert result["held_out"][0.10]["reach_rate"] == pytest.approx(1.0)

    def test_the_reference_goal_must_be_among_the_goals(self) -> None:
        rows = [row(3.0, 25.0), row(3.2, 25.0)]
        with pytest.raises(ValueError, match="not among the goals"):
            evaluate_affine_goal_baseline(rows, [0], [1], [0.08, 0.12], 0.10, TOL, STEP, RANGE)


class TestWhenAGlobalSlopeIsAndIsNotEnough:
    def test_a_homogeneous_population_transfers_perfectly(self) -> None:
        rows = [row(3.0 + 0.2 * index, 25.0) for index in range(6)]
        result = evaluate(rows, [0, 1, 2], [3, 4, 5])
        assert result["held_out"][0.08]["reach_rate"] == pytest.approx(1.0)
        assert result["held_out"][0.12]["reach_rate"] == pytest.approx(1.0)
        assert result["held_out"][0.12]["gap_to_oracle_pp"] == pytest.approx(0.0)

    def test_a_drawer_whose_slope_is_far_off_fails(self) -> None:
        """The mechanism: the force error is ``|k_i - k_global| * delta_goal``, and it fails
        once that exceeds the band half-width."""
        calibration = [row(3.0 + 0.1 * index, 25.0) for index in range(4)]
        outlier = row(3.0, 8.0)
        result = evaluate(calibration + [outlier], [0, 1, 2, 3], [4])
        assert result["held_out"][0.12]["reach_rate"] == pytest.approx(0.0)
        assert result["held_out"][0.12]["gap_to_oracle_pp"] > 0

    def test_the_gap_to_oracle_never_blames_the_baseline_for_an_unsolvable_goal(self) -> None:
        """If the Oracle cannot solve it either, the gap must be zero, not positive."""
        rows = [row(3.0, 25.0), row(3.2, 25.0), row(3.4, 25.0)]
        result = evaluate(rows, [0, 1], [2], goals=[0.10, 0.12])
        held = result["held_out"][0.12]
        assert held["gap_to_oracle_pp"] == pytest.approx((held["oracle_solvable"] - held["reached"]) / 1 * 100)


class TestReportedQuantities:
    def test_slope_error_is_measured_on_held_out_drawers_only(self) -> None:
        rows = [row(3.0, 25.0), row(3.2, 25.0), row(3.4, 12.0), row(3.6, 12.0)]
        result = evaluate(rows, [0, 1], [2, 3])
        assert result["slope_error"]["held_out_slope"]["n"] == 2
        assert result["slope_error"]["abs_error"]["median"] > 5.0

    def test_position_error_is_signed_and_also_absolute(self) -> None:
        rows = [row(3.0, 25.0), row(3.2, 25.0), row(3.4, 25.0)]
        result = evaluate(rows, [0, 1], [2])
        held = result["held_out"][0.08]
        assert held["position_error_mm"] is not None
        assert held["abs_position_error_mm"]["median"] >= abs(held["position_error_mm"]["median"]) - 1e-9

    def test_no_fittable_calibration_drawer_is_an_error(self) -> None:
        rows = [row(50.0, 25.0), row(3.0, 25.0)]
        with pytest.raises(ValueError, match="no slope to fit"):
            evaluate(rows, [0], [1])
