"""Unit tests for the task-conditioning audit. No Isaac Sim required.

The audit's conclusion rests on one comparison -- how far the optimum moves against how wide
the band is -- so these tests pin the band definition, the shift, and the transfer test that
turns the ratio into a success rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.task_conditioning import band_of, summarise_task_conditioning

STEP = 0.10
TOL = 0.0075
RANGE = (0.5, 6.5)


def row(centre: float, slope: float = 25.0, forces: np.ndarray | None = None, valid=None) -> dict:
    r"""A synthetic drawer whose achieved displacement is affine in the force.

    ``d(F) = 0.10 + (F - centre) / slope`` -- so ``centre`` is exactly the force that hits
    100 mm, and ``slope`` newtons per metre sets how sharply displacement tracks force.
    """
    grid = np.arange(0.25, 9.05, STEP) if forces is None else forces
    displacement = 0.10 + (grid - centre) / slope
    return {
        "hidden_state": {"mass": 8.0, "static_friction": 1.5, "dynamic_friction": 0.9, "damping": 6.0},
        "forces": [float(f) for f in grid],
        "displacement": [float(d) for d in displacement],
        "valid": [True] * len(grid) if valid is None else list(valid),
    }


class TestTheBandDefinition:
    def test_a_single_succeeding_force_has_one_cell_of_width(self) -> None:
        band = band_of([1.0, 1.1, 1.2], [False, True, False], STEP)
        assert band["width"] == pytest.approx(STEP)
        assert band["centre"] == pytest.approx(1.1)
        assert band["components"] == 1

    def test_it_takes_the_widest_component_not_the_union(self) -> None:
        """A union's centre can sit in the gap and fail, which is the whole reason."""
        band = band_of([1.0, 1.1, 1.2, 1.3, 1.4], [True, False, True, True, False], STEP)
        assert band["low"] == pytest.approx(1.2)
        assert band["high"] == pytest.approx(1.3)
        assert band["components"] == 2, "the disconnection must still be reported"

    def test_nothing_reaching_gives_none_rather_than_a_zero_band(self) -> None:
        assert band_of([1.0, 1.1], [False, False], STEP) is None

    def test_a_run_touching_the_end_of_the_grid_is_closed(self) -> None:
        band = band_of([1.0, 1.1, 1.2], [False, True, True], STEP)
        assert band["cells"] == 2 and band["high"] == pytest.approx(1.2)


class TestScoringOneSweepAgainstSeveralGoals:
    def test_a_further_goal_needs_more_force(self) -> None:
        summary = summarise_task_conditioning([row(3.0)], [0.08, 0.10, 0.12], TOL, STEP, RANGE)
        centres = [summary["per_goal"][g]["band_centre"]["median"] for g in (0.08, 0.10, 0.12)]
        assert centres[0] < centres[1] < centres[2]

    def test_the_shift_matches_the_synthetic_slope(self) -> None:
        """25 N/m over 20 mm is 0.5 N, which is what the audit measured on the real drawer."""
        summary = summarise_task_conditioning([row(3.0, slope=25.0)], [0.10, 0.12], TOL, STEP, RANGE)
        shift = summary["shift"]["0.1->0.12"]["abs_delta_centre"]
        assert shift["median"] == pytest.approx(0.5, abs=0.06)

    def test_the_shift_over_band_ratio_is_the_decision_quantity(self) -> None:
        """Below 1 the old optimum still works; above 1 it cannot."""
        summary = summarise_task_conditioning([row(3.0, slope=25.0)], [0.10, 0.12], TOL, STEP, RANGE)
        assert summary["shift"]["0.1->0.12"]["shift_over_band_width"]["median"] > 1.0

    def test_band_and_shift_scale_together_so_the_ratio_is_slope_free(self) -> None:
        r"""The mechanism, and the reason the audit's conclusion is robust.

        For a locally affine response ``d = d0 + (F - c)/k`` the band is ``2*eps_d*k`` wide and
        the optimum moves ``delta_goal*k`` between goals, so the ratio is
        ``delta_goal / (2*eps_d)`` -- independent of ``k``. The drawer's dynamics cannot rescue
        a transfer; only the goal step relative to the tolerance window can.
        """
        ratios = []
        for k in (12.0, 25.0, 60.0):
            summary = summarise_task_conditioning([row(3.0, slope=k)], [0.10, 0.12], TOL, STEP, RANGE)
            ratios.append(summary["shift"]["0.1->0.12"]["shift_over_band_width"]["median"])
        expected = 0.02 / (2 * TOL)
        assert all(r == pytest.approx(expected, rel=0.35) for r in ratios), ratios
        assert max(ratios) - min(ratios) < 0.6, f"the ratio should barely move with k: {ratios}"

    def test_invalid_forces_cannot_be_counted_as_reaching(self) -> None:
        base = row(3.0)
        blocked = row(3.0, valid=[False] * len(base["forces"]))
        assert summarise_task_conditioning([blocked], [0.10], TOL, STEP, RANGE)["per_goal"][0.10][
            "solvable"
        ] == 0

    def test_forces_outside_the_action_range_are_excluded(self) -> None:
        """A band reachable only at 8 N is not reachable by Setting V1."""
        far = row(8.0)
        summary = summarise_task_conditioning([far], [0.10], TOL, STEP, (0.5, 6.5))
        assert summary["per_goal"][0.10]["solvable"] == 0
        wide = summarise_task_conditioning([far], [0.10], TOL, STEP, (0.5, 9.0))
        assert wide["per_goal"][0.10]["solvable"] == 1


class TestTransfer:
    def test_the_middle_goals_optimum_transfers_to_itself_perfectly(self) -> None:
        summary = summarise_task_conditioning(
            [row(3.0), row(2.0), row(4.0)], [0.08, 0.10, 0.12], TOL, STEP, RANGE
        )
        assert summary["transfer"][0.10]["success_rate"] == pytest.approx(1.0)

    def test_it_fails_elsewhere_when_the_shift_exceeds_the_band(self) -> None:
        summary = summarise_task_conditioning(
            [row(3.0, slope=25.0)], [0.08, 0.10, 0.12], TOL, STEP, RANGE
        )
        assert summary["transfer"][0.08]["success_rate"] == pytest.approx(0.0)
        assert summary["transfer"][0.12]["success_rate"] == pytest.approx(0.0)

    def test_it_succeeds_when_the_goal_step_fits_inside_the_tolerance_window(self) -> None:
        """The null result the audit was checking for, and it is a property of the *task*.

        Goals 4 mm apart sit well inside the +-7.5 mm window, so one force serves both and a
        multi-goal experiment would measure nothing. That the real 20 mm spacing fails is
        therefore a statement about the spacing, not about the drawer.
        """
        summary = summarise_task_conditioning(
            [row(3.0, slope=25.0)], [0.098, 0.100, 0.102], TOL, STEP, RANGE
        )
        assert summary["transfer"][0.098]["success_rate"] == pytest.approx(1.0)
        assert summary["transfer"][0.102]["success_rate"] == pytest.approx(1.0)
        assert summary["shift"]["0.098->0.1"]["shift_over_band_width"]["median"] < 1.0

    def test_states_unsolvable_at_either_goal_are_excluded_from_the_rate(self) -> None:
        """Counting them would blame the transfer for a state nothing could solve."""
        summary = summarise_task_conditioning(
            [row(3.0), row(50.0)], [0.10, 0.12], TOL, STEP, RANGE
        )
        assert summary["transfer"][0.12]["states_solvable_at_both"] == 1


class TestArguments:
    def test_no_states_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="at least one hidden state"):
            summarise_task_conditioning([], [0.10], TOL, STEP, RANGE)

    def test_mismatched_array_lengths_are_refused(self) -> None:
        broken = row(3.0)
        broken["valid"] = broken["valid"][:-1]
        with pytest.raises(ValueError, match="same length"):
            summarise_task_conditioning([broken], [0.10], TOL, STEP, RANGE)
