"""Unit tests for the OOD feasibility summary. No Isaac Sim required.

The distinction these tests protect is the one the report exists for: "no force reaches the
goal" and "no force *the task allows* reaches the goal" are different findings with different
remedies, and collapsing them would hide which one an OOD failure is.
"""

from __future__ import annotations

import pytest

from probe_drawer.analysis.ood_feasibility import summarise_ood_feasibility

ALLOWED = (0.5, 6.5)


def row(
    *,
    required: float | None,
    novel: tuple[str, ...] = ("mass_high",),
    band: float = 0.3,
    invalid: float = 0.1,
    aborts: int = 0,
    closest_error: float = 0.002,
) -> dict:
    """One swept state. ``required`` of ``None`` means no force reached the goal."""
    in_range = required is not None and ALLOWED[0] <= required <= ALLOWED[1]
    return {
        "reach_any_force": required is not None,
        "reach_within_task_range": in_range,
        "required_force": required,
        "band_width": band if required is not None else 0.0,
        "novel_axes": list(novel),
        "sampled_axes": {
            "mass": 8.0,
            "static_friction": 1.5,
            "dynamic_friction_ratio": 0.6,
            "damping": 6.0,
        },
        "probe_moved": True,
        "closest_position_error": closest_error,
        "closest_force": required if required is not None else 6.5,
        "invalid_fraction": invalid,
        "safety_aborts": aborts,
    }


class TestTheThreeOutcomesAreCountedSeparately:
    def test_solvable_truncated_and_unsolvable_are_distinguished(self) -> None:
        rows = [row(required=2.0), row(required=8.0), row(required=None)]
        counts = summarise_ood_feasibility(rows, ALLOWED)["counts"]
        assert counts["solvable_within_task_range"] == 1
        assert counts["solvable_only_outside_task_range"] == 1
        assert counts["unsolvable_at_any_force"] == 1
        assert counts["solvable_any_force"] == 2

    def test_the_two_fractions_differ_when_a_state_is_only_truncated(self) -> None:
        """Reporting one number would call a truncated state unsolvable, or the reverse."""
        counts = summarise_ood_feasibility([row(required=2.0), row(required=8.0)], ALLOWED)["counts"]
        assert counts["fraction_solvable_within_task_range"] == pytest.approx(0.5)
        assert counts["fraction_solvable_any_force"] == pytest.approx(1.0)

    def test_a_force_exactly_on_the_ceiling_counts_as_allowed(self) -> None:
        counts = summarise_ood_feasibility([row(required=ALLOWED[1])], ALLOWED)["counts"]
        assert counts["solvable_within_task_range"] == 1


class TestTruncationIsAttributed:
    def test_the_forces_the_truncated_states_need_are_reported(self) -> None:
        rows = [row(required=7.0), row(required=9.0), row(required=2.0)]
        truncation = summarise_ood_feasibility(rows, ALLOWED)["truncation"]
        assert truncation["states_needing_more_than_allowed"] == 2
        assert truncation["required_force_above_ceiling"]["min"] == pytest.approx(7.0)
        assert truncation["required_force_above_ceiling"]["max"] == pytest.approx(9.0)

    def test_no_truncation_reports_none_rather_than_zero(self) -> None:
        """Zero would read as "they need 0 N", which is a different statement."""
        truncation = summarise_ood_feasibility([row(required=2.0)], ALLOWED)["truncation"]
        assert truncation["states_needing_more_than_allowed"] == 0
        assert truncation["required_force_above_ceiling"] is None

    def test_states_pressed_against_each_end_of_the_range_are_counted(self) -> None:
        """If solvable states pile up at a bound, the bound is doing the limiting."""
        rows = [row(required=ALLOWED[0]), row(required=ALLOWED[1]), row(required=3.0)]
        truncation = summarise_ood_feasibility(rows, ALLOWED)["truncation"]
        assert truncation["solvable_states_at_the_floor"] == 1
        assert truncation["solvable_states_at_the_ceiling"] == 1


class TestFailuresAreLocated:
    def test_failure_rate_is_relative_to_how_often_an_axis_appears(self) -> None:
        """An axis novel in many states but failing in none is not where the trouble is."""
        rows = [
            row(required=2.0, novel=("mass_high",)),
            row(required=2.0, novel=("mass_high",)),
            row(required=None, novel=("static_friction_high",)),
        ]
        rates = summarise_ood_feasibility(rows, ALLOWED)["novel_axis_rates"]
        assert rates["mass_high"] == {"states": 2, "failed": 0, "failure_rate": 0.0}
        assert rates["static_friction_high"]["failure_rate"] == pytest.approx(1.0)

    def test_axes_are_ordered_worst_first(self) -> None:
        rows = [
            row(required=2.0, novel=("mass_high",)),
            row(required=None, novel=("damping_high",)),
        ]
        rates = summarise_ood_feasibility(rows, ALLOWED)["novel_axis_rates"]
        assert list(rates)[0] == "damping_high"

    def test_a_state_novel_on_two_axes_is_charged_to_both(self) -> None:
        rows = [row(required=None, novel=("mass_high", "damping_low"))]
        rates = summarise_ood_feasibility(rows, ALLOWED)["novel_axis_rates"]
        assert rates["mass_high"]["failed"] == 1
        assert rates["damping_low"]["failed"] == 1

    def test_every_failure_is_listed_with_enough_to_diagnose_it(self) -> None:
        rows = [row(required=None, closest_error=0.031), row(required=2.0)]
        failures = summarise_ood_feasibility(rows, ALLOWED)["failures"]
        assert len(failures) == 1
        assert failures[0]["closest_position_error_mm"] == pytest.approx(31.0)
        assert failures[0]["reach_any_force"] is False

    def test_a_truncated_state_is_listed_as_a_failure_too(self) -> None:
        """It failed the task as posed, even though the drawer is physically openable."""
        failures = summarise_ood_feasibility([row(required=8.0)], ALLOWED)["failures"]
        assert len(failures) == 1
        assert failures[0]["reach_any_force"] is True
        assert failures[0]["required_force"] == pytest.approx(8.0)


class TestSafetyAndArguments:
    def test_aborts_are_totalled_and_attributed_to_states(self) -> None:
        rows = [row(required=2.0, aborts=3), row(required=2.0, aborts=0)]
        safety = summarise_ood_feasibility(rows, ALLOWED)["safety"]
        assert safety["total_aborts"] == 3
        assert safety["states_with_any_abort"] == 1

    def test_no_states_is_an_error_not_an_empty_report(self) -> None:
        with pytest.raises(ValueError, match="at least one state"):
            summarise_ood_feasibility([], ALLOWED)
