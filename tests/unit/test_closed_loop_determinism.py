"""Unit tests for the batch-order determinism gates. No Isaac Sim required.

These gates decided that the D047 warm-up fix does not work, so what matters is that they
reject what they are supposed to reject: a comparison that silently covers different hidden
states, and a run whose probe, chosen force or headline number moved with the schedule.
"""

from __future__ import annotations

import pytest

from probe_drawer.analysis.closed_loop_determinism import (
    FORCE_AGREEMENT_FLOOR,
    MAX_PROBE_MEDIAN_MM,
    MAX_PROBE_P90_MM,
    MAX_REACH_DELTA_PP,
    compare_batch_orders,
)

METHODS = ("fixed force", "A linear (1 feature)", "B ridge (summary)", "ACE + PSP")


def report(probe_mm: dict[str, float], forces: dict[str, float], reach: dict[str, float]) -> dict:
    """A deployment report with only the fields the comparison reads."""
    rows = []
    for xi_id, displacement in probe_mm.items():
        for method in METHODS:
            rows.append(
                {
                    "method": method,
                    "seed": 0 if method == "ACE + PSP" else None,
                    "xi_id": xi_id,
                    "probe_displacement": displacement / 1000.0,
                    "chosen_force": forces[xi_id],
                }
            )
    return {"rows": rows, "methods": {name: {"reach_success_rate": rate} for name, rate in reach.items()}}


def pair(probe_shift_mm: float = 0.0, force_shift: float = 0.0, reach_shift_pp: float = 0.0):
    """Two reports over the same 40 hidden states, differing by the given amounts."""
    ids = [f"xi{index:03d}" for index in range(40)]
    base_probe = {key: 6.7 for key in ids}
    base_force = {key: 3.0 for key in ids}
    base_reach = {name: 0.9 for name in METHODS}
    moved_probe = {key: 6.7 + probe_shift_mm for key in ids}
    moved_force = {key: 3.0 + force_shift for key in ids}
    moved_reach = {name: 0.9 + reach_shift_pp / 100.0 for name in METHODS}
    return (
        report(base_probe, base_force, base_reach),
        report(moved_probe, moved_force, moved_reach),
    )


class TestAPerfectlyDeterministicRun:
    def test_two_identical_reports_pass_every_gate(self) -> None:
        first, second = pair()
        result = compare_batch_orders(first, second)
        assert result["passes"]
        assert result["probe"]["identical"] == 40
        assert result["probe"]["median_mm"] == pytest.approx(0.0)

    def test_it_reports_the_thresholds_it_used(self) -> None:
        """A verdict without its thresholds cannot be re-checked later."""
        first, second = pair()
        thresholds = compare_batch_orders(first, second)["thresholds"]
        assert thresholds["max_probe_median_mm"] == MAX_PROBE_MEDIAN_MM
        assert thresholds["max_probe_p90_mm"] == MAX_PROBE_P90_MM
        assert thresholds["force_agreement_floor"] == FORCE_AGREEMENT_FLOOR
        assert thresholds["max_reach_delta_pp"] == MAX_REACH_DELTA_PP


class TestEachGateRejectsItsOwnFailure:
    def test_a_shifted_probe_fails_the_probe_gate_alone(self) -> None:
        first, second = pair(probe_shift_mm=2 * MAX_PROBE_MEDIAN_MM)
        gates = compare_batch_orders(first, second)["gates"]
        assert not gates["probe_is_the_same_measurement"]
        assert gates["chosen_force_is_stable"] and gates["reported_result_is_stable"]

    def test_a_probe_shift_just_inside_the_threshold_passes(self) -> None:
        first, second = pair(probe_shift_mm=0.99 * MAX_PROBE_MEDIAN_MM)
        assert compare_batch_orders(first, second)["gates"]["probe_is_the_same_measurement"]

    def test_a_changed_force_fails_the_force_gate_alone(self) -> None:
        """Checked on deterministic methods, whose choice cannot legitimately move."""
        first, second = pair(force_shift=0.05)
        gates = compare_batch_orders(first, second)["gates"]
        assert not gates["chosen_force_is_stable"]
        assert gates["probe_is_the_same_measurement"] and gates["reported_result_is_stable"]

    def test_a_moved_reach_rate_fails_the_result_gate_alone(self) -> None:
        first, second = pair(reach_shift_pp=2 * MAX_REACH_DELTA_PP)
        gates = compare_batch_orders(first, second)["gates"]
        assert not gates["reported_result_is_stable"]
        assert gates["probe_is_the_same_measurement"] and gates["chosen_force_is_stable"]

    def test_a_reach_shift_just_inside_the_threshold_passes(self) -> None:
        first, second = pair(reach_shift_pp=0.99 * MAX_REACH_DELTA_PP)
        assert compare_batch_orders(first, second)["gates"]["reported_result_is_stable"]


class TestItRefusesAMeaninglessComparison:
    def test_different_populations_raise_rather_than_return_a_number(self) -> None:
        """Comparing different drawers would produce a confident, meaningless answer."""
        first, second = pair()
        second["rows"] = [row for row in second["rows"] if row["xi_id"] != "xi000"]
        with pytest.raises(ValueError, match="different hidden states"):
            compare_batch_orders(first, second)

    def test_a_learned_method_is_not_used_for_the_force_gate(self) -> None:
        """Its choice may legitimately move if its input moved; only closed-form fits may not."""
        first, second = pair()
        forces = compare_batch_orders(first, second)["forces"]
        assert "ACE + PSP" not in forces
        assert set(forces) == {"fixed force", "A linear (1 feature)", "B ridge (summary)"}
