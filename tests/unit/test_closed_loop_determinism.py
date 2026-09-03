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
    summarise_permutations,
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


def permutation_report(label: int, rates: dict[str, float], probe_mm: float = 6.7) -> dict:
    """A deployment report for one slot permutation, with the fields the summary reads."""
    ids = [f"xi{index:03d}" for index in range(20)]
    rows = [
        {
            "method": name,
            "seed": None,
            "xi_id": xi_id,
            "probe_displacement": probe_mm / 1000.0,
            "chosen_force": 3.0,
        }
        for name in rates
        for xi_id in ids
    ]
    return {
        "slot_permutation": label,
        "num_test_states": len(ids),
        "rows": rows,
        "methods": {
            name: {
                "reach_success_rate": rate,
                "median_position_error_mm": 2.0,
                "per_seed": {},
            }
            for name, rate in rates.items()
        },
    }


ORDERED = ("teacher (privileged)", "ACE + PSP", "D GRU (history)", "B ridge (summary)")


class TestSummarisingPermutations:
    def test_it_reports_mean_sd_and_the_observed_range(self) -> None:
        reports = [
            permutation_report(index, dict(zip(ORDERED, rates, strict=True)))
            for index, rates in enumerate([(0.98, 0.90, 0.80, 0.60), (0.98, 0.94, 0.84, 0.64)])
        ]
        ace = summarise_permutations(reports)["methods"]["ACE + PSP"]["reach_success_pp"]
        assert ace["mean"] == pytest.approx(92.0)
        assert ace["min"] == pytest.approx(90.0)
        assert ace["max"] == pytest.approx(94.0)
        assert ace["sd"] == pytest.approx(2.0)

    def test_gaps_are_differenced_within_each_run_before_aggregating(self) -> None:
        """The case that matters: both terms move together, so the gap is stable even though
        each method's own spread is large. Differencing the aggregates would show the same
        mean but would hide that the *gap* never varied."""
        reports = [
            permutation_report(index, dict(zip(ORDERED, rates, strict=True)))
            for index, rates in enumerate([(0.99, 0.85, 0.75, 0.55), (0.99, 0.95, 0.85, 0.65)])
        ]
        summary = summarise_permutations(reports)
        assert summary["methods"]["ACE + PSP"]["reach_success_pp"]["sd"] == pytest.approx(5.0)
        gap = summary["gaps"]["ACE + PSP - D GRU"]
        assert gap["mean"] == pytest.approx(10.0)
        assert gap["sd"] == pytest.approx(0.0), "the gap is identical in both runs"

    def test_the_ordering_claim_is_fixed_not_inferred(self) -> None:
        """Otherwise "the ordering held" would just restate whatever the data happened to be."""
        reports = [
            permutation_report(index, dict(zip(ORDERED, rates, strict=True)))
            for index, rates in enumerate([(0.98, 0.90, 0.80, 0.60), (0.98, 0.90, 0.80, 0.60)])
        ]
        ordering = summarise_permutations(reports)["ordering"]
        assert ordering["claim"] == list(ORDERED)
        assert ordering["held_everywhere"]

    def test_one_permutation_breaking_the_ordering_is_reported(self) -> None:
        reports = [
            permutation_report(index, dict(zip(ORDERED, rates, strict=True)))
            for index, rates in enumerate([(0.98, 0.90, 0.80, 0.60), (0.98, 0.78, 0.82, 0.60)])
        ]
        ordering = summarise_permutations(reports)["ordering"]
        assert ordering["held_per_permutation"] == [True, False]
        assert not ordering["held_everywhere"]

    def test_probe_displacement_is_pooled_over_every_pair(self) -> None:
        reports = [
            permutation_report(0, dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True)), probe_mm=6.7),
            permutation_report(1, dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True)), probe_mm=6.9),
            permutation_report(2, dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True)), probe_mm=6.8),
        ]
        probe = summarise_permutations(reports)["probe_displacement"]
        assert probe["pairs"] == 3
        assert probe["max_mm"] == pytest.approx(0.2, abs=1e-6)


class TestSummaryRefusesAMeaninglessSpread:
    def test_a_single_report_is_not_a_spread(self) -> None:
        report = permutation_report(0, dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True)))
        with pytest.raises(ValueError, match="at least two reports"):
            summarise_permutations([report])

    def test_the_same_permutation_twice_is_rejected(self) -> None:
        """Two runs of one permutation measure repeatability, not slot sensitivity."""
        rates = dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True))
        with pytest.raises(ValueError, match="duplicate slot permutations"):
            summarise_permutations([permutation_report(1, rates), permutation_report(1, rates)])

    def test_different_populations_are_rejected(self) -> None:
        rates = dict(zip(ORDERED, (0.98, 0.90, 0.80, 0.60), strict=True))
        first, second = permutation_report(0, rates), permutation_report(1, rates)
        second["rows"] = [row for row in second["rows"] if row["xi_id"] != "xi000"]
        with pytest.raises(ValueError, match="different hidden states"):
            summarise_permutations([first, second])
