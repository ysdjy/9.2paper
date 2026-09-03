"""Unit tests for the stratified OOD evaluation. No Isaac Sim required.

The strata come from the feasibility sweep, which ran before any model was evaluated. These
tests are mostly about that provenance being respected: membership must come from the sweep's
flags, an in-distribution report must be refused rather than silently mis-stratified, and the
force bias must keep its sign convention -- negative means the model asked for too little, which
is the whole diagnosis of a silent-probe failure.
"""

from __future__ import annotations

import pytest

from probe_drawer.analysis.ood_evaluation import STRATA, summarise_ood_evaluation

GAPS = (("teacher - ACE", "teacher", "ACE"), ("ACE - GRU", "ACE", "GRU"))
GOAL = 0.10


def row(
    method: str,
    xi: str,
    *,
    seed: int | None = 0,
    reach: bool = True,
    displacement: float = 0.10,
    force: float = 3.0,
    required: float | None = 3.0,
    moved: bool = True,
    feasible: bool = True,
) -> dict:
    return {
        "method": method,
        "seed": seed,
        "xi_id": xi,
        "reach_success": reach,
        "total_displacement": displacement,
        "chosen_force": force,
        "oracle_required_force": required,
        "probe_moved_in_sweep": moved,
        "reach_within_task_range": feasible,
    }


def report(rows: list[dict]) -> dict:
    return {"task": {"goal_displacement": GOAL}, "rows": rows}


class TestStrataComeFromTheSweep:
    def test_membership_uses_the_sweeps_flags_not_the_outcome(self) -> None:
        """A stratification derived from model scores would let one be flattered."""
        rows = [
            row("ACE", "a", moved=True, feasible=True, reach=False),
            row("ACE", "b", moved=False, feasible=True, reach=True),
            row("ACE", "c", moved=False, feasible=False, reach=False),
        ]
        strata = summarise_ood_evaluation(report(rows), GAPS)["strata"]
        assert strata["all"]["states"] == 3
        assert strata["oracle_feasible"]["states"] == 2
        assert strata["responsive"]["states"] == 1
        assert strata["no_breakaway"]["states"] == 2
        assert strata["no_breakaway_feasible"]["states"] == 1

    def test_every_declared_stratum_is_produced(self) -> None:
        rows = [row("ACE", "a"), row("ACE", "b", moved=False)]
        strata = summarise_ood_evaluation(report(rows), GAPS)["strata"]
        assert set(strata) == {name for name, _ in STRATA}

    def test_an_in_distribution_report_is_refused(self) -> None:
        """Its rows carry no sweep flags, so every state would land in one bucket silently."""
        plain = {"task": {"goal_displacement": GOAL}, "rows": [{"method": "ACE", "seed": 0}]}
        with pytest.raises(ValueError, match="no feasibility flags"):
            summarise_ood_evaluation(plain, GAPS)

    def test_an_empty_report_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no feasibility flags"):
            summarise_ood_evaluation({"task": {"goal_displacement": GOAL}, "rows": []}, GAPS)


class TestForceBias:
    def test_asking_for_too_little_is_negative(self) -> None:
        """The sign convention the silent-probe diagnosis rests on."""
        rows = [row("ACE", "a", force=2.0, required=4.0)]
        stats = summarise_ood_evaluation(report(rows), GAPS)["strata"]["all"]["methods"]["ACE"]
        assert stats["median_force_bias"] == pytest.approx(-2.0)
        assert stats["under_forced_fraction"] == pytest.approx(1.0)

    def test_asking_for_too_much_is_positive(self) -> None:
        rows = [row("ACE", "a", force=6.0, required=4.0)]
        stats = summarise_ood_evaluation(report(rows), GAPS)["strata"]["all"]["methods"]["ACE"]
        assert stats["median_force_bias"] == pytest.approx(2.0)
        assert stats["under_forced_fraction"] == pytest.approx(0.0)

    def test_states_with_no_required_force_are_excluded_not_zeroed(self) -> None:
        """An infeasible state has no requirement; treating it as 0 N would invent a bias."""
        rows = [row("ACE", "a", force=6.0, required=None, feasible=False)]
        stats = summarise_ood_evaluation(report(rows), GAPS)["strata"]["all"]["methods"]["ACE"]
        assert stats["median_force_bias"] is None
        assert stats["under_forced_fraction"] is None


class TestReportedQuantities:
    def test_position_error_is_measured_against_the_reports_goal(self) -> None:
        """The rows record where the drawer ended, not what it aimed at."""
        rows = [row("ACE", "a", displacement=0.12)]
        stats = summarise_ood_evaluation(report(rows), GAPS)["strata"]["all"]["methods"]["ACE"]
        assert stats["median_position_error_mm"] == pytest.approx(20.0)

    def test_reach_is_pooled_and_also_split_by_seed(self) -> None:
        rows = [
            row("ACE", "a", seed=0, reach=True),
            row("ACE", "b", seed=0, reach=False),
            row("ACE", "a", seed=1, reach=True),
            row("ACE", "b", seed=1, reach=True),
        ]
        stats = summarise_ood_evaluation(report(rows), GAPS)["strata"]["all"]["methods"]["ACE"]
        assert stats["reach_pp"] == pytest.approx(75.0)
        assert stats["reach_per_seed"] == {0: pytest.approx(50.0), 1: pytest.approx(100.0)}
        assert stats["reach_sd_across_seeds"] == pytest.approx(25.0)

    def test_gaps_are_computed_inside_each_stratum(self) -> None:
        """A gap averaged across strata would hide that it vanishes in one of them."""
        rows = [
            row("teacher", "a", moved=True, reach=True),
            row("ACE", "a", moved=True, reach=True),
            row("teacher", "b", moved=False, reach=True),
            row("ACE", "b", moved=False, reach=False),
        ]
        strata = summarise_ood_evaluation(report(rows), GAPS)["strata"]
        assert strata["responsive"]["gaps"]["teacher - ACE"] == pytest.approx(0.0)
        assert strata["no_breakaway"]["gaps"]["teacher - ACE"] == pytest.approx(100.0)

    def test_a_method_absent_from_a_stratum_produces_no_gap(self) -> None:
        rows = [row("ACE", "a", moved=False)]
        strata = summarise_ood_evaluation(report(rows), GAPS)["strata"]
        assert strata["responsive"]["gaps"] == {}

    def test_an_unknown_stratum_name_is_refused(self) -> None:
        from probe_drawer.analysis.ood_evaluation import _members  # noqa: PLC0415

        with pytest.raises(ValueError, match="unknown stratum"):
            _members([row("ACE", "a")], "made-up")
