"""Unit tests for the fixed-budget probe's selection rule. No Isaac Sim required.

The rule decides the paper's probe, so it is tested against the failure modes it exists to
prevent rather than only against a happy path: a probe that cannot move the stiff end, one
that has already performed a third of the task, one that leaves the task unsolvable, and a
tie between a short probe and a marginally better long one.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.fixed_probe_calibration import (
    MAX_INTRUSION,
    MIN_REACH_COVERAGE,
    TIE_FRACTION,
    FixedProbeCandidate,
    XiOutcome,
    score_candidate,
    select_candidate,
)
from probe_drawer.analysis.probe_features import PROBE_FEATURES

GOAL = 0.10


def row(
    required_force: float | None = 3.0,
    *,
    moved: bool = True,
    displacement: float = 0.008,
    velocity: float = 0.02,
    safety_aborted: bool = False,
    features: tuple[float, ...] | None = None,
) -> XiOutcome:
    return XiOutcome(
        hidden_state={"mass": 8.0, "static_friction": 1.5, "dynamic_friction": 1.0, "damping": 6.0},
        moved=moved,
        post_probe_displacement=displacement,
        post_probe_velocity=velocity,
        safety_aborted=safety_aborted,
        features=features if features is not None else (0.0,) * len(PROBE_FEATURES),
        required_force=required_force,
    )


def informative_rows(count: int = 24, noise: float = 0.05, seed: int = 0) -> list[XiOutcome]:
    """Rows whose first feature predicts the required force, so the readout has something to find."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        signal = 1.0 + 3.0 * index / (count - 1)
        features = (signal, *rng.normal(size=len(PROBE_FEATURES) - 1))
        rows.append(row(required_force=signal + rng.normal(scale=noise), features=features))
    return rows


def candidate(force: float = 4.5, duration: float = 0.4) -> FixedProbeCandidate:
    return FixedProbeCandidate(peak_force=force, duration=duration)


class TestCandidate:
    def test_label_identifies_both_numbers(self) -> None:
        assert candidate(4.5, 0.4).label == "F4.5N_H0.4s"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"peak_force": 0.0}, "peak_force must be > 0"),
            ({"duration": -0.1}, "duration must be > 0"),
        ],
    )
    def test_rejects_non_physical(self, kwargs: dict, match: str) -> None:
        args = {"peak_force": 4.5, "duration": 0.4}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FixedProbeCandidate(**args)


class TestGates:
    def test_a_clean_candidate_passes_all_four(self) -> None:
        outcome = score_candidate(candidate(), informative_rows(), GOAL)
        assert outcome.gates == {
            "safe": True,
            "responsive": True,
            "non_intrusive": True,
            "task_remains_solvable": True,
        }
        assert outcome.passed

    def test_one_safety_abort_fails_the_safety_gate(self) -> None:
        rows = informative_rows()
        rows[3] = row(required_force=2.0, safety_aborted=True)
        outcome = score_candidate(candidate(), rows, GOAL)
        assert not outcome.gates["safe"]
        assert not outcome.passed
        assert outcome.metrics["safety_aborts"] == 1

    def test_a_drawer_that_never_moved_fails_the_responsiveness_gate(self) -> None:
        """The stiff end is exactly what the probe is for; a constant identifies nothing."""
        rows = informative_rows()
        rows[0] = row(required_force=None, moved=False)
        outcome = score_candidate(candidate(), rows, GOAL)
        assert not outcome.gates["responsive"]
        assert outcome.metrics["moved_fraction"] < 1.0

    def test_a_probe_that_travels_a_third_of_the_goal_fails_the_intrusion_gate(self) -> None:
        rows = informative_rows()
        rows[5] = row(required_force=2.0, displacement=(MAX_INTRUSION + 0.01) * GOAL)
        outcome = score_candidate(candidate(), rows, GOAL)
        assert not outcome.gates["non_intrusive"]
        assert outcome.metrics["max_intrusion_fraction"] > MAX_INTRUSION

    def test_intrusion_just_inside_the_ceiling_passes(self) -> None:
        rows = informative_rows()
        rows[5] = row(required_force=2.0, displacement=0.99 * MAX_INTRUSION * GOAL)
        assert score_candidate(candidate(), rows, GOAL).gates["non_intrusive"]

    def test_intrusion_is_measured_relative_to_the_goal_not_absolutely(self) -> None:
        """The same 25 mm probe is modest against a 100 mm goal and gross against a 40 mm one."""
        rows = informative_rows()
        rows[5] = row(required_force=2.0, displacement=0.025)
        assert score_candidate(candidate(), rows, 0.10).gates["non_intrusive"]
        assert not score_candidate(candidate(), rows, 0.04).gates["non_intrusive"]

    def test_an_unsolvable_task_fails_the_coverage_gate_even_when_the_probe_is_gentle(self) -> None:
        """Gate 4 is not implied by the first three: a probe can be safe, responsive and small
        and still leave too many hidden states with no force that reaches the goal."""
        rows = informative_rows()
        unsolved = int(len(rows) * (1.0 - MIN_REACH_COVERAGE)) + 2
        for index in range(unsolved):
            rows[index] = row(required_force=None)
        outcome = score_candidate(candidate(), rows, GOAL)
        assert outcome.gates["safe"] and outcome.gates["responsive"] and outcome.gates["non_intrusive"]
        assert not outcome.gates["task_remains_solvable"]

    def test_gates_are_reported_individually_so_a_rejection_names_its_cause(self) -> None:
        rows = informative_rows()
        rows[0] = row(required_force=None, moved=False, safety_aborted=True)
        failed = [name for name, ok in score_candidate(candidate(), rows, GOAL).gates.items() if not ok]
        assert set(failed) == {"safe", "responsive"}


class TestScore:
    def test_an_informative_probe_scores_better_than_an_uninformative_one(self) -> None:
        informative = score_candidate(candidate(), informative_rows(noise=0.02), GOAL)
        rng = np.random.default_rng(3)
        scrambled = [
            row(required_force=r.required_force, features=tuple(rng.normal(size=len(PROBE_FEATURES))))
            for r in informative_rows(noise=0.02)
        ]
        uninformative = score_candidate(candidate(), scrambled, GOAL)
        assert informative.score < uninformative.score

    def test_the_readout_uses_only_hidden_states_that_have_a_required_force(self) -> None:
        """There is no target for an unsolved state, and inventing one would invent a label."""
        rows = informative_rows()
        rows[0] = row(required_force=None)
        rows[1] = row(required_force=None)
        assert score_candidate(candidate(), rows, GOAL).readout["n"] == len(rows) - 2

    def test_an_unfittable_candidate_scores_infinity_rather_than_nan(self) -> None:
        """So that ``min`` cannot silently pick it: nan compares false against everything."""
        rows = [row(required_force=None) for _ in range(12)]
        outcome = score_candidate(candidate(), rows, GOAL)
        assert outcome.score == float("inf")

    def test_the_required_force_spread_is_reported(self) -> None:
        """Evidence that adaptation is needed at all: one force cannot serve every xi."""
        metrics = score_candidate(candidate(), informative_rows(), GOAL).metrics
        assert metrics["required_force_min"] < metrics["required_force_max"]
        assert metrics["required_force_ratio"] > 1.0


class TestSelection:
    def test_the_lowest_rmse_wins_among_passing_candidates(self) -> None:
        good = score_candidate(candidate(4.5, 0.4), informative_rows(noise=0.01, seed=1), GOAL)
        poor = score_candidate(candidate(3.5, 0.4), informative_rows(noise=0.6, seed=2), GOAL)
        assert select_candidate([poor, good]) is good

    def test_a_failing_candidate_never_wins_however_good_its_score(self) -> None:
        """A gate failure is not a worse score; adopting one would hide it from the report."""
        rows = informative_rows(noise=0.001)
        rows[0] = row(required_force=None, moved=False)
        best_but_broken = score_candidate(candidate(5.5, 0.4), rows, GOAL)
        acceptable = score_candidate(candidate(4.5, 0.4), informative_rows(noise=0.3), GOAL)
        assert best_but_broken.score < acceptable.score
        assert select_candidate([best_but_broken, acceptable]) is acceptable

    def test_nothing_is_selected_when_every_candidate_fails_a_gate(self) -> None:
        broken = informative_rows()
        broken[0] = row(required_force=None, moved=False)
        outcomes = [score_candidate(candidate(f, 0.4), broken, GOAL) for f in (3.5, 4.5)]
        assert select_candidate(outcomes) is None

    def test_a_tie_goes_to_the_shorter_probe(self) -> None:
        """Probe time is cost and nothing else, so it breaks ties rather than deciding them."""
        rows = informative_rows(noise=0.05, seed=5)
        short = score_candidate(candidate(4.5, 0.4), rows, GOAL)
        long = score_candidate(candidate(4.5, 0.8), rows, GOAL)
        assert short.score == pytest.approx(long.score)
        assert select_candidate([long, short]) is short

    def test_a_clearly_better_long_probe_beats_a_short_one(self) -> None:
        """The tie-break must not become a preference for speed over information."""
        short = score_candidate(candidate(4.5, 0.4), informative_rows(noise=0.8, seed=6), GOAL)
        long = score_candidate(candidate(4.5, 0.8), informative_rows(noise=0.01, seed=7), GOAL)
        assert long.score < short.score * (1.0 - TIE_FRACTION)
        assert select_candidate([short, long]) is long

    def test_an_empty_field_selects_nothing(self) -> None:
        assert select_candidate([]) is None


class TestArguments:
    def test_no_hidden_states_is_an_error_not_an_empty_report(self) -> None:
        with pytest.raises(ValueError, match="at least one hidden state"):
            score_candidate(candidate(), [], GOAL)

    def test_a_non_positive_goal_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="goal_displacement must be > 0"):
            score_candidate(candidate(), informative_rows(), 0.0)
