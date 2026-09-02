"""The dataset schema, and the splits that must not leak.

The leakage tests are the reason this module exists. Every other property here is cheap
bookkeeping; a leaking split silently inflates every number the paper reports, and it does so
without failing anything.
"""

from __future__ import annotations

import pytest

from probe_drawer.analysis.sweep import force_grid
from probe_drawer.dataset import (
    SPLIT_LEVELS,
    SplitCfg,
    TrainingSample,
    assert_no_leakage,
    candidate_id,
    model_input_fields,
    probe_id,
    split_samples,
    validate_probe_history,
    xi_id,
)
from probe_drawer.dataset.splits import NESTING
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_TASK

PROBE_TASK = RECOMMENDED_PROBE_TASK.as_dict()


def make_xi(mass: float = 8.0, static: float = 1.25, dynamic: float = 0.8, damping: float = 6.0) -> dict:
    return {"mass": mass, "static_friction": static, "dynamic_friction": dynamic, "damping": damping}


def make_sample(
    xi: dict,
    episode: int,
    force: float,
    *,
    success: bool = True,
    valid: bool = True,
    branch_index: int = 0,
) -> TrainingSample:
    probe = probe_id(xi, episode, PROBE_TASK)
    return TrainingSample(
        candidate_id=candidate_id(probe, force, MAIN_TASK.duration, MAIN_TASK.goal_displacement),
        probe_id=probe,
        xi_id=xi_id(xi),
        xi=xi,
        probe_history={"drawer_position": [0.0, 0.001, 0.003]},
        probe_summary={"duration": 0.5, "breakaway_force": 2.0},
        post_probe_state={"displacement": 0.0035, "velocity": 0.0002},
        candidate_peak_force=force,
        branch_index=branch_index,
        duration=MAIN_TASK.duration,
        goal_displacement=MAIN_TASK.goal_displacement,
        final_total_displacement=0.04,
        final_velocity=0.01,
        success=success,
        valid=valid,
    )


def make_dataset(num_states: int = 24, probes_per_state: int = 2, forces=(1.0, 1.5, 2.0)) -> list[TrainingSample]:
    """A dataset with the structure the real one has: many candidates share one probe."""
    return [
        make_sample(make_xi(mass=4.0 + index), episode, force, branch_index=position)
        for index in range(num_states)
        for episode in range(probes_per_state)
        for position, force in enumerate(forces)
    ]


class TestIdentifiers:
    def test_the_hidden_state_id_depends_only_on_the_four_values(self) -> None:
        assert xi_id(make_xi()) == xi_id(dict(reversed(list(make_xi().items()))))

    def test_a_different_hidden_state_gets_a_different_id(self) -> None:
        assert xi_id(make_xi(mass=8.0)) != xi_id(make_xi(mass=8.1))

    def test_an_incomplete_hidden_state_is_refused(self) -> None:
        partial = make_xi()
        del partial["damping"]
        with pytest.raises(ValueError, match="all four dimensions"):
            xi_id(partial)

    def test_repeats_of_one_drawer_are_separate_probes(self) -> None:
        """Otherwise two independent episodes would collapse into one group."""
        xi = make_xi()
        assert probe_id(xi, 0, PROBE_TASK) != probe_id(xi, 1, PROBE_TASK)

    def test_a_recalibrated_probe_gets_a_new_id(self) -> None:
        recalibrated = {**PROBE_TASK, "max_force": 7.0}
        assert probe_id(make_xi(), 0, PROBE_TASK) != probe_id(make_xi(), 0, recalibrated)

    def test_candidates_of_one_probe_differ_only_by_the_force(self) -> None:
        probe = probe_id(make_xi(), 0, PROBE_TASK)
        first = candidate_id(probe, 1.0, 1.5, 0.04)
        assert first != candidate_id(probe, 1.05, 1.5, 0.04)
        assert first == candidate_id(probe, 1.0, 1.5, 0.04)

    def test_the_same_row_judged_against_a_different_task_is_a_different_row(self) -> None:
        probe = probe_id(make_xi(), 0, PROBE_TASK)
        assert candidate_id(probe, 1.0, 1.5, 0.04) != candidate_id(probe, 1.0, 1.5, 0.05)


class TestSampleContract:
    def test_a_reset_row_is_not_a_training_sample(self) -> None:
        with pytest.raises(ValueError, match="sequential protocol"):
            TrainingSample(**{**make_sample(make_xi(), 0, 1.0).as_dict(), "protocol": "reset"})

    def test_the_post_probe_state_needs_both_position_and_velocity(self) -> None:
        payload = make_sample(make_xi(), 0, 1.0).as_dict()
        payload["post_probe_state"] = {"displacement": 0.003}
        with pytest.raises(ValueError, match="needs 'velocity'"):
            TrainingSample(**payload)

    def test_a_sample_round_trips_through_a_dict(self) -> None:
        sample = make_sample(make_xi(), 0, 2.0)
        assert TrainingSample.from_dict(sample.as_dict()) == sample

    def test_the_hidden_state_is_not_a_model_input(self) -> None:
        assert "xi" not in model_input_fields()
        assert "xi_id" not in model_input_fields()

    def test_the_label_is_not_a_model_input(self) -> None:
        for leak in ("final_total_displacement", "final_velocity", "success"):
            assert leak not in model_input_fields()

    def test_the_candidate_force_is_a_model_input(self) -> None:
        """The model is asked about a force, so it has to be able to see which one."""
        assert "candidate_peak_force" in model_input_fields()

    def test_the_branch_index_is_not_a_model_input(self) -> None:
        """It is generation bookkeeping. A model that read it would be reading an artefact of
        how the labels were produced, which a deployed robot has no analogue of."""
        assert "branch_index" not in model_input_fields()

    def test_the_branch_index_is_recorded(self) -> None:
        sample = make_sample(make_xi(), 0, 2.0, branch_index=17)
        assert sample.as_dict()["branch_index"] == 17
        assert TrainingSample.from_dict(sample.as_dict()).branch_index == 17

    def test_a_probe_history_of_deployable_channels_is_accepted(self) -> None:
        validate_probe_history({"drawer_position": [0.0], "commanded_force": [1.0]})

    def test_a_privileged_channel_in_the_probe_history_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be inputs"):
            validate_probe_history({"drawer_position": [0.0], "drawer_resistance_force": [1.0]})


class TestGroupedSplits:
    def test_splitting_per_row_is_not_offered(self) -> None:
        assert "candidate_id" not in SPLIT_LEVELS
        with pytest.raises(ValueError, match="leaks the probe"):
            SplitCfg(level="candidate_id")

    def test_a_hidden_state_split_does_not_leak(self) -> None:
        assert_no_leakage(split_samples(make_dataset()))

    def test_a_probe_split_does_not_leak_probes(self) -> None:
        assert_no_leakage(split_samples(make_dataset(), SplitCfg(level="probe_id")))

    def test_every_row_of_a_hidden_state_lands_in_one_subset(self) -> None:
        split = split_samples(make_dataset())
        for name, rows in (("train", split.train), ("val", split.val), ("test", split.test)):
            for sample in rows:
                assert split.groups[sample.xi_id] == name

    def test_all_three_subsets_are_populated(self) -> None:
        split = split_samples(make_dataset(num_states=40))
        counts = split.counts()["rows"]
        assert all(counts[name] > 0 for name in ("train", "val", "test")), counts

    def test_the_row_counts_add_up(self) -> None:
        rows = make_dataset()
        split = split_samples(rows)
        assert len(split.train) + len(split.val) + len(split.test) == len(rows)

    def test_the_split_is_stable_when_the_dataset_grows(self) -> None:
        """Adding hidden states must not move the existing ones between subsets.

        Without this, a model trained before the dataset grew can no longer be evaluated on
        the later test set, because part of it was in the earlier training set.
        """
        small = make_dataset(num_states=12)
        large = make_dataset(num_states=24)
        before = split_samples(small).groups
        after = split_samples(large).groups
        assert all(after[group] == subset for group, subset in before.items())

    def test_the_split_does_not_depend_on_row_order(self) -> None:
        rows = make_dataset()
        assert split_samples(rows).groups == split_samples(list(reversed(rows))).groups

    def test_a_different_salt_gives_a_different_partition(self) -> None:
        rows = make_dataset(num_states=40)
        assert split_samples(rows).groups != split_samples(rows, SplitCfg(salt="alternative")).groups

    def test_invalid_rows_are_kept_for_the_training_script_to_drop(self) -> None:
        """Dropping them here would hide how many were dropped."""
        rows = [make_sample(make_xi(mass=4.0 + index), 0, 1.0, valid=index % 2 == 0) for index in range(20)]
        split = split_samples(rows)
        assert sum(not sample.valid for group in (split.train, split.val, split.test) for sample in group) == 10

    def test_a_split_leaving_no_test_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leaves no test set"):
            SplitCfg(train_fraction=0.9, val_fraction=0.1)

    def test_a_hand_built_leaking_split_is_caught(self) -> None:
        """The detector has to fail on a leak, not only pass on clean splits."""
        rows = make_dataset(num_states=8)
        split = split_samples(rows)
        leaked = type(split)(
            train=split.train + split.test[:1],
            val=split.val,
            test=split.test,
            level=split.level,
            groups=split.groups,
        )
        with pytest.raises(AssertionError, match="the split leaks"):
            assert_no_leakage(leaked)

    def test_a_probe_split_reports_only_what_it_claims(self) -> None:
        """Two probes of one drawer may straddle a probe-level split; that is not a failure
        of the probe-level guarantee, and the checker must not pretend otherwise."""
        assert NESTING.index("probe_id") < NESTING.index("xi_id")
        rows = make_dataset(num_states=40, probes_per_state=4)
        split = split_samples(rows, SplitCfg(level="probe_id"))
        assert_no_leakage(split)
        straddling = {
            sample.xi_id
            for sample in split.train
            if sample.xi_id in {other.xi_id for other in split.test}
        }
        assert straddling, "expected a probe-level split to share hidden states, or the test proves nothing"


class TestForceGrid:
    def test_the_grid_includes_both_ends(self) -> None:
        grid = force_grid(1.0, 5.0, 0.1)
        assert grid[0] == 1.0
        assert grid[-1] == 5.0

    def test_the_spacing_is_exact_and_not_accumulated(self) -> None:
        """Repeated addition drifts; two sweeps must produce mergeable keys."""
        grid = force_grid(0.15, 5.0, 0.05)
        for index, value in enumerate(grid):
            assert value == pytest.approx(0.15 + index * 0.05, abs=1e-12)

    def test_the_phase10_supplements_merge_with_the_main_grid(self) -> None:
        """The dataset was assembled from three passes joined on exact force equality."""
        main = set(force_grid(1.0, 5.0, 0.1))
        low = set(force_grid(0.4, 0.9, 0.1))
        lower = set(force_grid(0.15, 0.35, 0.05))
        merged = sorted(main | low | lower)
        assert len(merged) == 52, merged
        assert merged[0] == 0.15
        assert merged[-1] == 5.0
        assert not (low & lower), "the supplements must not overlap each other"

    def test_a_single_point_grid_is_allowed(self) -> None:
        assert force_grid(2.0, 2.0, 0.1) == (2.0,)

    def test_a_non_positive_step_is_refused(self) -> None:
        with pytest.raises(ValueError, match="step must be > 0"):
            force_grid(1.0, 5.0, 0.0)

    def test_a_reversed_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            force_grid(5.0, 1.0, 0.1)
