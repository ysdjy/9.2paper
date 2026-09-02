"""Unit tests for the sweep record format and the Oracle acceptance logic. No Isaac Sim.

The acceptance conditions decide whether the paper's task is well posed, so they are tested
against synthetic landscapes where the right answer is known by construction: a landscape
where every hidden state needs the same force must be rejected for lack of discrimination,
one where the band covers the whole axis must be rejected for lack of selectivity, and so on.
"""

from __future__ import annotations

import pytest

from probe_drawer.analysis.oracle import AcceptanceThresholds, TaskCandidate, score_candidate, select_task_parameters
from probe_drawer.analysis.sweep import SweepDataset, SweepRecord, success_interval, xi_grid
from probe_drawer.envs import XI_FIELDS
from probe_drawer.evaluation import SuccessCriteria

CRITERIA = SuccessCriteria(goal_displacement=0.05, displacement_tolerance=0.01, velocity_tolerance=0.08)


def make_record(
    xi: tuple[float, float, float, float],
    peak_force: float,
    displacement: float,
    duration: float = 1.5,
    velocity: float = 0.02,
    valid: bool = True,
) -> SweepRecord:
    return SweepRecord(
        xi=dict(zip(XI_FIELDS, xi, strict=True)),
        peak_force=peak_force,
        duration=duration,
        final_displacement=displacement,
        final_velocity=velocity,
        peak_velocity=abs(velocity),
        peak_measured_force=peak_force,
        travel_fraction=displacement / 0.4,
        peak_lateral_drift=0.0005,
        peak_orientation_drift_deg=0.3,
        termination_reason="duration_completed",
        valid=valid,
        invalid_reasons=[] if valid else ["excessive_velocity"],
    )


def linear_landscape(
    num_states: int = 12,
    forces: tuple[float, ...] = tuple(round(1.0 + 0.25 * index, 2) for index in range(17)),
    slope: float = 0.02,
    offset: float = 0.0,
) -> SweepDataset:
    """A dataset where hidden state ``i`` needs force ``1 + i * 0.25`` to reach 50 mm.

    ``d = slope * (F - required_i)`` plus the goal, so each state has a band around its own
    required force and the bands march across the axis -- the shape a learnable, adaptation-
    requiring task has.
    """
    dataset = SweepDataset()
    for index in range(num_states):
        xi = (4.0 + index, 0.5 + 0.1 * index, 0.3 + 0.05 * index, 2.0 + index)
        required = 1.0 + 0.25 * index + offset
        for force in forces:
            displacement = CRITERIA.goal_displacement + slope * (force - required)
            dataset.extend([make_record(xi, force, displacement)])
    return dataset


class TestSweepRecord:
    def test_xi_vector_follows_the_canonical_order(self) -> None:
        record = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.05)
        assert record.xi_vector == (8.0, 2.0, 1.0, 6.0)

    def test_success_requires_position_velocity_and_validity(self) -> None:
        on_target = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.052, velocity=0.01)
        too_fast = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.052, velocity=0.2)
        off_target = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.09, velocity=0.01)
        invalid = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.052, velocity=0.01, valid=False)

        assert on_target.succeeds(CRITERIA)
        assert not too_fast.succeeds(CRITERIA)
        assert not off_target.succeeds(CRITERIA)
        assert not invalid.succeeds(CRITERIA)

    def test_round_trips_through_a_dict(self) -> None:
        record = make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.05)
        assert SweepRecord.from_dict(record.as_dict()) == record


class TestSweepDataset:
    def test_queries_and_index_agree(self) -> None:
        dataset = linear_landscape(num_states=4)
        key = dataset.xi_keys()[1]
        indexed = dataset.select(xi_key=key, duration=1.5)
        scanned = [row for row in dataset.records if row.xi_key == key and row.duration == 1.5]
        assert indexed == sorted(scanned, key=lambda row: (row.peak_force, row.duration))

    def test_index_is_invalidated_by_extend(self) -> None:
        dataset = linear_landscape(num_states=2)
        before = len(dataset.select(xi_key=dataset.xi_keys()[0], duration=1.5))
        dataset.extend([make_record(dataset.xi_keys()[0], 99.0, 0.05)])
        after = len(dataset.select(xi_key=dataset.xi_keys()[0], duration=1.5))
        assert after == before + 1

    def test_validity_rate_and_reason_counts(self) -> None:
        dataset = SweepDataset()
        dataset.extend([make_record((8.0, 2.0, 1.0, 6.0), 3.0, 0.05, valid=index % 2 == 0) for index in range(4)])
        assert dataset.validity_rate() == pytest.approx(0.5)
        assert dataset.invalid_reason_counts() == {"excessive_velocity": 2}

    def test_surface_shape_and_missing_points(self) -> None:
        dataset = linear_landscape(num_states=3)
        forces, durations, values = dataset.surface("final_displacement", dataset.xi_keys()[0])
        assert values.shape == (len(durations), len(forces))
        assert not values[0].mask.any() if hasattr(values[0], "mask") else True

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        dataset = linear_landscape(num_states=3)
        dataset.metadata["stage"] = "unit-test"
        loaded = SweepDataset.load(dataset.save(tmp_path / "sweep.json"))
        assert len(loaded) == len(dataset)
        assert loaded.metadata["stage"] == "unit-test"
        assert loaded.records[0] == dataset.records[0]


class TestXiGrid:
    def test_full_factorial_size_and_ratio_handling(self) -> None:
        grid = xi_grid((4.0, 8.0), (1.0, 2.0), (0.5, 1.0), (2.0, 6.0))
        assert len(grid) == 16
        for params in grid:
            assert params.joint_dynamic_friction <= params.joint_static_friction

    def test_rejects_a_ratio_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"friction_ratios must lie inside \[0, 1\]"):
            xi_grid((8.0,), (2.0,), (1.5,), (6.0,))

    def test_rejects_an_empty_axis(self) -> None:
        with pytest.raises(ValueError, match="masses must not be empty"):
            xi_grid((), (2.0,), (1.0,), (6.0,))


class TestSuccessInterval:
    def test_finds_a_contiguous_band_around_the_required_force(self) -> None:
        dataset = linear_landscape(num_states=6)
        report = success_interval(dataset, dataset.xi_keys()[2], CRITERIA, duration=1.5)

        assert report["any_success"]
        assert report["contiguous"]
        assert report["force_low"] <= report["best_force"] <= report["force_high"]
        # The band is where |d - d_goal| <= eps_d. With slope 0.02 m/N and eps_d 0.01 m the
        # analytic half-width is 0.5 N, so the grid points at +-0.5 N sit exactly on the
        # boundary -- and land just outside it once the arithmetic is done in binary
        # (|0.04 - 0.05| evaluates to 0.010000000000000009). The resolved band is therefore
        # the three inner grid points, 1.25 to 1.75 N.
        assert report["success_forces"] == [1.25, 1.5, 1.75]
        assert report["force_width"] == pytest.approx(0.5)

    def test_reports_no_success_when_the_goal_is_unreachable(self) -> None:
        dataset = linear_landscape(num_states=3)
        unreachable = SuccessCriteria(goal_displacement=10.0, displacement_tolerance=0.01, velocity_tolerance=0.08)
        report = success_interval(dataset, dataset.xi_keys()[0], unreachable, duration=1.5)
        assert not report["any_success"]
        assert report["success_forces"] == []


class TestAcceptance:
    def test_a_learnable_discriminating_landscape_is_accepted(self) -> None:
        score = score_candidate(linear_landscape(), TaskCandidate(1.5, 0.05, 0.015, 0.08))
        assert score.accepted, score.failures
        assert score.coverage == pytest.approx(1.0)
        assert score.discrimination > 0.5

    def test_a_landscape_needing_one_force_everywhere_is_rejected(self) -> None:
        """If every drawer needs the same force, a constant solves the task and no probe is needed."""
        dataset = linear_landscape(num_states=8, offset=0.0)
        # Collapse the requirement: rebuild with zero spread in the required force.
        flat = SweepDataset()
        for index, key in enumerate(dataset.xi_keys()):
            for force in dataset.forces():
                flat.extend([make_record(key, force, 0.05 + 0.02 * (force - 2.5))])
            _ = index
        score = score_candidate(flat, TaskCandidate(1.5, 0.05, 0.015, 0.08))
        assert not score.accepted
        assert any("discrimination" in failure for failure in score.failures)

    def test_an_over_tolerant_task_is_rejected_for_band_width(self) -> None:
        """A band covering most of the force axis means adaptation buys nothing."""
        score = score_candidate(
            linear_landscape(slope=0.0005), TaskCandidate(1.5, 0.05, 0.015, 0.08)
        )
        assert not score.accepted
        assert any("width" in failure for failure in score.failures)

    def test_a_loose_tolerance_relative_to_the_goal_is_rejected(self) -> None:
        score = score_candidate(linear_landscape(), TaskCandidate(1.5, 0.02, 0.015, 0.08))
        assert any("tolerance-ratio" in failure for failure in score.failures)

    def test_a_band_narrower_than_the_grid_step_is_flagged_not_accepted(self) -> None:
        """The remedy for an unresolved band is a finer sweep, so it must be distinguishable."""
        score = score_candidate(linear_landscape(slope=0.2), TaskCandidate(1.5, 0.05, 0.015, 0.08))
        assert not score.grid_resolves_band
        assert any("grid-resolution" in failure for failure in score.failures)

    def test_thresholds_are_reported_with_the_result(self) -> None:
        report = select_task_parameters(linear_landscape(), [TaskCandidate(1.5, 0.05, 0.015, 0.08)])
        assert set(report["thresholds"]) >= {"min_coverage", "min_discrimination", "max_tolerance_ratio"}

    def test_selection_prefers_the_most_discriminating_accepted_candidate(self) -> None:
        dataset = linear_landscape()
        candidates = [TaskCandidate(1.5, 0.05, 0.015, 0.08), TaskCandidate(1.5, 0.05, 0.0075, 0.08)]
        report = select_task_parameters(dataset, candidates)
        assert report["recommended"] is not None
        best = max(
            (score_candidate(dataset, candidate) for candidate in candidates),
            key=lambda score: (score.accepted, score.discrimination),
        )
        assert report["recommended"]["candidate"] == best.candidate.as_dict()

    def test_nothing_accepted_reports_why_instead_of_a_best_of_a_bad_bunch(self) -> None:
        impossible = TaskCandidate(1.5, 10.0, 0.015, 0.08)
        report = select_task_parameters(linear_landscape(), [impossible])
        assert report["recommended"] is None
        assert report["rejection_summary"]["failures_by_condition"]

    def test_custom_thresholds_are_honoured(self) -> None:
        strict = AcceptanceThresholds(min_coverage=1.01)
        score = score_candidate(linear_landscape(), TaskCandidate(1.5, 0.05, 0.015, 0.08), strict)
        assert not score.accepted
        assert any("coverage" in failure for failure in score.failures)
