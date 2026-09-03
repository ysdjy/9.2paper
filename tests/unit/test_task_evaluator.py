"""Unit tests for validity assessment and success labelling. No Isaac Sim required.

The evaluator reads nothing but a recorded ``ExecutionResult``, so it can be tested against
synthetic episodes -- which is the only way to exercise combinations (goal reached at high
speed, drift without a safety abort) that are hard to produce on demand in simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers import HISTORY_CHANNELS, SafetyLimits, TerminationReason
from probe_drawer.controllers.types import ExecutionResult, PullHistory
from probe_drawer.observations import OBSERVATION_SPECS, ChannelShape
from probe_drawer.evaluation import (
    DRAWER_TRAVEL_LIMIT,
    InvalidReason,
    OperatingRegionCfg,
    SuccessCriteria,
    assess_validity,
    evaluate_execution,
)

NUM_STEPS = 60
NUM_JOINTS = 7


def make_result(
    final_displacement: float = 0.15,
    final_velocity: float = 0.0,
    peak_velocity: float | None = None,
    lateral_drift: float = 0.0005,
    orientation_drift_deg: float = 0.3,
    termination_reason: TerminationReason = TerminationReason.DURATION_COMPLETED,
    active_steps: int = NUM_STEPS,
    non_finite: bool = False,
) -> ExecutionResult:
    """Build a synthetic single-environment execution with the given end state.

    The history is only as detailed as the validity checks need: a displacement ramp, a
    velocity trace whose maximum is controllable, and constant drift channels.
    """
    peak_velocity = final_velocity if peak_velocity is None else peak_velocity

    active = np.zeros((NUM_STEPS, 1), dtype=bool)
    active[:active_steps] = True

    velocity = np.zeros((NUM_STEPS, 1))
    velocity[:active_steps] = final_velocity
    velocity[active_steps // 2, 0] = peak_velocity

    displacement = np.linspace(0.0, final_displacement, NUM_STEPS).reshape(-1, 1)
    if non_finite:
        displacement[active_steps // 3, 0] = np.nan

    # Built from the channel registry rather than a hand-written argument list, so adding a
    # channel to PullHistory does not silently break every test in this file.
    trailing = {
        ChannelShape.SCALAR: (),
        ChannelShape.VEC3: (3,),
        ChannelShape.QUAT: (4,),
        ChannelShape.JOINTS: (NUM_JOINTS,),
    }
    channels = {
        name: np.zeros((NUM_STEPS, 1, *trailing[OBSERVATION_SPECS[name].shape]))
        for name in HISTORY_CHANNELS
    }
    ones = np.ones((NUM_STEPS, 1))
    channels.update(
        active=active,
        commanded_force=ones * 5.0,
        measured_force=ones * 5.0,
        drawer_position=displacement,
        drawer_velocity=velocity,
        drawer_velocity_raw=velocity,
        tcp_pull_axis_position=displacement,
        tcp_lateral_error=ones * lateral_drift,
        tcp_orientation_error=ones * np.radians(orientation_drift_deg),
    )
    history = PullHistory(time=np.arange(NUM_STEPS) / 60.0, **channels)
    return ExecutionResult(
        termination_reason=[termination_reason],
        duration=np.asarray([active_steps / 60.0]),
        final_displacement=np.asarray([final_displacement]),
        final_velocity=np.asarray([final_velocity]),
        peak_velocity=np.asarray([abs(peak_velocity)]),
        peak_commanded_force=np.asarray([5.0]),
        peak_measured_force=np.asarray([5.0]),
        safety_aborted=np.asarray([termination_reason is TerminationReason.SAFETY_ABORT]),
        history=history,
        parameters={"controller": "ExecutionPullController"},
    )


CRITERIA = SuccessCriteria(goal_displacement=0.15, displacement_tolerance=0.01, velocity_tolerance=0.02)


class TestValidity:
    def test_a_clean_episode_is_valid(self) -> None:
        report = assess_validity(make_result())
        assert report.verdicts[0].valid
        assert report.verdicts[0].reasons == []
        assert bool(report.valid[0])

    def test_a_safety_abort_is_never_valid(self) -> None:
        report = assess_validity(make_result(termination_reason=TerminationReason.SAFETY_ABORT))
        assert not report.verdicts[0].valid
        assert InvalidReason.SAFETY_ABORT in report.verdicts[0].reasons

    def test_reaching_the_mechanical_limit_is_invalid(self) -> None:
        cfg = OperatingRegionCfg()
        report = assess_validity(make_result(final_displacement=cfg.max_displacement + 0.01))
        assert InvalidReason.MECHANICAL_LIMIT in report.verdicts[0].reasons

    def test_the_mechanical_threshold_is_derived_from_the_measured_travel_limit(self) -> None:
        cfg = OperatingRegionCfg(mechanical_margin_fraction=0.8)
        assert cfg.max_displacement == pytest.approx(0.8 * DRAWER_TRAVEL_LIMIT)

    def test_excessive_velocity_is_invalid(self) -> None:
        cfg = OperatingRegionCfg()
        report = assess_validity(make_result(peak_velocity=cfg.max_peak_velocity + 0.1))
        assert InvalidReason.EXCESSIVE_VELOCITY in report.verdicts[0].reasons

    def test_excessive_lateral_drift_is_invalid(self) -> None:
        cfg = OperatingRegionCfg()
        report = assess_validity(make_result(lateral_drift=cfg.max_lateral_drift + 0.001))
        assert InvalidReason.EXCESSIVE_LATERAL_DRIFT in report.verdicts[0].reasons

    def test_excessive_orientation_drift_is_invalid(self) -> None:
        cfg = OperatingRegionCfg()
        report = assess_validity(make_result(orientation_drift_deg=cfg.max_orientation_drift_deg + 1.0))
        assert InvalidReason.EXCESSIVE_ORIENTATION_DRIFT in report.verdicts[0].reasons

    def test_an_immobile_drawer_is_invalid(self) -> None:
        report = assess_validity(make_result(final_displacement=1e-5))
        assert InvalidReason.NO_MEASURABLE_MOTION in report.verdicts[0].reasons

    def test_non_finite_history_is_invalid(self) -> None:
        report = assess_validity(make_result(non_finite=True))
        assert InvalidReason.NON_FINITE in report.verdicts[0].reasons

    def test_drift_after_an_environment_stopped_is_ignored(self) -> None:
        """Only the steps an environment was actually driven for count against it."""
        cfg = OperatingRegionCfg()
        result = make_result(active_steps=NUM_STEPS // 2, lateral_drift=cfg.max_lateral_drift * 0.5)
        result.history.tcp_lateral_error[NUM_STEPS // 2 :, 0] = cfg.max_lateral_drift * 10
        assert assess_validity(result).verdicts[0].valid

    def test_metrics_are_reported_alongside_the_verdict(self) -> None:
        verdict = assess_validity(make_result(final_displacement=0.2)).verdicts[0]
        assert verdict.metrics["final_displacement"] == pytest.approx(0.2)
        assert verdict.metrics["travel_fraction"] == pytest.approx(0.2 / DRAWER_TRAVEL_LIMIT)

    def test_assessment_is_deterministic(self) -> None:
        result = make_result(final_displacement=0.2, peak_velocity=0.1)
        first = assess_validity(result).as_dict()
        second = assess_validity(result).as_dict()
        assert first == second

    def test_rejects_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError, match="mechanical_margin_fraction"):
            OperatingRegionCfg(mechanical_margin_fraction=1.5)
        with pytest.raises(ValueError, match="max_peak_velocity must be > 0"):
            OperatingRegionCfg(max_peak_velocity=0.0)

    def test_thresholds_are_tighter_than_the_safety_limits(self) -> None:
        region, limits = OperatingRegionCfg(), SafetyLimits()
        assert region.max_lateral_drift < limits.max_lateral_error
        assert region.max_orientation_drift_deg < limits.max_orientation_error_deg
        assert region.max_peak_velocity < limits.max_drawer_velocity


class TestSuccess:
    def test_on_goal_and_at_rest_passes(self) -> None:
        report = evaluate_execution(make_result(final_displacement=0.152, final_velocity=0.005), CRITERIA)
        verdict = report.verdicts[0]
        assert verdict.success and verdict.displacement_ok and verdict.velocity_ok and verdict.valid

    def test_on_goal_but_still_moving_fails(self) -> None:
        """Passing through the goal at speed is not the same as being placed there (D020)."""
        report = evaluate_execution(make_result(final_displacement=0.15, final_velocity=0.2), CRITERIA)
        verdict = report.verdicts[0]
        assert verdict.displacement_ok
        assert not verdict.velocity_ok
        assert not verdict.success

    def test_off_goal_but_at_rest_fails(self) -> None:
        report = evaluate_execution(make_result(final_displacement=0.10, final_velocity=0.0), CRITERIA)
        verdict = report.verdicts[0]
        assert not verdict.displacement_ok
        assert verdict.velocity_ok
        assert not verdict.success

    def test_a_safety_abort_fails_even_if_the_goal_was_met(self) -> None:
        report = evaluate_execution(
            make_result(
                final_displacement=0.15,
                final_velocity=0.0,
                termination_reason=TerminationReason.SAFETY_ABORT,
            ),
            CRITERIA,
        )
        verdict = report.verdicts[0]
        assert verdict.displacement_ok and verdict.velocity_ok
        assert verdict.safety_aborted
        assert not verdict.success

    def test_an_invalid_operating_point_fails_even_if_the_goal_was_met(self) -> None:
        cfg = OperatingRegionCfg()
        criteria = SuccessCriteria(
            goal_displacement=cfg.max_displacement + 0.02, displacement_tolerance=0.01, velocity_tolerance=0.02
        )
        report = evaluate_execution(make_result(final_displacement=criteria.goal_displacement), criteria)
        verdict = report.verdicts[0]
        assert verdict.displacement_ok and verdict.velocity_ok
        assert not verdict.valid
        assert not verdict.success

    def test_displacement_error_is_signed(self) -> None:
        overshoot = evaluate_execution(make_result(final_displacement=0.18), CRITERIA).verdicts[0]
        undershoot = evaluate_execution(make_result(final_displacement=0.12), CRITERIA).verdicts[0]
        assert overshoot.displacement_error > 0
        assert undershoot.displacement_error < 0

    def test_the_tolerance_boundary_separates_pass_from_fail(self) -> None:
        """Just inside both tolerances passes, just outside either one fails.

        Exact equality is not asserted: ``0.15 + 0.01`` is not representable, so a test at
        the exact boundary would be measuring float arithmetic rather than the evaluator.
        """
        inside = evaluate_execution(
            make_result(
                final_displacement=CRITERIA.goal_displacement + 0.99 * CRITERIA.displacement_tolerance,
                final_velocity=0.99 * CRITERIA.velocity_tolerance,
            ),
            CRITERIA,
        )
        assert inside.verdicts[0].success

        outside_position = evaluate_execution(
            make_result(final_displacement=CRITERIA.goal_displacement + 1.01 * CRITERIA.displacement_tolerance),
            CRITERIA,
        )
        assert not outside_position.verdicts[0].displacement_ok

        outside_velocity = evaluate_execution(
            make_result(final_velocity=1.01 * CRITERIA.velocity_tolerance), CRITERIA
        )
        assert not outside_velocity.verdicts[0].velocity_ok

    def test_report_exposes_a_success_mask(self) -> None:
        report = evaluate_execution(make_result(), CRITERIA)
        assert report.success.dtype == bool
        assert report.success.shape == (1,)

    def test_report_serialises_criteria_and_validity(self) -> None:
        payload = evaluate_execution(make_result(), CRITERIA).as_dict()
        assert payload["criteria"] == CRITERIA.as_dict()
        assert "validity" in payload and "verdicts" in payload

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"goal_displacement": 0.0}, "goal_displacement must be > 0"),
            ({"displacement_tolerance": 0.0}, "displacement_tolerance must be > 0"),
            ({"velocity_tolerance": -0.1}, "velocity_tolerance must be > 0"),
        ],
    )
    def test_rejects_invalid_criteria(self, kwargs: dict, match: str) -> None:
        args = {"goal_displacement": 0.15, "displacement_tolerance": 0.01, "velocity_tolerance": 0.02}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            SuccessCriteria(**args)


class TestTheTwoSuccessDefinitions:
    """``reach_success`` and ``stable_success``, and the case that separates them (D046)."""

    def test_a_drawer_that_arrives_but_does_not_stop_reaches_without_being_stable(self) -> None:
        """The whole reason for the split.

        Under one combined label this episode and one that stopped 3 cm short are the same
        number, and the goal-distance sweep found that at 100 mm and beyond it is almost
        always *this* one. Reporting them together hid which half was failing.
        """
        report = evaluate_execution(make_result(final_displacement=0.15, final_velocity=0.2), CRITERIA)
        verdict = report.verdicts[0]
        assert verdict.reach_success
        assert not verdict.stable_success

    def test_a_drawer_that_stops_short_achieves_neither(self) -> None:
        verdict = evaluate_execution(make_result(final_displacement=0.10, final_velocity=0.0), CRITERIA).verdicts[0]
        assert not verdict.reach_success
        assert not verdict.stable_success

    def test_both_hold_when_the_drawer_is_placed_at_the_goal(self) -> None:
        verdict = evaluate_execution(make_result(final_displacement=0.152, final_velocity=0.005), CRITERIA).verdicts[0]
        assert verdict.reach_success and verdict.stable_success

    def test_stable_success_implies_reach_success(self) -> None:
        """Nested by construction, so no combination of the three booleans can invert them."""
        for displacement in (0.10, 0.148, 0.15, 0.152, 0.20):
            for velocity in (0.0, 0.01, 0.03, 0.2):
                verdict = evaluate_execution(
                    make_result(final_displacement=displacement, final_velocity=velocity), CRITERIA
                ).verdicts[0]
                assert not verdict.stable_success or verdict.reach_success

    def test_neither_label_survives_an_invalid_operating_point(self) -> None:
        """Validity gates the *primary* metric too -- reaching the goal unsafely is not a reach."""
        report = evaluate_execution(
            make_result(
                final_displacement=0.15, final_velocity=0.0, termination_reason=TerminationReason.SAFETY_ABORT
            ),
            CRITERIA,
        )
        verdict = report.verdicts[0]
        assert verdict.displacement_ok and verdict.velocity_ok
        assert not verdict.reach_success
        assert not verdict.stable_success

    def test_success_still_means_what_dataset_v0_recorded(self) -> None:
        """``success`` is the strict label. Redefining it would reinterpret 49,152 stored rows."""
        for displacement in (0.10, 0.15):
            for velocity in (0.005, 0.2):
                verdict = evaluate_execution(
                    make_result(final_displacement=displacement, final_velocity=velocity), CRITERIA
                ).verdicts[0]
                assert verdict.success == verdict.stable_success
                assert verdict.success == (verdict.displacement_ok and verdict.velocity_ok and verdict.valid)

    def test_the_report_exposes_both_masks(self) -> None:
        report = evaluate_execution(make_result(final_displacement=0.15, final_velocity=0.2), CRITERIA)
        assert report.reach_success.tolist() == [True]
        assert report.stable_success.tolist() == [False]
        assert report.success.tolist() == report.stable_success.tolist()

    def test_the_continuous_quantities_are_kept_alongside_the_flags(self) -> None:
        """A threshold can be revisited offline; a discarded measurement cannot."""
        payload = evaluate_execution(
            make_result(final_displacement=0.15, final_velocity=0.2), CRITERIA
        ).verdicts[0].as_dict()
        assert {"reach_success", "stable_success", "success"} <= set(payload)
        assert payload["terminal_velocity"] == pytest.approx(0.2)
        assert payload["displacement_error"] == pytest.approx(0.15 - CRITERIA.goal_displacement, abs=1e-12)
