"""Unit tests for causal differentiation, probe features and the selected experiment plan.

No Isaac Sim. The derivative estimators and the feature extraction are the pieces a deployed
policy would also have to run, so their causality and their channel provenance are asserted
here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from probe_drawer.analysis.probe_features import (
    BREAKAWAY_SPEED,
    PROBE_FEATURES,
    assert_features_are_deployable,
    extract_features,
    rank_correlation,
)
from probe_drawer.controllers import HISTORY_CHANNELS, TerminationReason
from probe_drawer.controllers.types import ProbeResult, PullHistory
from probe_drawer.experiment_plan import (
    MAIN_TASK,
    OOD_XI_RANGES,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    TRAINING_XI_RANGES,
)
from probe_drawer.observations import OBSERVATION_SPECS, ChannelShape
from probe_drawer.sensors import CausalDerivative

NUM_JOINTS = 7


class TestCausalDerivative:
    def test_differentiates_a_ramp_exactly(self) -> None:
        derivative = CausalDerivative(dt=0.1, window=1)
        for step in range(5):
            derivative.update(torch.tensor([2.0 * step * 0.1]))
        assert float(derivative.filtered) == pytest.approx(2.0)
        assert float(derivative.raw) == pytest.approx(2.0)

    def test_is_zero_before_two_samples(self) -> None:
        derivative = CausalDerivative(dt=0.1)
        derivative.update(torch.zeros(3))
        assert torch.equal(derivative.filtered, torch.zeros(3))
        assert torch.equal(derivative.raw, torch.zeros(3))

    def test_moving_average_cancels_a_two_step_alternation(self) -> None:
        """This is exactly the artefact the drawer's contact chatter produces (D009)."""
        derivative = CausalDerivative(dt=1.0, window=2)
        # Position advancing by 3 then 1 each step: mean rate 2, alternating raw rate.
        position = 0.0
        for step in range(6):
            position += 3.0 if step % 2 == 0 else 1.0
            derivative.update(torch.tensor([position]))
        assert float(derivative.filtered) == pytest.approx(2.0)
        assert float(derivative.raw) in (pytest.approx(1.0), pytest.approx(3.0))

    def test_uses_only_past_samples(self) -> None:
        """A causal estimator's output must not change when a *later* sample arrives."""
        derivative = CausalDerivative(dt=0.1, window=2)
        for value in (0.0, 1.0, 2.0):
            derivative.update(torch.tensor([value]))
        before = float(derivative.filtered)
        derivative.update(torch.tensor([100.0]))
        assert float(derivative.filtered) != before  # the new sample affects the new output
        # ...but the earlier output was computed without it, which is what causality means.
        replay = CausalDerivative(dt=0.1, window=2)
        for value in (0.0, 1.0, 2.0):
            replay.update(torch.tensor([value]))
        assert float(replay.filtered) == pytest.approx(before)

    def test_reset_clears_history(self) -> None:
        derivative = CausalDerivative(dt=0.1)
        derivative.update(torch.ones(2))
        derivative.update(torch.full((2,), 3.0))
        derivative.reset()
        assert derivative.value is None
        with pytest.raises(RuntimeError, match="no samples yet"):
            _ = derivative.raw

    def test_handles_vector_and_joint_shaped_signals(self) -> None:
        derivative = CausalDerivative(dt=0.5, window=2)
        for step in range(4):
            derivative.update(torch.full((3, NUM_JOINTS), float(step)))
        assert derivative.filtered.shape == (3, NUM_JOINTS)
        assert torch.allclose(derivative.filtered, torch.full((3, NUM_JOINTS), 2.0))

    def test_describes_its_own_filter(self) -> None:
        description = CausalDerivative(dt=1 / 60, window=4).describe()
        assert description["causal"] is True
        assert description["window_steps"] == 4
        assert description["lag_steps"] == pytest.approx(1.5)

    @pytest.mark.parametrize(("kwargs", "match"), [({"dt": 0.0}, "dt must be > 0"), ({"window": 0}, "window must be")])
    def test_rejects_invalid_configuration(self, kwargs: dict, match: str) -> None:
        args = {"dt": 0.1, "window": 2}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            CausalDerivative(**args)


def make_probe(
    breakaway_step: int = 10,
    num_steps: int = 40,
    speed: float = 0.02,
    termination: TerminationReason = TerminationReason.DISPLACEMENT_REACHED,
) -> ProbeResult:
    """A synthetic single-environment probe: still, then sliding at ``speed``."""
    trailing = {
        ChannelShape.SCALAR: (),
        ChannelShape.VEC3: (3,),
        ChannelShape.QUAT: (4,),
        ChannelShape.JOINTS: (NUM_JOINTS,),
    }
    channels = {
        name: np.zeros((num_steps, 1, *trailing[OBSERVATION_SPECS[name].shape])) for name in HISTORY_CHANNELS
    }
    time = np.arange(1, num_steps + 1) / 60.0
    velocity = np.zeros((num_steps, 1))
    velocity[breakaway_step:] = speed
    displacement = np.cumsum(velocity, axis=0) / 60.0

    channels.update(
        active=np.ones((num_steps, 1), dtype=bool),
        commanded_force=(2.0 + 8.0 * time).reshape(-1, 1),
        drawer_velocity=velocity,
        drawer_position=displacement,
        drawer_acceleration=np.gradient(velocity[:, 0]).reshape(-1, 1) * 60.0,
    )
    history = PullHistory(time=time, **channels)
    return ProbeResult(
        termination_reason=[termination],
        duration=np.asarray([time[-1]]),
        final_displacement=np.asarray([displacement[-1, 0]]),
        final_velocity=np.asarray([velocity[-1, 0]]),
        final_commanded_force=np.asarray([channels["commanded_force"][-1, 0]]),
        peak_measured_force=np.asarray([3.0]),
        reached_target=np.asarray([True]),
        history=history,
        parameters={"controller": "ProbePullController"},
    )


class TestProbeFeatures:
    def test_finds_the_breakaway_instant_and_force(self) -> None:
        features = extract_features(make_probe(breakaway_step=10, speed=0.02), 0)
        assert features.moved
        assert features.breakaway_time == pytest.approx(11 / 60.0)
        # The ramp is 2 + 8t, so the force at that instant follows from the time.
        assert features.breakaway_force == pytest.approx(2.0 + 8.0 * 11 / 60.0)

    def test_reports_not_moved_when_the_drawer_never_slides(self) -> None:
        features = extract_features(make_probe(speed=BREAKAWAY_SPEED / 2), 0)
        assert not features.moved
        assert features.mean_speed_after_breakaway == 0.0

    def test_later_breakaway_means_more_force_was_needed(self) -> None:
        early = extract_features(make_probe(breakaway_step=5), 0)
        late = extract_features(make_probe(breakaway_step=25), 0)
        assert late.breakaway_time > early.breakaway_time
        assert late.breakaway_force > early.breakaway_force

    def test_every_declared_feature_is_produced(self) -> None:
        features = extract_features(make_probe(), 0)
        assert len(features.as_vector()) == len(PROBE_FEATURES)
        assert set(PROBE_FEATURES) <= set(features.as_dict())

    def test_features_read_only_deployable_channels(self) -> None:
        assert_features_are_deployable()


class TestRankCorrelation:
    def test_perfect_monotone_relationships(self) -> None:
        assert rank_correlation([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
        assert rank_correlation([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_monotone_but_nonlinear_still_scores_one(self) -> None:
        """The reason for using ranks: the probe-to-force relationship need not be linear."""
        assert rank_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)

    def test_a_constant_input_has_no_information(self) -> None:
        assert np.isnan(rank_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))

    def test_ties_are_averaged_rather_than_ordered_arbitrarily(self) -> None:
        tied = rank_correlation([1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0])
        assert tied == pytest.approx(0.8944, abs=1e-3)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            rank_correlation([1.0, 2.0], [1.0])

    def test_too_few_points_is_not_a_correlation(self) -> None:
        assert np.isnan(rank_correlation([1.0, 2.0], [1.0, 2.0]))


class TestExperimentPlan:
    """The selected parameters must be internally consistent with each other."""

    def test_the_probe_barely_disturbs_the_task(self) -> None:
        """The probe's own travel counts towards ``d_total``, so it has to stay small (D027)."""
        ratio = RECOMMENDED_PROBE_TASK.target_displacement / MAIN_TASK.goal_displacement
        assert ratio < 0.15, f"the probe travels {ratio:.0%} of the goal; it should measure, not act"

    def test_the_probe_ramp_reaches_past_the_stiffest_drawer(self) -> None:
        """The probe must be able to break away every drawer the task covers.

        Only the *upper* end is asserted. The probe's ramp deliberately starts above the
        weakest task force (1.0 N against a 0.15 N band floor): a probe is not a scaled-down
        execution, and what matters is that its ramp passes through every drawer's breakaway
        force, which the calibration measured for all 108 hidden states.
        """
        _, high = MAIN_TASK.peak_force_range
        assert RECOMMENDED_PROBE_TASK.max_force >= high

    def test_the_probe_ramp_starts_below_the_median_required_force(self) -> None:
        """So the ramp has room to separate drawers rather than starting past most of them."""
        low, high = MAIN_TASK.peak_force_range
        assert RECOMMENDED_PROBE_TASK.initial_force < 0.5 * (low + high)

    def test_the_probe_ramp_fits_inside_its_budget(self) -> None:
        assert RECOMMENDED_PROBE_CFG.ramp_duration < RECOMMENDED_PROBE_CFG.max_probe_duration

    def test_the_probe_is_shorter_than_the_execution(self) -> None:
        assert RECOMMENDED_PROBE_CFG.max_probe_duration <= MAIN_TASK.duration

    def test_the_task_criteria_are_valid(self) -> None:
        criteria = MAIN_TASK.criteria
        assert criteria.goal_displacement == MAIN_TASK.goal_displacement
        assert criteria.displacement_tolerance / criteria.goal_displacement <= 0.30

    def test_the_execution_ramp_down_is_the_selected_one(self) -> None:
        """0.35: a short ramp-down leaves a low-resistance drawer no time to decelerate."""
        assert RECOMMENDED_EXECUTION_CFG.fall_fraction == pytest.approx(0.35)

    def test_the_execution_does_not_settle(self) -> None:
        """A settle would brake the pull axis and erase what the probe left (D029)."""
        assert RECOMMENDED_EXECUTION_CFG.settle_steps == 0

    def test_the_execution_releases_the_force_after_T(self) -> None:
        assert RECOMMENDED_EXECUTION_CFG.zero_force_cleanup_steps > 0

    def test_ood_ranges_strictly_contain_the_training_ranges(self) -> None:
        for name in ("mass", "static_friction", "dynamic_friction_ratio", "damping"):
            train_low, train_high = getattr(TRAINING_XI_RANGES, name)
            ood_low, ood_high = getattr(OOD_XI_RANGES, name)
            assert ood_low <= train_low and ood_high >= train_high, name
            assert (ood_low, ood_high) != (train_low, train_high), f"{name} is not extended at all"

    def test_ranges_convert_into_a_usable_randomizer_config(self) -> None:
        for ranges in (TRAINING_XI_RANGES, OOD_XI_RANGES):
            cfg = ranges.as_randomizer_cfg()
            assert cfg.mass_range == ranges.mass
            assert 0.0 <= cfg.dynamic_friction_ratio_range[0] <= cfg.dynamic_friction_ratio_range[1] <= 1.0

    def test_the_plan_serialises(self) -> None:
        for payload in (MAIN_TASK.as_dict(), RECOMMENDED_PROBE_TASK.as_dict(), TRAINING_XI_RANGES.as_dict()):
            assert payload
