"""Integration tests for the sequential probe-then-execute protocol.

These are the checks that make the protocol's three promises verifiable rather than
intended: nothing is reset after the probe, nothing is artificially quieted, and the task is
measured from before the probe. Each launches the shared Isaac Sim session from
``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers import ExecutionControllerCfg
from probe_drawer.evaluation import SuccessCriteria, evaluate_execution
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_TASK, SEQUENTIAL_TRANSITION_STEPS
from probe_drawer.protocols import InferenceTransitionCfg, SequentialProtocolCfg, SequentialPullProtocol

pytestmark = pytest.mark.isaacsim


@pytest.fixture
def protocol_cfg() -> SequentialProtocolCfg:
    return SequentialProtocolCfg(
        probe_task=RECOMMENDED_PROBE_TASK,
        duration=MAIN_TASK.duration,
        transition=InferenceTransitionCfg(steps=SEQUENTIAL_TRANSITION_STEPS),
    )


@pytest.fixture
def sequential(pull_system, protocol_cfg):
    """The shared system with a non-settling execution, wrapped in the protocol.

    The execution's settle is what would brake the pull axis, so the protocol refuses to run
    with it enabled; this fixture swaps it out and restores it afterwards.
    """
    original = pull_system.execution.cfg
    pull_system.execution.cfg = ExecutionControllerCfg(fall_fraction=0.35, settle_steps=0)
    try:
        yield SequentialPullProtocol(pull_system, protocol_cfg)
    finally:
        pull_system.execution.cfg = original


class TestProtocolGuardrails:
    def test_a_settling_execution_is_refused(self, pull_system, protocol_cfg) -> None:
        """A settle brakes the pull axis, which would erase what the probe left (D029)."""
        original = pull_system.execution.cfg
        pull_system.execution.cfg = ExecutionControllerCfg(settle_steps=30)
        try:
            with pytest.raises(ValueError, match="settle_steps = 0"):
                SequentialPullProtocol(pull_system, protocol_cfg)
        finally:
            pull_system.execution.cfg = original

    def test_a_negative_gap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="transition steps must be >= 0"):
            InferenceTransitionCfg(steps=-1)


class TestSequentialContinuity:
    def test_the_drawer_is_not_reset_after_the_probe(self, uniform_system, sequential) -> None:
        uniform_system("medium")
        episode = sequential.run(peak_force=2.0)

        assert np.all(episode.probe_displacement > 1e-4), "the probe must move the drawer"
        assert np.all(episode.pre_execution_displacement >= episode.probe_displacement - 1e-9), (
            "the execution must start from where the probe left the drawer, or further"
        )

    def test_the_probe_leaves_the_drawer_moving_and_the_gap_only_slows_it(
        self, uniform_system, sequential
    ) -> None:
        uniform_system("easy")
        episode = sequential.run(peak_force=1.0)
        transition = episode.transition

        assert np.any(np.abs(transition.velocity_before) > 1e-4)
        assert np.all(np.abs(transition.velocity_after) <= np.abs(transition.velocity_before) + 1e-9), (
            "the gap commands zero force and must never speed the drawer up or write its state"
        )

    def test_the_arm_configuration_carries_across_the_probe(self, uniform_system, pull_system, sequential) -> None:
        """The grasp is established once, at the top, and never re-established."""
        uniform_system("medium")
        episode = sequential.run(peak_force=2.0)
        finger_positions = pull_system.reader.finger_joint_position
        assert np.all(finger_positions.cpu().numpy() > 0.0), "the gripper is still closed on the handle"
        assert episode.execution.history.num_steps == pull_system.execution.steps_for(MAIN_TASK.duration)

    def test_the_gap_does_not_appear_in_either_history(self, uniform_system, pull_system, sequential) -> None:
        """The gap belongs to neither the probe the model sees nor the commanded duration (D030)."""
        uniform_system("medium")
        episode = sequential.run(peak_force=2.0)

        # 1e-6 rather than exact: the recorded durations are float32, the history clock float64.
        assert episode.probe.history.time[-1] == pytest.approx(episode.probe.duration.max(), abs=1e-6)
        assert episode.execution.history.time[-1] == pytest.approx(MAIN_TASK.duration, abs=1e-6)
        assert episode.transition.steps == SEQUENTIAL_TRANSITION_STEPS
        assert episode.transition.duration == pytest.approx(
            SEQUENTIAL_TRANSITION_STEPS * pull_system.step_dt, abs=1e-12
        )

    def test_the_gap_length_is_fixed_across_episodes(self, uniform_system, sequential) -> None:
        durations = []
        for _ in range(2):
            uniform_system("medium")
            durations.append(sequential.run(peak_force=2.0).transition.duration)
        assert durations[0] == pytest.approx(durations[1], abs=1e-12)


class TestTaskReference:
    def test_total_displacement_is_the_sum_of_the_parts(self, uniform_system, sequential) -> None:
        uniform_system("medium")
        episode = sequential.run(peak_force=2.5)
        assert np.allclose(
            episode.total_displacement,
            episode.pre_execution_displacement + episode.execution.final_displacement,
            atol=1e-12,
        )

    def test_the_probe_contribution_counts_towards_the_goal(self, uniform_system, sequential) -> None:
        """A 3 mm probe plus a 37 mm execution reaches a 40 mm goal (D027)."""
        uniform_system("medium")
        episode = sequential.run(peak_force=2.5, criteria=MAIN_TASK.criteria)
        verdict = episode.evaluation.verdicts[0]

        assert verdict.pre_execution_displacement > 1e-4
        assert verdict.total_displacement == pytest.approx(
            verdict.pre_execution_displacement + verdict.execution_displacement, abs=1e-12
        )
        assert verdict.displacement_error == pytest.approx(
            verdict.total_displacement - MAIN_TASK.goal_displacement, abs=1e-12
        )

    def test_ignoring_the_probe_would_change_the_label(self, uniform_system, sequential) -> None:
        """Evidence that the reference frame is load-bearing, not bookkeeping."""
        uniform_system("easy")
        episode = sequential.run(peak_force=1.0)
        criteria = SuccessCriteria(
            goal_displacement=float(episode.total_displacement[0]),
            displacement_tolerance=0.001,
            velocity_tolerance=1.0,
        )

        with_probe = evaluate_execution(
            episode.execution,
            criteria,
            pre_execution_displacement=episode.pre_execution_displacement,
        )
        without_probe = evaluate_execution(episode.execution, criteria)

        assert with_probe.verdicts[0].displacement_ok
        assert not without_probe.verdicts[0].displacement_ok


class TestExecutionIsolation:
    def test_the_execution_controller_never_sees_the_goal(self, sequential) -> None:
        """``run`` takes a force, a duration and a step observer -- nothing goal-shaped (D004).

        ``on_step`` was added for the diagnostic video recorder; it is called and never read
        back, so it cannot influence the run. The name check is the part that matters and is
        stricter than the exact list it replaces, which excluded a goal parameter only by
        accident of enumeration.
        """
        import inspect  # noqa: PLC0415

        parameters = list(inspect.signature(sequential.system.execution.run).parameters)
        assert parameters == ["peak_force", "duration", "on_step"]

        forbidden = ("goal", "target", "reference", "setpoint", "desired", "criteria", "tolerance")
        assert not [name for name in parameters for word in forbidden if word in name.lower()]

    def test_the_protocol_passes_no_goal_to_the_execution(self, uniform_system, sequential) -> None:
        uniform_system("medium")
        episode = sequential.run(peak_force=2.0, criteria=MAIN_TASK.criteria)
        recorded = episode.execution.parameters
        assert "goal_displacement" not in recorded
        assert not any("goal" in key for key in recorded)


class TestCandidateComparison:
    def test_per_environment_forces_are_applied_independently(self, uniform_system, sequential) -> None:
        """Several candidates share one protocol run, each with its own amplitude."""
        uniform_system("medium")
        forces = [1.5, 2.0, 2.5]
        episode = sequential.run(peak_force=forces)

        assert episode.peak_force.tolist() == pytest.approx(forces)
        assert episode.execution.peak_commanded_force.tolist() == pytest.approx(forces, rel=1e-4)
        # More force must move the drawer further, or the amplitudes were not independent.
        assert np.all(np.diff(episode.total_displacement) > 0)

    def test_the_normalised_profile_is_identical_across_candidates(self, uniform_system, sequential) -> None:
        """Only the amplitude may differ: phi(t/T) is the same curve for every candidate."""
        uniform_system("medium")
        forces = [1.5, 2.0, 2.5]
        history = sequential.run(peak_force=forces).execution.history

        reference = history.commanded_force[:, 0] / forces[0]
        for index, force in enumerate(forces[1:], start=1):
            assert np.allclose(history.commanded_force[:, index] / force, reference, atol=1e-6)

    def test_candidates_start_from_comparable_states(self, uniform_system, sequential) -> None:
        """The post-probe spread must stay well inside the task's position tolerance.

        It is not zero -- parallel environments are not bit-identical replicas, and a probe
        stopping on a displacement threshold can cross it a step early or late. What matters
        is that the spread cannot flip a success label on its own.
        """
        uniform_system("medium")
        episode = sequential.run(peak_force=[2.0, 2.0, 2.0])
        spread = float(episode.pre_execution_displacement.max() - episode.pre_execution_displacement.min())
        assert spread < 0.2 * MAIN_TASK.displacement_tolerance, (
            f"post-probe displacement spread {spread * 1000:.3f} mm is too large a fraction of "
            f"eps_d = {MAIN_TASK.displacement_tolerance * 1000:.1f} mm"
        )

    def test_rejects_a_wrong_length_force_list(self, uniform_system, pull_system, sequential) -> None:
        uniform_system("medium")
        with pytest.raises(ValueError, match="must have 1 or"):
            sequential.run(peak_force=[1.0] * (pull_system.env.num_envs + 1))

    def test_rejects_a_non_positive_force(self, uniform_system, sequential) -> None:
        uniform_system("medium")
        with pytest.raises(ValueError, match="must be > 0 N everywhere"):
            sequential.run(peak_force=[1.0, 0.0, 2.0])
