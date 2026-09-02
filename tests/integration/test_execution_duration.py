"""Integration tests for the execution controller's timing, profile and safety behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers import ExecutionControllerCfg, SafetyLimits, TerminationReason
from probe_drawer.evaluation import PROVISIONAL_VALIDATION_DURATION, PROVISIONAL_VALIDATION_PEAK_FORCE

pytestmark = pytest.mark.isaacsim

#: The provisional operating point Phases 6-8 validated at. Not the paper's final values.
REFERENCE_PEAK_FORCE = PROVISIONAL_VALIDATION_PEAK_FORCE
REFERENCE_DURATION = PROVISIONAL_VALIDATION_DURATION


class TestDuration:
    @pytest.mark.parametrize("duration", [0.5, 1.0, 2.0])
    def test_runs_for_the_commanded_duration(self, uniform_system, pull_system, duration: float) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=duration)

        assert result.termination_reason[0] is TerminationReason.DURATION_COMPLETED
        assert result.duration[0] == pytest.approx(duration, abs=pull_system.step_dt)
        assert result.history.num_steps == pull_system.execution.steps_for(duration)

    def test_duration_is_not_decided_by_displacement(self, uniform_system, pull_system) -> None:
        """The same duration must be executed no matter how far the drawer travels."""
        durations = []
        displacements = []
        for preset_name in ("easy", "hard"):
            uniform_system(preset_name)
            result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
            durations.append(float(result.duration[0]))
            displacements.append(float(result.final_displacement[0]))

        assert durations[0] == pytest.approx(durations[1], abs=1e-9)
        assert displacements[0] > 2.0 * displacements[1], (
            "the two presets must travel very different distances, otherwise this test "
            "would pass even if displacement did control the duration"
        )


class TestForceProfile:
    def test_starts_and_ends_at_essentially_zero_force(self, uniform_system, pull_system) -> None:
        """The profile satisfies phi(0) == phi(1) == 0; the discretisation costs 2 % of peak.

        Commands are held for a whole control step, so the last command of the episode is
        the one issued at ``T - step_dt`` rather than at ``T``. With a 10 % smoothstep fall
        that is ``phi(1 - dt/T)``, about 2 % of the peak -- not exactly zero, and not a
        defect. ``phi(1) == 0`` exactly is asserted in ``tests/unit/test_force_profiles.py``.
        """
        uniform_system("medium")
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history
        commanded = history.commanded_force[:, 0]

        assert commanded[0] == pytest.approx(0.0, abs=1e-6), "the first command must be exactly zero"
        assert 0.0 <= commanded[-1] <= 0.03 * REFERENCE_PEAK_FORCE

    def test_plateau_reaches_the_commanded_peak(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history
        assert history.commanded_force[:, 0].max() == pytest.approx(REFERENCE_PEAK_FORCE, rel=1e-4)

    def test_normalised_shape_is_the_same_for_every_peak_force(self, uniform_system, pull_system) -> None:
        """Execution Test 1: F(t)/F_peak must be one and the same curve."""
        normalised = []
        for peak in (3.0, 5.0, 7.0):
            uniform_system("hard")
            history = pull_system.execution.run(peak_force=peak, duration=REFERENCE_DURATION).history
            normalised.append(history.commanded_force[:, 0] / peak)

        reference = normalised[0]
        for curve in normalised[1:]:
            assert curve.shape == reference.shape
            assert np.allclose(curve, reference, atol=1e-5)

    def test_command_never_steps(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history
        cfg = ExecutionControllerCfg()

        # Analytic bound on |dphi/dtau| for smoothstep is 1.5, spread over rise_fraction * T.
        bound = 1.5 * REFERENCE_PEAK_FORCE / (cfg.rise_fraction * REFERENCE_DURATION) * pull_system.step_dt
        assert np.abs(np.diff(history.commanded_force[:, 0])).max() <= bound * 1.05


class TestSafety:
    @pytest.fixture
    def tight_limits(self, pull_system):
        """Temporarily install a limit the reference execution is certain to violate.

        Triggering the abort with a deliberately tight limit rather than with a huge force
        keeps the test deterministic: at 30 N the arm's behaviour is chaotic and may or may
        not cross any particular limit, which would make the test flaky rather than strict.
        """
        original = pull_system.execution.safety
        pull_system.execution.safety = SafetyLimits(max_drawer_velocity=0.02)
        yield
        pull_system.execution.safety = original

    def test_a_violated_limit_aborts_the_execution(self, uniform_system, pull_system, tight_limits) -> None:
        """The only permitted early stop is an absolute safety violation."""
        uniform_system("easy")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)

        assert bool(result.safety_aborted[0])
        assert result.termination_reason[0] is TerminationReason.SAFETY_ABORT
        assert result.duration[0] < REFERENCE_DURATION

    def test_the_same_execution_completes_with_the_project_limits(self, uniform_system, pull_system) -> None:
        """Control: the abort above came from the limit, not from the execution itself."""
        uniform_system("easy")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)

        assert not bool(result.safety_aborted[0])
        assert result.termination_reason[0] is TerminationReason.DURATION_COMPLETED

    def test_a_large_force_keeps_the_simulation_finite(self, uniform_system, pull_system) -> None:
        """Whatever a 30 N pull does, it must not produce NaNs or unbounded state."""
        uniform_system("easy")
        result = pull_system.execution.run(peak_force=30.0, duration=REFERENCE_DURATION)

        for name, array in result.history.as_arrays().items():
            assert np.all(np.isfinite(array)), f"{name} contains non-finite values"
        assert abs(result.final_displacement[0]) < 1.0

    def test_a_profile_above_the_absolute_force_limit_is_refused(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        limit = SafetyLimits().max_commanded_force
        with pytest.raises(ValueError, match="above the absolute safety limit"):
            pull_system.execution.run(peak_force=limit + 1.0, duration=0.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"peak_force": 0.0}, "peak_force must be > 0"), ({"duration": 0.0}, "duration must be > 0")],
    )
    def test_rejects_non_physical_arguments(self, pull_system, kwargs: dict, match: str) -> None:
        args = {"peak_force": 5.0, "duration": 1.0}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            pull_system.execution.run(**args)


class TestZeroForceCleanup:
    """The result is snapshotted at ``T``; the cleanup afterwards must not touch it (D022)."""

    @staticmethod
    def _commanded_pull_force(pull_system) -> float:
        """The pull-axis force in the action the environment is currently holding."""
        action = pull_system.env.action_manager.action
        direction = pull_system.pull_axis.direction(pull_system.env.device)
        return float(action[0, 7:10] @ direction)

    def test_the_pull_force_is_released_after_the_duration(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        assert self._commanded_pull_force(pull_system) == pytest.approx(0.0, abs=1e-6)

    def test_without_cleanup_a_residual_command_would_remain(self, uniform_system, pull_system) -> None:
        """Control: shows the cleanup is what zeroes the command, not the profile."""
        original = pull_system.execution.cfg
        pull_system.execution.cfg = ExecutionControllerCfg(zero_force_cleanup_steps=0)
        try:
            uniform_system("medium")
            pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
            residual = self._commanded_pull_force(pull_system)
        finally:
            pull_system.execution.cfg = original
        assert residual > 0.0
        assert residual < 0.03 * REFERENCE_PEAK_FORCE

    def test_cleanup_steps_are_absent_from_the_history(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        assert result.history.num_steps == pull_system.execution.steps_for(REFERENCE_DURATION)
        assert result.history.time[-1] == pytest.approx(REFERENCE_DURATION, abs=1e-9)
        assert result.parameters["post_execution_steps_excluded_from_result"] > 0

    def test_the_snapshot_predates_the_cleanup(self, uniform_system, pull_system) -> None:
        """``d(T)`` and ``v(T)`` must be the values at ``T``, not after the release."""
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)

        # The drawer keeps coasting once the force is released, so the state now differs
        # from the snapshot -- which is exactly what proves the snapshot came first.
        assert result.final_displacement[0] == pytest.approx(result.history.drawer_position[-1, 0], abs=1e-9)
        assert float(pull_system.reader.drawer_position[0]) > result.final_displacement[0]

    def test_cleanup_step_count_is_configurable_and_recorded(self, uniform_system, pull_system) -> None:
        original = pull_system.execution.cfg
        pull_system.execution.cfg = ExecutionControllerCfg(
            zero_force_cleanup_steps=3, post_execution_settle_steps=4
        )
        try:
            uniform_system("medium")
            result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=0.5)
        finally:
            pull_system.execution.cfg = original
        assert result.parameters["post_execution_steps_excluded_from_result"] == 7
        assert result.history.num_steps == pull_system.execution.steps_for(0.5)

    def test_the_next_episode_does_not_inherit_a_pull_force(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=0.5)
        uniform_system("medium")
        first = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=0.5)
        assert first.history.commanded_force[0, 0] == pytest.approx(0.0, abs=1e-6)


class TestPeakVelocity:
    def test_peak_velocity_is_reported_and_at_least_the_terminal_speed(
        self, uniform_system, pull_system
    ) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        assert result.peak_velocity[0] >= abs(result.final_velocity[0]) - 1e-9
        assert result.peak_velocity[0] == pytest.approx(
            float(np.abs(result.history.drawer_velocity[:, 0]).max()), rel=1e-6
        )

    def test_peak_velocity_grows_with_force(self, uniform_system, pull_system) -> None:
        peaks = []
        for peak_force in (3.0, 6.0):
            uniform_system("medium")
            peaks.append(
                float(pull_system.execution.run(peak_force=peak_force, duration=1.0).peak_velocity[0])
            )
        assert peaks[1] > peaks[0]


class TestHeldAxisStability:
    def test_the_five_held_degrees_of_freedom_hold_at_the_reference_point(
        self, uniform_system, pull_system
    ) -> None:
        uniform_system("medium")
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history

        assert history.tcp_lateral_error.max() < 0.005
        assert np.degrees(history.tcp_orientation_error).max() < 3.0


class TestResultContract:
    def test_result_reports_the_commanded_peak(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        assert result.peak_commanded_force[0] == pytest.approx(REFERENCE_PEAK_FORCE, rel=1e-4)

    def test_measured_force_is_an_independent_signal(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        assert result.peak_measured_force[0] != pytest.approx(result.peak_commanded_force[0], abs=1e-3)

    def test_parameters_record_everything_needed_to_replay(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        for key in ("controller", "peak_force", "duration", "profile", "config", "safety", "step_dt"):
            assert key in result.parameters, key
