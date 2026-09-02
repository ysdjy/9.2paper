"""Integration tests for the probe's stop conditions and recorded history.

These launch Isaac Sim (see ``conftest.py``) and assert on physical behaviour, not just on
the absence of exceptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.controllers import ProbeControllerCfg, TerminationReason

pytestmark = pytest.mark.isaacsim

#: Task parameters from the documented example in ``docs/API.md``.
DEFAULT_TASK = {"initial_force": 2.0, "max_force": 10.0, "target_displacement": 0.005, "max_velocity": 0.05}


def _run(pull_system, cfg: ProbeControllerCfg | None = None, **overrides):
    pull_system.probe.cfg = cfg or ProbeControllerCfg()
    return pull_system.probe.run(**{**DEFAULT_TASK, **overrides})


class TestForceRamp:
    def test_commanded_force_rises_monotonically_from_initial_to_at_most_max(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = _run(pull_system)
        history = result.history
        # Environments that stop early are zero-padded to the longest one, so only the
        # steps this environment was actually driven for are part of its probe input.
        commanded = history.commanded_force[history.active_steps(0), 0]

        assert commanded[0] == pytest.approx(DEFAULT_TASK["initial_force"], abs=0.2)
        assert np.all(np.diff(commanded) >= -1e-6), "the probe input must never decrease"
        assert commanded.max() <= DEFAULT_TASK["max_force"] + 1e-4

    def test_force_rate_matches_the_configured_ramp(self, uniform_system, pull_system) -> None:
        uniform_system("hard")
        cfg = ProbeControllerCfg(ramp_duration=1.0)
        result = _run(pull_system, cfg)
        history = result.history

        expected_rate = (DEFAULT_TASK["max_force"] - DEFAULT_TASK["initial_force"]) / cfg.ramp_duration
        driven = history.active_steps(0)
        measured_rate = np.polyfit(history.time[driven], history.commanded_force[driven, 0], 1)[0]
        assert measured_rate == pytest.approx(expected_rate, rel=0.02)


class TestDisplacementStop:
    def test_stops_at_the_target_displacement(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = _run(pull_system)

        assert result.termination_reason[0] is TerminationReason.DISPLACEMENT_REACHED
        assert bool(result.reached_target[0])
        # One control step of overshoot is unavoidable: the condition is checked after the step.
        target = DEFAULT_TASK["target_displacement"]
        assert target <= result.final_displacement[0] <= target * 1.5


class TestVelocityStop:
    def test_a_low_velocity_limit_stops_the_probe_on_velocity(self, uniform_system, pull_system) -> None:
        uniform_system("easy")
        result = _run(pull_system, max_velocity=0.004, target_displacement=0.2)

        assert result.termination_reason[0] is TerminationReason.VELOCITY_LIMIT
        assert result.final_velocity[0] >= 0.004 * 0.9
        assert not bool(result.reached_target[0])


class TestForceLimitStop:
    def test_an_unreachable_target_stops_the_probe_at_max_force(self, uniform_system, pull_system) -> None:
        uniform_system("hard")
        # A target the drawer cannot reach and a velocity limit it cannot trip, so the only
        # remaining task condition is the force limit.
        result = _run(pull_system, target_displacement=0.35, max_velocity=5.0)

        assert result.termination_reason[0] is TerminationReason.MAX_FORCE_REACHED
        assert result.final_commanded_force[0] == pytest.approx(DEFAULT_TASK["max_force"], rel=1e-3)
        # The probe stops on the step *after* the command first reaches max_force, because
        # the stop conditions are evaluated once the step has been simulated.
        assert result.duration[0] == pytest.approx(
            ProbeControllerCfg().ramp_duration + pull_system.step_dt, abs=0.02
        )


class TestTimeoutStop:
    def test_a_ramp_longer_than_the_budget_stops_the_probe_on_timeout(self, uniform_system, pull_system) -> None:
        uniform_system("hard")
        cfg = ProbeControllerCfg(ramp_duration=5.0, max_probe_duration=0.5)
        result = _run(pull_system, cfg, target_displacement=0.35, max_velocity=5.0)

        assert result.termination_reason[0] is TerminationReason.TIMEOUT
        assert result.duration[0] == pytest.approx(cfg.max_probe_duration, abs=0.02)

    def test_the_probe_always_terminates(self, uniform_system, pull_system) -> None:
        """No stop condition may be left unreachable, whatever the drawer does."""
        uniform_system("hard")
        result = _run(pull_system, target_displacement=1.0, max_velocity=100.0)
        assert all(reason is not None for reason in result.termination_reason)
        assert result.duration[0] <= ProbeControllerCfg().max_probe_duration + 1e-6


class TestHistory:
    def test_every_signal_is_recorded_for_every_step_and_environment(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = _run(pull_system)
        history = result.history

        assert history.num_steps > 1
        assert history.num_envs == pull_system.env.num_envs
        for name, array in history.as_arrays().items():
            assert array.shape[0] == history.num_steps, name
            assert np.all(np.isfinite(array)), f"{name} contains non-finite values"

    def test_the_active_mask_covers_exactly_the_driven_steps(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = _run(pull_system)
        history = result.history

        for env_index in range(history.num_envs):
            driven = history.active_steps(env_index)
            assert driven[0], "every environment is driven on the first step"
            # Once an environment stops it never restarts, so the mask is a prefix.
            assert np.all(np.diff(driven.astype(int)) <= 0)
            expected_steps = int(round(result.duration[env_index] / pull_system.step_dt))
            assert driven.sum() == expected_steps

    def test_time_advances_by_one_control_step(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        history = _run(pull_system).history
        assert np.allclose(np.diff(history.time), pull_system.step_dt, atol=1e-9)

    def test_measured_force_is_not_a_copy_of_the_command(self, uniform_system, pull_system) -> None:
        """The whole project depends on these being different quantities (D006)."""
        uniform_system("medium")
        history = _run(pull_system).history
        commanded = history.commanded_force[:, 0]
        measured = history.measured_force[:, 0]

        assert not np.allclose(commanded, measured, atol=1e-3)
        assert np.abs(commanded - measured).max() > 0.1

    def test_privileged_and_raw_velocity_are_both_recorded(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        history = _run(pull_system).history
        assert "drawer_velocity" in history.as_arrays()
        assert "drawer_velocity_raw" in history.as_arrays()

    def test_displacement_is_measured_from_the_pull_start(self, uniform_system, pull_system) -> None:
        uniform_system("medium")
        result = _run(pull_system)
        assert abs(result.history.drawer_position[0, 0]) < 1e-3
        assert "reference_drawer_position" in result.parameters


class TestHeldAxisStability:
    def test_the_five_held_degrees_of_freedom_do_not_drift(self, uniform_system, pull_system) -> None:
        """Probe Test 5: hybrid control is only correct if y, z and orientation hold."""
        uniform_system("medium")
        history = _run(pull_system).history

        assert history.tcp_lateral_error.max() < 0.002, "TCP drifted more than 2 mm off the pull axis"
        assert np.degrees(history.tcp_orientation_error).max() < 2.0, "TCP rotated more than 2 degrees"

    def test_the_pull_axis_is_free_to_move(self, uniform_system, pull_system) -> None:
        """The complement of the test above: the force-controlled axis must not be held."""
        uniform_system("medium")
        history = _run(pull_system).history
        assert history.tcp_pull_axis_position[-1, 0] > 0.9 * history.drawer_position[-1, 0]
