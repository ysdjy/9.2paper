"""Integration tests proving the hidden dynamics reach the simulation and change behaviour.

All three presets run in parallel environments of the *same* simulation, so nothing but
``xi = [mass, friction, damping]`` differs between the trajectories being compared.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.envs import REFERENCE_DURATION, REFERENCE_PEAK_FORCE, DynamicsRandomizer, preset

pytestmark = pytest.mark.isaacsim


@pytest.fixture
def preset_dynamics(pull_system, randomizer, preset_order):
    """Reset and give environment ``i`` the ``preset_order[i]`` preset."""
    pull_system.reset()
    return randomizer.apply(pull_system.env, [preset(name) for name in preset_order])


class TestParametersReachTheSimulation:
    def test_every_parameter_reads_back_as_requested(self, preset_dynamics) -> None:
        assert preset_dynamics.consistent
        for key in ("drawer_mass", "joint_friction", "joint_damping", "joint_stiffness"):
            requested = [getattr(p, key) for p in preset_dynamics.requested]
            assert preset_dynamics.readback[key] == pytest.approx(requested, rel=1e-4, abs=1e-6)

    def test_static_and_dynamic_friction_are_both_set(self, preset_dynamics) -> None:
        """A stiff drawer must resist while sliding, not only while starting."""
        assert preset_dynamics.readback["joint_dynamic_friction"] == pytest.approx(
            preset_dynamics.readback["joint_friction"], rel=1e-4, abs=1e-6
        )

    def test_the_targeted_joint_and_body_are_the_top_drawer(self, preset_dynamics) -> None:
        assert preset_dynamics.notes["drawer_joint"] == "drawer_top_joint"
        assert preset_dynamics.notes["drawer_body"] == "drawer_top"

    def test_other_cabinet_joints_are_untouched(self, pull_system, preset_dynamics) -> None:
        cabinet = pull_system.env.scene["cabinet"]
        for joint_name in ("drawer_bottom_joint", "door_left_joint", "door_right_joint"):
            index = cabinet.find_joints(joint_name)[0][0]
            assert float(cabinet.data.joint_friction_coeff[0, index]) == pytest.approx(0.0)

    def test_the_moving_mass_accounts_for_the_handle(self, preset_dynamics) -> None:
        assert preset_dynamics.handle_mass > 0.0
        for total, params in zip(preset_dynamics.total_moving_mass, preset_dynamics.requested, strict=True):
            assert total == pytest.approx(params.drawer_mass + preset_dynamics.handle_mass)


class TestExecutionResponseDiffers:
    def test_the_same_force_profile_produces_clearly_different_displacements(
        self, pull_system, preset_dynamics
    ) -> None:
        result = pull_system.execution.run(peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION)
        displacement = result.final_displacement

        assert np.all(np.diff(displacement) < 0), (
            f"expected easy > medium > hard, got {displacement.tolist()}"
        )
        ratios = displacement[:-1] / displacement[1:]
        assert np.all(ratios >= 1.5), f"presets are not clearly separated: ratios {ratios.tolist()}"

    def test_velocity_trajectories_differ(self, pull_system, preset_dynamics) -> None:
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history
        peak_velocity = np.abs(history.drawer_velocity).max(axis=0)
        assert np.all(np.diff(peak_velocity) < 0), f"peak velocities not ordered: {peak_velocity.tolist()}"

    def test_the_commanded_force_is_identical_across_presets(self, pull_system, preset_dynamics) -> None:
        """The input must be the same, or the displacement comparison proves nothing."""
        history = pull_system.execution.run(
            peak_force=REFERENCE_PEAK_FORCE, duration=REFERENCE_DURATION
        ).history
        for env_index in range(1, history.num_envs):
            assert np.allclose(history.commanded_force[:, env_index], history.commanded_force[:, 0])


class TestProbeResponseDiffers:
    def test_the_standardised_probe_distinguishes_the_presets(self, pull_system, preset_dynamics) -> None:
        result = pull_system.probe.run(
            initial_force=2.0, max_force=10.0, target_displacement=0.005, max_velocity=0.05
        )
        assert np.all(np.diff(result.duration) > 0), (
            f"a stiffer drawer must take longer to reach the probe target, got {result.duration.tolist()}"
        )
        assert np.all(np.diff(result.final_commanded_force) > 0), (
            f"a stiffer drawer must need more force, got {result.final_commanded_force.tolist()}"
        )


class TestPrivilegedState:
    def test_the_applied_dynamics_are_retrievable_for_logging(
        self, randomizer, preset_dynamics, preset_order
    ) -> None:
        current = randomizer.get_current_params()
        assert current is preset_dynamics
        assert [p["name"] for p in current.as_dict()["requested"]] == list(preset_order)

    def test_sampling_produces_parameters_that_apply_cleanly(self, pull_system) -> None:
        randomizer = DynamicsRandomizer(seed=11)
        pull_system.reset()
        applied = randomizer.apply(pull_system.env, randomizer.sample(pull_system.env.num_envs))
        assert applied.consistent
