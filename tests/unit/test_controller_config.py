"""Unit tests for configuration validation and pull-axis geometry. No Isaac Sim required."""

from __future__ import annotations

import pytest
import torch

from probe_drawer.controllers import ExecutionControllerCfg, ProbeControllerCfg, SafetyLimits, TerminationReason
from probe_drawer.controllers.probe_pull_controller import ProbePullController
from probe_drawer.envs import (
    PRESETS,
    DynamicsParameters,
    DynamicsRandomizer,
    GraspConfiguration,
    load_grasp_configuration,
    preset,
)
from probe_drawer.sensors import PullAxis


class TestPullAxis:
    def test_default_is_the_measured_cabinet_axis(self) -> None:
        """The official cabinet opens along base -x; measured in run_official_drawer.py."""
        axis = PullAxis()
        assert (axis.index, axis.sign) == (0, -1.0)
        assert axis.name == "-x"
        assert torch.equal(axis.direction(), torch.tensor([-1.0, 0.0, 0.0]))

    def test_masks_are_complementary(self) -> None:
        """Exactly one axis is force-controlled and the other five are pose-held."""
        for index in (0, 1, 2):
            axis = PullAxis(index=index, sign=1.0)
            motion = axis.motion_control_axes()
            wrench = axis.wrench_control_axes()
            assert sum(motion) == 5
            assert sum(wrench) == 1
            assert motion[index] == 0
            assert wrench[index] == 1
            assert all(m + w == 1 for m, w in zip(motion[:3], wrench[:3], strict=True))

    def test_rotational_axes_are_never_force_controlled(self) -> None:
        for index in (0, 1, 2):
            assert PullAxis(index=index).wrench_control_axes()[3:] == (0, 0, 0)

    @pytest.mark.parametrize(("kwargs", "match"), [({"index": 3}, "index must be"), ({"sign": 0.0}, "sign must be")])
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PullAxis(**kwargs)


class TestProbeControllerCfg:
    def test_defaults_are_self_consistent(self) -> None:
        cfg = ProbeControllerCfg()
        assert cfg.ramp_duration + cfg.hold_after_max_force < cfg.max_probe_duration, (
            "the force-limit stop must be reachable inside the time budget, "
            "otherwise the probe can only ever terminate on timeout"
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ramp_duration": 0.0}, "ramp_duration must be > 0"),
            ({"hold_after_max_force": -0.1}, "hold_after_max_force must be >= 0"),
            ({"max_probe_duration": 0.0}, "max_probe_duration must be > 0"),
            ({"settle_steps": -1}, "settle_steps must be >= 0"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ProbeControllerCfg(**kwargs)


class TestProbeTaskParameterValidation:
    """``run`` must reject non-physical task parameters before touching the simulation."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"initial_force": -1.0}, "initial_force must be >= 0"),
            ({"initial_force": 10.0, "max_force": 2.0}, "monotonically non-decreasing"),
            ({"target_displacement": 0.0}, "target_displacement must be > 0"),
            ({"max_velocity": -0.1}, "max_velocity must be > 0"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        args = {"initial_force": 2.0, "max_force": 10.0, "target_displacement": 0.005, "max_velocity": 0.05}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ProbePullController._validate_task_parameters(**args)  # type: ignore[arg-type]

    def test_accepts_the_documented_example(self) -> None:
        ProbePullController._validate_task_parameters(2.0, 10.0, 0.005, 0.05)


class TestExecutionControllerCfg:
    def test_rejects_negative_settle(self) -> None:
        with pytest.raises(ValueError, match="settle_steps must be >= 0"):
            ExecutionControllerCfg(settle_steps=-1)

    def test_rise_and_fall_default_to_ten_percent(self) -> None:
        cfg = ExecutionControllerCfg()
        assert (cfg.rise_fraction, cfg.fall_fraction) == (0.1, 0.1)


class TestSafetyLimits:
    def test_serialises_every_field(self) -> None:
        limits = SafetyLimits()
        assert set(limits.as_dict()) == set(vars(limits))


class TestTerminationReason:
    def test_execution_nominal_reason_is_duration(self) -> None:
        assert TerminationReason.DURATION_COMPLETED.value == "duration_completed"

    def test_values_are_stable_strings(self) -> None:
        """Logged episodes carry these strings, so renaming one silently breaks old logs."""
        assert {r.value for r in TerminationReason} == {
            "displacement_reached",
            "velocity_limit",
            "max_force_reached",
            "timeout",
            "duration_completed",
            "safety_abort",
        }


class TestDynamicsParameters:
    def test_presets_are_ordered_by_difficulty(self) -> None:
        easy, medium, hard = (PRESETS[n] for n in ("easy", "medium", "hard"))
        for attribute in ("drawer_mass", "joint_friction", "joint_damping"):
            values = [getattr(p, attribute) for p in (easy, medium, hard)]
            assert values == sorted(values), f"{attribute} must not decrease from easy to hard: {values}"

    def test_presets_have_no_drive_stiffness(self) -> None:
        """Stiffness would be a fourth hidden parameter; see DECISIONS D008."""
        assert all(p.joint_stiffness == 0.0 for p in PRESETS.values())

    def test_unknown_preset_lists_the_valid_ones(self) -> None:
        with pytest.raises(KeyError, match="Available: "):
            preset("impossible")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"drawer_mass": 0.0}, "drawer_mass must be > 0"),
            ({"joint_friction": -1.0}, "joint_friction must be >= 0"),
            ({"joint_damping": -1.0}, "joint_damping must be >= 0"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        args = {"drawer_mass": 5.0, "joint_friction": 1.0, "joint_damping": 1.0}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            DynamicsParameters(**args)


class TestDynamicsRandomizer:
    def test_sampling_is_seeded_and_reproducible(self) -> None:
        first = DynamicsRandomizer(seed=7).sample(4)
        second = DynamicsRandomizer(seed=7).sample(4)
        assert [p.as_dict() for p in first] == [p.as_dict() for p in second]

    def test_samples_stay_inside_the_configured_ranges(self) -> None:
        randomizer = DynamicsRandomizer(seed=3)
        cfg = randomizer.cfg
        for params in randomizer.sample(64):
            assert cfg.mass_range[0] <= params.drawer_mass <= cfg.mass_range[1]
            assert cfg.friction_range[0] <= params.joint_friction <= cfg.friction_range[1]
            assert cfg.damping_range[0] <= params.joint_damping <= cfg.damping_range[1]

    def test_current_params_is_none_before_apply(self) -> None:
        assert DynamicsRandomizer().get_current_params() is None

    def test_broadcast_rejects_a_wrong_length_sequence(self) -> None:
        params = [PRESETS["easy"], PRESETS["hard"]]
        with pytest.raises(ValueError, match="Expected 1 or 3 parameter sets, got 2"):
            DynamicsRandomizer._broadcast(params, 3)

    def test_broadcast_expands_a_single_parameter_set(self) -> None:
        expanded = DynamicsRandomizer._broadcast(PRESETS["medium"], 3)
        assert expanded == [PRESETS["medium"]] * 3


class TestGraspConfiguration:
    """The recorded grasp is a checked-in artefact, so its invariants are worth asserting."""

    @staticmethod
    def _configuration() -> GraspConfiguration:
        return load_grasp_configuration()

    def test_recorded_configuration_covers_every_franka_joint(self) -> None:
        joint_pos = self._configuration().joint_pos
        expected = {f"panda_joint{i}" for i in range(1, 8)} | {"panda_finger_joint1", "panda_finger_joint2"}
        assert set(joint_pos) == expected

    def test_fingers_are_recorded_off_centre(self) -> None:
        """If they were equal there would be no imbalance to correct, and no need for D010."""
        values = list(self._configuration().finger_equilibrium.values())
        assert abs(values[0] - values[1]) > 1e-3

    def test_closed_command_deflects_both_fingers_equally(self) -> None:
        configuration = self._configuration()
        squeeze = 0.006
        command = configuration.closed_gripper_command(squeeze)
        deflections = [configuration.finger_equilibrium[name] - value for name, value in command.items()]
        assert deflections == pytest.approx([squeeze, squeeze])

    def test_closed_command_never_opens_a_finger(self) -> None:
        configuration = self._configuration()
        smallest = min(configuration.finger_equilibrium.values())
        with pytest.raises(ValueError, match="exceeds the smallest recorded finger equilibrium"):
            configuration.closed_gripper_command(smallest + 1e-4)

    def test_rejects_non_positive_squeeze(self) -> None:
        with pytest.raises(ValueError, match="squeeze must be > 0"):
            self._configuration().closed_gripper_command(0.0)

    def test_shipped_default_squeeze_is_usable_with_the_recorded_grasp(self) -> None:
        """Guards against a future grasp recording the shipped default cannot squeeze."""
        # Mirrors ProbeDrawerEnvCfg.grip_squeeze, which cannot be imported without Isaac Sim.
        default_grip_squeeze = 0.006
        command = self._configuration().closed_gripper_command(default_grip_squeeze)
        assert all(value >= 0.0 for value in command.values())
