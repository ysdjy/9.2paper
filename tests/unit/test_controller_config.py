"""Unit tests for configuration validation and pull-axis geometry. No Isaac Sim required."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from probe_drawer.controllers import ExecutionControllerCfg, ProbeControllerCfg, SafetyLimits, TerminationReason
from probe_drawer.controllers.force_profiles import TrapezoidForceProfile
from probe_drawer.controllers.probe_pull_controller import ProbePullController
from probe_drawer.envs import (
    PRESETS,
    XI_FIELDS,
    DynamicsParameters,
    DynamicsRandomizer,
    DynamicsRandomizerCfg,
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
            ({"fixed_rise_fraction": 0.0}, r"fixed_rise_fraction must lie in \(0, 1\)"),
            ({"fixed_fall_fraction": 1.0}, r"fixed_fall_fraction must lie in \(0, 1\)"),
            ({"fixed_rise_fraction": 0.7, "fixed_fall_fraction": 0.5}, "must be <= 1"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ProbeControllerCfg(**kwargs)

    def test_the_fixed_budget_profile_starts_and_ends_at_zero_force(self) -> None:
        """"Fixed release -> Probe END" is a property of the shape, not of a stop condition.

        The probe hands the drawer to the execution with the force already at zero, so the
        post-probe state is a coasting drawer rather than a loaded one. If the fractions ever
        made ``phi(1) > 0`` the probe would end mid-pull and the recorded ``v_post`` would
        describe a state the execution never starts from.
        """
        cfg = ProbeControllerCfg()
        profile = TrapezoidForceProfile(
            peak_force=1.0,
            duration=0.8,
            rise_fraction=cfg.fixed_rise_fraction,
            fall_fraction=cfg.fixed_fall_fraction,
            shape=cfg.fixed_shape,
        )
        assert profile.normalized(0.0) == pytest.approx(0.0, abs=1e-12)
        assert profile.normalized(1.0) == pytest.approx(0.0, abs=1e-12)
        plateau = 0.5 * (cfg.fixed_rise_fraction + (1.0 - cfg.fixed_fall_fraction))
        assert profile.normalized(plateau) == pytest.approx(1.0, abs=1e-12)

    def test_the_release_gets_more_of_the_budget_than_the_rise(self) -> None:
        """Not symmetry for its own sake: the release is the informative half (D044)."""
        cfg = ProbeControllerCfg()
        assert cfg.fixed_fall_fraction > cfg.fixed_rise_fraction


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


class TestFixedBudgetProbe:
    """The Setting V1 probe mode. It shares a controller with ``run`` and differs in kind."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"peak_force": -1.0}, "peak_force must be >= 0"),
            ({"duration": 0.0}, "duration must be > 0"),
        ],
    )
    def test_rejects_non_physical_arguments_before_touching_the_simulation(
        self, kwargs: dict, match: str
    ) -> None:
        args = {"peak_force": 2.0, "duration": 0.8}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ProbePullController.run_fixed_budget(None, **args)  # type: ignore[arg-type]

    def test_the_null_amplitude_is_permitted_and_is_not_an_oversight(self) -> None:
        """Zero force is the *passive observation* control, not bad input.

        ``scripts/audit_probe_value.py`` compares the frozen 3.5 N probe against a 1.0 N weak
        excitation and against spending the same budget applying nothing at all. All three are
        the same trapezoid at three amplitudes, which is only one code path if zero is a legal
        amplitude. A negative force is still refused.
        """
        # Called unbound, so it clears validation and then fails on the missing ``self.cfg``.
        # Reaching that point is the assertion: no ValueError was raised for a zero amplitude.
        with pytest.raises(AttributeError, match="cfg"):
            ProbePullController.run_fixed_budget(None, peak_force=0.0, duration=0.3)  # type: ignore[arg-type]

    def test_it_takes_no_task_parameters(self) -> None:
        """The point of the mode: one excitation for every hidden state and every task.

        A ``d_goal``, a target displacement or a velocity limit in this signature would make
        the probe task-dependent, which is the property Phase 12's response-triggered probe
        had and Setting V1 gives up (D044). Enumerated exactly, in the manner of the D004
        guard, so adding a parameter has to be a deliberate edit to this list.
        """
        parameters = list(inspect.signature(ProbePullController.run_fixed_budget).parameters)
        assert parameters == ["self", "peak_force", "duration", "on_step"]

        forbidden = ("goal", "target", "displacement", "velocity", "criteria", "tolerance", "alpha")
        assert not [name for name in parameters for word in forbidden if word in name.lower()]

    def test_no_task_stop_condition_can_fire_in_fixed_budget_mode(self) -> None:
        """Only safety may cut it short, and safety is applied by ``run_profile`` itself.

        Checked on the flag rather than through a run because the guarantee is structural:
        the empty tuple is what makes every healthy history the same length, and a future
        edit that appends a condition unconditionally would break that silently.
        """
        pretend = SimpleNamespace(_fixed_budget=True)
        conditions = ProbePullController._stop_conditions(pretend, 0.5, torch.tensor([1.0]))  # type: ignore[arg-type]
        assert conditions == ()

    def test_the_ramp_mode_still_has_its_stop_conditions(self) -> None:
        """The counterpart, so the test above cannot pass by disabling both modes."""
        pretend = SimpleNamespace(
            _fixed_budget=False,
            drawer_displacement=torch.tensor([0.001]),
            reader=SimpleNamespace(drawer_velocity=torch.tensor([0.01])),
            step_dt=1.0 / 60.0,
            _target_displacement=0.003,
            _max_force=6.0,
            _max_velocity=0.08,
            _stop_force_time=1.0,
        )
        reasons = [reason for reason, _ in ProbePullController._stop_conditions(pretend, 0.5, torch.tensor([2.0]))]  # type: ignore[arg-type]
        assert reasons == [
            TerminationReason.DISPLACEMENT_REACHED,
            TerminationReason.VELOCITY_LIMIT,
            TerminationReason.MAX_FORCE_REACHED,
        ]


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
        """Logged episodes carry these strings, so renaming one silently breaks old logs.

        Adding a member is safe and is why this asserts a superset relationship for the
        original six rather than equality: ``AT_REST`` joined for the response-triggered
        probe, whose coast ends because the drawer *stopped* -- the opposite of
        ``VELOCITY_LIMIT``, which means it got too fast.
        """
        values = {r.value for r in TerminationReason}
        assert {
            "displacement_reached",
            "velocity_limit",
            "max_force_reached",
            "timeout",
            "duration_completed",
            "safety_abort",
        } <= values
        assert values == {
            "displacement_reached",
            "velocity_limit",
            "at_rest",
            "max_force_reached",
            "timeout",
            "duration_completed",
            "safety_abort",
        }


class TestDynamicsParameters:
    """The main paper's hidden state is exactly four dimensional (DECISIONS D015)."""

    def test_xi_is_four_dimensional(self) -> None:
        assert XI_FIELDS == (
            "drawer_mass",
            "joint_static_friction",
            "joint_dynamic_friction",
            "joint_damping",
        )
        assert len(PRESETS["medium"].as_vector()) == 4

    def test_as_vector_follows_the_canonical_order(self) -> None:
        params = DynamicsParameters(7.0, 4.0, 2.0, 5.0)
        assert params.as_vector() == (7.0, 4.0, 2.0, 5.0)

    def test_as_dict_carries_every_xi_field_and_no_removed_one(self) -> None:
        payload = PRESETS["medium"].as_dict()
        assert set(XI_FIELDS) <= set(payload)
        assert "joint_friction" not in payload, "the merged friction field is gone"
        assert "joint_stiffness" not in payload, "stiffness is not part of xi (D008)"

    def test_presets_are_ordered_by_difficulty(self) -> None:
        easy, medium, hard = (PRESETS[n] for n in ("easy", "medium", "hard"))
        for attribute in XI_FIELDS:
            values = [getattr(p, attribute) for p in (easy, medium, hard)]
            assert values == sorted(values), f"{attribute} must not decrease from easy to hard: {values}"

    def test_every_preset_respects_the_physx_friction_ordering(self) -> None:
        for name, params in PRESETS.items():
            assert params.joint_dynamic_friction <= params.joint_static_friction, name

    def test_asymmetric_friction_presets_exist(self) -> None:
        """The friction split must be exercisable on its own, not only via easy/medium/hard."""
        assert PRESETS["sticky"].friction_ratio < 0.5
        assert PRESETS["slippery"].friction_ratio < 0.5
        assert PRESETS["medium"].friction_ratio == pytest.approx(1.0)

    def test_friction_ratio_is_defined_without_static_friction(self) -> None:
        assert DynamicsParameters(5.0, 0.0, 0.0, 1.0).friction_ratio == pytest.approx(1.0)

    def test_unknown_preset_lists_the_valid_ones(self) -> None:
        with pytest.raises(KeyError, match="Available: "):
            preset("impossible")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"drawer_mass": 0.0}, "drawer_mass must be > 0"),
            ({"joint_static_friction": -1.0}, "joint_static_friction must be >= 0"),
            ({"joint_dynamic_friction": -1.0}, "joint_dynamic_friction must be >= 0"),
            ({"joint_damping": -1.0}, "joint_damping must be >= 0"),
        ],
    )
    def test_rejects_invalid(self, kwargs: dict, match: str) -> None:
        args = {
            "drawer_mass": 5.0,
            "joint_static_friction": 1.0,
            "joint_dynamic_friction": 1.0,
            "joint_damping": 1.0,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            DynamicsParameters(**args)

    def test_rejects_dynamic_friction_above_static(self) -> None:
        """PhysX discards such a write silently, so it must never reach the simulator."""
        with pytest.raises(ValueError, match="PhysX enforces"):
            DynamicsParameters(5.0, 1.0, 5.0, 1.0)


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
            assert cfg.static_friction_range[0] <= params.joint_static_friction <= cfg.static_friction_range[1]
            assert cfg.damping_range[0] <= params.joint_damping <= cfg.damping_range[1]
            low, high = cfg.dynamic_friction_ratio_range
            assert low - 1e-9 <= params.friction_ratio <= high + 1e-9

    def test_every_sample_is_writable_by_physx(self) -> None:
        """Sampling a ratio rather than an absolute value makes this true by construction."""
        for params in DynamicsRandomizer(seed=5).sample(256):
            assert params.joint_dynamic_friction <= params.joint_static_friction

    def test_sampling_spans_the_friction_asymmetry(self) -> None:
        ratios = [p.friction_ratio for p in DynamicsRandomizer(seed=9).sample(256)]
        low, high = DynamicsRandomizerCfg().dynamic_friction_ratio_range
        assert min(ratios) < low + 0.1 * (high - low)
        assert max(ratios) > high - 0.1 * (high - low)

    def test_rejects_a_ratio_range_above_one(self) -> None:
        with pytest.raises(ValueError, match="must lie inside"):
            DynamicsRandomizerCfg(dynamic_friction_ratio_range=(0.5, 1.5))

    def test_rejects_unordered_ranges(self) -> None:
        with pytest.raises(ValueError, match="must be ordered"):
            DynamicsRandomizerCfg(mass_range=(10.0, 2.0))

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
