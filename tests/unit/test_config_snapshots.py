"""Keep ``configs/*.yaml`` honest.

The dataclasses are the single source of truth for configuration; ``configs/*.yaml`` is a
snapshot of their defaults so the current settings can be read without opening the code.
A snapshot nobody checks is worse than no snapshot, so these tests rebuild the expected
contents from the dataclasses and fail -- printing the corrected mapping -- if the files
have drifted.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest
import yaml

from probe_drawer.controllers import ExecutionControllerCfg, HybridPullOSCCfg, ProbeControllerCfg, SafetyLimits
from probe_drawer.envs import PRESETS, XI_FIELDS, DynamicsRandomizerCfg, HybridPullControlCfg
from probe_drawer.evaluation import (
    DRAWER_TRAVEL_LIMIT,
    PROVISIONAL_VALIDATION_DURATION,
    PROVISIONAL_VALIDATION_PEAK_FORCE,
    OperatingRegionCfg,
)
from probe_drawer.sensors import DrawerStateCfg
from probe_drawer.utils import project_root


def _snapshot(name: str) -> dict:
    path = project_root() / "configs" / name
    assert path.is_file(), f"missing configuration snapshot {path}"
    return yaml.safe_load(path.read_text())


def _as_yaml_types(payload: dict) -> dict:
    """Normalise a dataclass mapping the way PyYAML round-trips it (tuples become lists)."""
    return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


def _assert_section(snapshot: dict, section: str, expected: dict, name: str) -> None:
    __tracebackhide__ = True
    actual = snapshot.get(section)
    if actual != expected:
        pytest.fail(
            f"configs/{name} section {section!r} has drifted from the dataclass defaults.\n"
            f"expected:\n{yaml.safe_dump({section: expected}, sort_keys=False)}"
            f"found:\n{yaml.safe_dump({section: actual}, sort_keys=False)}"
        )


class TestControllerSnapshot:
    def test_hybrid_pull_control_matches(self) -> None:
        cfg = HybridPullControlCfg()
        expected = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        _assert_section(_snapshot("controller.yaml"), "hybrid_pull_control", expected, "controller.yaml")

    def test_derived_axes_match(self) -> None:
        axis = HybridPullControlCfg().pull_axis()
        snapshot = _snapshot("controller.yaml")
        _assert_section(snapshot, "derived_pull_axis", axis.as_dict(), "controller.yaml")
        _assert_section(
            snapshot,
            "derived_osc_axes",
            {
                "motion_control_axes_task": list(axis.motion_control_axes()),
                "contact_wrench_control_axes_task": list(axis.wrench_control_axes()),
            },
            "controller.yaml",
        )

    def test_osc_and_safety_and_reader_match(self) -> None:
        snapshot = _snapshot("controller.yaml")
        _assert_section(snapshot, "hybrid_pull_osc", asdict(HybridPullOSCCfg()), "controller.yaml")
        _assert_section(snapshot, "safety_limits", SafetyLimits().as_dict(), "controller.yaml")
        _assert_section(snapshot, "drawer_state", _as_yaml_types(asdict(DrawerStateCfg())), "controller.yaml")


class TestProbeSnapshot:
    def test_probe_controller_matches(self) -> None:
        _assert_section(_snapshot("probe.yaml"), "probe_controller", ProbeControllerCfg().as_dict(), "probe.yaml")

    def test_example_task_parameters_are_valid(self) -> None:
        """The documented example must actually be accepted by the controller."""
        from probe_drawer.controllers.probe_pull_controller import ProbePullController  # noqa: PLC0415

        ProbePullController._validate_task_parameters(**_snapshot("probe.yaml")["example_task_parameters"])


class TestExecutionSnapshot:
    def test_execution_controller_matches(self) -> None:
        _assert_section(
            _snapshot("execution.yaml"), "execution_controller", ExecutionControllerCfg().as_dict(), "execution.yaml"
        )

    def test_reference_operating_point_matches(self) -> None:
        _assert_section(
            _snapshot("execution.yaml"),
            "reference_operating_point",
            {
                "peak_force": PROVISIONAL_VALIDATION_PEAK_FORCE,
                "duration": PROVISIONAL_VALIDATION_DURATION,
            },
            "execution.yaml",
        )


class TestDynamicsSnapshot:
    def test_presets_match(self) -> None:
        expected = {name: params.as_dict() for name, params in PRESETS.items()}
        _assert_section(_snapshot("dynamics.yaml"), "presets", expected, "dynamics.yaml")

    def test_randomizer_matches(self) -> None:
        expected = _as_yaml_types(asdict(DynamicsRandomizerCfg()))
        _assert_section(_snapshot("dynamics.yaml"), "randomizer", expected, "dynamics.yaml")

    def test_hidden_state_is_exactly_the_four_xi_fields(self) -> None:
        assert _snapshot("dynamics.yaml")["hidden_state"] == list(XI_FIELDS)

    def test_every_snapshotted_preset_respects_the_physx_friction_ordering(self) -> None:
        for name, params in _snapshot("dynamics.yaml")["presets"].items():
            assert params["joint_dynamic_friction"] <= params["joint_static_friction"], name


class TestEvaluationSnapshot:
    def test_operating_region_matches(self) -> None:
        _assert_section(
            _snapshot("evaluation.yaml"), "operating_region", OperatingRegionCfg().as_dict(), "evaluation.yaml"
        )

    def test_travel_limit_matches_the_measured_asset(self) -> None:
        assert _snapshot("evaluation.yaml")["drawer_travel_limit"] == DRAWER_TRAVEL_LIMIT

    def test_validity_thresholds_are_tighter_than_the_safety_limits(self) -> None:
        """Validity decides whether an episode is usable; safety only stops divergence."""
        region = OperatingRegionCfg()
        limits = SafetyLimits()
        assert region.max_lateral_drift < limits.max_lateral_error
        assert region.max_orientation_drift_deg < limits.max_orientation_error_deg
        assert region.max_peak_velocity < limits.max_drawer_velocity


class TestExperimentPlanSnapshot:
    """The selected experiment parameters, mirrored for review."""

    def test_every_section_matches_the_plan(self) -> None:
        from probe_drawer.experiment_plan import (  # noqa: PLC0415
            MAIN_TASK,
            OOD_XI_RANGES,
            PHASE9_RESET_TASK,
            RECOMMENDED_EXECUTION_CFG,
            RECOMMENDED_PROBE_CFG,
            RECOMMENDED_PROBE_TASK,
            SEQUENTIAL_TRANSITION_STEPS,
            TRAINING_XI_RANGES,
        )

        snapshot = _snapshot("experiment_plan.yaml")
        assert snapshot["sequential_transition_steps"] == SEQUENTIAL_TRANSITION_STEPS
        for section, expected in (
            ("main_task", MAIN_TASK.as_dict()),
            ("phase9_reset_task", PHASE9_RESET_TASK.as_dict()),
            ("probe_task", RECOMMENDED_PROBE_TASK.as_dict()),
            ("probe_controller", RECOMMENDED_PROBE_CFG.as_dict()),
            ("execution_controller", RECOMMENDED_EXECUTION_CFG.as_dict()),
            ("training_xi", TRAINING_XI_RANGES.as_dict()),
            ("ood_xi", OOD_XI_RANGES.as_dict()),
        ):
            _assert_section(snapshot, section, expected, "experiment_plan.yaml")

    def test_provenance_is_recorded(self) -> None:
        """A selected parameter with no cited sweep is a guess by another name."""
        provenance = _snapshot("experiment_plan.yaml")["_provenance"]
        for key in ("main_task", "transition", "probe", "xi_ranges"):
            assert "scripts/" in provenance[key]


class TestEverySnapshotNamesItsSource:
    @pytest.mark.parametrize(
        "name",
        [
            "controller.yaml",
            "probe.yaml",
            "execution.yaml",
            "dynamics.yaml",
            "evaluation.yaml",
            "experiment_plan.yaml",
        ],
    )
    def test_source_is_declared_and_importable(self, name: str) -> None:
        import importlib  # noqa: PLC0415

        sources = _snapshot(name)["_source"]
        assert sources, f"configs/{name} must declare where its values come from"
        for target in sources.values():
            module_name, _, attribute = target.partition(":")
            module = importlib.import_module(module_name)
            assert hasattr(module, attribute), f"{target} does not exist"

    def test_no_stale_snapshot_files(self) -> None:
        """Every YAML in configs/ is either a snapshot tested above or a recorded artefact."""
        known = {
            "controller.yaml",
            "probe.yaml",
            "execution.yaml",
            "dynamics.yaml",
            "evaluation.yaml",
            "experiment_plan.yaml",
            "grasp_pose.yaml",
        }
        found = {path.name for path in (project_root() / "configs").glob("*.yaml")}
        assert found == known, f"unexpected or missing configuration files: {found ^ known}"
