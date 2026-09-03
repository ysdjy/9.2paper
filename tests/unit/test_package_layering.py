"""The boundaries that keep this repository from becoming a pile of experiments.

Four phases of exploration have left modules that answered a question and are kept for their
answer. The risk is not that they exist -- it is that the pipeline quietly starts importing
them, and then "experimental" stops meaning anything. These tests are cheap and they fail
loudly the first time a boundary is crossed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import probe_drawer.controllers as controllers
from probe_drawer.utils import project_root

#: The packages that make up the paper's pipeline. Nothing here may import an experiment.
PIPELINE_PACKAGES = (
    "controllers",
    "dataset",
    "envs",
    "evaluation",
    "logging",
    "models",
    "protocols",
    "sensors",
    "state_machines",
    "training",
)

#: The two task-level controllers the paper has. A third would be a third thing to rule out.
PUBLIC_CONTROLLERS = ("ProbePullController", "ExecutionPullController")


def source_files(package: str) -> list[Path]:
    return sorted((project_root() / "src" / "probe_drawer" / package).rglob("*.py"))


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by a file, from its AST rather than by regex."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestExperimentsStayOutOfThePipeline:
    @pytest.mark.parametrize("package", PIPELINE_PACKAGES)
    def test_no_pipeline_module_imports_an_experiment(self, package: str) -> None:
        offenders = [
            f"{path.name}: {name}"
            for path in source_files(package)
            for name in imported_modules(path)
            if "experimental" in name
        ]
        assert not offenders, offenders

    def test_no_pipeline_module_imports_a_baseline(self) -> None:
        """The RMA2 baseline may call the main method; the main method may never call it."""
        offenders = [
            f"{package}/{path.name}: {name}"
            for package in PIPELINE_PACKAGES
            for path in source_files(package)
            for name in imported_modules(path)
            if name.startswith("baselines") or name.startswith("rma2")
        ]
        assert not offenders, offenders

    def test_experiments_are_not_re_exported(self) -> None:
        """Reaching into an experiment should be visible at the call site as a full path."""
        init = (project_root() / "src" / "probe_drawer" / "experimental" / "__init__.py").read_text()
        tree = ast.parse(init)
        assert not [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ], "experimental/__init__.py should document, not re-export"


class TestPublicControllerSurface:
    def test_exactly_two_task_level_controllers_are_public(self) -> None:
        exported = {
            name
            for name in controllers.__all__
            if name.endswith("Controller") and name != "BasePullController"
        }
        assert exported == set(PUBLIC_CONTROLLERS), exported

    def test_the_response_probe_is_not_in_the_controller_api(self) -> None:
        assert not [name for name in controllers.__all__ if "Response" in name]
        assert not hasattr(controllers, "ResponseProbeController")

    def test_the_response_probe_is_still_reachable_for_reproduction(self) -> None:
        """Demoted, not deleted -- the Phase 12 comparison must stay runnable."""
        from probe_drawer.experimental.response_probe import ResponseProbeController  # noqa: PLC0415

        assert ResponseProbeController.__name__ == "ResponseProbeController"

    def test_the_pull_system_wires_only_the_two_public_controllers(self) -> None:
        source = (project_root() / "src" / "probe_drawer" / "pull_system.py").read_text()
        assert "ResponseProbe" not in source
