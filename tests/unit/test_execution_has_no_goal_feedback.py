"""Enforce, mechanically, that the execution controller has no goal feedback.

Spec section 27, Execution Test 3 makes this a mandatory review item.  A review item that
depends on somebody remembering to review is worth exactly as much as the reviewer's
attention, so it is a test instead: these assertions fail if anyone reintroduces a
displacement- or goal-triggered stop into the execution path.

See ``docs/DECISIONS.md`` D004.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from probe_drawer.controllers import execution_pull_controller
from probe_drawer.controllers.base_pull_controller import BasePullController
from probe_drawer.controllers.execution_pull_controller import ExecutionPullController

#: Names that must not appear anywhere in the execution controller's code.
FORBIDDEN_NAMES = ("d_goal", "goal", "target_displacement", "epsilon", "success")


def _source() -> str:
    return Path(inspect.getsourcefile(execution_pull_controller)).read_text()  # type: ignore[arg-type]


def _code_only(source: str) -> ast.Module:
    """Parse the module, so that docstrings and comments are excluded from the checks."""
    return ast.parse(source)


def _identifiers(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


class TestNoGoalFeedback:
    @pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
    def test_no_goal_identifier_in_the_execution_controller(self, forbidden: str) -> None:
        assert forbidden not in _identifiers(_code_only(_source())), (
            f"{forbidden!r} appears in the execution controller's code. The goal displacement "
            "must stay out of the low-level control loop (DECISIONS D004)."
        )

    def test_run_takes_only_peak_force_duration_and_an_observer(self) -> None:
        """The signature stays minimal, and nothing goal-shaped may join it (D004).

        ``on_step`` was added for the diagnostic video recorder. It is an *observer*: the
        controller calls it and never reads it back, so it cannot influence the run. The
        allowance is explicit and narrow rather than a relaxation -- the second assertion
        below is new, and forbids by name the whole family of parameters this guard exists
        to keep out, which the old exact-list check only excluded incidentally.
        """
        parameters = list(inspect.signature(ExecutionPullController.run).parameters)
        assert parameters == ["self", "peak_force", "duration", "on_step"]

        forbidden = ("goal", "target", "reference", "setpoint", "desired", "criteria", "tolerance")
        offenders = [name for name in parameters for word in forbidden if word in name.lower()]
        assert not offenders, offenders

    def test_the_step_observer_cannot_influence_the_run(self) -> None:
        """``on_step``'s return value must be discarded, or it would be a feedback path."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(BasePullController.run_profile)))
        # One parse, walked once: comparing nodes across two parses would compare different
        # objects and the identity check would never hold.
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "on_step"
        ]
        assert calls, "run_profile should call on_step"
        discarded = {
            id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)
        }
        for call in calls:
            assert id(call) in discarded, "on_step's return value must be discarded"

    def test_stop_conditions_is_empty(self) -> None:
        """The task-stop hook must return nothing at all, whatever the state."""
        controller = ExecutionPullController.__new__(ExecutionPullController)
        assert tuple(ExecutionPullController._stop_conditions(controller, elapsed=1.0, commanded_force=None)) == ()

    def test_stop_conditions_reads_no_state(self) -> None:
        """It must not read anything off ``self``: that is the first step towards feedback."""
        source = textwrap.dedent(inspect.getsource(ExecutionPullController._stop_conditions))
        self_attributes = [
            node.attr
            for node in ast.walk(_code_only(source))
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"
        ]
        assert self_attributes == []

    def test_execution_result_carries_no_success_field(self) -> None:
        from probe_drawer.controllers.types import ExecutionResult  # noqa: PLC0415

        assert not any("success" in name or "goal" in name for name in ExecutionResult.__dataclass_fields__)
