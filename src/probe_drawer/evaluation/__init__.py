"""Turning recorded executions into labels: validity, then success.

Nothing here touches the simulation or the controllers. Every judgement is made after the
fact from an :class:`~probe_drawer.controllers.types.ExecutionResult`, which is what keeps
the goal displacement out of the control loop (``docs/DECISIONS.md`` D004).
"""

from .force_selection import ForceSelection, SelectionCfg, select_forces, select_nearest
from .operating_region import (
    DRAWER_TRAVEL_LIMIT,
    PROVISIONAL_VALIDATION_DURATION,
    PROVISIONAL_VALIDATION_PEAK_FORCE,
    InvalidReason,
    OperatingRegionCfg,
    ValidityReport,
    ValidityVerdict,
    assess_validity,
)
from .task_evaluator import EvaluationReport, ExecutionVerdict, SuccessCriteria, evaluate_execution

__all__ = [
    "DRAWER_TRAVEL_LIMIT",
    "PROVISIONAL_VALIDATION_DURATION",
    "PROVISIONAL_VALIDATION_PEAK_FORCE",
    "EvaluationReport",
    "ExecutionVerdict",
    "ForceSelection",
    "InvalidReason",
    "OperatingRegionCfg",
    "SelectionCfg",
    "SuccessCriteria",
    "ValidityReport",
    "ValidityVerdict",
    "assess_validity",
    "evaluate_execution",
    "select_forces",
    "select_nearest",
]
