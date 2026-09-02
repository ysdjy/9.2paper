r"""Whether an execution accomplished the task. Deliberately outside the controller.

The execution controller is given a peak force and a duration and nothing else; it never
learns what the goal was (``docs/DECISIONS.md`` D004). Turning its output into a success
label is this module's job, and it happens after the fact, from recorded values only.

Success has three parts, all required:

.. math::

    |d(T) - d_\text{goal}| \le \epsilon_d
    \quad\wedge\quad
    |v(T)| \le \epsilon_v
    \quad\wedge\quad
    \text{valid operating point}

The terminal-velocity term is not decoration. A drawer that reaches the goal at
0.4 m/s has not been placed there -- it is passing through, and a moment later it is
somewhere else or against its end stop. Requiring ``|v(T)|`` to be small is what makes
"reached the goal" mean "came to rest at the goal" (``docs/DECISIONS.md`` D020).

Validity is delegated to :mod:`probe_drawer.evaluation.operating_region`, which also
subsumes the safety check: a safety-aborted episode is never a valid operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probe_drawer.controllers.types import ExecutionResult
from probe_drawer.evaluation.operating_region import (
    InvalidReason,
    OperatingRegionCfg,
    ValidityReport,
    assess_validity,
)

__all__ = ["EvaluationReport", "ExecutionVerdict", "SuccessCriteria", "evaluate_execution"]


@dataclass(frozen=True)
class SuccessCriteria:
    """The task definition an execution is judged against.

    Args:
        goal_displacement: :math:`d_\\text{goal}`, how far the drawer should end up open (m).
        displacement_tolerance: :math:`\\epsilon_d` (m). Must be positive.
        velocity_tolerance: :math:`\\epsilon_v`, the largest terminal speed that still counts
            as having come to rest (m/s). Must be positive.
    """

    goal_displacement: float
    displacement_tolerance: float
    velocity_tolerance: float

    def __post_init__(self) -> None:
        if self.goal_displacement <= 0.0:
            raise ValueError(f"goal_displacement must be > 0 m, got {self.goal_displacement}.")
        if self.displacement_tolerance <= 0.0:
            raise ValueError(f"displacement_tolerance must be > 0 m, got {self.displacement_tolerance}.")
        if self.velocity_tolerance <= 0.0:
            raise ValueError(f"velocity_tolerance must be > 0 m/s, got {self.velocity_tolerance}.")

    def as_dict(self) -> dict:
        return {
            "goal_displacement": self.goal_displacement,
            "displacement_tolerance": self.displacement_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
        }


@dataclass
class ExecutionVerdict:
    """One environment's success label, with every term that produced it.

    Attributes:
        success: All three requirements met.
        displacement_ok: :math:`|d(T) - d_\\text{goal}| \\le \\epsilon_d`.
        velocity_ok: :math:`|v(T)| \\le \\epsilon_v`.
        valid: The operating point is usable (which includes not having safety-aborted).
        displacement_error: Signed :math:`d(T) - d_\\text{goal}` (m); positive means overshoot.
        terminal_velocity: :math:`v(T)` (m/s).
        invalid_reasons: Why the operating point was rejected, if it was.
    """

    success: bool
    displacement_ok: bool
    velocity_ok: bool
    valid: bool
    displacement_error: float
    terminal_velocity: float
    invalid_reasons: list[InvalidReason] = field(default_factory=list)

    @property
    def safety_aborted(self) -> bool:
        """Whether an absolute safety limit ended the episode."""
        return InvalidReason.SAFETY_ABORT in self.invalid_reasons

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "displacement_ok": self.displacement_ok,
            "velocity_ok": self.velocity_ok,
            "valid": self.valid,
            "displacement_error": self.displacement_error,
            "terminal_velocity": self.terminal_velocity,
            "invalid_reasons": [reason.value for reason in self.invalid_reasons],
        }


@dataclass
class EvaluationReport:
    """Per-environment verdicts for one execution, plus the criteria used."""

    verdicts: list[ExecutionVerdict]
    criteria: SuccessCriteria
    validity: ValidityReport

    @property
    def success(self) -> np.ndarray:
        """Boolean success mask, shape ``(num_envs,)``."""
        return np.asarray([verdict.success for verdict in self.verdicts], dtype=bool)

    def as_dict(self) -> dict:
        return {
            "criteria": self.criteria.as_dict(),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
            "validity": self.validity.as_dict(),
        }


def evaluate_execution(
    result: ExecutionResult,
    criteria: SuccessCriteria,
    operating_region: OperatingRegionCfg | None = None,
) -> EvaluationReport:
    """Label one execution, per environment.

    Args:
        result: What the execution controller returned.
        criteria: The task definition to judge against.
        operating_region: Validity thresholds. Project defaults when omitted.

    Returns:
        An :class:`EvaluationReport`. Nothing in it is written back to the controller or
        the environment.
    """
    validity = assess_validity(result, operating_region)
    verdicts: list[ExecutionVerdict] = []

    for index in range(result.num_envs):
        displacement_error = float(result.final_displacement[index]) - criteria.goal_displacement
        terminal_velocity = float(result.final_velocity[index])

        displacement_ok = abs(displacement_error) <= criteria.displacement_tolerance
        velocity_ok = abs(terminal_velocity) <= criteria.velocity_tolerance
        verdict = validity.verdicts[index]

        verdicts.append(
            ExecutionVerdict(
                success=bool(displacement_ok and velocity_ok and verdict.valid),
                displacement_ok=displacement_ok,
                velocity_ok=velocity_ok,
                valid=verdict.valid,
                displacement_error=displacement_error,
                terminal_velocity=terminal_velocity,
                invalid_reasons=list(verdict.reasons),
            )
        )

    return EvaluationReport(verdicts=verdicts, criteria=criteria, validity=validity)
