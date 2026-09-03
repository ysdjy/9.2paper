r"""Whether an execution accomplished the task. Deliberately outside the controller.

The execution controller is given a peak force and a duration and nothing else; it never
learns what the goal was (``docs/DECISIONS.md`` D004). Turning its output into a success
label is this module's job, and it happens after the fact, from recorded values only.

Two success definitions, nested
-------------------------------
Setting V1 reports both, and says which is which (``docs/DECISIONS.md`` D046):

.. math::

    \text{reach} &: \quad |d(T) - d_\text{goal}| \le \epsilon_d
        \;\wedge\; \text{valid operating point} \\
    \text{stable} &: \quad \text{reach} \;\wedge\; |v(T)| \le \epsilon_v

**reach_success** is the primary metric: did the adaptation put the drawer where the task
asked, from an unknown hidden state, without leaving the usable operating region. That is
the question the paper poses, and one number answers it.

**stable_success** is the secondary one, and it is *kept*, not dropped. A drawer that
arrives at the goal at 0.4 m/s has not been placed there -- it is passing through, and a
moment later it is somewhere else or against its end stop (D020). What Phase 13 changed is
only the reporting: one combined number made a *positioning* failure and a *braking*
failure indistinguishable, and the goal-distance sweep showed they are not the same
failure -- past roughly 100 mm the terminal-velocity term is what fails first while the
position term is still comfortably met. Splitting them is what let that be seen.

Both are derived from the same three booleans, so they cannot disagree, and every
continuous quantity behind them -- ``displacement_error``, ``terminal_velocity`` -- stays on
the verdict. A threshold can be revisited offline; a discarded measurement cannot.

Validity is delegated to :mod:`probe_drawer.evaluation.operating_region`, which also
subsumes the safety check: a safety-aborted episode is never a valid operating point.

Where ``d(T)`` is measured from
------------------------------
``d_goal`` is defined relative to the drawer's position at the **start of the task, before
the probe** (``docs/DECISIONS.md`` D027). In the sequential protocol the probe itself moves
the drawer, so the quantity the task is judged on is

.. math:: d_\text{total}(T) = d_\text{probe} + d_\text{execution}(T),

not the execution segment alone. A probe that travelled 3 mm followed by an execution that
travelled 47 mm has reached the 50 mm goal. Pass the probe's contribution as
``pre_execution_displacement``; omitting it measures the execution segment alone, which is
the reset protocol Phase 9 used.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    ``success`` is a derived property rather than a stored field, so the three booleans
    below are the single source of truth and no recorded label can disagree with them.

    Attributes:
        displacement_ok: :math:`|d(T) - d_\\text{goal}| \\le \\epsilon_d`.
        velocity_ok: :math:`|v(T)| \\le \\epsilon_v`.
        valid: The operating point is usable (which includes not having safety-aborted).
        displacement_error: Signed :math:`d_\\text{total}(T) - d_\\text{goal}` (m); positive
            means overshoot.
        total_displacement: :math:`d_\\text{total}(T)`, measured from the task's start (m).
        execution_displacement: What this execution segment contributed on its own (m).
        pre_execution_displacement: What happened before it -- the probe, in the sequential
            protocol (m).
        terminal_velocity: :math:`v(T)` (m/s).
        invalid_reasons: Why the operating point was rejected, if it was.
    """

    displacement_ok: bool
    velocity_ok: bool
    valid: bool
    displacement_error: float
    total_displacement: float
    execution_displacement: float
    pre_execution_displacement: float
    terminal_velocity: float
    invalid_reasons: list[InvalidReason] = field(default_factory=list)

    @property
    def reach_success(self) -> bool:
        """**Primary metric**: ended up at the goal, from a usable operating point.

        Says nothing about whether the drawer stopped there; that is :attr:`stable_success`.
        """
        return bool(self.displacement_ok and self.valid)

    @property
    def stable_success(self) -> bool:
        """**Secondary metric**: reached the goal *and* came to rest there."""
        return bool(self.reach_success and self.velocity_ok)

    @property
    def success(self) -> bool:
        """The strict label, identical to :attr:`stable_success`.

        Kept under its original name and meaning on purpose: it is the label Dataset v0 was
        generated with, and redefining it would change how 49,152 stored rows read without
        changing a byte of them. New code should name which of the two it means.
        """
        return self.stable_success

    @property
    def safety_aborted(self) -> bool:
        """Whether an absolute safety limit ended the episode."""
        return InvalidReason.SAFETY_ABORT in self.invalid_reasons

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "reach_success": self.reach_success,
            "stable_success": self.stable_success,
            "displacement_ok": self.displacement_ok,
            "velocity_ok": self.velocity_ok,
            "valid": self.valid,
            "displacement_error": self.displacement_error,
            "total_displacement": self.total_displacement,
            "execution_displacement": self.execution_displacement,
            "pre_execution_displacement": self.pre_execution_displacement,
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
        """Strict-success mask, shape ``(num_envs,)``. Identical to :attr:`stable_success`."""
        return self.stable_success

    @property
    def reach_success(self) -> np.ndarray:
        """Primary-metric mask, shape ``(num_envs,)``."""
        return np.asarray([verdict.reach_success for verdict in self.verdicts], dtype=bool)

    @property
    def stable_success(self) -> np.ndarray:
        """Secondary-metric mask, shape ``(num_envs,)``."""
        return np.asarray([verdict.stable_success for verdict in self.verdicts], dtype=bool)

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
    pre_execution_displacement: np.ndarray | Sequence[float] | None = None,
) -> EvaluationReport:
    """Label one execution, per environment.

    Args:
        result: What the execution controller returned.
        criteria: The task definition to judge against.
        operating_region: Validity thresholds. Project defaults when omitted.
        pre_execution_displacement: Drawer displacement between the task's start and the
            start of this execution (m), per environment -- the probe's contribution in the
            sequential protocol. Omitting it judges the execution segment alone.

    Returns:
        An :class:`EvaluationReport`. Nothing in it is written back to the controller or
        the environment.
    """
    validity = assess_validity(result, operating_region, pre_execution_displacement)
    verdicts: list[ExecutionVerdict] = []

    for index in range(result.num_envs):
        metrics = validity.verdicts[index].metrics
        total_displacement = metrics["final_displacement"]
        displacement_error = total_displacement - criteria.goal_displacement
        terminal_velocity = float(result.final_velocity[index])

        displacement_ok = abs(displacement_error) <= criteria.displacement_tolerance
        velocity_ok = abs(terminal_velocity) <= criteria.velocity_tolerance
        verdict = validity.verdicts[index]

        verdicts.append(
            ExecutionVerdict(
                displacement_ok=displacement_ok,
                velocity_ok=velocity_ok,
                valid=verdict.valid,
                displacement_error=displacement_error,
                total_displacement=total_displacement,
                execution_displacement=metrics["execution_displacement"],
                pre_execution_displacement=metrics["pre_execution_displacement"],
                terminal_velocity=terminal_velocity,
                invalid_reasons=list(verdict.reasons),
            )
        )

    return EvaluationReport(verdicts=verdicts, criteria=criteria, validity=validity)
