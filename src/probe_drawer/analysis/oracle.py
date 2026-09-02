r"""The Oracle success landscape, and choosing the task from it.

:math:`S_\text{oracle}(F_\text{peak}, T \mid \xi, d_\text{goal})` is not a model. It is the
label the physics gives: run the execution, apply the success definition, record the answer.
This module reads a completed sweep and answers the two questions that decide whether the
research problem is even well posed.

**Is the task learnable?** For a fixed :math:`(T, d_\text{goal})` each hidden state has a
band of peak forces that succeed. The band must exist for most hidden states, must be wide
enough that a model has some tolerance, and must contain more than a single grid point --
otherwise there is nothing to fit.

**Is adaptation necessary?** The bands must sit at *different* forces for different hidden
states. If one force succeeded everywhere, a constant would solve the task and a probe would
be pointless. :attr:`CandidateScore.discrimination` measures exactly this, and it is the
quantity the recommendation maximises.

The five acceptance conditions below are the task-design requirements made testable. A
candidate that fails any of them is rejected with the reason recorded, so a rejection is
reviewable rather than a silent absence from the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from probe_drawer.analysis.sweep import SweepDataset, success_interval
from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = [
    "ACCEPTANCE",
    "AcceptanceThresholds",
    "CandidateScore",
    "TaskCandidate",
    "score_candidate",
    "select_task_parameters",
]


@dataclass(frozen=True)
class TaskCandidate:
    """One possible task definition: how long, how far, and how precisely."""

    duration: float
    goal_displacement: float
    displacement_tolerance: float
    velocity_tolerance: float

    @property
    def criteria(self) -> SuccessCriteria:
        return SuccessCriteria(
            goal_displacement=self.goal_displacement,
            displacement_tolerance=self.displacement_tolerance,
            velocity_tolerance=self.velocity_tolerance,
        )

    def as_dict(self) -> dict:
        return {
            "duration": self.duration,
            "goal_displacement": self.goal_displacement,
            "displacement_tolerance": self.displacement_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
        }


@dataclass(frozen=True)
class AcceptanceThresholds:
    """When a task definition is fit to build the paper on.

    Args:
        min_coverage: Fraction of hidden states that must have at least one succeeding
            force. Below this the task is unachievable for too much of the training
            distribution.
        min_discrimination: Required spread of the per-hidden-state optimal force, as
            ``(max - min) / median``. This is the whole point of the study: if the force a
            drawer needs barely varies, one constant would do and no probe is needed.
        min_relative_width: Smallest acceptable median band width, as ``width / centre``. A
            model has to predict a number; if only a 2 % force error is tolerated the task is
            a knife edge no regression can be expected to hit.
        max_relative_width: Largest acceptable median band width, same units. A band
            covering most of the force axis means the task tolerates anything and adaptation
            buys nothing.
        max_travel_fraction: Largest travel fraction any succeeding episode may reach, so
            the goal is not adjacent to the mechanical end stop.
        min_contiguous_fraction: Fraction of hidden states whose succeeding forces form an
            unbroken run. A band with holes in it is not a band, and a regression trained
            against one would be fitting noise.
        max_tolerance_ratio: Largest acceptable ``eps_d / d_goal``. A goal of 20 mm with a
            15 mm tolerance is not a positioning task; requiring the tolerance to be a
            modest fraction of the goal is what keeps the task meaningful.
        require_grid_resolved: Reject candidates whose success band is narrower than
            1.5 grid steps. Such a candidate is not necessarily bad -- the *sweep* is too
            coarse to characterise it -- so accepting it would mean reporting a band width
            that is really the grid spacing.

    Note what is *not* here: a minimum number of succeeding grid points, which would measure
    the sweep's force resolution rather than the task. That concern is handled by
    :attr:`require_grid_resolved` instead, and the remedy when it bites is a finer force
    grid.
    """

    min_coverage: float = 0.80
    min_discrimination: float = 0.50
    min_relative_width: float = 0.10
    max_relative_width: float = 0.60
    max_travel_fraction: float = 0.70
    min_contiguous_fraction: float = 0.95
    max_tolerance_ratio: float = 0.30
    require_grid_resolved: bool = True


#: The thresholds used for the recommendation. Recorded in ``docs/EXPERIMENT_SPACE.md``.
ACCEPTANCE = AcceptanceThresholds()


@dataclass
class CandidateScore:
    """How well one task definition satisfies the five acceptance conditions."""

    candidate: TaskCandidate
    intervals: list[dict]
    coverage: float
    discrimination: float
    median_relative_width: float
    median_success_levels: float
    median_absolute_width: float
    grid_step: float
    max_travel_fraction: float
    contiguous_fraction: float
    failures: list[str]

    @property
    def grid_resolves_band(self) -> bool:
        """Whether the swept force grid is fine enough to see the band's width.

        Not an acceptance condition: when this is false the sweep is too coarse, and the
        remedy is a finer force grid rather than a different task.
        """
        return bool(self.median_absolute_width >= 1.5 * self.grid_step)

    @property
    def accepted(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate.as_dict(),
            "coverage": self.coverage,
            "discrimination": self.discrimination,
            "median_relative_width": self.median_relative_width,
            "median_success_levels": self.median_success_levels,
            "median_absolute_width": self.median_absolute_width,
            "grid_step": self.grid_step,
            "grid_resolves_band": self.grid_resolves_band,
            "max_travel_fraction": self.max_travel_fraction,
            "contiguous_fraction": self.contiguous_fraction,
            "accepted": self.accepted,
            "failures": self.failures,
            "num_hidden_states": len(self.intervals),
            "num_with_success": sum(1 for row in self.intervals if row["any_success"]),
        }


def score_candidate(
    dataset: SweepDataset,
    candidate: TaskCandidate,
    thresholds: AcceptanceThresholds = ACCEPTANCE,
) -> CandidateScore:
    """Evaluate one task definition against the whole hidden-state grid."""
    criteria = candidate.criteria
    intervals = [
        success_interval(dataset, xi_key, criteria, candidate.duration) for xi_key in dataset.xi_keys()
    ]
    achieved = [row for row in intervals if row["any_success"]]

    coverage = len(achieved) / len(intervals) if intervals else 0.0
    centres = [row["force_centre"] for row in achieved]
    widths = [row["relative_width"] for row in achieved if row["relative_width"] is not None]
    levels = [len(row["success_forces"]) for row in achieved]

    discrimination = 0.0
    if len(centres) >= 2 and float(np.median(centres)) > 0:
        discrimination = float((max(centres) - min(centres)) / np.median(centres))

    # Reuse the per-hidden-state rows already gathered above rather than rescanning the
    # whole dataset: candidate scoring runs hundreds of times per report.
    max_travel = max(
        (
            record.travel_fraction
            for xi_key in dataset.xi_keys()
            for record in dataset.select(xi_key=xi_key, duration=candidate.duration)
            if record.succeeds(criteria)
        ),
        default=0.0,
    )

    swept_forces = dataset.forces()
    grid_step = min(
        (high - low for low, high in zip(swept_forces, swept_forces[1:], strict=False)), default=float("inf")
    )
    absolute_widths = [row["force_width"] for row in achieved]

    failures: list[str] = []
    if coverage < thresholds.min_coverage:
        failures.append(f"coverage {coverage:.2f} < {thresholds.min_coverage:.2f}")
    if discrimination < thresholds.min_discrimination:
        failures.append(f"discrimination {discrimination:.2f} < {thresholds.min_discrimination:.2f}")
    if widths and float(np.median(widths)) > thresholds.max_relative_width:
        failures.append(f"width {float(np.median(widths)):.2f} > {thresholds.max_relative_width:.2f}")
    if widths and float(np.median(widths)) < thresholds.min_relative_width:
        failures.append(f"width {float(np.median(widths)):.2f} < {thresholds.min_relative_width:.2f}")
    if not widths:
        failures.append("width no-succeeding-hidden-states")
    if max_travel > thresholds.max_travel_fraction:
        failures.append(f"travel fraction {max_travel:.2f} > {thresholds.max_travel_fraction:.2f}")
    contiguous_fraction = sum(1 for row in achieved if row["contiguous"]) / len(achieved) if achieved else 0.0
    if contiguous_fraction < thresholds.min_contiguous_fraction:
        failures.append(f"contiguity {contiguous_fraction:.2f} < {thresholds.min_contiguous_fraction:.2f}")
    tolerance_ratio = candidate.displacement_tolerance / candidate.goal_displacement
    if tolerance_ratio > thresholds.max_tolerance_ratio:
        failures.append(f"tolerance-ratio {tolerance_ratio:.2f} > {thresholds.max_tolerance_ratio:.2f}")
    median_absolute_width = float(np.median(absolute_widths)) if absolute_widths else 0.0
    if thresholds.require_grid_resolved and median_absolute_width < 1.5 * grid_step:
        failures.append(f"grid-resolution band {median_absolute_width:.2f} N < 1.5 x step {grid_step:.2f} N")

    return CandidateScore(
        candidate=candidate,
        intervals=intervals,
        coverage=coverage,
        discrimination=discrimination,
        median_relative_width=float(np.median(widths)) if widths else float("nan"),
        median_success_levels=float(np.median(levels)) if levels else 0.0,
        median_absolute_width=median_absolute_width,
        grid_step=grid_step,
        max_travel_fraction=max_travel,
        contiguous_fraction=contiguous_fraction,
        failures=failures,
    )


def select_task_parameters(
    dataset: SweepDataset,
    candidates: Sequence[TaskCandidate],
    thresholds: AcceptanceThresholds = ACCEPTANCE,
) -> dict:
    """Score every candidate and recommend the most discriminating accepted one.

    Discrimination is the tie-breaker rather than coverage or precision because it is the
    property the research question depends on: a task where every drawer needs a different
    force is a task where a probe earns its place.

    Returns:
        A report with every candidate's score, the recommendation, and -- if nothing was
        accepted -- the reasons, so the next step is to change the experiment rather than to
        lower the bar.
    """
    scores = [score_candidate(dataset, candidate, thresholds) for candidate in candidates]
    accepted = [score for score in scores if score.accepted]
    best = max(accepted, key=lambda score: score.discrimination) if accepted else None

    return {
        "thresholds": {
            "min_coverage": thresholds.min_coverage,
            "min_discrimination": thresholds.min_discrimination,
            "min_relative_width": thresholds.min_relative_width,
            "max_relative_width": thresholds.max_relative_width,
            "max_travel_fraction": thresholds.max_travel_fraction,
            "min_contiguous_fraction": thresholds.min_contiguous_fraction,
            "max_tolerance_ratio": thresholds.max_tolerance_ratio,
            "require_grid_resolved": thresholds.require_grid_resolved,
        },
        "num_candidates": len(scores),
        "num_accepted": len(accepted),
        "scores": [score.as_dict() for score in scores],
        "recommended": best.as_dict() if best else None,
        "recommended_intervals": best.intervals if best else None,
        "rejection_summary": _rejection_summary(scores) if not accepted else None,
    }


def _rejection_summary(scores: Sequence[CandidateScore]) -> dict:
    """Which condition eliminated the most candidates, and how close the best came."""
    counts: dict[str, int] = {}
    for score in scores:
        for failure in score.failures:
            key = failure.split()[0]
            counts[key] = counts.get(key, 0) + 1
    closest = min(scores, key=lambda score: len(score.failures), default=None)
    return {
        "failures_by_condition": dict(sorted(counts.items(), key=lambda item: -item[1])),
        "closest_candidate": closest.as_dict() if closest else None,
    }
