"""Choosing a peak force from a predicted success landscape.

This is *not* SPC. It is a search: score a grid of candidate forces and take the best one.
It lives in ``evaluation/`` rather than in a controller on purpose -- the execution
controller still takes only ``(peak_force, duration)`` and must never learn what a goal is
(D004). The search happens outside, before the execution starts, exactly as a deployed system
would run its model between the probe and the pull.

The grid is 0.05 N over the task's force range, which is the resolution the Phase 10 Oracle
resolved the success band at. Finer would imply a precision the labels do not have.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["ForceSelection", "SelectionCfg", "select_forces"]


@dataclass(frozen=True)
class SelectionCfg:
    """How the candidate search is run.

    Args:
        force_range: Grid bounds (N). Defaults are set by the caller from ``MAIN_TASK``.
        step: Grid spacing (N).
        abstain_below: If the best score falls below this, the selection is flagged as
            low-confidence. Nothing is abstained from -- this round still executes the
            argmax -- but the flag and the score are recorded, because "the model could not
            find a force it believed in" is the signal a future abstention policy needs and
            it is free to collect now (§25.1).
    """

    force_range: tuple[float, float] = (0.15, 4.5)
    step: float = 0.05
    abstain_below: float = 0.5

    def grid(self) -> np.ndarray:
        low, high = self.force_range
        return np.round(np.arange(low, high + 1e-9, self.step), 6)


@dataclass
class ForceSelection:
    """What the search chose, and how sure it was.

    Attributes:
        force: The selected peak force per environment (N).
        score: The score at the selection.
        best_index: Index into the grid.
        low_confidence: Whether the best score fell below ``abstain_below``.
        landscape: ``(num_envs, grid)`` scores, kept so a landscape can be plotted or a
            future abstention rule can be fitted without re-running the simulator.
        grid: The forces scored.
    """

    force: np.ndarray
    score: np.ndarray
    best_index: np.ndarray
    low_confidence: np.ndarray
    landscape: np.ndarray
    grid: np.ndarray

    def as_dict(self) -> dict:
        return {
            "force": self.force.tolist(),
            "score": self.score.tolist(),
            "low_confidence": self.low_confidence.tolist(),
            "grid": self.grid.tolist(),
        }


def select_forces(
    score_fn: Callable[[np.ndarray], np.ndarray],
    num_envs: int,
    cfg: SelectionCfg | None = None,
) -> ForceSelection:
    """Score every grid force for every environment and take each one's argmax.

    Args:
        score_fn: Given a ``(num_envs,)`` array of candidate forces, returns a ``(num_envs,)``
            array of scores. Called once per grid point, so the caller decides what a score
            is -- a success probability for a landscape model, or negative distance for a
            force regressor.
        num_envs: How many environments are being selected for.
        cfg: Grid and confidence settings.

    Returns:
        A :class:`ForceSelection`. Deterministic: ties resolve to the lowest force, which is
        the conservative choice for a drawer.
    """
    cfg = cfg or SelectionCfg()
    grid = cfg.grid()
    landscape = np.empty((num_envs, len(grid)), dtype=float)
    for index, force in enumerate(grid):
        scores = np.asarray(score_fn(np.full(num_envs, force, dtype=float)), dtype=float)
        if scores.shape != (num_envs,):
            raise ValueError(f"score_fn must return {(num_envs,)} scores, got {scores.shape}.")
        landscape[:, index] = scores

    best_index = landscape.argmax(axis=1)
    rows = np.arange(num_envs)
    best_score = landscape[rows, best_index]
    return ForceSelection(
        force=grid[best_index],
        score=best_score,
        best_index=best_index,
        low_confidence=best_score < cfg.abstain_below,
        landscape=landscape,
        grid=grid,
    )


def select_nearest(predicted: Sequence[float], cfg: SelectionCfg | None = None) -> ForceSelection:
    """Snap a directly regressed force onto the grid.

    So a force regressor and a landscape model are compared on the same set of executable
    forces, rather than one of them being allowed arbitrary precision.
    """
    cfg = cfg or SelectionCfg()
    grid = cfg.grid()
    predicted = np.asarray(predicted, dtype=float)
    distance = np.abs(grid[None, :] - predicted[:, None])
    best_index = distance.argmin(axis=1)
    rows = np.arange(len(predicted))
    return ForceSelection(
        force=grid[best_index],
        score=-distance[rows, best_index],
        best_index=best_index,
        low_confidence=np.zeros(len(predicted), dtype=bool),
        landscape=-distance,
        grid=grid,
    )
