r"""Summarising an out-of-distribution Oracle sweep.

Turns the per-state rows from ``scripts/sweep_ood_feasibility.py`` into the four answers the
question needs, and keeps the distinction that makes it useful:

**Solvable at all** vs **solvable within the task's own force range.** The sweep deliberately
runs past ``SETTING_V1_TASK.peak_force_range``, because "no force reaches the goal" and "no
force *we allow* reaches the goal" have different remedies -- the first is a drawer the rig
cannot open, the second is a truncated action range. Reporting only one number would hide which
of the two an OOD failure is.

**Where the failures are.** An unsolvable fraction is only actionable with a location, so the
failures are tallied by which axes of the hidden state are novel and in which direction. That
is what says whether the range is unreasonable on one axis or merely hard everywhere.

Pure: it reads recorded rows and touches nothing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = ["summarise_ood_feasibility"]


def _spread(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarise_ood_feasibility(rows: list[dict], allowed_range: tuple[float, float]) -> dict:
    """Aggregate an OOD sweep.

    Args:
        rows: Per-state records, each with ``reach_any_force``, ``reach_within_task_range``,
            ``required_force``, ``band_width``, ``novel_axes``, ``invalid_fraction``,
            ``safety_aborts`` and ``closest_position_error``.
        allowed_range: The task's ``peak_force_range``, used to report truncation.

    Returns:
        ``counts``, ``required_force``, ``band_width``, ``truncation``, ``failures``,
        ``novel_axis_rates`` and ``safety``.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("summarise_ood_feasibility needs at least one state.")

    low, high = allowed_range
    solvable = [row for row in rows if row["reach_any_force"]]
    in_range = [row for row in rows if row["reach_within_task_range"]]
    # Solvable, but only by a force the task does not permit. This is the truncation case, and
    # it is the one that a range change would fix -- unlike the genuinely unsolvable states.
    truncated = [row for row in solvable if not row["reach_within_task_range"]]
    unsolvable = [row for row in rows if not row["reach_any_force"]]

    required = [row["required_force"] for row in solvable]

    # Failure rate per novel axis, against how often that axis appears at all: a direction that
    # is novel in many states but fails in none is not where the trouble is.
    appearances: Counter = Counter()
    failures: Counter = Counter()
    for row in rows:
        for axis in row["novel_axes"]:
            appearances[axis] += 1
            if not row["reach_within_task_range"]:
                failures[axis] += 1

    return {
        "counts": {
            "states": len(rows),
            "solvable_any_force": len(solvable),
            "solvable_within_task_range": len(in_range),
            "solvable_only_outside_task_range": len(truncated),
            "unsolvable_at_any_force": len(unsolvable),
            "fraction_solvable_within_task_range": len(in_range) / len(rows),
            "fraction_solvable_any_force": len(solvable) / len(rows),
        },
        "required_force": _spread(required),
        "band_width": _spread([row["band_width"] for row in solvable]),
        "truncation": {
            "allowed_range": [low, high],
            "states_needing_more_than_allowed": len(truncated),
            "required_force_above_ceiling": _spread(
                [row["required_force"] for row in truncated]
            ),
            "solvable_states_at_the_ceiling": sum(
                1 for row in solvable if row["required_force"] is not None and row["required_force"] > high - 1e-9
            ),
            "solvable_states_at_the_floor": sum(
                1 for row in solvable if row["required_force"] is not None and row["required_force"] < low + 1e-9
            ),
        },
        "novel_axis_rates": {
            axis: {
                "states": appearances[axis],
                "failed": failures[axis],
                "failure_rate": failures[axis] / appearances[axis],
            }
            for axis in sorted(appearances, key=lambda name: -failures[name] / appearances[name])
        },
        "failures": [
            {
                "novel_axes": row["novel_axes"],
                "sampled_axes": row["sampled_axes"],
                "probe_moved": row["probe_moved"],
                "closest_position_error_mm": row["closest_position_error"] * 1000,
                "closest_force": row["closest_force"],
                "invalid_fraction": row["invalid_fraction"],
                "reach_any_force": row["reach_any_force"],
                "required_force": row["required_force"],
            }
            for row in rows
            if not row["reach_within_task_range"]
        ],
        "safety": {
            "states_with_any_abort": sum(1 for row in rows if row["safety_aborts"]),
            "total_aborts": sum(row["safety_aborts"] for row in rows),
            "median_invalid_fraction": float(np.median([row["invalid_fraction"] for row in rows])),
        },
    }
