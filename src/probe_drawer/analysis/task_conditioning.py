r"""Does the task condition actually change the action, and by enough to be worth studying?

Setting V1 hands the model :math:`(d_\text{goal}, T_\text{goal})` as a condition and adapts
only :math:`F_\text{peak}` (``docs/DECISIONS.md`` D044). Within Dataset v1 both conditions are
constant, so they teach the network nothing -- they are in the input for contract symmetry, and
whether a *multi-goal* experiment would be worth running is an open question this module helps
answer offline.

The measurement rests on a structural fact rather than on a re-run: **neither controller reads
the goal.** ``ExecutionPullController.run`` takes a force and a duration (D004) and the
fixed-budget probe takes an amplitude and a budget (D044), so one force sweep produces the
episodes for *every* goal and the goals differ only in how those episodes are scored. Validity
is goal-independent too -- the operating region bounds travel and drift, not the target -- so
the only term that moves is :math:`|d(T) - d_\text{goal}| \le \epsilon_d`.

That makes the comparison exact: the three goals are not three experiments, they are three
readings of one.

The quantity the question turns on is not whether the required force changes -- of course a
further goal needs more force -- but whether it changes by **more than the width of the success
band**. If the bands at 80 and 120 mm still contain the 100 mm optimum, a single-goal model
transfers for free and a multi-goal experiment measures nothing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "band_of",
    "summarise_task_conditioning",
]


def _runs(mask: list[bool]) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of ``mask``, as inclusive ``(start, stop)`` index pairs."""
    runs, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def band_of(forces: list[float], reaching: list[bool], step: float) -> dict | None:
    """The widest contiguous band of succeeding forces, and its centre.

    The **widest** component rather than the union, because the centre of a union that spans a
    gap can itself fail. The number of components is reported so a disconnected success set is
    visible rather than averaged over.

    Args:
        forces: The swept grid, ascending.
        reaching: Whether each force reached the goal.
        step: Grid spacing, used to give a single succeeding force a width of one cell.

    Returns:
        ``{"low", "high", "centre", "width", "components", "cells"}``, or ``None`` if nothing
        reached.
    """
    runs = _runs(list(reaching))
    if not runs:
        return None
    start, stop = max(runs, key=lambda pair: pair[1] - pair[0])
    low, high = float(forces[start]), float(forces[stop])
    return {
        "low": low,
        "high": high,
        "centre": (low + high) / 2.0,
        "width": high - low + step,
        "components": len(runs),
        "cells": stop - start + 1,
    }


def summarise_task_conditioning(
    rows: list[dict], goals: list[float], tolerance: float, step: float, action_range: tuple[float, float]
) -> dict:
    r"""Score one force sweep against several goals and compare the resulting action maps.

    Args:
        rows: One entry per hidden state, each with ``hidden_state``, ``forces`` (ascending),
            ``displacement`` (the achieved :math:`d_\text{total}(T)` per force) and ``valid``
            (per force).
        goals: Goal displacements to score, in metres.
        tolerance: :math:`\epsilon_d` in metres.
        step: Force grid spacing in newtons.
        action_range: The frozen ``peak_force_range``, for reporting truncation.

    Returns:
        ``per_goal``, ``shift``, ``transfer`` and ``per_state``.

    Raises:
        ValueError: If ``rows`` is empty or a row's arrays disagree in length.
    """
    if not rows:
        raise ValueError("summarise_task_conditioning needs at least one hidden state.")
    for row in rows:
        if not len(row["forces"]) == len(row["displacement"]) == len(row["valid"]):
            raise ValueError("forces, displacement and valid must have the same length.")

    low_bound, high_bound = action_range
    bands: dict[float, list[dict | None]] = {}
    for goal in goals:
        bands[goal] = []
        for row in rows:
            reaching = [
                bool(valid) and abs(achieved - goal) <= tolerance
                for achieved, valid in zip(row["displacement"], row["valid"], strict=True)
            ]
            in_range = [
                hit and low_bound - 1e-9 <= force <= high_bound + 1e-9
                for hit, force in zip(reaching, row["forces"], strict=True)
            ]
            band = band_of(row["forces"], in_range, step)
            bands[goal].append(band)

    def spread(values: list[float]) -> dict | None:
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    per_goal = {}
    for goal in goals:
        solved = [band for band in bands[goal] if band is not None]
        per_goal[goal] = {
            "states": len(rows),
            "solvable": len(solved),
            "solvable_fraction": len(solved) / len(rows),
            "band_width": spread([band["width"] for band in solved]),
            "band_centre": spread([band["centre"] for band in solved]),
            "required_force": spread([band["low"] for band in solved]),
            "disconnected": sum(1 for band in solved if band["components"] > 1),
            "at_action_ceiling": sum(1 for band in solved if band["high"] >= high_bound - 1e-9),
        }

    # How far the optimum moves between goals, per hidden state -- only over states solvable
    # at both, since a shift is undefined otherwise.
    shift = {}
    for first, second in zip(goals[:-1], goals[1:], strict=False):
        deltas = [
            second_band["centre"] - first_band["centre"]
            for first_band, second_band in zip(bands[first], bands[second], strict=True)
            if first_band is not None and second_band is not None
        ]
        widths = [
            0.5 * (first_band["width"] + second_band["width"])
            for first_band, second_band in zip(bands[first], bands[second], strict=True)
            if first_band is not None and second_band is not None
        ]
        shift[f"{first}->{second}"] = {
            "delta_centre": spread(deltas),
            "abs_delta_centre": spread([abs(value) for value in deltas]),
            # The number the decision turns on: a shift smaller than the band it has to leave
            # means the old optimum still works.
            "shift_over_band_width": spread(
                [abs(d) / w for d, w in zip(deltas, widths, strict=True) if w > 0]
            ),
        }

    # Transfer: take the middle goal's optimum, snap it to the grid, and ask whether it still
    # reaches the other goals. This is what a single-goal model would do if deployed unchanged.
    middle = goals[len(goals) // 2]
    transfer = {}
    for goal in goals:
        usable, hits = 0, 0
        for row, source, target in zip(rows, bands[middle], bands[goal], strict=True):
            if source is None or target is None:
                continue
            usable += 1
            index = int(np.argmin([abs(f - source["centre"]) for f in row["forces"]]))
            achieved = row["displacement"][index]
            hits += bool(row["valid"][index]) and abs(achieved - goal) <= tolerance
        transfer[goal] = {
            "source_goal": middle,
            "states_solvable_at_both": usable,
            "reached": hits,
            "success_rate": hits / usable if usable else float("nan"),
        }

    per_state = [
        {
            "hidden_state": row["hidden_state"],
            "bands": {
                goal: (None if bands[goal][index] is None else dict(bands[goal][index]))
                for goal in goals
            },
        }
        for index, row in enumerate(rows)
    ]
    return {
        "goals": list(goals),
        "tolerance": tolerance,
        "force_step": step,
        "action_range": list(action_range),
        "per_goal": per_goal,
        "shift": shift,
        "transfer": transfer,
        "per_state": per_state,
    }
