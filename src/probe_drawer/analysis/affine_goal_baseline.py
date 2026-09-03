r"""Would one global slope solve the multi-goal problem?

`docs/TASK_CONDITIONING.md` established that the goal moves the required force by 1.29x the
success band, so a single-goal force fails at a new goal. It also showed *why*: for a locally
affine force-displacement response the ratio is :math:`\Delta d_\text{goal} / 2\epsilon_d`,
which is arithmetic rather than dynamics. That raises the obvious follow-up, and it is the one
that decides whether a task-conditioned experiment is worth running at all:

    if the mapping is near-affine, is **one global slope** enough?

If it is, a multi-goal model would mostly be learning a constant, and the honest baseline to
beat is not a fixed force but

.. math:: F(g) = F_{100} + k_\text{global}\,(g - 0.10)

This module estimates :math:`k_\text{global}` on a **calibration** subset and evaluates that
formula on a **held-out** subset, with the split fixed by a content-addressed permutation
before any physics is read.

Two things are deliberately generous to the baseline, because the question is whether a global
slope *suffices*, not whether this is a deployable method:

* it is handed the **correct** :math:`F_{100}` for each held-out drawer, taken from that
  drawer's own Oracle band centre. A real system would have to infer it from the probe.
* the slope is fitted on the same distribution it is tested on.

So the numbers below are an **upper bound** on what a global correction can do. If it still
falls short of the Oracle, no simpler correction will close the gap.
"""

from __future__ import annotations

import numpy as np

from probe_drawer.analysis.task_conditioning import band_of

__all__ = ["evaluate_affine_goal_baseline", "per_state_slopes"]


def per_state_slopes(
    rows: list[dict], goals: list[float], tolerance: float, step: float, action_range: tuple[float, float]
) -> list[dict]:
    r"""Each drawer's Oracle optimum at every goal, and its own least-squares slope.

    The slope is fitted per drawer across the goals, so a drawer solvable at fewer than two
    goals gets ``None`` -- a slope through one point is not a slope.

    Returns:
        One entry per row with ``centres`` (goal to band centre, or ``None``), ``slope`` and
        ``goals_solved``.
    """
    low_bound, high_bound = action_range
    output = []
    for row in rows:
        centres: dict[float, float | None] = {}
        for goal in goals:
            reaching = [
                bool(valid)
                and abs(achieved - goal) <= tolerance
                and low_bound - 1e-9 <= force <= high_bound + 1e-9
                for achieved, valid, force in zip(
                    row["displacement"], row["valid"], row["forces"], strict=True
                )
            ]
            band = band_of(row["forces"], reaching, step)
            centres[goal] = None if band is None else band["centre"]

        solved = [(goal, centre) for goal, centre in centres.items() if centre is not None]
        slope = None
        if len(solved) >= 2:
            x = np.array([goal for goal, _ in solved], dtype=float)
            y = np.array([centre for _, centre in solved], dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
        output.append(
            {
                "hidden_state": row["hidden_state"],
                "centres": centres,
                "slope": slope,
                "goals_solved": len(solved),
            }
        )
    return output


def evaluate_affine_goal_baseline(
    rows: list[dict],
    calibration: list[int],
    held_out: list[int],
    goals: list[float],
    reference_goal: float,
    tolerance: float,
    step: float,
    action_range: tuple[float, float],
) -> dict:
    r"""Fit one slope on ``calibration`` and apply it to ``held_out``.

    Args:
        rows: Per-drawer sweep tables (``forces``, ``displacement``, ``valid``).
        calibration: Row indices the slope may be fitted on.
        held_out: Row indices it is evaluated on. Must not overlap ``calibration``.
        goals: Goal displacements in metres, including ``reference_goal``.
        reference_goal: The goal whose Oracle force the baseline is *given* (0.10 m).
        tolerance: :math:`\epsilon_d` in metres.
        step: Force grid spacing in newtons.
        action_range: The frozen ``peak_force_range``.

    Returns:
        ``k_global``, ``calibration``, ``held_out`` (per goal), ``slope_error`` and
        ``per_state``.

    Raises:
        ValueError: If the subsets overlap, if ``reference_goal`` is not among ``goals``, or if
            no calibration drawer yields a slope.
    """
    if set(calibration) & set(held_out):
        raise ValueError(
            f"the subsets overlap on {sorted(set(calibration) & set(held_out))}; the slope would "
            "be fitted on part of its own test set."
        )
    if reference_goal not in goals:
        raise ValueError(f"reference_goal {reference_goal} is not among the goals {goals}.")

    states = per_state_slopes(rows, goals, tolerance, step, action_range)

    fitted = [states[index]["slope"] for index in calibration if states[index]["slope"] is not None]
    if not fitted:
        raise ValueError("no calibration drawer is solvable at two or more goals; no slope to fit.")
    k_global = float(np.mean(fitted))

    def spread(values) -> dict | None:
        if len(values) == 0:
            return None
        array = np.asarray(values, dtype=float)
        return {
            "n": int(array.size),
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "sd": float(array.std()),
            "max": float(array.max()),
        }

    held = {}
    for goal in goals:
        reached, errors, oracle, usable, applied = 0, [], 0, 0, []
        for index in held_out:
            row, state = rows[index], states[index]
            reference = state["centres"][reference_goal]
            if reference is None:
                # The baseline is defined as a correction to the 100 mm force; without one
                # there is nothing to correct, so the drawer is outside its domain rather than
                # a failure of it. Counted separately.
                continue
            usable += 1
            oracle += state["centres"][goal] is not None

            predicted = reference + k_global * (goal - reference_goal)
            applied.append(predicted)
            # Snapped to the swept grid, because only swept forces have measured physics. The
            # 0.05 N quantisation is small against a ~0.35 N band but is not nothing, and it is
            # reported rather than interpolated away.
            nearest = int(np.argmin([abs(force - predicted) for force in row["forces"]]))
            achieved = row["displacement"][nearest]
            errors.append((achieved - goal) * 1000.0)
            reached += bool(row["valid"][nearest]) and abs(achieved - goal) <= tolerance

        held[goal] = {
            "states_with_a_reference": usable,
            "oracle_solvable": oracle,
            "reached": reached,
            "reach_rate": reached / usable if usable else float("nan"),
            "oracle_rate": oracle / usable if usable else float("nan"),
            "gap_to_oracle_pp": (oracle - reached) / usable * 100 if usable else float("nan"),
            "position_error_mm": spread(errors),
            "abs_position_error_mm": spread(np.abs(errors)),
            "applied_force": spread(applied),
        }

    held_slopes = [states[index]["slope"] for index in held_out if states[index]["slope"] is not None]
    return {
        "k_global": k_global,
        "reference_goal": reference_goal,
        "calibration": {
            "states": len(calibration),
            "states_with_a_slope": len(fitted),
            "slope": spread(fitted),
        },
        "held_out": held,
        "slope_error": {
            "held_out_slope": spread(held_slopes),
            "abs_error": spread([abs(value - k_global) for value in held_slopes]),
            "relative_error": spread(
                [abs(value - k_global) / abs(k_global) for value in held_slopes]
            )
            if k_global
            else None,
        },
        "per_state": states,
    }
