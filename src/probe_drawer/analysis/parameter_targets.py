r"""What should a single-point regressor be asked to predict?

In one dimension this was not a question: a hidden state's succeeding forces formed an
interval and its midpoint worked for 104 of 105 solvable states, so "the middle of the
success band" was both the obvious target and a safe one.

In two dimensions it stops being obvious, and the reason is the point of Phase 12. A hidden
state may have many succeeding :math:`(F, T)` pairs, so a regressor asked for *one* answer
needs a rule for which. Three rules are defensible and they disagree:

``centroid``
    The mean of the succeeding points. The natural least-squares target -- and the one that
    can fail outright, because the mean of a curved band need not lie in the band.
``min_cost``
    The cheapest succeeding point under :math:`J = \hat{F} + \lambda \hat{T}` on normalised
    axes. Operationally sensible: pull as gently and as briefly as the task allows.
``max_margin``
    The succeeding point furthest from any failure. The most robust choice, and the one a
    deployed system would want when its own prediction is imperfect.

**The measurement that matters is whether each target succeeds.** A target's realised success
rate is a hard ceiling on any regressor trained toward it: a perfect regressor hitting that
target exactly inherits its failures.

Measured on the coarse sweep over 46 solvable hidden states, and the result is worth stating
carefully because it cuts against the easy story:

* ``centroid`` succeeds for **80 %**. Its 20 % failure is real evidence of curvature -- the
  mean of a hyperbola-like band falls off the band -- and it is a ceiling on any regression
  trained toward centroids.
* ``min_cost`` and ``max_margin`` succeed for **100 %**, but that is a tautology rather than
  a finding: both are *selected from* the success set. It does establish that a single point
  is always representable, so "the landscape is needed because no single answer exists" would
  be false here.

The honest conclusion is therefore narrower than "landscape modelling is necessary": a single
point always exists, and the difficulty is that the good ones (``max_margin``, ``min_cost``)
are a less smooth function of the hidden state than the centroid is, while the smooth one
(``centroid``) carries a 20 % ceiling.

Choosing the target that makes the baseline look worst would be cheating (Phase 12's red
line), so all three are computed, all three are reported, and the baseline is trained on
whichever has the highest ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from probe_drawer.analysis.landscape_2d import success_mask
from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = ["ParameterTarget", "TARGET_RULES", "compare_targets", "targets_for_state"]

#: Weight on duration in the ``min_cost`` rule.
#:
#: One: on normalised axes a full sweep of the force range and a full sweep of the duration
#: range cost the same. There is no physical exchange rate between newtons and seconds, so
#: any other value would be a preference dressed as a measurement -- and the reported
#: sensitivity to it is more useful than a tuned value.
DURATION_COST_WEIGHT = 1.0

TARGET_RULES = ("centroid", "min_cost", "max_margin")


@dataclass
class ParameterTarget:
    """One rule's answer for one hidden state.

    Attributes:
        rule: Which of :data:`TARGET_RULES` produced it.
        force, duration: The parameter pair, in N and s.
        on_grid: Whether the pair is a swept grid point. The centroid usually is not, so it
            is reported both raw and snapped.
        succeeds: Whether the *executed* outcome at this pair -- or at the nearest swept point
            to it -- satisfies the task. This is the number that bounds a regressor.
        snapped_force, snapped_duration: The nearest swept grid point, which is what the
            success verdict was read from.
        grid_steps_to_failure: Chebyshev distance to the nearest failing swept point. Larger
            is more robust to a regressor's own error.
    """

    rule: str
    force: float
    duration: float
    on_grid: bool
    succeeds: bool
    snapped_force: float
    snapped_duration: float
    grid_steps_to_failure: float

    def as_dict(self) -> dict:
        return asdict(self)


def _snap(value: float, axis: np.ndarray) -> int:
    return int(np.argmin(np.abs(axis - value)))


def _margin_grid(success: np.ndarray, swept: np.ndarray) -> np.ndarray:
    """Chebyshev distance from each succeeding cell to the nearest failing swept cell.

    Uses the same convention as :mod:`landscape_2d`: the grid's border counts as a failure
    source, because beyond the sweep nothing is known and a region touching the edge must not
    be credited with unbounded margin.
    """
    from collections import deque  # noqa: PLC0415

    distance = np.full(success.shape, np.inf)
    queue: deque[tuple[int, int]] = deque()
    for index in zip(*np.nonzero(swept & ~success), strict=True):
        distance[index] = 0.0
        queue.append(index)
    rows, columns = success.shape
    for row in range(rows):
        for column in (0, columns - 1):
            if success[row, column] and distance[row, column] > 1.0:
                distance[row, column] = 1.0
                queue.append((row, column))
    for column in range(columns):
        for row in (0, rows - 1):
            if success[row, column] and distance[row, column] > 1.0:
                distance[row, column] = 1.0
                queue.append((row, column))

    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    while queue:
        row, column = queue.popleft()
        for delta_row, delta_column in steps:
            neighbour = (row + delta_row, column + delta_column)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < columns):
                continue
            if success[neighbour] and distance[neighbour] > distance[row, column] + 1:
                distance[neighbour] = distance[row, column] + 1
                queue.append(neighbour)
    return np.where(success, distance, np.nan)


def targets_for_state(
    dataset, xi_key: tuple[float, ...], criteria: SuccessCriteria, duration_weight: float = DURATION_COST_WEIGHT
) -> dict[str, ParameterTarget]:
    """All three rules' answers for one hidden state, each with its realised outcome.

    Returns an empty mapping when the hidden state has no succeeding point at all -- there is
    nothing for a regressor to aim at, and pretending otherwise would put an unreachable
    target into the training signal.
    """
    masks = success_mask(dataset, xi_key, criteria)
    forces, durations = masks["forces"], masks["durations"]
    success, swept = masks["success"], masks["swept"]
    if not success.any():
        return {}

    rows, columns = np.nonzero(success)
    margin = _margin_grid(success, swept)

    force_span = float(forces[-1] - forces[0]) or 1.0
    duration_span = float(durations[-1] - durations[0]) or 1.0
    normalised_force = (forces[columns] - forces[0]) / force_span
    normalised_duration = (durations[rows] - durations[0]) / duration_span

    proposals = {
        "centroid": (float(forces[columns].mean()), float(durations[rows].mean())),
        "min_cost": None,
        "max_margin": None,
    }
    cheapest = int(np.argmin(normalised_force + duration_weight * normalised_duration))
    proposals["min_cost"] = (float(forces[columns[cheapest]]), float(durations[rows[cheapest]]))
    safest = np.unravel_index(int(np.nanargmax(margin)), margin.shape)
    proposals["max_margin"] = (float(forces[safest[1]]), float(durations[safest[0]]))

    results = {}
    for rule, (force, duration) in proposals.items():
        column, row = _snap(force, forces), _snap(duration, durations)
        exact = bool(np.isclose(forces[column], force) and np.isclose(durations[row], duration))
        results[rule] = ParameterTarget(
            rule=rule,
            force=force,
            duration=duration,
            on_grid=exact,
            succeeds=bool(success[row, column]),
            snapped_force=float(forces[column]),
            snapped_duration=float(durations[row]),
            grid_steps_to_failure=float(margin[row, column]) if success[row, column] else 0.0,
        )
    return results


def compare_targets(dataset, criteria: SuccessCriteria, duration_weight: float = DURATION_COST_WEIGHT) -> dict:
    """Each rule's realised success rate over every solvable hidden state.

    The headline is ``success_rate`` per rule: the fraction of hidden states for which a
    *perfect* regressor aiming at that rule would actually succeed. Anything below 1.0 is a
    ceiling that no amount of training can lift, and the gap between the best rule and 1.0 is
    the part of the task that a single predicted point cannot express.
    """
    per_state = []
    for key in dataset.xi_keys():
        targets = targets_for_state(dataset, key, criteria, duration_weight)
        if not targets:
            continue
        per_state.append(
            {
                "xi": dict(
                    zip(("mass", "static_friction", "dynamic_friction", "damping"), key, strict=True)
                ),
                "targets": {rule: target.as_dict() for rule, target in targets.items()},
            }
        )

    summary = {}
    for rule in TARGET_RULES:
        outcomes = [entry["targets"][rule] for entry in per_state]
        succeeded = [entry for entry in outcomes if entry["succeeds"]]
        summary[rule] = {
            "states": len(outcomes),
            "successes": len(succeeded),
            "success_rate": len(succeeded) / len(outcomes) if outcomes else float("nan"),
            "on_grid_fraction": float(np.mean([entry["on_grid"] for entry in outcomes])) if outcomes else float("nan"),
            "median_grid_steps_to_failure": float(
                np.median([entry["grid_steps_to_failure"] for entry in succeeded])
            )
            if succeeded
            else float("nan"),
            "median_force": float(np.median([entry["force"] for entry in outcomes])) if outcomes else float("nan"),
            "median_duration": float(np.median([entry["duration"] for entry in outcomes]))
            if outcomes
            else float("nan"),
        }

    best = max(TARGET_RULES, key=lambda rule: summary[rule]["success_rate"])
    return {
        "duration_cost_weight": duration_weight,
        "solvable_states": len(per_state),
        "per_rule": summary,
        "best_rule": best,
        "best_rule_success_rate": summary[best]["success_rate"],
        # The honest structural number, and it is the centroid's -- not the best rule's.
        #
        # ``min_cost`` and ``max_margin`` are *selected from* the success set, so they succeed
        # by construction and their 100 % is a tautology, not a finding. Reporting it as "a
        # single point suffices" would be true and useless.
        #
        # What is not tautological is that the **centroid** -- the natural least-squares
        # target, and the one an unthinking direct regression would be trained on -- fails
        # whenever the region is curved enough that its own mean falls outside it. That rate
        # is a hard ceiling on any regressor aimed at the centroid, however accurate.
        "centroid_failure_rate": 1.0 - summary["centroid"]["success_rate"],
        "fair_baseline_target": best,
        "per_state": per_state,
    }
