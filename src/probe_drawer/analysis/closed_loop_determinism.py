r"""Does a closed-loop deployment depend on the order its batches ran in?

The question D047 raised. A deployment run splits the test hidden states into batches of
``num_envs``, probes each batch once, and branches every method off that one probe. If the
simulator carried nothing between batches, then a given hidden state would be probed
identically no matter which batch it landed in -- and the run's numbers would be a property of
the models rather than of the schedule.

Phase 13 measured that it does not hold: batch 1 was bit-identical across runs while batches 2
and 3 were not, because ``system.reset()`` does not clear everything PhysX carries (the same
non-restorable contact manifolds and friction anchors that ``docs/COUNTERFACTUAL_BRANCHING.md``
documents for snapshots).

This module compares two deployment reports that used **different batch orders** over the same
hidden states and the same checkpoints, and applies three gates. The thresholds are fixed here,
in the module, rather than chosen once the numbers are in.

The gates, and where each threshold comes from
----------------------------------------------
**A -- the probe is the same measurement.** The probe is the model's whole input, so if it
differs the comparison is not about the models at all. Judged against the task's position
tolerance :math:`\epsilon_d` = 7.5 mm: a median disagreement above 0.10 mm is a systematic
shift rather than noise, and a p90 above 10 % of :math:`\epsilon_d` can flip a label on its own.

**B -- the chosen force is stable.** The selection grid is 0.05 N, so two runs that measured
the same probe should land on the same grid point. Checked on the *deterministic* methods only
(fixed force, and the two closed-form fits): a learned model's choice can legitimately differ if
its input differed, but a closed-form fit on an unchanged training split cannot.

**C -- the reported result is stable.** With 88 hidden states, one episode is 1.1 pp. Two is the
most that should move, and anything larger means the schedule is visible in the headline number.

Nothing here touches the simulator; it reads two saved reports.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FORCE_AGREEMENT_FLOOR",
    "MAX_PROBE_MEDIAN_MM",
    "MAX_PROBE_P90_MM",
    "MAX_REACH_DELTA_PP",
    "compare_batch_orders",
]

#: Largest acceptable median disagreement in probe displacement, same xi, two orders (mm).
MAX_PROBE_MEDIAN_MM = 0.10

#: Largest acceptable 90th-percentile disagreement (mm). 10 % of ``eps_d`` = 7.5 mm.
MAX_PROBE_P90_MM = 0.75

#: Smallest acceptable fraction of drawers given an identical force by a deterministic method.
FORCE_AGREEMENT_FLOOR = 0.90

#: Largest acceptable change in a method's reach success between orders (percentage points).
#: Two episodes out of 88.
MAX_REACH_DELTA_PP = 2.0

#: Methods whose chosen force cannot legitimately change between two runs of one checkpoint set.
DETERMINISTIC_METHODS = ("fixed force", "A linear (1 feature)", "B ridge (summary)")


def _probe_by_xi(report: dict) -> dict[str, float]:
    """One probe displacement per hidden state. Every method shares it, so any row will do."""
    return {row["xi_id"]: float(row["probe_displacement"]) for row in report["rows"]}


def _forces_by_xi(report: dict, method: str) -> dict[str, float]:
    return {
        row["xi_id"]: float(row["chosen_force"])
        for row in report["rows"]
        if row["method"] == method and row["seed"] is None
    }


def compare_batch_orders(first: dict, second: dict) -> dict:
    """Compare two deployment reports over the same hidden states in different batch orders.

    Args:
        first: A report from ``scripts/evaluate_closed_loop.py``.
        second: Another, over the same hidden states and checkpoints, batched differently.

    Returns:
        A mapping with ``probe``, ``forces``, ``reach`` and ``gates`` sections, plus ``passes``.
        Every gate is reported individually so a failure names what moved.

    Raises:
        ValueError: If the two reports do not cover the same hidden states -- comparing
            different populations would produce a meaningless number rather than an error.
    """
    left, right = _probe_by_xi(first), _probe_by_xi(second)
    shared = sorted(set(left) & set(right))
    if not shared or set(left) != set(right):
        raise ValueError(
            f"the two reports cover different hidden states: {len(left)} and {len(right)}, "
            f"{len(shared)} shared. They must be the same population."
        )

    differences = np.array([abs(left[key] - right[key]) for key in shared]) * 1000.0
    probe = {
        "hidden_states": len(shared),
        "median_mm": float(np.median(differences)),
        "p90_mm": float(np.percentile(differences, 90)),
        "max_mm": float(differences.max()),
        "identical": int(np.sum(differences < 1e-9)),
    }

    forces = {}
    for method in DETERMINISTIC_METHODS:
        a, b = _forces_by_xi(first, method), _forces_by_xi(second, method)
        keys = sorted(set(a) & set(b))
        if not keys:
            continue
        same = sum(1 for key in keys if abs(a[key] - b[key]) < 1e-9)
        forces[method] = {"drawers": len(keys), "identical": same, "agreement": same / len(keys)}

    reach = {}
    for method in sorted(set(first["methods"]) & set(second["methods"])):
        a = first["methods"][method]["reach_success_rate"] * 100.0
        b = second["methods"][method]["reach_success_rate"] * 100.0
        reach[method] = {"first": a, "second": b, "delta_pp": b - a}

    worst_force = min((values["agreement"] for values in forces.values()), default=0.0)
    worst_reach = max((abs(values["delta_pp"]) for values in reach.values()), default=float("inf"))
    gates = {
        "probe_is_the_same_measurement": (
            probe["median_mm"] <= MAX_PROBE_MEDIAN_MM and probe["p90_mm"] <= MAX_PROBE_P90_MM
        ),
        "chosen_force_is_stable": worst_force >= FORCE_AGREEMENT_FLOOR,
        "reported_result_is_stable": worst_reach <= MAX_REACH_DELTA_PP,
    }
    return {
        "probe": probe,
        "forces": forces,
        "reach": reach,
        "worst_force_agreement": worst_force,
        "worst_reach_delta_pp": worst_reach,
        "thresholds": {
            "max_probe_median_mm": MAX_PROBE_MEDIAN_MM,
            "max_probe_p90_mm": MAX_PROBE_P90_MM,
            "force_agreement_floor": FORCE_AGREEMENT_FLOOR,
            "max_reach_delta_pp": MAX_REACH_DELTA_PP,
        },
        "gates": gates,
        "passes": all(gates.values()),
    }
