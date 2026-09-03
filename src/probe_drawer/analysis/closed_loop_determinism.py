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
    "EXPECTED_ORDERING",
    "FORCE_AGREEMENT_FLOOR",
    "MAX_PROBE_MEDIAN_MM",
    "MAX_PROBE_P90_MM",
    "MAX_REACH_DELTA_PP",
    "REPORTED_GAPS",
    "compare_batch_orders",
    "summarise_permutations",
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


#: The ordering the paper claims, strongest first.
#:
#: Stated here rather than inferred from whichever run is being summarised, so "the ordering
#: held" is a check against a fixed claim and not a restatement of the data.
EXPECTED_ORDERING = (
    "teacher (privileged)",
    "ACE + PSP",
    "D GRU (history)",
    "B ridge (summary)",
)

#: The pairwise differences the paper reports, as ``(name, better, worse)``.
REPORTED_GAPS = (
    ("ACE + PSP - D GRU", "ACE + PSP", "D GRU (history)"),
    ("ACE + PSP - B ridge", "ACE + PSP", "B ridge (summary)"),
    ("teacher - ACE + PSP", "teacher (privileged)", "ACE + PSP"),
)


def _spread(values: list[float]) -> dict:
    """Mean, population sd, and the range actually observed.

    Population sd rather than a sample estimate: with five permutations the question is how
    much *these* runs differed, not an inference about a wider population that does not exist.
    The min-max is reported alongside because with n = 5 the range is the more honest summary
    and a reader should see both.
    """
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def summarise_permutations(reports: list[dict]) -> dict:
    """Aggregate several deployment runs that differ only in the slot permutation.

    Turns D047's schedule sensitivity into an error bar: every run measures every method, so
    the spread across runs is what the environment-slot assignment is worth, and the pairwise
    gaps can be computed *within* each run before being aggregated -- which is the right order,
    because a gap between two methods measured on the same slots is the quantity that matters.

    Args:
        reports: Two or more reports from ``scripts/evaluate_closed_loop.py``, over the same
            hidden states and checkpoints, with different ``slot_permutation`` values.

    Returns:
        ``methods``, ``gaps``, ``probe_displacement``, ``ordering`` and ``permutations``.

    Raises:
        ValueError: If fewer than two reports are given, if they do not cover the same hidden
            states, or if a permutation appears twice -- each of which would make the spread
            mean something other than what it claims.
    """
    if len(reports) < 2:
        raise ValueError(f"need at least two reports to measure a spread, got {len(reports)}.")

    populations = {frozenset(_probe_by_xi(report)) for report in reports}
    if len(populations) != 1:
        raise ValueError("the reports cover different hidden states; the spread would be meaningless.")

    labels = [report.get("slot_permutation", index) for index, report in enumerate(reports)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate slot permutations {labels}; each run must be a distinct one.")

    names = sorted(set.intersection(*(set(report["methods"]) for report in reports)))
    rate = {
        name: [report["methods"][name]["reach_success_rate"] * 100.0 for report in reports]
        for name in names
    }

    methods = {
        name: {
            "reach_success_pp": _spread(rate[name]),
            "median_position_error_mm": _spread(
                [report["methods"][name]["median_position_error_mm"] for report in reports]
            ),
        }
        for name in names
    }

    # Within-run differences, then aggregated. Differencing the aggregates instead would hide
    # that a gap can be stable while both of its terms move together.
    gaps = {}
    for label, better, worse in REPORTED_GAPS:
        if better not in rate or worse not in rate:
            continue
        gaps[label] = _spread(
            [high - low for high, low in zip(rate[better], rate[worse], strict=True)]
        )

    ranked = [name for name in EXPECTED_ORDERING if name in rate]
    per_run = [
        all(
            rate[ranked[position]][index] > rate[ranked[position + 1]][index]
            for position in range(len(ranked) - 1)
        )
        for index in range(len(reports))
    ]

    # How much the *same* drawer's probe moved between permutations, pooled over every pair.
    probe = [_probe_by_xi(report) for report in reports]
    shared = sorted(probe[0])
    pairwise = [
        abs(probe[left][key] - probe[right][key]) * 1000.0
        for left in range(len(probe))
        for right in range(left + 1, len(probe))
        for key in shared
    ]

    return {
        "permutations": labels,
        "methods": methods,
        "gaps": gaps,
        "probe_displacement": {
            "pairs": len(probe) * (len(probe) - 1) // 2,
            "median_mm": float(np.median(pairwise)),
            "p90_mm": float(np.percentile(pairwise, 90)),
            "max_mm": float(np.max(pairwise)),
        },
        "ordering": {
            "claim": list(ranked),
            "held_per_permutation": per_run,
            "held_everywhere": all(per_run),
        },
    }
