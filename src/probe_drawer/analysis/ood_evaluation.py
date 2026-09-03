r"""Stratifying an out-of-distribution deployment.

A single out-of-distribution number is close to uninterpretable, because it mixes three
different situations that the feasibility pilot already separated
(``docs/OOD_FEASIBILITY.md``):

* states the task cannot reach inside the frozen force range at all -- 3 of 64, where every
  method must fail and a low score says nothing about adaptation;
* states the probe **does** move, which is the regime the probe was calibrated for;
* states the probe barely moves -- 17 of 64, every one with :math:`\mu_s` above the training
  maximum -- where the drawer is usually still openable but the model's only input is close to
  silent.

So the summary reports the same methods over several strata, with the membership taken from the
**feasibility sweep**, which ran before any model was evaluated. That ordering matters: a
stratification chosen after seeing model scores would be a way of finding a subset that flatters
one of them.

The quantity that answers the interesting question is ``force_bias``: the chosen force minus the
force the Oracle says that state needs. On a state the probe hardly moved, a model that reads
"nothing happened" as "this is easy" will choose *too little* force and show a negative bias; a
model that infers "it did not move because it is stiff" will show a bias near zero. That is a
different question from whether it succeeded, and it is the one that says whether the failure is
inference or actuation.
"""

from __future__ import annotations

import numpy as np

__all__ = ["STRATA", "summarise_ood_evaluation"]

#: The strata, as ``(name, predicate, description)``. Membership comes from the feasibility
#: sweep's own flags, not from anything measured during the evaluation.
STRATA: tuple[tuple[str, str], ...] = (
    ("all", "every out-of-distribution state, whatever the Oracle says"),
    ("oracle_feasible", "solvable by some force inside the frozen action range"),
    ("responsive", "the probe broke the drawer away"),
    ("no_breakaway", "the probe did not break the drawer away"),
    ("no_breakaway_feasible", "probe silent, but the task is still solvable in range"),
)


def _members(rows: list[dict], stratum: str) -> list[dict]:
    if stratum == "all":
        return rows
    if stratum == "oracle_feasible":
        return [row for row in rows if row.get("reach_within_task_range")]
    if stratum == "responsive":
        return [row for row in rows if row.get("probe_moved_in_sweep")]
    if stratum == "no_breakaway":
        return [row for row in rows if row.get("probe_moved_in_sweep") is False]
    if stratum == "no_breakaway_feasible":
        return [
            row
            for row in rows
            if row.get("probe_moved_in_sweep") is False and row.get("reach_within_task_range")
        ]
    raise ValueError(f"unknown stratum {stratum!r}; expected one of {[name for name, _ in STRATA]}")


def _method_stats(rows: list[dict], goal: float) -> dict:
    """One method's numbers on one stratum, pooled over seeds, plus the per-seed spread.

    ``goal`` comes from the report's task rather than the rows: a deployment row records where
    the drawer ended up, not what it was aiming at, and every row in one report shares the goal.
    """
    if not rows:
        return {"states": 0}
    errors = [abs(row["total_displacement"] - goal) * 1000 for row in rows]
    forces = [row["chosen_force"] for row in rows]
    biases = [
        row["chosen_force"] - row["oracle_required_force"]
        for row in rows
        if row.get("oracle_required_force") is not None
    ]
    per_seed: dict = {}
    for seed in sorted({row["seed"] for row in rows if row["seed"] is not None}):
        subset = [row for row in rows if row["seed"] == seed]
        per_seed[seed] = float(np.mean([r["reach_success"] for r in subset])) * 100

    rates = list(per_seed.values())
    return {
        "episodes": len(rows),
        "states": len({row["xi_id"] for row in rows}),
        "reach_pp": float(np.mean([row["reach_success"] for row in rows])) * 100,
        "reach_sd_across_seeds": float(np.std(rates)) if len(rates) > 1 else 0.0,
        "reach_per_seed": per_seed,
        "median_position_error_mm": float(np.median(errors)),
        "median_chosen_force": float(np.median(forces)),
        "chosen_force_range": [float(np.min(forces)), float(np.max(forces))],
        # Negative means the model asked for less than the state needed -- under-forcing.
        "median_force_bias": float(np.median(biases)) if biases else None,
        "mean_force_bias": float(np.mean(biases)) if biases else None,
        "under_forced_fraction": (
            float(np.mean([bias < 0 for bias in biases])) if biases else None
        ),
    }


def summarise_ood_evaluation(report: dict, gaps: tuple[tuple[str, str, str], ...]) -> dict:
    """Stratify a deployment report and compute the requested pairwise gaps.

    Args:
        report: A deployment report produced with ``--ood-report``, whose rows therefore carry
            the sweep's ``reach_within_task_range``, ``probe_moved_in_sweep`` and
            ``oracle_required_force``.
        gaps: ``(label, better, worse)`` triples to difference within each stratum.

    Returns:
        ``strata``, each with ``description``, ``states``, ``methods`` and ``gaps``.

    Raises:
        ValueError: If the rows lack the sweep's flags, which means the report came from an
            in-distribution deployment and cannot be stratified this way.
    """
    rows = report["rows"]
    goal = float(report["task"]["goal_displacement"])
    if not rows or "probe_moved_in_sweep" not in rows[0]:
        raise ValueError(
            "these rows carry no feasibility flags; re-run evaluate_closed_loop.py with "
            "--ood-report so the sweep's strata travel with the deployment."
        )

    methods = sorted({row["method"] for row in rows})
    output: dict = {}
    for name, description in STRATA:
        members = _members(rows, name)
        stats = {
            method: _method_stats([r for r in members if r["method"] == method], goal)
            for method in methods
        }
        stratum_gaps = {}
        for label, better, worse in gaps:
            high, low = stats.get(better, {}), stats.get(worse, {})
            if high.get("episodes") and low.get("episodes"):
                stratum_gaps[label] = high["reach_pp"] - low["reach_pp"]
        output[name] = {
            "description": description,
            "states": len({row["xi_id"] for row in members}),
            "methods": stats,
            "gaps": stratum_gaps,
        }
    return {"strata": output, "methods": methods}
