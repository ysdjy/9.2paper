"""Phase 12C/12D/12E -- what shape is the two-dimensional success region, and does it matter?

This is the hard gate. Phase 11 established that with ``T`` fixed the success set is a
contiguous interval whose midpoint works for 104 of 105 solvable hidden states, so a model
predicting the whole landscape has no *structural* advantage over one predicting a single
number. Opening ``T`` makes richer structure *possible*; this script decides whether it is
*actual*.

It answers, from the coarse sweep and nothing else:

1. Is the region's location and shape a function of the hidden state, or roughly fixed?
2. Is it convex? Measured operationally: does the mean of two succeeding parameter pairs
   succeed?
3. Is it connected, or are there separated islands?
4. Where does the valid operating region cut the box, and what should the formal candidate
   range be?

A negative answer is a real result and is reported as one. The point of the gate is that
Dataset v1 costs hours; if the second axis buys no structure, the honest move is to stop and
say so rather than to generate 147 000 rows and discover it later.

No simulator. Usage::

    python scripts/analyze_landscape_2d.py --dataset outputs/logs/landscape_2d_coarse.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from probe_drawer.analysis.landscape_2d import analyse_landscape, success_mask
from probe_drawer.analysis.sweep import SweepDataset
from probe_drawer.experiment_plan import MAIN_TASK
from probe_drawer.utils import git_commit, project_root

#: A hidden state's region must move by more than this, relative to the swept box, before the
#: landscape is called hidden-state dependent.
#:
#: 10 % of the box is a tenth of the whole parameter range -- far beyond any plausible
#: measurement noise, and comfortably more than the 0.05 N / 0.05 s resolution a fine sweep
#: would use. Below that the regions would be effectively interchangeable and a single fixed
#: parameter pair would serve every drawer.
CENTROID_SHIFT_THRESHOLD = 0.10

#: Midpoint failure rate above which the region is called non-convex in a way that matters.
#:
#: Not zero: a grid samples a curved boundary, so a genuinely convex region can show a few
#: failures where the boundary cuts a cell. 5 % is well above that and is also the level at
#: which one candidate in twenty from an averaged target would fail outright.
MIDPOINT_FAILURE_THRESHOLD = 0.05


def valid_region_edges(dataset: SweepDataset, criteria) -> dict:
    """Where the box is unusable, marginalised over hidden states.

    Reported per axis value rather than as a single box, because the two axes fail for
    different physical reasons and at different places: too little force never breaks the
    drawer loose, too much overshoots, too short a ``T`` cannot deliver the impulse, and too
    long a ``T`` leaves the drawer still moving when the force reaches zero.
    """
    forces, durations = dataset.forces(), dataset.durations()
    keys = dataset.xi_keys()
    per_force = {force: {"swept": 0, "valid": 0, "success": 0} for force in forces}
    per_duration = {duration: {"swept": 0, "valid": 0, "success": 0} for duration in durations}
    reasons_by_force: dict[float, Counter] = {force: Counter() for force in forces}
    reasons_by_duration: dict[float, Counter] = {duration: Counter() for duration in durations}

    for row in dataset.records:
        for bucket, reasons, key in (
            (per_force, reasons_by_force, row.peak_force),
            (per_duration, reasons_by_duration, row.duration),
        ):
            bucket[key]["swept"] += 1
            bucket[key]["valid"] += int(row.valid)
            bucket[key]["success"] += int(row.succeeds(criteria))
            for reason in row.invalid_reasons:
                reasons[key][reason] += 1

    def summarise(bucket: dict, reasons: dict) -> list[dict]:
        return [
            {
                "value": float(key),
                "swept": counts["swept"],
                "valid_fraction": counts["valid"] / counts["swept"] if counts["swept"] else float("nan"),
                "success_fraction": counts["success"] / counts["swept"] if counts["swept"] else float("nan"),
                "top_invalid_reason": reasons[key].most_common(1)[0][0] if reasons[key] else None,
            }
            for key, counts in bucket.items()
        ]

    return {
        "hidden_states": len(keys),
        "per_force": summarise(per_force, reasons_by_force),
        "per_duration": summarise(per_duration, reasons_by_duration),
    }


def proposed_box(edges: dict, min_success_fraction: float = 0.01) -> dict:
    """The narrowest ``(F, T)`` box that keeps every axis value worth sweeping.

    An axis value is kept if *some* hidden state succeeds there. Trimming on validity alone
    would keep large regions that are physically fine and useless -- a 0.15 N force is
    perfectly valid and never reaches the goal -- and trimming on a high success fraction
    would throw away the forces only the stiffest drawers need, which are exactly the ones
    that make the task discriminative.
    """
    def keep(entries: list[dict]) -> list[float]:
        return [entry["value"] for entry in entries if entry["success_fraction"] >= min_success_fraction]

    forces, durations = keep(edges["per_force"]), keep(edges["per_duration"])
    return {
        "min_success_fraction": min_success_fraction,
        "force_range": [min(forces), max(forces)] if forces else None,
        "duration_range": [min(durations), max(durations)] if durations else None,
        "forces_dropped": [
            entry["value"] for entry in edges["per_force"] if entry["success_fraction"] < min_success_fraction
        ],
        "durations_dropped": [
            entry["value"] for entry in edges["per_duration"] if entry["success_fraction"] < min_success_fraction
        ],
    }


def structure_verdict(metrics: list, dataset: SweepDataset) -> dict:
    """Does the second axis buy structure the first one lacked?"""
    solvable = [entry for entry in metrics if entry.success_points > 0]
    forces, durations = dataset.forces(), dataset.durations()
    force_span = forces[-1] - forces[0]
    duration_span = durations[-1] - durations[0]

    centroids = np.array([entry.centroid for entry in solvable]) if solvable else np.zeros((0, 2))
    normalised = (
        np.column_stack(
            [(centroids[:, 0] - forces[0]) / force_span, (centroids[:, 1] - durations[0]) / duration_span]
        )
        if len(centroids)
        else np.zeros((0, 2))
    )

    # Only hidden states whose region the grid actually resolves can testify about topology.
    # A region three cells wide cannot be shown non-convex by a three-cell-wide grid, and
    # including it would let resolution masquerade as structure (the Phase 12 red line).
    resolved = [entry for entry in solvable if entry.resolution["sufficient_for_topology"]]
    midpoint_rates = np.array([entry.midpoint["rate"] for entry in resolved], dtype=float)
    finite_rates = midpoint_rates[np.isfinite(midpoint_rates)]
    disconnected = [entry for entry in resolved if entry.components > 1]
    disconnected_diagonal = [entry for entry in resolved if entry.components_diagonal > 1]
    orthogonally_convex = [entry for entry in resolved if entry.resolution["orthogonally_convex"]]
    areas = np.array([entry.success_fraction for entry in solvable])
    orientations = np.array([entry.orientation_deg for entry in solvable], dtype=float)
    finite_orientations = orientations[np.isfinite(orientations)]

    # Does the region move with xi? Compared against the *within-state* extent, so a region
    # that is merely large is not mistaken for a region that moves.
    centroid_spread = (
        float(np.linalg.norm(normalised.max(axis=0) - normalised.min(axis=0))) if len(normalised) > 1 else 0.0
    )

    return {
        "hidden_states": len(metrics),
        "solvable": len(solvable),
        "coverage": len(solvable) / len(metrics) if metrics else 0.0,
        "success_fraction": {
            "median": float(np.median(areas)) if len(areas) else float("nan"),
            "min": float(areas.min()) if len(areas) else float("nan"),
            "max": float(areas.max()) if len(areas) else float("nan"),
        },
        "centroid_normalised_spread": centroid_spread,
        "centroid_force_range": [float(centroids[:, 0].min()), float(centroids[:, 0].max())] if len(centroids) else None,
        "centroid_duration_range": (
            [float(centroids[:, 1].min()), float(centroids[:, 1].max())] if len(centroids) else None
        ),
        "orientation_deg": {
            "median": float(np.median(finite_orientations)) if len(finite_orientations) else float("nan"),
            "spread": float(finite_orientations.max() - finite_orientations.min())
            if len(finite_orientations)
            else float("nan"),
        },
        "midpoint_failure_rate": {
            "median": float(np.median(finite_rates)) if len(finite_rates) else float("nan"),
            "mean": float(finite_rates.mean()) if len(finite_rates) else float("nan"),
            "max": float(finite_rates.max()) if len(finite_rates) else float("nan"),
            "states_above_threshold": int((finite_rates > MIDPOINT_FAILURE_THRESHOLD).sum()),
            "threshold": MIDPOINT_FAILURE_THRESHOLD,
        },
        "topology_resolved_states": len(resolved),
        "disconnected_states": len(disconnected),
        "disconnected_states_8_connectivity": len(disconnected_diagonal),
        "orthogonally_convex_states": len(orthogonally_convex),
        "median_columns_spanned": float(np.median([entry.resolution["columns_spanned"] for entry in solvable]))
        if solvable
        else float("nan"),
        "median_rows_spanned": float(np.median([entry.resolution["rows_spanned"] for entry in solvable]))
        if solvable
        else float("nan"),
        "row_contiguity_median": float(
            np.nanmedian([entry.row_contiguity for entry in solvable])
        ) if solvable else float("nan"),
        "column_contiguity_median": float(
            np.nanmedian([entry.column_contiguity for entry in solvable])
        ) if solvable else float("nan"),
        "elongation_median": float(
            np.nanmedian([entry.elongation for entry in solvable])
        ) if solvable else float("nan"),
        # The gate. Any one of these is enough to make a landscape model structurally
        # motivated; none of them means the honest answer is that it is not.
        "evidence": {
            # This one the coarse grid can settle: it compares *where* regions sit, not their
            # internal shape, so it needs no resolution inside a region.
            "region_depends_on_hidden_state": centroid_spread > CENTROID_SHIFT_THRESHOLD,
            # These require a resolved region, and 8-connectivity for the disconnection claim
            # -- a staircase is not two islands.
            "non_convex": bool(len(finite_rates))
            and float(np.median(finite_rates)) > MIDPOINT_FAILURE_THRESHOLD,
            "any_state_non_convex": int((finite_rates > MIDPOINT_FAILURE_THRESHOLD).sum()) > 0,
            "disconnected": len(disconnected_diagonal) > 0,
        },
        "caveats": {
            "states_too_thin_for_topology": len(solvable) - len(resolved),
            "four_connected_disconnection_is_a_staircase_artefact": len(disconnected)
            > len(disconnected_diagonal),
            "regions_are_orthogonally_convex": len(orthogonally_convex) == len(resolved) and bool(resolved),
        },
    }


def xi_dependence(metrics: list) -> dict:
    """Rank correlation between each hidden dimension and each region descriptor.

    Spearman rather than Pearson: the relationships in this project have consistently been
    monotone and curved (Phase 10 measured 0.910 against 0.841 for the leading probe
    feature), so a linear coefficient understates dependence that is plainly there.
    """
    from probe_drawer.analysis.probe_features import rank_correlation  # noqa: PLC0415

    solvable = [entry for entry in metrics if entry.success_points > 0 and entry.centroid]
    if len(solvable) < 5:
        return {"skipped": "too few solvable hidden states"}

    dimensions = ("mass", "static_friction", "dynamic_friction", "damping")
    descriptors = {
        "centroid_force": [entry.centroid[0] for entry in solvable],
        "centroid_duration": [entry.centroid[1] for entry in solvable],
        "success_fraction": [entry.success_fraction for entry in solvable],
        "orientation_deg": [entry.orientation_deg for entry in solvable],
        "elongation": [entry.elongation for entry in solvable],
        "midpoint_failure_rate": [entry.midpoint["rate"] for entry in solvable],
    }
    return {
        name: {
            dimension: rank_correlation([entry.xi[dimension] for entry in solvable], values)
            for dimension in dimensions
        }
        for name, values in descriptors.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--examples", type=int, default=6, help="Representative states to print in detail.")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_absolute():
        path = project_root() / path
    dataset = SweepDataset.load(path)
    criteria = MAIN_TASK.criteria

    metrics = [analyse_landscape(dataset, key, criteria) for key in dataset.xi_keys()]
    edges = valid_region_edges(dataset, criteria)
    box = proposed_box(edges)
    verdict = structure_verdict(metrics, dataset)
    dependence = xi_dependence(metrics)

    report = {
        "dataset": str(path),
        "stage": dataset.metadata.get("stage"),
        "git_commit": git_commit(),
        "episodes": len(dataset),
        "validity_rate": dataset.validity_rate(),
        "invalid_reasons": dataset.invalid_reason_counts(),
        "forces": dataset.forces(),
        "durations": dataset.durations(),
        "task": MAIN_TASK.as_dict(),
        "valid_region_edges": edges,
        "proposed_box": box,
        "structure": verdict,
        "xi_dependence": dependence,
        "per_hidden_state": [entry.as_dict() for entry in metrics],
    }

    output = Path(args.output) if args.output else path.with_name(f"{path.stem}_analysis.json")
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report, metrics, args.examples)
    print(f"[land] report written: {output}")
    print("=" * 78 + "\n")


def _print(report: dict, metrics: list, examples: int) -> None:
    verdict = report["structure"]
    print("\n" + "=" * 78)
    print(f"[land] dataset  : {report['dataset']} ({report['stage']})")
    print(
        f"[land] episodes : {report['episodes']}, {report['validity_rate'] * 100:.1f} % valid"
    )
    print(f"[land] invalid  : {report['invalid_reasons']}")
    print(
        f"[land] grid     : F {report['forces'][0]:.2f}-{report['forces'][-1]:.2f} N x "
        f"T {report['durations'][0]:.2f}-{report['durations'][-1]:.2f} s"
    )

    print("[land]")
    print("[land] validity and success along each axis:")
    print(f"[land]   {'F (N)':>7} {'valid':>7} {'success':>8}   |   {'T (s)':>6} {'valid':>7} {'success':>8}")
    per_force, per_duration = report["valid_region_edges"]["per_force"], report["valid_region_edges"]["per_duration"]
    for index in range(max(len(per_force), len(per_duration))):
        left = (
            f"{per_force[index]['value']:7.2f} {per_force[index]['valid_fraction'] * 100:6.1f}% "
            f"{per_force[index]['success_fraction'] * 100:7.1f}%"
            if index < len(per_force)
            else " " * 23
        )
        right = (
            f"{per_duration[index]['value']:6.2f} {per_duration[index]['valid_fraction'] * 100:6.1f}% "
            f"{per_duration[index]['success_fraction'] * 100:7.1f}%"
            if index < len(per_duration)
            else ""
        )
        print(f"[land]   {left}   |   {right}")

    box = report["proposed_box"]
    print("[land]")
    print(
        f"[land] proposed box: F {box['force_range']} N x T {box['duration_range']} s "
        f"(kept where some hidden state succeeds)"
    )
    print(f"[land]   dropped forces   : {box['forces_dropped']}")
    print(f"[land]   dropped durations: {box['durations_dropped']}")

    print("[land]")
    print(
        f"[land] coverage : {verdict['solvable']}/{verdict['hidden_states']} hidden states have a "
        f"succeeding (F,T) ({verdict['coverage'] * 100:.1f} %)"
    )
    area = verdict["success_fraction"]
    print(
        f"[land] success area fraction: median {area['median'] * 100:.1f} %, "
        f"range {area['min'] * 100:.1f}-{area['max'] * 100:.1f} %"
    )
    print(
        f"[land] region centroid: F {verdict['centroid_force_range']} N, "
        f"T {verdict['centroid_duration_range']} s, normalised spread "
        f"{verdict['centroid_normalised_spread']:.3f}"
    )
    print(
        f"[land] orientation: median {verdict['orientation_deg']['median']:.1f} deg, "
        f"spread {verdict['orientation_deg']['spread']:.1f} deg; "
        f"elongation median {verdict['elongation_median']:.2f}"
    )
    print(
        f"[land] contiguity : {verdict['row_contiguity_median'] * 100:.1f} % of T rows and "
        f"{verdict['column_contiguity_median'] * 100:.1f} % of F columns unbroken"
    )
    midpoint = verdict["midpoint_failure_rate"]
    print(
        f"[land] midpoint failure rate (resolved states only): median {midpoint['median'] * 100:.2f} %, "
        f"mean {midpoint['mean'] * 100:.2f} %, max {midpoint['max'] * 100:.2f} %; "
        f"{midpoint['states_above_threshold']} states above {midpoint['threshold'] * 100:.0f} %"
    )
    if verdict["caveats"]["states_too_thin_for_topology"]:
        print(
            f"[land]   ({verdict['caveats']['states_too_thin_for_topology']} states excluded as "
            "too thin for this grid to judge)"
        )
    print(
        f"[land] region size  : median {verdict['median_columns_spanned']:.0f} F columns x "
        f"{verdict['median_rows_spanned']:.0f} T rows; "
        f"{verdict['topology_resolved_states']}/{verdict['solvable']} states are wide enough "
        f"(>=4 x 4) for the grid to say anything about topology"
    )
    print(
        f"[land] disconnected : {verdict['disconnected_states']} states under 4-connectivity, "
        f"{verdict['disconnected_states_8_connectivity']} under 8-connectivity "
        f"({verdict['orthogonally_convex_states']} are orthogonally convex, i.e. every row and "
        f"column is one unbroken run)"
    )
    caveats = verdict["caveats"]
    if caveats["four_connected_disconnection_is_a_staircase_artefact"]:
        print(
            "[land]   -> the 4-connected count is inflated by staircase artefacts: a thin strip "
            "that steps one column per row is 4-disconnected and physically one band"
        )
    if caveats["regions_are_orthogonally_convex"]:
        print(
            "[land]   -> every resolved region is orthogonally convex, which is what a smooth "
            "monotone F-T trade-off band looks like"
        )

    print("[land]")
    print("[land] xi dependence (Spearman):")
    dependence = report["xi_dependence"]
    if "skipped" in dependence:
        print(f"[land]   {dependence['skipped']}")
    else:
        print(f"[land]   {'descriptor':>22} {'mass':>7} {'mu_s':>7} {'mu_d':>7} {'b':>7}")
        for name, values in dependence.items():
            print(
                f"[land]   {name:>22} "
                + " ".join(f"{values[dim]:+7.3f}" for dim in ("mass", "static_friction", "dynamic_friction", "damping"))
            )

    print("[land]")
    solvable = sorted(
        (entry for entry in metrics if entry.success_points > 0), key=lambda entry: entry.centroid[0]
    )
    chosen = [solvable[index] for index in np.linspace(0, len(solvable) - 1, min(examples, len(solvable))).astype(int)]
    print("[land] representative hidden states:")
    print(
        f"[land]   {'m':>5} {'mu_s':>5} {'mu_d':>5} {'b':>5} | {'area%':>6} {'F range':>12} "
        f"{'T range':>12} {'4c/8c':>5} {'mid%':>6} {'orient':>7}"
    )
    for entry in chosen:
        print(
            f"[land]   {entry.xi['mass']:5.1f} {entry.xi['static_friction']:5.2f} "
            f"{entry.xi['dynamic_friction']:5.2f} {entry.xi['damping']:5.1f} | "
            f"{entry.success_fraction * 100:5.1f}% "
            f"{entry.force_extent[0]:5.2f}-{entry.force_extent[1]:5.2f} "
            f"{entry.duration_extent[0]:5.2f}-{entry.duration_extent[1]:5.2f} "
            f"{entry.components:2d}/{entry.components_diagonal:1d} {entry.midpoint['rate'] * 100:5.1f}% "
            f"{entry.orientation_deg:6.1f}"
        )

    print("[land]")
    evidence = verdict["evidence"]
    print("[land] GATE -- does the second axis buy structure?")
    for name, held in evidence.items():
        print(f"[land]   {'YES' if held else 'no ':>3}  {name}")
    print(
        f"[land] VERDICT: {'PROCEED to Dataset v1' if any(evidence.values()) else 'STOP -- report and reconsider'}"
    )


if __name__ == "__main__":
    main()
