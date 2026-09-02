"""Phase 10 -- how much did the reset cost, and does the probe still predict the answer?

Two questions, both pure analysis on datasets already collected.

**What does the probe do to the task?** Phase 9's Oracle reset the drawer between the probe
and the execution, so every execution started closed and at rest. Phase 10's does not. The
comparison is per hidden state, on the same task: the force each one needs, the width of its
success band, and whether it is solvable at all. A small difference would mean the reset was
a harmless simplification; a large one means the sequential Oracle is the only admissible
ground truth. Either way the paper trains on the sequential protocol -- the point of the
comparison is to know *how much* the reset was hiding.

**Does a scalar probe feature still track the required force?** Phase 9 measured
``|rho| = 0.969`` between the probe's best feature and the reset Oracle's required force.
That was under a protocol where the probe's effect was thrown away. Recomputing it against
the sequential Oracle is what says whether a simple predictor remains plausible, and it is
the number that decides how strong a baseline the eventual model has to beat.

Usage::

    python scripts/compare_reset_vs_sequential.py
    python scripts/compare_reset_vs_sequential.py --task phase9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from probe_drawer.analysis.probe_features import PROBE_FEATURES, rank_correlation
from probe_drawer.analysis.sweep import SweepDataset, success_interval
from probe_drawer.experiment_plan import MAIN_TASK, PHASE9_RESET_TASK
from probe_drawer.utils import project_root


def pearson(left: list[float], right: list[float]) -> float:
    """Linear correlation, reported next to the rank correlation so the shape is visible.

    A rank correlation near 1 with a much lower Pearson means the relationship is monotone
    but strongly curved -- which is what a breakaway-dominated system should look like.
    """
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def compare(reset: SweepDataset, sequential: SweepDataset, criteria, duration: float) -> dict:
    """Per-hidden-state comparison of the two protocols on one task."""
    shared = [key for key in sequential.xi_keys() if key in set(reset.xi_keys())]
    rows: list[dict] = []

    for key in shared:
        reset_interval = success_interval(reset, key, criteria, duration)
        sequential_interval = success_interval(sequential, key, criteria, duration)
        row = {
            "xi": list(key),
            "reset_success": reset_interval["any_success"],
            "sequential_success": sequential_interval["any_success"],
        }
        if reset_interval["any_success"]:
            row.update(
                reset_best=reset_interval["best_force"],
                reset_low=reset_interval["force_low"],
                reset_high=reset_interval["force_high"],
                reset_width=reset_interval["force_width"],
            )
        if sequential_interval["any_success"]:
            row.update(
                sequential_best=sequential_interval["best_force"],
                sequential_low=sequential_interval["force_low"],
                sequential_high=sequential_interval["force_high"],
                sequential_width=sequential_interval["force_width"],
            )
        if reset_interval["any_success"] and sequential_interval["any_success"]:
            row["force_shift"] = sequential_interval["best_force"] - reset_interval["best_force"]
            row["force_ratio"] = sequential_interval["best_force"] / reset_interval["best_force"]
        rows.append(row)

    both = [row for row in rows if "force_shift" in row]
    shifts = [row["force_shift"] for row in both]
    ratios = [row["force_ratio"] for row in both]

    return {
        "task": {
            "duration": duration,
            "goal_displacement": criteria.goal_displacement,
            "displacement_tolerance": criteria.displacement_tolerance,
            "velocity_tolerance": criteria.velocity_tolerance,
        },
        "hidden_states_compared": len(shared),
        "reset_coverage": sum(row["reset_success"] for row in rows) / len(rows) if rows else 0.0,
        "sequential_coverage": sum(row["sequential_success"] for row in rows) / len(rows) if rows else 0.0,
        "solvable_in_both": len(both),
        "solvable_only_in_reset": sum(1 for row in rows if row["reset_success"] and not row["sequential_success"]),
        "solvable_only_in_sequential": sum(1 for row in rows if row["sequential_success"] and not row["reset_success"]),
        "force_shift_mean": float(np.mean(shifts)) if shifts else float("nan"),
        "force_shift_median": float(np.median(shifts)) if shifts else float("nan"),
        "force_shift_abs_max": float(np.max(np.abs(shifts))) if shifts else float("nan"),
        "force_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
        "force_ratio_range": [float(np.min(ratios)), float(np.max(ratios))] if ratios else None,
        "reset_median_width": float(np.median([row["reset_width"] for row in both])) if both else float("nan"),
        "sequential_median_width": (
            float(np.median([row["sequential_width"] for row in both])) if both else float("nan")
        ),
        "rank_correlation_of_required_force": rank_correlation(
            [row["reset_best"] for row in both], [row["sequential_best"] for row in both]
        ),
        "rows": rows,
    }


def probe_predictivity(sequential: SweepDataset, criteria, duration: float) -> dict:
    """Correlate each probe feature with the force the sequential Oracle says is needed.

    A hidden state's probe features vary a little from episode to episode, so each one's
    features are averaged over its own rows before correlating -- the question is whether the
    *hidden state's* probe signature predicts its required force, not whether one episode does.
    """
    features: dict[str, list[float]] = {name: [] for name in PROBE_FEATURES}
    required: list[float] = []

    for key in sequential.xi_keys():
        interval = success_interval(sequential, key, criteria, duration)
        if not interval["any_success"]:
            continue
        rows = [row for row in sequential.select(xi_key=key, duration=duration) if row.probe_features]
        if not rows:
            continue
        required.append(interval["best_force"])
        for name in PROBE_FEATURES:
            features[name].append(float(np.mean([row.probe_features[name] for row in rows])))

    correlations = {
        name: {
            "spearman": rank_correlation(values, required),
            "pearson": pearson(values, required),
        }
        for name, values in features.items()
    }
    finite = {
        name: abs(payload["spearman"])
        for name, payload in correlations.items()
        if np.isfinite(payload["spearman"])
    }
    best = max(finite, key=lambda name: finite[name]) if finite else None

    return {
        "hidden_states": len(required),
        "required_force_range": [min(required), max(required)] if required else None,
        "correlations": correlations,
        "best_feature": best,
        "best_spearman": finite.get(best) if best else float("nan"),
        # The pairs the correlations were computed from, so a figure plots the same numbers
        # rather than recomputing them from a possibly different dataset.
        "feature_values": {**features, "required_force": required},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        type=str,
        default=None,
        help="Phase 9 reset dataset. Defaults to outputs/logs/sweep_fine_fall035.json.",
    )
    parser.add_argument(
        "--sequential",
        type=str,
        default=None,
        help="Phase 10 sequential dataset. Defaults to outputs/logs/sequential_oracle_fall035.json.",
    )
    parser.add_argument(
        "--task",
        choices=("phase10", "phase9", "both"),
        default="both",
        help="Which task definition to compare on.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logs = project_root() / "outputs" / "logs"
    reset = SweepDataset.load(Path(args.reset) if args.reset else logs / "sweep_fine_fall035.json")
    sequential = SweepDataset.load(
        Path(args.sequential) if args.sequential else logs / "sequential_oracle_fall035.json"
    )
    duration = MAIN_TASK.duration

    tasks = {"phase10": MAIN_TASK, "phase9": PHASE9_RESET_TASK}
    chosen = tasks if args.task == "both" else {args.task: tasks[args.task]}

    report: dict = {
        "reset_dataset": {
            "rows": len(reset),
            "forces": [reset.forces()[0], reset.forces()[-1], len(reset.forces())],
            "fall_fraction": reset.metadata.get("fall_fraction"),
            "protocol": "reset",
        },
        "sequential_dataset": {
            "rows": len(sequential),
            "forces": [sequential.forces()[0], sequential.forces()[-1], len(sequential.forces())],
            "fall_fraction": sequential.metadata.get("fall_fraction"),
            "transition_steps": sequential.metadata.get("transition_steps"),
            "protocol": "sequential",
        },
        "comparisons": {name: compare(reset, sequential, task.criteria, duration) for name, task in chosen.items()},
        "probe_predictivity": probe_predictivity(sequential, MAIN_TASK.criteria, duration),
    }

    _print(report)
    output = Path(args.output) if args.output else logs / "reset_vs_sequential.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[compare] report written : {output}")
    print("=" * 78 + "\n")


def _print(report: dict) -> None:
    print("\n" + "=" * 78)
    for label in ("reset_dataset", "sequential_dataset"):
        info = report[label]
        print(
            f"[compare] {info['protocol']:>10} : {info['rows']} rows, "
            f"F {info['forces'][0]:.2f}-{info['forces'][1]:.2f} N ({info['forces'][2]} values), "
            f"fall={info['fall_fraction']}"
            + (f", gap={info['transition_steps']} steps" if info.get("transition_steps") else "")
        )
    print(
        "[compare] the two grids differ in resolution (0.25 N against 0.05-0.10 N), so band "
        "widths are compared with that in mind"
    )

    for name, comparison in report["comparisons"].items():
        task = comparison["task"]
        print("[compare]")
        print(
            f"[compare] task '{name}': T={task['duration']:g} s d_goal={task['goal_displacement'] * 1000:g} mm "
            f"eps_d={task['displacement_tolerance'] * 1000:g} mm eps_v={task['velocity_tolerance']:g} m/s"
        )
        print(
            f"[compare]   coverage        : reset {comparison['reset_coverage']:.3f}  "
            f"sequential {comparison['sequential_coverage']:.3f}"
        )
        print(
            f"[compare]   solvable        : both {comparison['solvable_in_both']}, "
            f"reset only {comparison['solvable_only_in_reset']}, "
            f"sequential only {comparison['solvable_only_in_sequential']}"
        )
        print(
            f"[compare]   required force  : shift median {comparison['force_shift_median']:+.3f} N, "
            f"mean {comparison['force_shift_mean']:+.3f} N, largest |shift| "
            f"{comparison['force_shift_abs_max']:.3f} N"
        )
        if comparison["force_ratio_range"]:
            print(
                f"[compare]   ratio sequential/reset: median {comparison['force_ratio_median']:.3f}, "
                f"range {comparison['force_ratio_range'][0]:.3f}-{comparison['force_ratio_range'][1]:.3f}"
            )
        print(
            f"[compare]   median band     : reset {comparison['reset_median_width']:.2f} N  "
            f"sequential {comparison['sequential_median_width']:.2f} N"
        )
        print(
            f"[compare]   rank corr of required force between protocols: "
            f"{comparison['rank_correlation_of_required_force']:+.4f}"
        )
        if comparison["solvable_only_in_sequential"] > 5 * max(comparison["solvable_only_in_reset"], 1):
            print(
                "[compare]   NOTE: the reset dataset's coverage here is limited by its own force grid "
                "(1.00 N floor, 0.25 N spacing), not by the reset. This task is simply not expressible "
                "on that grid, which is why it was re-swept. The fair protocol comparison is the task "
                "the reset Oracle was itself selected for."
            )

    predictivity = report["probe_predictivity"]
    print("[compare]")
    print(
        f"[compare] probe feature vs SEQUENTIAL required force, {predictivity['hidden_states']} hidden states "
        f"(required force {predictivity['required_force_range'][0]:.2f}-{predictivity['required_force_range'][1]:.2f} N):"
    )
    print(f"[compare]   {'feature':>30} {'spearman':>10} {'pearson':>10}")
    ordered = sorted(
        predictivity["correlations"].items(),
        key=lambda item: -abs(item[1]["spearman"]) if np.isfinite(item[1]["spearman"]) else 0.0,
    )
    for feature, payload in ordered:
        print(f"[compare]   {feature:>30} {payload['spearman']:10.4f} {payload['pearson']:10.4f}")
    print(
        f"[compare]   best feature: {predictivity['best_feature']} "
        f"|rho| = {predictivity['best_spearman']:.4f}"
    )


if __name__ == "__main__":
    main()
