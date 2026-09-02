"""Phase 10 figures -- the task refinement and the protocol change, drawn.

Seven figures, all from reports already on disk, so this needs no simulator:

======  ====================================================  ==========================
Figure  What it shows                                         Source
======  ====================================================  ==========================
A       Each hidden state's success band on the fine grid     ``task_refinement.json``
B       Coverage against the position tolerance               ``task_refinement.json``
C       Coverage against the terminal-velocity tolerance      ``task_refinement.json``
D       The three ramp-down fractions compared                ``task_refinement.json``
E       Reset against sequential, per hidden state             ``reset_vs_sequential.json``
F       Required force across the hidden-state grid           ``task_refinement.json``
G       Probe feature against the sequential required force   ``reset_vs_sequential.json``
======  ====================================================  ==========================

Usage::

    python scripts/plot_phase10.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe_drawer.experiment_plan import MAIN_TASK  # noqa: E402
from probe_drawer.utils import project_root  # noqa: E402

#: The ramp-down fraction the task was selected with, so it can be highlighted.
SELECTED_FALL = "fall=0.35"

#: Axis labels for the four hidden dimensions, in registry order.
XI_LABELS = ("mass $m$ (kg)", "static friction $\\mu_s$", "dynamic friction $\\mu_d$", "damping $b$ (N$\\cdot$s/m)")


def plots_dir() -> Path:
    directory = project_root() / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_force_intervals(refinement: dict, path: Path) -> Path:
    """A -- every hidden state's success band, sorted by the force it needs.

    The point of the figure is the vertical spread: if every drawer needed the same force
    there would be nothing to adapt to. The bands are what a predictor has to land inside.
    """
    intervals = [row for row in refinement["selected_intervals"] if row["any_success"]]
    intervals.sort(key=lambda row: row["best_force"])
    missing = len(refinement["selected_intervals"]) - len(intervals)

    figure, axis = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    positions = np.arange(len(intervals))
    lows = np.array([row["force_low"] for row in intervals])
    highs = np.array([row["force_high"] for row in intervals])
    bests = np.array([row["best_force"] for row in intervals])

    axis.vlines(positions, lows, highs, color="tab:blue", alpha=0.55, linewidth=2.0, label="success band")
    axis.plot(positions, bests, ".", color="tab:red", markersize=4.0, label="closest to the goal")
    axis.set_xlabel(f"hidden state, sorted by required force ({len(intervals)} solvable, {missing} not)")
    axis.set_ylabel("$F_\\mathrm{peak}$ (N)")
    axis.set_title(
        f"A. Success bands on the fine grid "
        f"($d_\\mathrm{{goal}}$={MAIN_TASK.goal_displacement * 1000:g} mm, "
        f"$\\epsilon_d$={MAIN_TASK.displacement_tolerance * 1000:g} mm, "
        f"$\\epsilon_v$={MAIN_TASK.velocity_tolerance:g} m/s, grid {refinement['selected']['grid_step']:g} N)"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _coverage_curves(curves: dict, kind: str, tolerances_key: str, axis, scale: float, unit: str) -> None:
    """One panel of B or C: coverage against a tolerance, one line per goal."""
    tolerances = np.array(curves[tolerances_key]) * scale
    for goal, coverages in sorted(curves[kind].items(), key=lambda item: float(item[0])):
        axis.plot(tolerances, coverages, "o-", markersize=4.0, label=f"$d_\\mathrm{{goal}}$={float(goal) * 1000:g} mm")
    axis.axhline(0.80, color="grey", linestyle="--", linewidth=1.0)
    axis.text(tolerances[0], 0.81, "coverage floor 0.80", color="grey", fontsize=8)
    axis.set_ylabel("coverage (fraction of hidden states with a succeeding force)")
    axis.set_xlabel(unit)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.3)


def plot_coverage_vs_eps_d(refinement: dict, path: Path) -> Path:
    """B -- how much position precision the task can afford.

    This is the trade-off the task selection turns on. Tightening ``eps_d`` costs coverage,
    and the selected point is the tightest tolerance that keeps coverage above the floor
    *and* leaves a success band wide enough to aim at.
    """
    curves = refinement["tolerance_curves"][SELECTED_FALL]
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    _coverage_curves(curves, "coverage_vs_eps_d", "displacement_tolerances", axis, 1000.0, "$\\epsilon_d$ (mm)")
    axis.axvline(
        MAIN_TASK.displacement_tolerance * 1000.0, color="tab:red", linestyle=":", linewidth=1.5, label="selected"
    )
    axis.set_title(f"B. Coverage against position tolerance ({SELECTED_FALL}, $\\epsilon_v$=0.03 m/s)")
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_coverage_vs_eps_v(refinement: dict, path: Path) -> Path:
    """C -- how much terminal-velocity precision the task can afford.

    Steeper than B, and for a physical reason: a drawer still moving at ``T`` has not been
    placed, and whether it can stop in time is set by the ramp-down, not by the force.
    """
    curves = refinement["tolerance_curves"][SELECTED_FALL]
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    _coverage_curves(curves, "coverage_vs_eps_v", "velocity_tolerances", axis, 1.0, "$\\epsilon_v$ (m/s)")
    axis.axvline(MAIN_TASK.velocity_tolerance, color="tab:red", linestyle=":", linewidth=1.5, label="selected")
    axis.set_title(f"C. Coverage against terminal-velocity tolerance ({SELECTED_FALL})")
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_fall_fraction_comparison(refinement: dict, path: Path) -> Path:
    """D -- the three ramp-down fractions, at the selected goal.

    A longer ramp-down gives a low-resistance drawer time to decelerate before ``T``, which
    is what the terminal-velocity condition needs. The figure is the evidence that 0.35 is
    not a preference.
    """
    goal = f"{MAIN_TASK.goal_displacement:g}"
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

    for label, curves in sorted(refinement["tolerance_curves"].items()):
        style = {"linewidth": 2.5, "marker": "o"} if label == SELECTED_FALL else {"linewidth": 1.2, "marker": "."}
        axes[0].plot(
            np.array(curves["displacement_tolerances"]) * 1000.0,
            curves["coverage_vs_eps_d"][goal],
            label=label,
            **style,
        )
        axes[1].plot(curves["velocity_tolerances"], curves["coverage_vs_eps_v"][goal], label=label, **style)

    for axis, selected, unit in (
        (axes[0], MAIN_TASK.displacement_tolerance * 1000.0, "$\\epsilon_d$ (mm)"),
        (axes[1], MAIN_TASK.velocity_tolerance, "$\\epsilon_v$ (m/s)"),
    ):
        axis.axvline(selected, color="tab:red", linestyle=":", linewidth=1.5)
        axis.axhline(0.80, color="grey", linestyle="--", linewidth=1.0)
        axis.set_xlabel(unit)
        axis.set_ylabel("coverage")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=9)

    figure.suptitle(f"D. Ramp-down fraction, at $d_\\mathrm{{goal}}$={MAIN_TASK.goal_displacement * 1000:g} mm")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_reset_vs_sequential(comparison: dict, path: Path) -> Path:
    """E -- what the probe did to the force the task needs.

    Left: the required force under each protocol, per hidden state. Points below the diagonal
    are drawers the probe made easier. Right: the same as a ratio.

    The ratio histogram is the panel that matters, and it is bimodal rather than centred: one
    cluster near 0.55-0.65 and another near 0.95-1.00. So the reset was not a uniform
    rescaling that a calibration factor could undo -- the probe's benefit depends on the
    hidden state, which is precisely the thing a model has to infer.
    """
    rows = [row for row in comparison["comparisons"]["phase9"]["rows"] if "force_ratio" in row]
    reset = np.array([row["reset_best"] for row in rows])
    sequential = np.array([row["sequential_best"] for row in rows])
    ratios = np.array([row["force_ratio"] for row in rows])

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    limit = max(reset.max(), sequential.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], color="grey", linestyle="--", linewidth=1.0, label="no effect")
    axes[0].plot(reset, sequential, "o", markersize=5.0, alpha=0.7, color="tab:blue")
    axes[0].set_xlabel("required $F_\\mathrm{peak}$, reset protocol (N)")
    axes[0].set_ylabel("required $F_\\mathrm{peak}$, sequential protocol (N)")
    axes[0].set_title(
        f"per hidden state (n={len(rows)}), rank corr "
        f"{comparison['comparisons']['phase9']['rank_correlation_of_required_force']:+.3f}"
    )
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(ratios, bins=20, color="tab:blue", alpha=0.75)
    axes[1].axvline(1.0, color="grey", linestyle="--", linewidth=1.0, label="no effect")
    median = float(np.median(ratios))
    axes[1].axvline(median, color="tab:red", linestyle=":", linewidth=1.8, label=f"median {median:.3f}")
    axes[1].set_xlabel("sequential / reset required force")
    axes[1].set_ylabel("hidden states")
    axes[1].set_title("bimodal: the probe helps compliant drawers, barely helps stiff ones")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    task = comparison["comparisons"]["phase9"]["task"]
    figure.suptitle(
        f"E. Reset against sequential, on the task both grids express "
        f"($d_\\mathrm{{goal}}$={task['goal_displacement'] * 1000:g} mm, "
        f"$\\epsilon_d$={task['displacement_tolerance'] * 1000:g} mm, "
        f"$\\epsilon_v$={task['velocity_tolerance']:g} m/s)"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_required_force_across_xi(refinement: dict, path: Path) -> Path:
    """F -- the required force against each hidden dimension separately.

    Four panels rather than one, because the question is not "is there spread" (figure A
    already showed that) but "which dimension causes it". The answer is stark: dynamic
    friction sets the required force almost by itself, static friction is secondary, and mass
    and damping barely move the median at all. The damping panel is the identifiability
    limitation made visible -- and also its consolation, since a dimension that does not
    change the answer costs little to leave unidentified.
    """
    intervals = [row for row in refinement["selected_intervals"] if row["any_success"]]
    values = np.array([row["xi"] for row in intervals])
    forces = np.array([row["best_force"] for row in intervals])

    figure, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), constrained_layout=True, sharey=True)
    for index, (axis, label) in enumerate(zip(axes, XI_LABELS, strict=True)):
        levels = sorted(set(values[:, index]))
        axis.boxplot(
            [forces[values[:, index] == level] for level in levels],
            positions=np.arange(len(levels)),
            widths=0.55,
        )
        axis.set_xticks(np.arange(len(levels)))
        # mu_d is derived (mu_s times the ratio) and so has twelve levels rather than three
        # or four; its labels only fit rotated.
        axis.set_xticklabels([f"{level:g}" for level in levels], rotation=90 if len(levels) > 5 else 0, fontsize=8)
        axis.set_xlabel(label)
        axis.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("required $F_\\mathrm{peak}$ (N)")
    figure.suptitle(
        f"F. Required force against each hidden dimension, sequential protocol (n={len(intervals)})"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_probe_predictivity(comparison: dict, path: Path) -> Path:
    """G -- is a scalar probe feature still enough?

    Left: every feature's rank and linear correlation with the required force. Right: the
    strongest one drawn against the required force, which is where a curved but monotone
    relationship shows itself -- and where the residual spread is the error a scalar
    predictor cannot avoid.
    """
    predictivity = comparison["probe_predictivity"]
    correlations = predictivity["correlations"]
    ordered = sorted(correlations.items(), key=lambda item: -abs(item[1]["spearman"]))
    names = [name for name, _ in ordered]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    positions = np.arange(len(names))
    axes[0].barh(positions - 0.2, [abs(correlations[n]["spearman"]) for n in names], height=0.4, label="|Spearman|")
    axes[0].barh(positions + 0.2, [abs(correlations[n]["pearson"]) for n in names], height=0.4, label="|Pearson|")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([name.replace("_", " ") for name in names], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("correlation with the required $F_\\mathrm{peak}$")
    axes[0].set_title(f"all probe features (n={predictivity['hidden_states']} hidden states)")
    axes[0].grid(alpha=0.3, axis="x")
    axes[0].legend(fontsize=9)

    best = predictivity["best_feature"]
    pairs = predictivity.get("feature_values")
    if not pairs:
        raise SystemExit(
            "The comparison report predates the stored feature values. Re-run "
            "scripts/compare_reset_vs_sequential.py so the panel plots the same numbers the "
            "correlations were computed from."
        )
    axes[1].plot(pairs[best], pairs["required_force"], "o", markersize=5.0, alpha=0.7, color="tab:blue")
    axes[1].set_xlabel(best.replace("_", " "))
    axes[1].set_ylabel("required $F_\\mathrm{peak}$ (N)")
    axes[1].set_title(
        f"strongest feature: Spearman {correlations[best]['spearman']:+.3f}, "
        f"Pearson {correlations[best]['pearson']:+.3f}"
    )
    axes[1].grid(alpha=0.3)

    figure.suptitle("G. Probe feature against the sequential required force")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=str, default=None, help="task_refinement.json")
    parser.add_argument("--comparison", type=str, default=None, help="reset_vs_sequential.json")
    args = parser.parse_args()

    logs = project_root() / "outputs" / "logs"
    refinement = json.loads(Path(args.refinement or logs / "task_refinement.json").read_text())
    comparison = json.loads(Path(args.comparison or logs / "reset_vs_sequential.json").read_text())
    if refinement.get("selected") is None:
        raise SystemExit("The task refinement has no selected candidate; nothing to plot.")

    written = [
        plot_force_intervals(refinement, plots_dir() / "phase10_a_force_intervals.png"),
        plot_coverage_vs_eps_d(refinement, plots_dir() / "phase10_b_coverage_vs_eps_d.png"),
        plot_coverage_vs_eps_v(refinement, plots_dir() / "phase10_c_coverage_vs_eps_v.png"),
        plot_fall_fraction_comparison(refinement, plots_dir() / "phase10_d_fall_fraction.png"),
        plot_reset_vs_sequential(comparison, plots_dir() / "phase10_e_reset_vs_sequential.png"),
        plot_required_force_across_xi(refinement, plots_dir() / "phase10_f_force_across_xi.png"),
        plot_probe_predictivity(comparison, plots_dir() / "phase10_g_probe_predictivity.png"),
    ]
    for path in written:
        print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
