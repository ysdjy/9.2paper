"""Phase 9M -- the figures that carry the Phase 9 argument. Needs no Isaac Sim.

Reads the sweep datasets and the Oracle report and produces the high-information-density
figures rather than one plot per signal:

``experiment_space_surfaces.png``
    ``(F_peak, T) -> d(T)`` and ``-> v(T)`` for representative hidden states, with the
    goal and the terminal-velocity tolerance marked.
``experiment_space_validity.png``
    Where the usable operating region is, and which condition invalidates the rest.
``oracle_success_landscape.png``
    The Oracle labels themselves: success against ``F_peak`` for every hidden state,
    ordered by the force each one needs.
``oracle_force_intervals.png``
    The per-hidden-state success bands, which is the figure that shows adaptation is
    necessary: they do not overlap into one band.
``execution_drift_vs_operating_point.png``
    Held-axis drift against force, displacement and velocity, which answers whether the
    drift seen in Phase 8 was the operating point or the controller.

Usage::

    python scripts/plot_experiment_space.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe_drawer.analysis.sweep import SweepDataset  # noqa: E402
from probe_drawer.evaluation import DRAWER_TRAVEL_LIMIT, OperatingRegionCfg  # noqa: E402
from probe_drawer.utils import project_root  # noqa: E402


def plots_dir() -> Path:
    directory = project_root() / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def xi_label(xi_key: tuple[float, ...]) -> str:
    mass, static, dynamic, damping = xi_key
    return f"m={mass:g} $\\mu_s$={static:g} $\\mu_d$={dynamic:g} b={damping:g}"


def plot_surfaces(dataset: SweepDataset, path: Path) -> Path:
    """``(F_peak, T)`` surfaces of ``d(T)`` and ``v(T)`` for a spread of hidden states."""
    keys = dataset.xi_keys()
    chosen = [keys[index] for index in np.linspace(0, len(keys) - 1, 4).astype(int)]
    durations = dataset.durations()

    figure, axes = plt.subplots(2, len(chosen), figsize=(4.0 * len(chosen), 7.0), constrained_layout=True)
    for column, key in enumerate(chosen):
        forces, _, displacement = dataset.surface("final_displacement", key)
        _, _, velocity = dataset.surface("final_velocity", key)
        for row_index, duration in enumerate(durations):
            axes[0, column].plot(forces, displacement[row_index] * 1000, marker=".", label=f"T={duration:g} s")
            axes[1, column].plot(forces, np.abs(velocity[row_index]), marker=".", label=f"T={duration:g} s")
        axes[0, column].set_title(xi_label(key), fontsize=8)
        axes[0, column].axhline(50.0, color="k", linestyle="--", linewidth=0.8, label="$d_{goal}$ = 50 mm")
        axes[1, column].axhline(0.08, color="k", linestyle="--", linewidth=0.8, label="$\\epsilon_v$ = 0.08 m/s")
        for axis in (axes[0, column], axes[1, column]):
            axis.grid(alpha=0.3)
            axis.set_xlabel("$F_{peak}$ [N]")
        axes[0, column].set_ylabel("d(T) [mm]" if column == 0 else "")
        axes[1, column].set_ylabel("|v(T)| [m/s]" if column == 0 else "")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    figure.suptitle("Execution response surfaces: the same command, four hidden states", fontsize=11)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_validity(dataset: SweepDataset, path: Path) -> Path:
    """A heatmap of usable fraction over ``(F_peak, T)``, plus what invalidates the rest."""
    forces, durations = dataset.forces(), dataset.durations()
    fraction = np.full((len(durations), len(forces)), np.nan)
    for row, duration in enumerate(durations):
        for column, force in enumerate(forces):
            rows = dataset.select(duration=duration, peak_force=force)
            if rows:
                fraction[row, column] = sum(record.valid for record in rows) / len(rows)

    reasons = dataset.invalid_reason_counts()
    figure, (heat, bars) = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    image = heat.imshow(fraction, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    heat.set_xticks(range(len(forces)), [f"{force:g}" for force in forces], rotation=90, fontsize=7)
    heat.set_yticks(range(len(durations)), [f"{duration:g}" for duration in durations])
    heat.set(xlabel="$F_{peak}$ [N]", ylabel="T [s]", title="fraction of hidden states in the valid region")
    figure.colorbar(image, ax=heat)

    bars.barh(list(reasons)[::-1], list(reasons.values())[::-1], color="tab:red", alpha=0.75)
    bars.set(xlabel="swept points rejected", title="why points are outside the valid region")
    bars.grid(alpha=0.3, axis="x")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_success_landscape(dataset: SweepDataset, report: dict, path: Path) -> Path:
    """The Oracle labels: success over ``(hidden state, F_peak)`` at the chosen task."""
    candidate = report["recommended"]["candidate"]
    criteria_duration = candidate["duration"]
    intervals = {tuple(row["xi"]): row for row in report["recommended_intervals"]}
    ordered = sorted(
        intervals.values(),
        key=lambda row: row["force_centre"] if row["any_success"] else float("inf"),
    )
    forces = dataset.forces()
    grid = np.zeros((len(ordered), len(forces)))

    from probe_drawer.evaluation import SuccessCriteria  # noqa: PLC0415

    criteria = SuccessCriteria(
        goal_displacement=candidate["goal_displacement"],
        displacement_tolerance=candidate["displacement_tolerance"],
        velocity_tolerance=candidate["velocity_tolerance"],
    )
    for row_index, row in enumerate(ordered):
        key = tuple(round(value, 6) for value in row["xi"])
        for record in dataset.select(xi_key=key, duration=criteria_duration):
            column = forces.index(record.peak_force)
            grid[row_index, column] = 2.0 if record.succeeds(criteria) else (1.0 if record.valid else 0.0)

    figure, axis = plt.subplots(figsize=(9.0, 6.5), constrained_layout=True)
    image = axis.imshow(
        grid, aspect="auto", origin="lower", cmap=matplotlib.colors.ListedColormap(["#d9d9d9", "#9ecae1", "#08519c"])
    )
    axis.set(
        xlabel="$F_{peak}$ [N]",
        ylabel="hidden states, ordered by the force they need",
        title=(
            f"Oracle success landscape: T={criteria_duration:g} s, "
            f"$d_{{goal}}$={candidate['goal_displacement'] * 1000:g} mm, "
            f"$\\epsilon_d$={candidate['displacement_tolerance'] * 1000:g} mm, "
            f"$\\epsilon_v$={candidate['velocity_tolerance']:g} m/s"
        ),
    )
    axis.set_xticks(range(len(forces)), [f"{force:g}" for force in forces], rotation=90, fontsize=7)
    colorbar = figure.colorbar(image, ax=axis, ticks=[0.33, 1.0, 1.67])
    colorbar.ax.set_yticklabels(["invalid", "valid, misses", "success"])
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_force_intervals(report: dict, path: Path) -> Path:
    """Per-hidden-state success bands: the figure that shows one force cannot serve all."""
    rows = [row for row in report["recommended_intervals"] if row["any_success"]]
    rows.sort(key=lambda row: row["force_centre"])
    positions = np.arange(len(rows))
    low = np.array([row["force_low"] for row in rows])
    high = np.array([row["force_high"] for row in rows])
    best = np.array([row["best_force"] for row in rows])

    figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    axis.hlines(positions, low, high, color="tab:blue", linewidth=2.5, alpha=0.7, label="success band")
    axis.plot(best, positions, "o", color="tab:red", markersize=3.5, label="best force")
    for force, style, label in (
        (float(np.median(best)), "--", "median required force"),
        (float(best.min()), ":", "min / max required"),
        (float(best.max()), ":", None),
    ):
        axis.axvline(force, color="k", linestyle=style, linewidth=0.9, label=label)
    axis.set(
        xlabel="$F_{peak}$ [N]",
        ylabel="hidden states, ordered by required force",
        title=(
            f"Success force band per hidden state: {best.min():.2f}-{best.max():.2f} N, "
            f"a {best.max() / best.min():.1f}x range"
        ),
    )
    axis.grid(alpha=0.3, axis="x")
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_drift(dataset: SweepDataset, path: Path) -> Path:
    """Held-axis drift against force, displacement and speed, for every swept point.

    Answers the Phase 8 question directly: if drift only grows with displacement and speed,
    it is the operating point; if it grows with force at small displacement, it is the
    controller.
    """
    region = OperatingRegionCfg()
    records = dataset.records
    drift = np.array([record.peak_lateral_drift * 1000 for record in records])
    valid = np.array([record.valid for record in records])

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    for axis, values, label in (
        (axes[0], np.array([record.peak_force for record in records]), "$F_{peak}$ [N]"),
        (axes[1], np.array([record.final_displacement * 1000 for record in records]), "d(T) [mm]"),
        (axes[2], np.array([record.peak_velocity for record in records]), "peak |v| [m/s]"),
    ):
        axis.scatter(values[valid], drift[valid], s=6, alpha=0.35, label="valid", color="tab:blue")
        axis.scatter(values[~valid], drift[~valid], s=6, alpha=0.35, label="invalid", color="tab:red")
        axis.axhline(region.max_lateral_drift * 1000, color="k", linestyle="--", linewidth=0.9, label="validity limit")
        axis.set(xlabel=label, yscale="log")
        axis.grid(alpha=0.3)
    axes[1].axvline(
        region.max_displacement * 1000, color="tab:green", linestyle=":", linewidth=1.2, label="mechanical margin"
    )
    axes[0].set_ylabel("peak TCP lateral drift [mm]")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle(
        f"Held-axis drift against the operating point (drawer travel limit {DRAWER_TRAVEL_LIMIT * 1000:g} mm)",
        fontsize=11,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default=None, help="Sweep dataset for the surfaces and drift.")
    parser.add_argument("--landscape", type=str, default=None, help="Oracle landscape report.")
    parser.add_argument("--coarse", type=str, default=None, help="Coarse sweep, used for the validity map.")
    args = parser.parse_args()

    logs = project_root() / "outputs" / "logs"
    landscape = json.loads(Path(args.landscape or logs / "oracle_landscape.json").read_text())
    if landscape.get("recommended") is None:
        raise SystemExit("The Oracle landscape has no accepted candidate; nothing to plot.")

    fine = SweepDataset.load(args.dataset or Path(landscape["recommended"]["dataset"]))
    coarse = SweepDataset.load(args.coarse or logs / "sweep_execution_coarse.json")

    written = [
        plot_surfaces(fine, plots_dir() / "experiment_space_surfaces.png"),
        plot_validity(coarse, plots_dir() / "experiment_space_validity.png"),
        plot_success_landscape(fine, landscape, plots_dir() / "oracle_success_landscape.png"),
        plot_force_intervals(landscape, plots_dir() / "oracle_force_intervals.png"),
        plot_drift(fine, plots_dir() / "execution_drift_vs_operating_point.png"),
    ]
    for path in written:
        print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
