"""Phase 12 figures -- what the two-dimensional success region actually looks like.

The point of these is to make the gate's verdict checkable by eye rather than only by
statistic. A midpoint-failure rate of 10 % means one thing if the region is a curved band and
quite another if it is two islands, and a heatmap settles which in a way a number cannot.

======  ==================================================================  ================
Figure  What it shows                                                       Source
======  ==================================================================  ================
A       ``F x T -> success`` for representative hidden states               2-D sweep
B       Every hidden state's success region, overlaid                       2-D sweep
C       Region centroid, extent and orientation against xi                  sweep + analysis
D       A concrete success-success pair whose midpoint fails                sweep + analysis
E       ``F x T -> d(T)`` and ``-> v(T)``, and which constraint binds       2-D sweep
F       Validity map and why each region is invalid                         2-D sweep
======  ==================================================================  ================

Figures E and F are the physics behind the shape: the success band is the intersection of
"far enough" and "slow enough", and seeing the two surfaces separately shows which one each
edge of the band comes from.

Usage::

    python scripts/plot_phase12.py --dataset outputs/logs/landscape_2d_fine.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe_drawer.experimental.landscape_2d import analyse_landscape, success_mask  # noqa: E402
from probe_drawer.analysis.sweep import SweepDataset  # noqa: E402
from probe_drawer.experiment_plan import MAIN_TASK  # noqa: E402
from probe_drawer.utils import project_root  # noqa: E402

#: Hidden states shown individually, chosen by where their region sits rather than at random,
#: so the panels span the whole force range instead of clustering.
PANELS = 6


def plots_dir() -> Path:
    directory = project_root() / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def label(xi: dict) -> str:
    return (
        f"m={xi['mass']:.1f} $\\mu_s$={xi['static_friction']:.2f} "
        f"$\\mu_d$={xi['dynamic_friction']:.2f} b={xi['damping']:.1f}"
    )


def spread_of_states(metrics: list, count: int) -> list:
    """Solvable states spanning the range of region locations, not a random sample."""
    solvable = sorted((entry for entry in metrics if entry.centroid), key=lambda entry: entry.centroid[0])
    if not solvable:
        return []
    picks = np.linspace(0, len(solvable) - 1, min(count, len(solvable))).astype(int)
    return [solvable[index] for index in picks]


def plot_success_heatmaps(dataset: SweepDataset, metrics: list, path: Path) -> Path:
    """A -- the success region for a spread of hidden states.

    Grey is swept-and-failed, blue is success, white is unswept. The axes are shared so the
    regions' *relative* positions are readable, which is the whole point: they do not overlap.
    """
    chosen = spread_of_states(metrics, PANELS)
    columns = 3
    rows = int(np.ceil(len(chosen) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.4 * columns, 3.6 * rows), constrained_layout=True, sharex=True, sharey=True
    )
    flat = np.atleast_1d(axes).ravel()

    for axis, entry in zip(flat, chosen, strict=False):
        masks = success_mask(dataset, tuple(entry.xi.values()), MAIN_TASK.criteria)
        forces, durations = masks["forces"], masks["durations"]
        picture = np.where(masks["success"], 2.0, np.where(masks["swept"], 1.0, 0.0))
        axis.pcolormesh(
            forces, durations, picture, cmap="Blues", vmin=0, vmax=2, shading="nearest"
        )
        if entry.centroid:
            axis.plot(*entry.centroid, "x", color="tab:red", markersize=9, markeredgewidth=2)
        axis.set_title(
            f"{label(entry.xi)}\narea {entry.success_fraction * 100:.1f} %, "
            f"{entry.components}/{entry.components_diagonal} comp (4c/8c), "
            f"midpoint fail {entry.midpoint['rate'] * 100:.0f} %",
            fontsize=8,
        )
        axis.grid(alpha=0.2)
    for axis in flat[len(chosen) :]:
        axis.axis("off")
    for axis in flat[-columns:]:
        axis.set_xlabel("$F_\\mathrm{peak}$ (N)")
    for index in range(0, len(flat), columns):
        flat[index].set_ylabel("$T$ (s)")

    figure.suptitle(
        f"A. Success region in $(F_\\mathrm{{peak}}, T)$ -- blue succeeds, grey fails "
        f"($d_\\mathrm{{goal}}$={MAIN_TASK.goal_displacement * 1000:g} mm, red x = centroid)"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_region_overlay(dataset: SweepDataset, metrics: list, path: Path) -> Path:
    """B -- every region at once, coloured by dynamic friction.

    The single most informative figure of the phase: if the regions tile the force axis in
    order of ``mu_d`` and share the ``T`` axis, then the hidden state sets ``F`` and leaves
    ``T`` largely free, which bears directly on whether a second axis was worth opening.
    """
    solvable = [entry for entry in metrics if entry.centroid]
    if not solvable:
        raise SystemExit("no solvable hidden states to overlay")
    frictions = np.array([entry.xi["dynamic_friction"] for entry in solvable])
    colours = plt.cm.viridis((frictions - frictions.min()) / max(float(np.ptp(frictions)), 1e-9))

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    for entry, colour in zip(solvable, colours, strict=True):
        masks = success_mask(dataset, tuple(entry.xi.values()), MAIN_TASK.criteria)
        rows, columns = np.nonzero(masks["success"])
        axes[0].plot(
            masks["forces"][columns], masks["durations"][rows], ".", color=colour, markersize=2.5, alpha=0.55
        )
    axes[0].set_xlabel("$F_\\mathrm{peak}$ (N)")
    axes[0].set_ylabel("$T$ (s)")
    axes[0].set_title(f"all {len(solvable)} success regions, coloured by $\\mu_d$")
    axes[0].grid(alpha=0.3)

    # The force interval each hidden state succeeds in, ordered by mu_d: the tiling, if any.
    order = np.argsort(frictions)
    for position, index in enumerate(order):
        entry = solvable[index]
        axes[1].plot(
            entry.force_extent, [position, position], "-", color=colours[index], linewidth=3.0, solid_capstyle="butt"
        )
    axes[1].set_xlabel("$F_\\mathrm{peak}$ extent of the success region (N)")
    axes[1].set_ylabel("hidden state, ordered by $\\mu_d$")
    axes[1].set_title("the force axis is partitioned by dynamic friction")
    axes[1].grid(alpha=0.3, axis="x")

    mappable = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(frictions.min(), frictions.max())
    )
    figure.colorbar(mappable, ax=axes[1], label="$\\mu_d$ (N)")
    figure.suptitle("B. Where each drawer's success region sits")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_region_vs_xi(metrics: list, path: Path) -> Path:
    """C -- region descriptors against the hidden state.

    Four panels, one per hidden dimension, each showing the centroid's force and duration.
    A flat duration panel next to a steep force panel is the finding: the hidden state moves
    the region along ``F`` and not along ``T``.
    """
    solvable = [entry for entry in metrics if entry.centroid]
    dimensions = ("mass", "static_friction", "dynamic_friction", "damping")
    names = {
        "mass": "$m$ (kg)",
        "static_friction": "$\\mu_s$ (N)",
        "dynamic_friction": "$\\mu_d$ (N)",
        "damping": "$b$ (N$\\cdot$s/m)",
    }

    figure, axes = plt.subplots(2, 4, figsize=(16.0, 6.4), constrained_layout=True, sharey="row")
    for column, dimension in enumerate(dimensions):
        values = [entry.xi[dimension] for entry in solvable]
        axes[0, column].plot(values, [entry.centroid[0] for entry in solvable], "o", markersize=4, alpha=0.7)
        axes[1, column].plot(
            values, [entry.centroid[1] for entry in solvable], "o", markersize=4, alpha=0.7, color="tab:orange"
        )
        axes[1, column].set_xlabel(names[dimension])
        for row in (0, 1):
            axes[row, column].grid(alpha=0.3)
    axes[0, 0].set_ylabel("centroid $F_\\mathrm{peak}$ (N)")
    axes[1, 0].set_ylabel("centroid $T$ (s)")
    figure.suptitle(
        "C. The hidden state moves the region along $F$, not along $T$ "
        "(top row: force centroid; bottom row: duration centroid)"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_midpoint_example(dataset: SweepDataset, metrics: list, path: Path) -> Path:
    """D -- a concrete pair of succeeding parameters whose mean fails.

    Only drawn if one exists in a region the grid actually resolves. If none does, the figure
    is skipped and that absence is the result -- it is not something to go looking for at a
    resolution that would manufacture it.
    """
    candidates = [
        entry
        for entry in metrics
        if entry.resolution["sufficient_for_topology"] and entry.midpoint["examples"]
    ]
    if not candidates:
        return None
    entry = max(candidates, key=lambda item: item.midpoint["rate"])
    masks = success_mask(dataset, tuple(entry.xi.values()), MAIN_TASK.criteria)
    forces, durations = masks["forces"], masks["durations"]

    figure, axis = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    picture = np.where(masks["success"], 2.0, np.where(masks["swept"], 1.0, 0.0))
    axis.pcolormesh(forces, durations, picture, cmap="Blues", vmin=0, vmax=2, shading="nearest")

    for number, example in enumerate(entry.midpoint["examples"][:3], start=1):
        (row_a, column_a), (row_b, column_b) = example["a"], example["b"]
        row_m, column_m = example["midpoint"]
        axis.plot(
            [forces[column_a], forces[column_b]],
            [durations[row_a], durations[row_b]],
            "-o",
            color="tab:green",
            markersize=7,
            linewidth=1.5,
            label="two succeeding parameters" if number == 1 else None,
        )
        axis.plot(
            forces[column_m],
            durations[row_m],
            "X",
            color="tab:red",
            markersize=12,
            markeredgewidth=2,
            label="their mean, which fails" if number == 1 else None,
        )

    axis.set_xlabel("$F_\\mathrm{peak}$ (N)")
    axis.set_ylabel("$T$ (s)")
    axis.set_title(
        f"D. Averaging two good answers can fail\n{label(entry.xi)} -- "
        f"midpoint failure rate {entry.midpoint['rate'] * 100:.0f} % "
        f"({entry.midpoint['pairs_whose_midpoint_fails']}/{entry.midpoint['pairs_checked']} pairs)",
        fontsize=10,
    )
    axis.legend(loc="best", fontsize=9)
    axis.grid(alpha=0.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_physics_surfaces(dataset: SweepDataset, metrics: list, path: Path) -> Path:
    """E -- the two surfaces the success band is the intersection of.

    ``d(T)`` says how far the drawer went, ``v(T)`` how fast it was still moving. The band is
    where the first is within ``eps_d`` of the goal *and* the second is under ``eps_v``, and
    plotting them separately shows which condition each edge comes from.
    """
    chosen = spread_of_states(metrics, 3)
    figure, axes = plt.subplots(3, len(chosen), figsize=(4.6 * len(chosen), 10.5), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for column, entry in enumerate(chosen):
        key = tuple(entry.xi.values())
        forces, durations, displacement = dataset.surface("final_displacement", key)
        _, _, velocity = dataset.surface("final_velocity", key)
        masks = success_mask(dataset, key, MAIN_TASK.criteria)

        first = axes[0, column].pcolormesh(
            forces, durations, displacement * 1000.0, cmap="viridis", shading="nearest"
        )
        axes[0, column].contour(
            forces, durations, displacement * 1000.0,
            levels=[MAIN_TASK.goal_displacement * 1000.0], colors="white", linewidths=2.0,
        )
        figure.colorbar(first, ax=axes[0, column], label="$d(T)$ (mm)")
        axes[0, column].set_title(f"{label(entry.xi)}\nwhite line = $d_\\mathrm{{goal}}$", fontsize=8)

        second = axes[1, column].pcolormesh(
            forces, durations, np.abs(velocity), cmap="magma", shading="nearest",
            vmin=0.0, vmax=4.0 * MAIN_TASK.velocity_tolerance,
        )
        axes[1, column].contour(
            forces, durations, np.abs(velocity),
            levels=[MAIN_TASK.velocity_tolerance], colors="cyan", linewidths=2.0,
        )
        figure.colorbar(second, ax=axes[1, column], label="$|v(T)|$ (m/s)")
        axes[1, column].set_title("cyan line = $\\epsilon_v$", fontsize=8)

        axes[2, column].pcolormesh(
            forces, durations,
            np.where(masks["success"], 2.0, np.where(masks["swept"], 1.0, 0.0)),
            cmap="Blues", vmin=0, vmax=2, shading="nearest",
        )
        axes[2, column].set_title("their intersection = success", fontsize=8)
        axes[2, column].set_xlabel("$F_\\mathrm{peak}$ (N)")
        for row in range(3):
            axes[row, column].grid(alpha=0.2)
    for row, name in enumerate(("$T$ (s)", "$T$ (s)", "$T$ (s)")):
        axes[row, 0].set_ylabel(name)

    figure.suptitle("E. The success band is 'far enough' AND 'slow enough'")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_validity_map(dataset: SweepDataset, path: Path) -> Path:
    """F -- where the box is unusable, and why.

    Marginalised over hidden states, because validity turns out to depend far more on
    ``(F, T)`` than on the drawer: long executions drift and hard ones overshoot regardless
    of what is inside the cabinet.
    """
    forces, durations = dataset.forces(), dataset.durations()
    shape = (len(durations), len(forces))
    swept = np.zeros(shape)
    valid = np.zeros(shape)
    reasons: dict[str, np.ndarray] = {}

    for row in dataset.records:
        index = (durations.index(row.duration), forces.index(row.peak_force))
        swept[index] += 1
        valid[index] += int(row.valid)
        for reason in row.invalid_reasons:
            reasons.setdefault(reason, np.zeros(shape))[index] += 1

    ranked = sorted(reasons.items(), key=lambda item: -item[1].sum())[:3]
    figure, axes = plt.subplots(1, 1 + len(ranked), figsize=(4.6 * (1 + len(ranked)), 4.4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    with np.errstate(invalid="ignore"):
        fraction = np.where(swept > 0, valid / swept, np.nan)
    first = axes[0].pcolormesh(forces, durations, fraction, cmap="RdYlGn", vmin=0, vmax=1, shading="nearest")
    figure.colorbar(first, ax=axes[0], label="valid fraction")
    axes[0].set_title("validity, over all hidden states")
    axes[0].set_ylabel("$T$ (s)")

    for axis, (reason, counts) in zip(axes[1:], ranked, strict=True):
        with np.errstate(invalid="ignore"):
            share = np.where(swept > 0, counts / swept, np.nan)
        mesh = axis.pcolormesh(forces, durations, share, cmap="magma", vmin=0, vmax=1, shading="nearest")
        figure.colorbar(mesh, ax=axis, label="fraction")
        axis.set_title(reason.replace("_", " "), fontsize=9)

    for axis in axes:
        axis.set_xlabel("$F_\\mathrm{peak}$ (N)")
        axis.grid(alpha=0.2)
    figure.suptitle("F. The valid operating region in two dimensions, and what bounds it")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--analysis", type=str, default=None, help="Analysis report, for cross-checks.")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_absolute():
        path = project_root() / path
    dataset = SweepDataset.load(path)
    metrics = [analyse_landscape(dataset, key, MAIN_TASK.criteria) for key in dataset.xi_keys()]
    directory = plots_dir()
    stage = dataset.metadata.get("stage", "2d")

    written = [
        plot_success_heatmaps(dataset, metrics, directory / f"phase12_a_success_{stage}.png"),
        plot_region_overlay(dataset, metrics, directory / f"phase12_b_regions_{stage}.png"),
        plot_region_vs_xi(metrics, directory / f"phase12_c_region_vs_xi_{stage}.png"),
        plot_physics_surfaces(dataset, metrics, directory / f"phase12_e_surfaces_{stage}.png"),
        plot_validity_map(dataset, directory / f"phase12_f_validity_{stage}.png"),
    ]
    midpoint = plot_midpoint_example(dataset, metrics, directory / f"phase12_d_midpoint_{stage}.png")
    if midpoint is None:
        print("[plot] no resolvable midpoint failure to draw; figure D skipped, which is itself the result")
    else:
        written.append(midpoint)

    if args.analysis:
        report = json.loads(Path(args.analysis).read_text())
        print(f"[plot] analysis says: {report['structure']['evidence']}")
    for item in written:
        print(f"[plot] wrote {item}")


if __name__ == "__main__":
    main()
