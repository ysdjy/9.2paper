"""Phase 11 figures -- the dataset, the models, and the one that carries the argument.

Nine figures, no simulator. All from a dataset directory and a training run directory.

======  ==================================================================  ==============
Figure  What it shows                                                       Needs
======  ==================================================================  ==============
A       Hidden-state sampling coverage                                      dataset
B       Probe sequence-length distribution                                  dataset
C       Candidate force against success rate                                dataset
D       Success balance across train / val / test                           dataset + split
E       Privileged teacher: reliability, and example landscapes             run
F       ACE + PSP: example landscapes, against the teacher's                run
G       Predicted force against the reference force                         run
H       Closed-loop physical success by method                              closed_loop.json
I       Scalar-baseline residual against the success-band width             dataset + run
======  ==================================================================  ==============

Figure I is the point of the phase. Phase 10 measured |rho| = 0.91 between one scalar probe
feature and the force a drawer needs, which sounds like a solved problem. I puts that
correlation's *residual* next to the width of the window a force has to land in. If the
residual is wider than the window, a strong correlation still selects the wrong force, and
that is precisely the gap a learned landscape has to close.

Usage::

    python scripts/plot_phase11.py --dataset outputs/dataset_v0 --run outputs/training/run_XXXX
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples  # noqa: E402
from probe_drawer.dataset.schema import XI_DIMENSIONS  # noqa: E402
from probe_drawer.experiment_plan import MAIN_TASK  # noqa: E402
from probe_drawer.models.baselines import STRONGEST_FEATURE, FeatureRegression  # noqa: E402
from probe_drawer.training import reference_force_per_probe  # noqa: E402
from probe_drawer.utils import project_root  # noqa: E402

XI_LABELS = {
    "mass": "mass $m$ (kg)",
    "static_friction": "static friction $\\mu_s$ (N)",
    "dynamic_friction": "dynamic friction $\\mu_d$ (N)",
    "damping": "damping $b$ (N$\\cdot$s/m)",
}

#: Median success-band width the Phase 10 Oracle measured at the selected task.
#:
#: The reference figure I compares a predictor's error against: a force has to land inside a
#: window this wide, so an error of the same size is a coin flip
#: (``docs/ORACLE_LANDSCAPE.md``).
SUCCESS_BAND_WIDTH = 0.20


def plots_dir() -> Path:
    directory = project_root() / "outputs" / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def plot_xi_coverage(store: DatasetStore, path: Path) -> Path:
    """A -- did the Sobol draw actually cover the box?

    Pairwise projections rather than marginals: a sequence can have perfect marginals and
    still leave holes in the interior, and the interior is where the drawers live.
    """
    values = np.array([[row["xi"][name] for name in XI_DIMENSIONS] for row in store.hidden_states])
    pairs = [(0, 1), (0, 3), (1, 2), (2, 3)]

    figure, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.8), constrained_layout=True)
    for axis, (first, second) in zip(axes, pairs, strict=True):
        axis.plot(values[:, first], values[:, second], ".", markersize=3.5, alpha=0.6)
        axis.set_xlabel(XI_LABELS[XI_DIMENSIONS[first]])
        axis.set_ylabel(XI_LABELS[XI_DIMENSIONS[second]])
        axis.grid(alpha=0.3)
    figure.suptitle(
        f"A. Hidden-state coverage, {len(values)} scrambled-Sobol draws "
        f"($\\mu_d = \\mathrm{{ratio}} \\times \\mu_s$, so $\\mu_d \\leq \\mu_s$ by construction)"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_sequence_lengths(store: DatasetStore, path: Path) -> Path:
    """B -- why the loader pads dynamically.

    A probe stops when the drawer has moved 3 mm, so the recording is as long as that drawer
    needs. The spread is the whole reason histories are stored ragged.
    """
    lengths = np.array(store.sequence_lengths())
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    axis.hist(lengths, bins=range(int(lengths.min()), int(lengths.max()) + 2), color="tab:blue", alpha=0.8)
    axis.axvline(lengths.mean(), color="tab:red", linestyle=":", linewidth=1.8, label=f"mean {lengths.mean():.1f}")
    axis.set_xlabel("probe history length (control steps at 60 Hz)")
    axis.set_ylabel("probes")
    axis.set_title(
        f"B. Probe length varies with the drawer: {lengths.min()}-{lengths.max()} steps "
        f"({lengths.min() / 60:.2f}-{lengths.max() / 60:.2f} s)"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_force_success(store: DatasetStore, path: Path) -> Path:
    """C -- where in the force range the positives are, and how many per probe."""
    rows = store.candidates
    forces = np.array([row["candidate_peak_force"] for row in rows])
    success = np.array([bool(row["success"]) for row in rows])
    per_probe = defaultdict(int)
    for row in rows:
        per_probe[row["probe_id"]] += int(bool(row["success"]))
    counts = np.array(list(per_probe.values()))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    low, high = MAIN_TASK.peak_force_range
    edges = np.linspace(low, high, 25)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rates = [
        success[(forces >= a) & (forces < b)].mean() if ((forces >= a) & (forces < b)).any() else 0.0
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]
    axes[0].bar(centres, np.array(rates) * 100, width=(high - low) / 26, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("candidate $F_\\mathrm{peak}$ (N)")
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_title(f"success against force ({success.mean() * 100:.1f} % overall)")
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].hist(counts, bins=range(int(counts.max()) + 2), color="tab:blue", alpha=0.8)
    axes[1].set_xlabel("succeeding candidates per probe")
    axes[1].set_ylabel("probes")
    share = (counts >= 1).mean() * 100
    axes[1].set_title(f"{share:.1f} % of probes have at least one")
    axes[1].grid(alpha=0.3, axis="y")

    figure.suptitle("C. Candidate forces and their labels")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_split_balance(store: DatasetStore, path: Path) -> Path:
    """D -- the splits must be comparable, or a difference between them is the split."""
    cfg = SplitCfg(**store.read_splits()["cfg"])
    split = split_samples(store.load_samples(), cfg)
    subsets = {"train": split.train, "val": split.val, "test": split.test}

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    names = list(subsets)
    axes[0].bar(names, [len(subsets[name]) for name in names], color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("candidate rows")
    axes[0].set_title(f"rows (split on {cfg.level})")

    axes[1].bar(
        names,
        [len({sample.xi_id for sample in subsets[name]}) for name in names],
        color="tab:green",
        alpha=0.8,
    )
    axes[1].set_ylabel("hidden states")
    axes[1].set_title("groups -- disjoint by construction")

    rates = [np.mean([sample.success for sample in subsets[name]]) * 100 if subsets[name] else 0.0 for name in names]
    axes[2].bar(names, rates, color="tab:orange", alpha=0.8)
    axes[2].set_ylabel("positive rate (%)")
    axes[2].set_title("label balance")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
    figure.suptitle("D. Split composition")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_scalar_residual(store: DatasetStore, path: Path) -> Path:
    """I -- why 0.91 is not enough.

    Left: the scalar feature against the force each probe needs, with the fit. Right: the
    fit's residual, against the width of the window a force must land in. When the residual
    distribution is wider than the band, a strong correlation still misses.
    """
    cfg = SplitCfg(**store.read_splits()["cfg"])
    split = split_samples(store.load_samples(), cfg)
    targets = reference_force_per_probe(split.train)
    model = FeatureRegression(features=(STRONGEST_FEATURE,)).fit(
        split.train, [targets[sample.probe_id] for sample in split.train]
    )

    test_targets = reference_force_per_probe(split.test)
    seen: dict[str, tuple[float, float]] = {}
    for sample in split.test:
        if sample.probe_id not in seen:
            seen[sample.probe_id] = (
                float(sample.probe_summary[STRONGEST_FEATURE]),
                test_targets[sample.probe_id],
            )
    feature = np.array([value for value, _ in seen.values()])
    required = np.array([value for _, value in seen.values()])
    predicted = model.predict([_Row({STRONGEST_FEATURE: value}) for value in feature])
    residual = predicted - required

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    axes[0].plot(feature, required, "o", markersize=4, alpha=0.5, label="test probes")
    order = np.argsort(feature)
    axes[0].plot(feature[order], predicted[order], "-", color="tab:red", linewidth=2, label="linear fit")
    axes[0].set_xlabel(STRONGEST_FEATURE.replace("_", " "))
    axes[0].set_ylabel("required $F_\\mathrm{peak}$ (N)")
    axes[0].set_title(f"the scalar predictor (Spearman |rho| = 0.91 in Phase 10)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].hist(residual, bins=40, color="tab:blue", alpha=0.75, label="prediction error")
    axes[1].axvspan(
        -SUCCESS_BAND_WIDTH / 2,
        SUCCESS_BAND_WIDTH / 2,
        color="tab:green",
        alpha=0.25,
        label=f"success band ({SUCCESS_BAND_WIDTH:.2f} N wide)",
    )
    inside = float(np.mean(np.abs(residual) <= SUCCESS_BAND_WIDTH / 2)) * 100
    axes[1].set_xlabel("predicted $-$ required force (N)")
    axes[1].set_ylabel("test probes")
    axes[1].set_title(
        f"only {inside:.0f} % of errors fit inside the band "
        f"(MAE {np.abs(residual).mean():.3f} N)"
    )
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.suptitle("I. A strong correlation is not a solved task")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


class _Row:
    def __init__(self, summary: dict) -> None:
        self.probe_summary = summary


def plot_run_figures(run: Path, directory: Path) -> list[Path]:
    """E, F, G -- reliability, example landscapes, and predicted against reference force."""
    comparison = json.loads((run / "comparison.json").read_text())
    written = []

    # E/F -- reliability curves, from whatever the run recorded.
    figure, axis = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.0, label="perfect calibration")
    plotted = False
    for name in ("teacher (privileged)", "ACE + PSP"):
        rows = [row for row in comparison["results"].get(name, []) if row["split"] == "test"]
        if not rows or "curve" not in rows[0]:
            continue
        curve = [point for point in rows[0]["curve"] if point.get("count")]
        axis.plot(
            [point["confidence"] for point in curve],
            [point["accuracy"] for point in curve],
            "o-",
            label=name,
        )
        plotted = True
    axis.set_xlabel("predicted P(success)")
    axis.set_ylabel("observed success rate")
    axis.set_title("E/F. Reliability on the test split")
    axis.grid(alpha=0.3)
    axis.legend()
    path = directory / "phase11_ef_reliability.png"
    if plotted:
        figure.savefig(path, dpi=150)
        written.append(path)
    plt.close(figure)

    # G -- the summary table as a bar chart, which is what the run actually stores.
    summary = comparison["summary"]
    methods = [name for name in summary if "test" in summary[name]]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    positions = np.arange(len(methods))
    for axis, key, label in (
        (axes[0], "selected_success_rate_feasible_only", "selected force succeeds (feasible probes, %)"),
        (axes[1], "force_mae", "force MAE (N)"),
    ):
        values = [summary[name]["test"].get(key, {}).get("mean", np.nan) for name in methods]
        errors = [summary[name]["test"].get(key, {}).get("std", 0.0) for name in methods]
        scale = 100.0 if "rate" in key else 1.0
        axis.barh(positions, np.array(values) * scale, xerr=np.array(errors) * scale, alpha=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(methods, fontsize=9)
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(alpha=0.3, axis="x")
    figure.suptitle("G. Offline comparison on the test split (mean +- sd over seeds)")
    path = directory / "phase11_g_offline_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    return written


def plot_closed_loop(payload: dict, path: Path) -> Path:
    """H -- the number the paper reports: physical success on unseen drawers."""
    methods = sorted(payload["methods"], key=lambda name: -payload["methods"][name]["success_rate"])
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    positions = np.arange(len(methods))
    axes[0].barh(
        positions,
        [payload["methods"][name]["success_rate"] * 100 for name in methods],
        alpha=0.85,
        color="tab:blue",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(methods, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("physical task success (%)")
    axes[0].set_title(f"{payload['num_test_states']} unseen hidden states")
    axes[0].grid(alpha=0.3, axis="x")

    for name in methods:
        rows = [row for row in payload["rows"] if row["method"] == name]
        errors = np.abs([row["total_displacement"] - payload["task"]["goal_displacement"] for row in rows]) * 1000
        axes[1].plot(np.sort(errors), np.linspace(0, 1, len(errors)), label=name, linewidth=1.8)
    axes[1].axvline(
        payload["task"]["displacement_tolerance"] * 1000,
        color="grey",
        linestyle="--",
        label="$\\epsilon_d$",
    )
    axes[1].set_xlabel("$|d_\\mathrm{total}(T) - d_\\mathrm{goal}|$ (mm)")
    axes[1].set_ylabel("cumulative fraction of drawers")
    axes[1].set_xlim(0, 60)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    figure.suptitle("H. Closed-loop deployment: probe, choose a force, pull")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = project_root() / root
    store = DatasetStore(root)
    directory = plots_dir()

    written = [
        plot_xi_coverage(store, directory / "phase11_a_xi_coverage.png"),
        plot_sequence_lengths(store, directory / "phase11_b_sequence_lengths.png"),
        plot_force_success(store, directory / "phase11_c_force_success.png"),
        plot_split_balance(store, directory / "phase11_d_split_balance.png"),
        plot_scalar_residual(store, directory / "phase11_i_scalar_residual.png"),
    ]

    if args.run:
        run = Path(args.run)
        if not run.is_absolute():
            run = project_root() / run
        written += plot_run_figures(run, directory)
        closed_loop = run / "closed_loop.json"
        if closed_loop.exists():
            written.append(
                plot_closed_loop(json.loads(closed_loop.read_text()), directory / "phase11_h_closed_loop.png")
            )
        else:
            print(f"[plot] no closed_loop.json in {run}; figure H skipped")

    for path in written:
        print(f"[plot] wrote {path}")


if __name__ == "__main__":
    main()
