"""Phase 11H/11I/11J/11K -- train the baselines, the teacher and the student, and compare.

No simulator. Reads a dataset, takes the grouped split it already carries, and runs, in this
order and for a reason:

1. **A fixed force.** The floor. A learned model that cannot beat one force for every drawer
   is not adapting at all.
2. **Baselines A-D.** Linear on one scalar probe feature, ridge on all of them, an MLP on
   them, and a GRU on the raw history. ``D`` sees exactly what ACE sees and predicts a force
   directly, so a gap between ``D`` and ACE + PSP is attributable to modelling the landscape
   rather than to having more input.
3. **The privileged teacher.** Told ``xi``. This is a *gate*: if a model given the four
   hidden values cannot predict the success landscape, the landscape is not learnable from a
   probe either and the student's numbers would mean nothing.
4. **The student (ACE + PSP).** The deployable one, optionally distilling the teacher's
   landscape.
5. **An input ablation** over how much of the probe the encoder sees.

Every model is scored the same way: pick one force per probe, and ask whether that force
actually succeeded. Candidate-level AUROC is reported too, but it is not the task.

Usage::

    python scripts/train_models.py --dataset outputs/dataset_v0
    python scripts/train_models.py --dataset outputs/dataset_v0 --seeds 0 1 2 --ablation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from probe_drawer.dataset import DatasetStore, SplitCfg, assert_no_leakage, split_samples
from probe_drawer.models import PspCfg
from probe_drawer.models.baselines import (
    STRONGEST_FEATURE,
    FeatureRegression,
    FixedForceBaseline,
    GruForceRegressor,
    MlpForceRegressor,
    summary_matrix,
)
from probe_drawer.training import (
    FeatureScaler,
    SampleDataset,
    TrainCfg,
    evaluate,
    make_loader,
    reference_force_per_probe,
    save_run,
    selection_metrics,
    train_student,
    train_teacher,
)
from probe_drawer.utils import git_commit, project_root

#: Probe summary features baselines B and C may use.
#:
#: Every one is a scalar computed from the probe recording, so they are all deployable. The
#: strongest is listed first for readability; baseline A uses that one alone.
SUMMARY_FEATURES = (
    STRONGEST_FEATURE,
    "duration",
    "final_commanded_force",
    "breakaway_force",
    "breakaway_time",
    "final_displacement",
    "final_velocity",
    "mean_speed_after_breakaway",
    "peak_acceleration",
)

#: Encoder input ablation: how much of the probe does the model actually need?
#:
#: ``ACE-2`` against ``ACE-4`` is the comparison that answers the question -- position and
#: velocity against the full deployable set. The intermediate rungs show where the value
#: appears. Wrist force is deliberately absent (D018); it is recorded in the dataset as a
#: diagnostic and can be added later without regenerating anything.
ABLATIONS = {
    "ACE-1 (F,d)": ("commanded_force", "drawer_position"),
    "ACE-2 (F,d,v)": ("commanded_force", "drawer_position", "drawer_velocity"),
    "ACE-3 (F,d,v,a)": (
        "commanded_force",
        "drawer_position",
        "drawer_velocity",
        "drawer_acceleration",
    ),
    "ACE-4 (7 channels)": (
        "commanded_force",
        "drawer_position",
        "drawer_velocity",
        "drawer_acceleration",
        "tcp_pull_axis_position",
        "tcp_pull_axis_velocity",
        "tcp_pull_axis_acceleration",
    ),
}


def _label(sample, name: str) -> float:
    """One row's label under the configured name, refusing an absent one (D046)."""
    value = getattr(sample, name)
    if value is None:
        raise ValueError(f"this dataset does not record {name!r}; pass --label success.")
    return float(value)


def force_selection_score(
    samples: list, predicted_force: np.ndarray, reference: dict[str, float], label: str
) -> dict:
    """Score a model that outputs a *force* rather than a success probability.

    A force regressor names one force per probe, which is generally not a candidate that was
    executed. So it is scored by proximity: among that probe's candidates, the one closest to
    the prediction is the one it effectively chose, and its recorded label is the outcome.
    That is exactly what a deployed system would get if it were restricted to the candidates
    the dataset can answer for.
    """
    per_probe: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for sample, prediction in zip(samples, predicted_force, strict=True):
        per_probe[sample.probe_id].append(
            (sample.candidate_peak_force, _label(sample, label), float(prediction))
        )

    probes, forces, labels, scores = [], [], [], []
    for probe, rows in per_probe.items():
        # One prediction per probe: a force regressor does not vary with the candidate, and
        # averaging guards against tiny numerical differences between its rows.
        predicted = float(np.mean([row[2] for row in rows]))
        for force, outcome, _ in rows:
            probes.append(probe)
            forces.append(force)
            labels.append(outcome)
            # Closest candidate wins, so "score" is negative distance.
            scores.append(-abs(force - predicted))

    metrics = selection_metrics(probes, forces, labels, scores, reference)
    metrics.pop("selected_force", None)
    predicted_by_probe = {probe: float(np.mean([row[2] for row in rows])) for probe, rows in per_probe.items()}
    errors = [
        abs(predicted_by_probe[probe] - reference[probe]) for probe in predicted_by_probe if probe in reference
    ]
    metrics["predicted_force_mae"] = float(np.mean(errors)) if errors else float("nan")
    return metrics


def train_force_regressor(
    model: torch.nn.Module,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    targets: dict[str, dict[str, float]],
    epochs: int,
    seed: int,
    features: tuple[str, ...] | None = None,
) -> torch.nn.Module:
    """Fit a neural force regressor (baseline C or D) by mean squared error.

    Targets are the per-probe reference force -- the candidate whose displacement landed
    closest to the goal -- which is defined for every probe including the ones no candidate
    solved, so the hardest cases are not silently dropped.
    """
    torch.manual_seed(seed)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loader = make_loader(train_dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_state, best_error = None, float("inf")
    for _ in range(epochs):
        model.train()
        for batch in loader:
            # The loader shuffles, so each row's target is recovered through its probe id
            # rather than by position.
            target = torch.tensor(
                [targets["train"][probe] for probe in batch.probe_ids], dtype=torch.float32
            )
            optimiser.zero_grad(set_to_none=True)
            prediction = model(batch) if features is None else model(_feature_tensor(batch, features))
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            predictions = _predict_force(model, val_dataset, features)
        errors = [
            abs(float(np.mean(predictions[start:stop])) - targets["val"][probe])
            for probe, (start, stop) in _probe_spans(val_dataset).items()
            if probe in targets["val"]
        ]
        error = float(np.mean(errors)) if errors else float("inf")
        if error < best_error:
            best_error, best_state = error, {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _probe_spans(dataset: SampleDataset) -> dict[str, tuple[int, int]]:
    """``probe_id -> (start, stop)`` over the dataset's sample order."""
    spans: dict[str, tuple[int, int]] = {}
    for index, sample in enumerate(dataset.samples):
        start, stop = spans.get(sample.probe_id, (index, index))
        spans[sample.probe_id] = (min(start, index), max(stop, index) + 1)
    return spans


#: Probe summary features, looked up by probe id.
#:
#: Populated once per run from the dataset, because the batch carries only identifiers -- the
#: summary is a per-probe scalar set, not a per-row one, so putting it in every batch would
#: duplicate it 32 times over.
SUMMARY_BY_PROBE: dict[str, dict] = {}


def _feature_tensor(batch, features: tuple[str, ...]) -> torch.Tensor:
    rows = [[float(SUMMARY_BY_PROBE[probe][name]) for name in features] for probe in batch.probe_ids]
    return torch.tensor(rows, dtype=torch.float32)


def _predict_force(model, dataset: SampleDataset, features: tuple[str, ...] | None) -> np.ndarray:
    loader = make_loader(dataset, batch_size=512, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values = model(batch) if features is None else model(_feature_tensor(batch, features))
            out.append(values.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--baseline-epochs", type=int, default=40)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--ablation", action="store_true", help="Run the encoder input ablation.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        choices=("reach_success", "success"),
        help=(
            "Which label to train and score on. Defaults to 'reach_success' for a Setting V1 "
            "dataset and 'success' for Dataset v0, which predates the split (D046)."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = project_root() / root
    store = DatasetStore(root)

    split_payload = store.read_splits()
    cfg = SplitCfg(**{key: value for key, value in split_payload["cfg"].items()})
    samples = store.load_samples()

    # Default from the data rather than from a constant: a Dataset v0 row has no
    # ``reach_success`` to train on, and picking the label here means the failure is a clear
    # message at startup rather than a nan several epochs in (D046).
    label_field = args.label or ("success" if samples[0].reach_success is None else "reach_success")
    print(f"[train] label   : {label_field}")

    split = split_samples(samples, cfg)
    assert_no_leakage(split)

    channels = tuple(store.manifest["history_channels"])
    scaler = FeatureScaler.fit(split.train, channels)
    subsets = {
        "train": SampleDataset(split.train, channels, scaler),
        "val": SampleDataset(split.val, channels, scaler),
        "test": SampleDataset(split.test, channels, scaler),
    }
    targets = {name: reference_force_per_probe(subset.samples) for name, subset in subsets.items()}
    SUMMARY_BY_PROBE.update({probe["probe_id"]: probe["summary"] for probe in store.probes})

    run_root = Path(args.output) if args.output else project_root() / "outputs" / "training" / (
        f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"[train] dataset : {root} ({store.manifest.get('dataset_version')})")
    print(
        f"[train] split   : {cfg.level}, "
        + ", ".join(f"{name} {len(subset)} rows" for name, subset in subsets.items())
        + f" (dropped {sum(subset.dropped for subset in subsets.values())} invalid)"
    )
    print(f"[train] channels: {len(channels)} -> {list(channels)}")
    print(f"[train] seeds   : {list(args.seeds)}")
    print(f"[train] run dir : {run_root}")

    results: dict[str, list[dict]] = defaultdict(list)

    # 1 -- the floor.
    fixed = FixedForceBaseline().fit(
        [sample.candidate_peak_force for sample in subsets["train"].samples],
        [_label(sample, label_field) for sample in subsets["train"].samples],
    )
    for name in ("val", "test"):
        predicted = fixed.predict(len(subsets[name].samples))
        results["fixed force"].append(
            {"split": name, "seed": None, "force": fixed.force, **force_selection_score(subsets[name].samples, predicted, targets[name], label_field)}
        )
    print(f"[train] fixed force baseline: {fixed.force:.2f} N")

    # 2 -- closed-form baselines, which have no seed.
    for label, features, alpha in (
        ("A linear (1 feature)", (STRONGEST_FEATURE,), 0.0),
        ("B ridge (summary)", SUMMARY_FEATURES, 10.0),
    ):
        model = FeatureRegression(features=features, alpha=alpha).fit(
            subsets["train"].samples, [targets["train"][s.probe_id] for s in subsets["train"].samples]
        )
        for name in ("val", "test"):
            predicted = model.predict(subsets[name].samples)
            results[label].append(
                {"split": name, "seed": None, **force_selection_score(subsets[name].samples, predicted, targets[name], label_field)}
            )

    # 3-5 -- everything with a seed.
    for seed in args.seeds:
        psp = PspCfg()

        mlp = MlpForceRegressor(len(SUMMARY_FEATURES))
        _fit_summary_mlp(mlp, subsets, targets, SUMMARY_FEATURES, args.baseline_epochs, seed)
        for name in ("val", "test"):
            predicted = _predict_summary_mlp(mlp, subsets[name], SUMMARY_FEATURES)
            results["C MLP (summary)"].append(
                {"split": name, "seed": seed, **force_selection_score(subsets[name].samples, predicted, targets[name], label_field)}
            )

        gru = GruForceRegressor(len(channels), psp)
        train_force_regressor(gru, subsets["train"], subsets["val"], targets, args.baseline_epochs, seed)
        # Checkpointed like the neural models, so the closed-loop evaluation can deploy the
        # same weights rather than refitting and getting a different baseline.
        gru_dir = run_root / f"gru_seed{seed}"
        gru_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": gru.state_dict(), "psp": psp.as_dict()}, gru_dir / "best.pt")
        for name in ("val", "test"):
            predicted = _predict_force(gru, subsets[name], None)
            results["D GRU (history)"].append(
                {"split": name, "seed": seed, **force_selection_score(subsets[name].samples, predicted, targets[name], label_field)}
            )

        teacher = train_teacher(
            subsets["train"],
            subsets["val"],
            TrainCfg(epochs=args.epochs, seed=seed, device=args.device, label=label_field),
            psp,
        )
        for name in ("val", "test"):
            results["teacher (privileged)"].append(
                {"split": name, "seed": seed, **evaluate(teacher.restore_best(), subsets[name], teacher.cfg)}
            )
        save_run(run_root / f"teacher_seed{seed}", teacher, {"val": results["teacher (privileged)"][-2], "test": results["teacher (privileged)"][-1]}, {"psp": psp.as_dict()})

        student_cfg = TrainCfg(
            epochs=args.epochs,
            seed=seed,
            device=args.device,
            distillation_weight=args.distillation_weight,
            latent_weight=0.0,
            label=label_field,
        )
        student = train_student(
            subsets["train"], subsets["val"], student_cfg, len(channels), teacher.model, psp
        )
        for name in ("val", "test"):
            results["ACE + PSP"].append(
                {"split": name, "seed": seed, **evaluate(student.restore_best(), subsets[name], student_cfg)}
            )
        save_run(run_root / f"student_seed{seed}", student, {"val": results["ACE + PSP"][-2], "test": results["ACE + PSP"][-1]}, {"psp": psp.as_dict()})
        print(f"[train] seed {seed} done")

        if args.ablation:
            for label, subset_channels in ABLATIONS.items():
                ablation_scaler = FeatureScaler.fit(split.train, subset_channels)
                ablation_subsets = {
                    name: SampleDataset(getattr(split, name), subset_channels, ablation_scaler)
                    for name in ("train", "val", "test")
                }
                trained = train_student(
                    ablation_subsets["train"],
                    ablation_subsets["val"],
                    student_cfg,
                    len(subset_channels),
                    teacher.model,
                    psp,
                )
                results[f"ablation {label}"].append(
                    {
                        "split": "test",
                        "seed": seed,
                        "channels": len(subset_channels),
                        **evaluate(trained.restore_best(), ablation_subsets["test"], student_cfg),
                    }
                )

    summary = _summarise(results)
    payload = {
        "dataset": store.describe(),
        "dataset_version": store.manifest.get("dataset_version"),
        "git_commit": git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "split": {**split_payload["cfg"], "counts": split.counts()},
        "channels": list(channels),
        "seeds": list(args.seeds),
        "scaler": scaler.as_dict(),
        "summary_features": list(SUMMARY_FEATURES),
        "results": {name: rows for name, rows in results.items()},
        "summary": summary,
    }
    (run_root / "comparison.json").write_text(json.dumps(payload, indent=2, default=float))
    _print(summary, run_root)


def _fit_summary_mlp(model, subsets, targets, features, epochs, seed) -> None:
    """Full-batch fit: the summary features are one small matrix, so minibatching adds only
    noise and a shuffle to get wrong."""
    torch.manual_seed(seed)
    values = torch.tensor(summary_matrix(subsets["train"].samples, features), dtype=torch.float32)
    model.fit_scaler(values)
    target = torch.tensor(
        [targets["train"][s.probe_id] for s in subsets["train"].samples], dtype=torch.float32
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(values), target)
        loss.backward()
        optimiser.step()


def _predict_summary_mlp(model, dataset, features) -> np.ndarray:
    values = torch.tensor(summary_matrix(dataset.samples, features), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(values).numpy()


def _summarise(results: dict[str, list[dict]]) -> dict:
    """Mean and spread across seeds, per model and split."""
    summary: dict[str, dict] = {}
    for name, rows in results.items():
        summary[name] = {}
        for split in ("val", "test"):
            selected = [row for row in rows if row["split"] == split]
            if not selected:
                continue
            summary[name][split] = {
                "seeds": len(selected),
                **{
                    key: {
                        "mean": float(np.nanmean([row[key] for row in selected])),
                        "std": float(np.nanstd([row[key] for row in selected])),
                    }
                    for key in (
                        "selected_success_rate",
                        "selected_success_rate_feasible_only",
                        "force_mae",
                    )
                    if key in selected[0]
                },
                **{
                    key: {
                        "mean": float(np.nanmean([row[key] for row in selected if key in row])),
                    }
                    for key in ("auroc", "auprc", "brier", "ece")
                    if key in selected[0]
                },
            }
    return summary


def _print(summary: dict, run_root: Path) -> None:
    print("[train]")
    print(
        f"[train] {'model':>22} {'split':>5} {'sel. success':>14} {'feasible only':>15} "
        f"{'force MAE':>10} {'AUROC':>7}"
    )
    for name, splits in summary.items():
        for split, values in splits.items():
            selected = values.get("selected_success_rate", {})
            feasible = values.get("selected_success_rate_feasible_only", {})
            mae = values.get("force_mae", {})
            auroc = values.get("auroc", {}).get("mean", float("nan"))
            print(
                f"[train] {name:>22} {split:>5} "
                f"{selected.get('mean', float('nan')) * 100:8.2f} +-{selected.get('std', 0) * 100:4.1f} % "
                f"{feasible.get('mean', float('nan')) * 100:9.2f} +-{feasible.get('std', 0) * 100:4.1f} % "
                f"{mae.get('mean', float('nan')):10.3f} {auroc:7.3f}"
            )
    print("[train]")
    print(f"[train] comparison written: {run_root / 'comparison.json'}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
