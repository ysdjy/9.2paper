"""Stage B -- latent distillation. `tau_p -> adapter -> z_probe ~= stopgrad(z_priv)`.

RMA²'s second stage. The privileged encoder and the parameter head are taken from a trained
Stage A and frozen; only the adapter learns, and its only objective is the latent MSE. No
force loss reaches it -- that is what makes this Stage B rather than Stage C.

Stage B seed `k` distils from Stage A seed `k`, so the three runs are three independent
teacher-student pairs.

No Isaac Sim. Reads the main project's Dataset v1 and this baseline's Stage A checkpoints.

Usage::

    python baselines/rma2/scripts/train_rma2_adapter.py --dataset outputs/dataset_v1 \
        --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_drawer.dataset import DatasetStore, SplitCfg, assert_no_leakage, split_samples  # noqa: E402
from probe_drawer.experiment_plan import TRAINING_XI_RANGES  # noqa: E402
from probe_drawer.models import PspCfg  # noqa: E402
from probe_drawer.training import FeatureScaler, SampleDataset, reference_force_per_probe  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root  # noqa: E402
from rma2.config import StageACfg, StageBCfg  # noqa: E402
from rma2.model import build_stage_a, build_stage_b  # noqa: E402
from rma2.trainer import force_mae, train_stage_b  # noqa: E402
from rma2.trainer import _per_probe_predictions as per_probe_predictions  # noqa: E402


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="outputs/dataset_v1")
    parser.add_argument("--stage-a", type=str, default="baselines/rma2/checkpoints/stage_a")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="baselines/rma2/checkpoints/stage_b")
    args = parser.parse_args()

    def absolute(path: str) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else project_root() / resolved

    root, stage_a_root, run_root = absolute(args.dataset), absolute(args.stage_a), absolute(args.output)
    store = DatasetStore(root)
    stage_a_summary = json.loads((stage_a_root / "summary.json").read_text())

    split = split_samples(store.load_samples(), SplitCfg(**store.read_splits()["cfg"]))
    assert_no_leakage(split)
    channels = tuple(store.manifest["history_channels"])
    scaler = FeatureScaler.fit(split.train, channels)
    subsets = {
        name: SampleDataset(getattr(split, name), channels, scaler)
        for name in ("train", "val", "test")
    }
    targets = {name: reference_force_per_probe(subset.samples) for name, subset in subsets.items()}

    psp = PspCfg()
    force_range = tuple(stage_a_summary["force_range"])
    stage_a_cfg = StageACfg(latent_dim=stage_a_summary["cfg"]["latent_dim"])
    base = StageBCfg(device=args.device)
    if args.epochs is not None:
        base = replace(base, epochs=args.epochs)
    run_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 88)
    print(f"[stageB] dataset : {root} ({store.manifest.get('dataset_version')})")
    print(f"[stageB] stage A : {stage_a_root} (encoder + head frozen)")
    print(f"[stageB] adapter : AdaptationContextEncoder, {len(channels)} channels -> z in R^{psp.z_dim}")
    print(f"[stageB] loss    : ||z_probe - stopgrad(z_priv)||^2 only. No force loss.")
    print(f"[stageB] selected on val latent MSE; val force MAE recorded as a diagnostic only.")
    print(f"[stageB] cfg     : {base.epochs} epochs, Adam lr {base.learning_rate} (RMA2's own)")
    print(f"[stageB] seeds   : {list(args.seeds)}  (Stage B seed k distils Stage A seed k)")

    results = []
    for seed in args.seeds:
        stage_a = build_stage_a(stage_a_cfg, force_range, TRAINING_XI_RANGES.as_dict())
        weights = torch.load(stage_a_root / f"seed{seed}" / "best.pt", weights_only=True)
        stage_a.load_state_dict(weights["state_dict"])

        model = build_stage_b(stage_a, len(channels), psp)
        cfg = replace(base, seed=seed)
        trained = train_stage_b(model, subsets["train"], subsets["val"], targets, cfg)
        best = trained.restore_best()

        first, last = trained.history[0], trained.history[trained.best_epoch + 1]
        row = {
            "seed": seed,
            "best_epoch": trained.best_epoch,
            "val_latent_mse_before": first["val_latent_mse"],
            "val_latent_mse_best": trained.best_val_latent_mse,
            "val_force_mae_at_best": last["val_force_mae"],
            "frozen_parameter_groups": sorted({n.split(".")[0] for n in trained.frozen_parameters}),
            "trainable_parameters": sum(
                p.numel() for n, p in best.named_parameters() if n.startswith("adapter.")
            ),
        }
        for name in ("val", "test"):
            predictions = per_probe_predictions(best, subsets[name], cfg.device)
            row[f"{name}_force_mae"] = force_mae(predictions, targets[name])
        results.append(row)

        seed_root = run_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best.state_dict()}, seed_root / "best.pt")
        (seed_root / "config.json").write_text(
            json.dumps(
                {
                    "stage_b": cfg.as_dict(),
                    "stage_a": stage_a_cfg.as_dict(),
                    "psp": psp.as_dict(),
                    "force_range": list(force_range),
                    "frozen": trained.frozen_parameters,
                    "best_epoch": trained.best_epoch,
                },
                indent=2,
            )
        )
        (seed_root / "history.csv").write_text(
            "epoch,train_latent_mse,val_latent_mse,val_force_mae\n"
            + "\n".join(
                f"{r['epoch']},{r['train_latent_mse']:.8f},{r['val_latent_mse']:.8f},{r['val_force_mae']:.6f}"
                for r in trained.history
            )
        )
        print(
            f"[stageB] seed {seed}: latent MSE {first['val_latent_mse']:.5f} -> "
            f"{trained.best_val_latent_mse:.5f} (epoch {trained.best_epoch}); "
            f"test force MAE {row['test_force_mae']:.4f} N"
        )

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "dataset": str(root),
        "stage_a": str(stage_a_root),
        "channels": list(channels),
        "scaler": scaler.as_dict(),
        "force_range": list(force_range),
        "cfg": base.as_dict(),
        "psp": psp.as_dict(),
        "seeds": list(args.seeds),
        "per_seed": results,
        "test_force_mae": {
            "mean": float(np.mean([r["test_force_mae"] for r in results])),
            "sd": float(np.std([r["test_force_mae"] for r in results])),
        },
        "val_latent_mse": {
            "before": float(np.mean([r["val_latent_mse_before"] for r in results])),
            "after": float(np.mean([r["val_latent_mse_best"] for r in results])),
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[stageB]")
    latent = summary["val_latent_mse"]
    reduction = 1.0 - latent["after"] / latent["before"] if latent["before"] else float("nan")
    print(f"[stageB] latent MSE    : {latent['before']:.5f} -> {latent['after']:.5f} "
          f"({reduction * 100:.1f} % reduction)")
    mae = summary["test_force_mae"]
    print(f"[stageB] test force MAE: {mae['mean']:.4f} +- {mae['sd']:.4f} N "
          f"(Stage A was {stage_a_summary['test_force_mae']['mean']:.4f})")
    print(f"[stageB] written       : {run_root}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
