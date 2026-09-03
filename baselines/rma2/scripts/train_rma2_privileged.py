"""Stage A -- privileged direct adaptation. `xi -> z_priv -> F_peak*`.

The RMA²-style baseline's ceiling, and the missing cell in the paper's comparison: a
*privileged* model with a *point* output. Against the main project's numbers it splits the
ACE + PSP over Direct GRU gap into its two attributable halves --

    Direct GRU  --(+ privileged input)-->  Stage A  --(+ landscape & search)-->  teacher
       point, probe                     point, xi                        landscape, xi

-- so the question it answers is: knowing the hidden state exactly, what can a point regressor
do at all?

No Isaac Sim. Reads the main project's Dataset v1 and its split; writes nothing into it.

Usage::

    python baselines/rma2/scripts/train_rma2_privileged.py --dataset outputs/dataset_v1 \
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
from probe_drawer.experiment_plan import SETTING_V1_TASK, TRAINING_XI_RANGES  # noqa: E402
from probe_drawer.training import FeatureScaler, SampleDataset, reference_force_per_probe  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, git_commit, project_root  # noqa: E402
from rma2.config import StageACfg  # noqa: E402
from rma2.model import build_stage_a  # noqa: E402
from rma2.trainer import force_mae, train_stage_a  # noqa: E402
from rma2.trainer import _per_probe_predictions as per_probe_predictions  # noqa: E402


def main() -> None:
    enable_unbuffered_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="outputs/dataset_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=None, help="Overrides StageACfg.epochs.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="baselines/rma2/checkpoints/stage_a")
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.is_absolute():
        root = project_root() / root
    store = DatasetStore(root)

    split_cfg = SplitCfg(**store.read_splits()["cfg"])
    samples = store.load_samples()
    split = split_samples(samples, split_cfg)
    assert_no_leakage(split)

    # Identical to the main project's pipeline, so the only difference from Direct GRU is the
    # model: same channels, same scaler fitted on train only, same invalid-row dropping, and
    # the same target dictionary computed on what survives.
    channels = tuple(store.manifest["history_channels"])
    scaler = FeatureScaler.fit(split.train, channels)
    subsets = {
        name: SampleDataset(getattr(split, name), channels, scaler)
        for name in ("train", "val", "test")
    }
    targets = {name: reference_force_per_probe(subset.samples) for name, subset in subsets.items()}

    force_range = SETTING_V1_TASK.peak_force_range
    run_root = Path(args.output)
    if not run_root.is_absolute():
        run_root = project_root() / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 84)
    print(f"[stageA] dataset : {root} ({store.manifest.get('dataset_version')})")
    print(f"[stageA] split   : train {len(subsets['train'])} rows, val {len(subsets['val'])}, "
          f"test {len(subsets['test'])} (dropped {subsets['train'].dropped} invalid in train)")
    print(f"[stageA] target  : reference_force_per_probe, shared with Direct GRU / ridge / linear")
    print(f"[stageA] output  : F_peak squashed into {force_range} N")
    print(f"[stageA] seeds   : {list(args.seeds)}")

    base = StageACfg(device=args.device)
    if args.epochs is not None:
        base = replace(base, epochs=args.epochs)
    print(f"[stageA] cfg     : latent {base.latent_dim}, encoder {list(base.encoder_units)}, "
          f"head {list(base.head_units)}, {base.epochs} epochs, lr {base.learning_rate}")

    results = []
    for seed in args.seeds:
        cfg = replace(base, seed=seed)
        model = build_stage_a(cfg, force_range, TRAINING_XI_RANGES.as_dict())
        if seed == args.seeds[0]:
            print(f"[stageA] params  : {sum(p.numel() for p in model.parameters())}")
        trained = train_stage_a(model, subsets["train"], subsets["val"], targets, cfg)
        best = trained.restore_best()

        row = {"seed": seed, "best_epoch": trained.best_epoch}
        for name in ("val", "test"):
            predictions = per_probe_predictions(best, subsets[name], cfg.device)
            row[f"{name}_force_mae"] = force_mae(predictions, targets[name])
            row[f"{name}_probes"] = len(predictions)
        results.append(row)

        seed_root = run_root / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best.state_dict()}, seed_root / "best.pt")
        (seed_root / "config.json").write_text(
            json.dumps(
                {
                    "stage_a": cfg.as_dict(),
                    "force_range": list(force_range),
                    "xi_ranges": TRAINING_XI_RANGES.as_dict(),
                    "dataset": str(root),
                    "best_epoch": trained.best_epoch,
                },
                indent=2,
            )
        )
        (seed_root / "history.csv").write_text(
            "epoch,train_mse,val_force_mae\n"
            + "\n".join(
                f"{r['epoch']},{r['train_mse']:.6f},{r['val_force_mae']:.6f}" for r in trained.history
            )
        )
        print(f"[stageA] seed {seed}: best epoch {trained.best_epoch}, "
              f"val force MAE {row['val_force_mae']:.4f} N, test {row['test_force_mae']:.4f} N")

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "dataset": str(root),
        "dataset_version": store.manifest.get("dataset_version"),
        "channels": list(channels),
        "scaler": scaler.as_dict(),
        "force_range": list(force_range),
        "cfg": base.as_dict(),
        "seeds": list(args.seeds),
        "per_seed": results,
        "test_force_mae": {
            "mean": float(np.mean([r["test_force_mae"] for r in results])),
            "sd": float(np.std([r["test_force_mae"] for r in results])),
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("[stageA]")
    mae = summary["test_force_mae"]
    print(f"[stageA] test force MAE : {mae['mean']:.4f} +- {mae['sd']:.4f} N over {len(results)} seeds")
    print(f"[stageA] written        : {run_root}")
    print(f"[stageA] next           : baselines/rma2/scripts/eval_rma2_closed_loop.py")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
