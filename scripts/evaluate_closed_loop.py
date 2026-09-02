"""Phase 11L -- the real test: probe an unseen drawer, choose a force, pull, and see.

Everything before this is offline. Here the trained models go back into Isaac Sim and face
hidden states from the **test** split, which no model saw in any form. For each drawer::

    INITIAL -> Probe -> 8-step inference gap -> [model chooses F] -> Execution -> Evaluate

The model sees the probe recording and the post-probe state, and nothing else. ``xi`` reaches
only the privileged teacher, which is here as an upper bound and is not a deployable method.

**All methods share one probe.** After the gap the state is snapshotted, and each method's
chosen force restores it and executes. So they face an identical drawer *and* an identical
probe -- the fairest comparison available, and the reason the branching validation in Phase
11B had to come first. The alternative, one probe per method, would let a lucky probe decide
the ranking.

Force selection is a search over a 0.05 N grid, done in ``evaluation/force_selection.py``.
The execution controller still takes only ``(peak_force, duration)``; it has not learned what
a goal is.

Usage::

    python scripts/evaluate_closed_loop.py --headless --run outputs/training/run_XXXX \\
        --dataset outputs/dataset_v0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=str, required=True, help="Training run directory.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset the models were trained on.")
parser.add_argument("--seed", type=int, default=0, help="Which trained seed to deploy.")
parser.add_argument("--num-xi", type=int, default=64, help="Test hidden states to evaluate (0 = all).")
parser.add_argument("--num_envs", type=int, default=32, help="Drawers in parallel.")
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from probe_drawer.analysis.probe_features import extract_features  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import (  # noqa: E402
    OperatingRegionCfg,
    SelectionCfg,
    evaluate_execution,
    select_forces,
    select_nearest,
)
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.models import PspCfg, build_student, build_teacher  # noqa: E402
from probe_drawer.models.baselines import STRONGEST_FEATURE, FeatureRegression, GruForceRegressor  # noqa: E402
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.dataset.schema import XI_DIMENSIONS  # noqa: E402
from probe_drawer.training import FeatureScaler, reference_force_per_probe  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)


def build_system(num_envs: int) -> PullSystem:
    execution = ExecutionControllerCfg(
        rise_fraction=RECOMMENDED_EXECUTION_CFG.rise_fraction,
        fall_fraction=RECOMMENDED_EXECUTION_CFG.fall_fraction,
        shape=RECOMMENDED_EXECUTION_CFG.shape,
        settle_steps=0,
        zero_force_cleanup_steps=RECOMMENDED_EXECUTION_CFG.zero_force_cleanup_steps,
        post_execution_settle_steps=RECOMMENDED_EXECUTION_CFG.post_execution_settle_steps,
    )
    return PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=RECOMMENDED_PROBE_CFG,
            execution=execution,
        )
    )


def probe_tensors(probe, indices, channels, scaler, device):
    """The probe recording as a padded batch, exactly as the DataLoader would build it.

    Built here rather than reusing ``SampleDataset`` because these histories have just been
    measured and never went to disk; the padding and the scaling must still match what
    training saw, so the same scaler and the same channel order are used.
    """
    sequences = [
        np.stack([probe.history.channel(name, index).astype(np.float32) for name in channels], axis=1)
        for index in indices
    ]
    lengths = torch.tensor([len(values) for values in sequences], dtype=torch.long)
    longest = int(lengths.max())
    padded = torch.zeros(len(sequences), longest, len(channels), dtype=torch.float32)
    for row, values in enumerate(sequences):
        padded[row, : len(values)] = torch.from_numpy(scaler.transform_history(values).astype(np.float32))
    return padded.to(device), lengths


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    run_root = Path(args_cli.run)
    if not run_root.is_absolute():
        run_root = project_root() / run_root
    dataset_root = Path(args_cli.dataset)
    if not dataset_root.is_absolute():
        dataset_root = project_root() / dataset_root

    comparison = json.loads((run_root / "comparison.json").read_text())
    channels = tuple(comparison["channels"])
    scaler = FeatureScaler.from_dict(comparison["scaler"])
    psp = PspCfg()

    store = DatasetStore(dataset_root)
    split_cfg = SplitCfg(**store.read_splits()["cfg"])
    split = split_samples(store.load_samples(), split_cfg)
    xi_by_id = {row["xi_id"]: row["xi"] for row in store.hidden_states}
    test_ids = sorted({sample.xi_id for sample in split.test})
    if args_cli.num_xi:
        test_ids = test_ids[: args_cli.num_xi]
    print("\n" + "=" * 78)
    print(f"[deploy] run     : {run_root}")
    print(f"[deploy] dataset : {dataset_root} ({store.manifest.get('dataset_version')})")
    print(f"[deploy] test xi : {len(test_ids)} hidden states, never seen in any split")

    # --- the models, loaded from the run ---
    student = build_student(len(channels), psp)
    student.load_state_dict(
        torch.load(run_root / f"student_seed{args_cli.seed}" / "best.pt", weights_only=True)["state_dict"]
    )
    student.eval()
    teacher = build_teacher(psp)
    teacher.load_state_dict(
        torch.load(run_root / f"teacher_seed{args_cli.seed}" / "best.pt", weights_only=True)["state_dict"]
    )
    teacher.eval()

    # The two closed-form baselines are refitted here rather than checkpointed: they are a
    # handful of coefficients over the training split, so refitting is exact and cheaper than
    # serialising them.
    train_samples = split.train
    targets = reference_force_per_probe(train_samples)
    linear = FeatureRegression(features=(STRONGEST_FEATURE,)).fit(
        train_samples, [targets[sample.probe_id] for sample in train_samples]
    )
    fixed_force = float(
        next(row for row in comparison["results"]["fixed force"] if row["split"] == "test")["force"]
    )

    gru = GruForceRegressor(len(channels), psp)
    gru_path = run_root / f"gru_seed{args_cli.seed}" / "best.pt"
    has_gru = gru_path.exists()
    if has_gru:
        gru.load_state_dict(torch.load(gru_path, weights_only=True)["state_dict"])
        gru.eval()

    selection_cfg = SelectionCfg(force_range=MAIN_TASK.peak_force_range, step=0.05)
    region = OperatingRegionCfg()
    criteria = MAIN_TASK.criteria

    system = build_system(min(args_cli.num_envs, len(test_ids)))
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    num_envs = system.env.num_envs
    rows: list[dict] = []

    try:
        for start in range(0, len(test_ids), num_envs):
            batch_ids = test_ids[start : start + num_envs]
            padded_ids = batch_ids + [batch_ids[-1]] * (num_envs - len(batch_ids))
            parameters = [
                DynamicsParameters(
                    name=state_id[:8],
                    drawer_mass=xi_by_id[state_id]["mass"],
                    joint_static_friction=xi_by_id[state_id]["static_friction"],
                    joint_dynamic_friction=xi_by_id[state_id]["dynamic_friction"],
                    joint_damping=xi_by_id[state_id]["damping"],
                )
                for state_id in padded_ids
            ]
            randomizer.apply(system.env, parameters)
            system.reset()

            task_start = system.reader.drawer_position.clone()
            probe = system.probe.run(**RECOMMENDED_PROBE_TASK.as_kwargs())
            system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
            pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
            post_velocity = system.reader.drawer_velocity.cpu().numpy().copy()
            snapshot = capture_snapshot(system, label="deployment")

            history, lengths = probe_tensors(probe, range(num_envs), channels, scaler, "cpu")
            post_probe = torch.from_numpy(
                scaler.transform_post_probe(
                    np.stack([pre_execution, post_velocity], axis=1)
                ).astype(np.float32)
            )
            xi_tensor = torch.tensor(
                [[xi_by_id[state_id][name] for name in XI_DIMENSIONS] for state_id in padded_ids],
                dtype=torch.float32,
            )
            features = [extract_features(probe, index).as_dict() for index in range(num_envs)]

            choices = {
                "fixed force": np.full(num_envs, fixed_force),
                "A linear (1 feature)": select_nearest(
                    [float(linear.predict([_FeatureRow(row)])[0]) for row in features], selection_cfg
                ).force,
                "teacher (privileged)": _landscape_choice(
                    teacher, xi_tensor, post_probe, scaler, num_envs, selection_cfg, privileged=True
                ),
                "ACE + PSP": _landscape_choice(
                    student, (history, lengths), post_probe, scaler, num_envs, selection_cfg
                ),
            }
            if has_gru:
                with torch.no_grad():
                    predicted = gru(_Batch(history, lengths, post_probe)).numpy()
                choices["D GRU (history)"] = select_nearest(predicted, selection_cfg).force

            for method, forces in choices.items():
                restore_snapshot(system, snapshot)
                result = system.execution.run(peak_force=[float(f) for f in forces], duration=MAIN_TASK.duration)
                evaluation = evaluate_execution(
                    result, criteria, region, pre_execution_displacement=pre_execution
                )
                for index, state_id in enumerate(batch_ids):
                    verdict = evaluation.verdicts[index]
                    rows.append(
                        {
                            "method": method,
                            "xi_id": state_id,
                            "xi": xi_by_id[state_id],
                            "chosen_force": float(forces[index]),
                            "total_displacement": verdict.total_displacement,
                            "final_velocity": verdict.terminal_velocity,
                            "success": bool(verdict.success),
                            "valid": bool(verdict.valid),
                            "invalid_reasons": [reason.value for reason in verdict.invalid_reasons],
                            "probe_displacement": float(pre_execution[index]),
                        }
                    )
            print(
                f"[deploy] batch {start // num_envs + 1}/{-(-len(test_ids) // num_envs)} "
                f"({time.perf_counter() - started:.0f} s)"
            )
    finally:
        system.close()

    report = _summarise(rows, len(test_ids))
    report.update(
        {
            "run": str(run_root),
            "dataset": str(dataset_root),
            "seed": args_cli.seed,
            "selection": {"grid_step": selection_cfg.step, "range": list(selection_cfg.force_range)},
            "task": MAIN_TASK.as_dict(),
            "git_commit": git_commit(),
            "environment": collect_environment_info().as_dict(),
            "rows": rows,
        }
    )
    output = Path(args_cli.output) if args_cli.output else run_root / "closed_loop.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report, output)


class _FeatureRow:
    """Adapts a probe-feature dict to what ``FeatureRegression`` reads."""

    def __init__(self, summary: dict) -> None:
        self.probe_summary = summary


class _Batch:
    """The three fields the deployed models read. Deliberately not the whole ``ProbeBatch``:
    a deployed model has no labels and no ``xi``, and this makes that structural."""

    def __init__(self, history, lengths, post_probe) -> None:
        self.history = history
        self.lengths = lengths
        self.post_probe = post_probe


def _landscape_choice(model, inputs, post_probe, scaler, num_envs, cfg, privileged: bool = False):
    """Scan the force grid through a success-landscape model and take each drawer's argmax."""

    def score(forces: np.ndarray) -> np.ndarray:
        scaled = torch.tensor(
            [scaler.transform_force(float(value)) for value in forces], dtype=torch.float32
        )
        with torch.no_grad():
            if privileged:
                logits = model.head(model.encoder(inputs), scaled, post_probe)
            else:
                history, lengths = inputs
                logits = model.head(model.encoder(history, lengths), scaled, post_probe)
        return torch.sigmoid(logits).numpy()

    return select_forces(score, num_envs, cfg).force


def _summarise(rows: list[dict], num_states: int) -> dict:
    methods = sorted({row["method"] for row in rows})
    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        valid = [row for row in selected if row["valid"]]
        summary[method] = {
            "drawers": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "success_rate_valid_only": float(np.mean([row["success"] for row in valid])) if valid else float("nan"),
            "invalid_rate": 1.0 - len(valid) / len(selected),
            "median_displacement_error_mm": float(
                np.median([abs(row["total_displacement"] - MAIN_TASK.goal_displacement) for row in selected]) * 1000
            ),
            "median_terminal_velocity": float(np.median([abs(row["final_velocity"]) for row in selected])),
            "mean_chosen_force": float(np.mean([row["chosen_force"] for row in selected])),
            "chosen_force_spread": [
                float(np.min([row["chosen_force"] for row in selected])),
                float(np.max([row["chosen_force"] for row in selected])),
            ],
        }
    return {"num_test_states": num_states, "methods": summary}


def _print(report: dict, output: Path) -> None:
    print("[deploy]")
    print(
        f"[deploy] {'method':>22} {'success':>9} {'valid only':>11} {'invalid':>8} "
        f"{'|d-goal| med':>13} {'F range':>14}"
    )
    for method, values in sorted(report["methods"].items(), key=lambda item: -item[1]["success_rate"]):
        low, high = values["chosen_force_spread"]
        print(
            f"[deploy] {method:>22} {values['success_rate'] * 100:8.1f}% "
            f"{values['success_rate_valid_only'] * 100:10.1f}% {values['invalid_rate'] * 100:7.1f}% "
            f"{values['median_displacement_error_mm']:12.2f}mm {low:6.2f}-{high:5.2f} N"
        )
    print("[deploy]")
    print(f"[deploy] report written: {output}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
