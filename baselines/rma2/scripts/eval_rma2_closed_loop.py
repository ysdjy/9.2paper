"""Deploy Stage A in physics, beside the two methods it has to be read against.

Stage A, the privileged teacher and Direct GRU are all deployed **in one Isaac Sim session,
from the same probe snapshots**, for the reason D047 records: absolute closed-loop rates carry
session history, so a Stage A number measured in its own run could not be differenced against
the main table. Within a run every method branches from one snapshot of one probe, so the
comparison is exact even though the absolute values are not portable.

The reference methods are loaded read-only from the main project's run directory. This script
imports ``probe_drawer`` and writes nothing into it.

Usage::

    python baselines/rma2/scripts/eval_rma2_closed_loop.py --headless \
        --stage-a baselines/rma2/checkpoints/stage_a --run outputs/training/v1 \
        --dataset outputs/dataset_v1 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage-a", type=str, default="baselines/rma2/checkpoints/stage_a")
parser.add_argument("--stage-b", type=str, default="baselines/rma2/checkpoints/stage_b")
parser.add_argument("--run", type=str, default="outputs/training/v1", help="Main run, for the references.")
parser.add_argument("--dataset", type=str, default="outputs/dataset_v1")
parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
parser.add_argument("--num-xi", type=int, default=0, help="0 = every test hidden state.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--output", type=str, default="baselines/rma2/checkpoints/stage_a/closed_loop.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_drawer.controllers import ProbeControllerCfg  # noqa: E402
from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples  # noqa: E402
from probe_drawer.dataset.schema import XI_DIMENSIONS  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import (  # noqa: E402
    OperatingRegionCfg,
    SelectionCfg,
    evaluate_execution,
    select_forces,
    select_nearest,
)
from probe_drawer.experiment_plan import (  # noqa: E402
    RECOMMENDED_EXECUTION_CFG,
    SEQUENTIAL_TRANSITION_STEPS,
    TRAINING_XI_RANGES,
    MainTask,
)
from probe_drawer.models import PspCfg, build_student, build_teacher  # noqa: E402
from probe_drawer.models.baselines import GruForceRegressor  # noqa: E402
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.training import FeatureScaler, reference_force_per_probe  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)
from rma2.config import StageACfg  # noqa: E402
from rma2.config import StageBCfg  # noqa: E402,F401  (recorded in the report for provenance)
from rma2.model import build_stage_a, build_stage_b  # noqa: E402


class _Batch:
    """The fields a deployed model reads. Stage A reads ``xi``; the others never see it."""

    def __init__(self, history, lengths, post_probe, task_condition, xi) -> None:
        self.history = history
        self.lengths = lengths
        self.post_probe = post_probe
        self.task_condition = task_condition
        self.xi = xi


def absolute(path: str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else project_root() / resolved


def probe_tensors(probe, count, channels, scaler):
    """The probe recording as a padded batch, scaled exactly as training saw it.

    Setting V1's probe is a fixed budget, so every sequence is the same length and the padding
    is a formality -- kept so the tensor contract matches the training loader's.
    """
    sequences = [
        np.stack([probe.history.channel(name, index).astype(np.float32) for name in channels], axis=1)
        for index in range(count)
    ]
    lengths = torch.tensor([len(values) for values in sequences], dtype=torch.long)
    padded = torch.zeros(len(sequences), int(lengths.max()), len(channels), dtype=torch.float32)
    for row, values in enumerate(sequences):
        padded[row, : len(values)] = torch.from_numpy(scaler.transform_history(values).astype(np.float32))
    return padded, lengths


def landscape_choice(head, context, batch, scaler, count, cfg):
    """Scan the force grid through a success-landscape head and take each drawer's argmax.

    The context is passed in already computed: it does not depend on the candidate force, so
    re-encoding it per grid point would be waste, and taking it as an argument is what lets
    one function serve both the privileged teacher and ACE + PSP.
    """

    def score(forces: np.ndarray) -> np.ndarray:
        scaled = torch.tensor([scaler.transform_force(float(v)) for v in forces], dtype=torch.float32)
        with torch.no_grad():
            logits = head(context, scaled, batch.post_probe, batch.task_condition)
        return torch.sigmoid(logits).numpy()

    return select_forces(score, count, cfg).force


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    stage_a_root, stage_b_root, run_root, dataset_root = (
        absolute(args_cli.stage_a),
        absolute(args_cli.stage_b),
        absolute(args_cli.run),
        absolute(args_cli.dataset),
    )
    summary = json.loads((stage_a_root / "summary.json").read_text())
    stage_b_summary = json.loads((stage_b_root / "summary.json").read_text())
    comparison = json.loads((run_root / "comparison.json").read_text())
    channels = tuple(comparison["channels"])
    scaler = FeatureScaler.from_dict(comparison["scaler"])
    psp = PspCfg()
    seeds = tuple(args_cli.seeds)

    store = DatasetStore(dataset_root)
    manifest = store.manifest
    task = MainTask(
        **{**manifest["main_task"], "peak_force_range": tuple(manifest["main_task"]["peak_force_range"])}
    )
    split = split_samples(store.load_samples(), SplitCfg(**store.read_splits()["cfg"]))
    xi_by_id = {row["xi_id"]: row["xi"] for row in store.hidden_states}
    test_ids = sorted({sample.xi_id for sample in split.test})
    if args_cli.num_xi:
        test_ids = test_ids[: args_cli.num_xi]

    # Per-hidden-state reference force, averaged over that state's probes: the closed loop
    # re-probes, so no single dataset probe is "the" reference for it.
    test_targets = reference_force_per_probe(split.test)
    by_state: dict[str, list[float]] = defaultdict(list)
    for sample in split.test:
        if sample.probe_id in test_targets:
            by_state[sample.xi_id].append(test_targets[sample.probe_id])
    reference_by_state = {key: float(np.mean(values)) for key, values in by_state.items()}

    def load(build, root: Path, name: str, seed: int):
        path = root / (f"seed{seed}" if name == "stage_a" else f"{name}_seed{seed}") / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no {name} checkpoint for seed {seed} at {path}.")
        model = build()
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        return model.eval()

    stage_a = {
        seed: load(
            lambda: build_stage_a(
                StageACfg(latent_dim=summary["cfg"]["latent_dim"]),
                tuple(summary["force_range"]),
                TRAINING_XI_RANGES.as_dict(),
            ),
            stage_a_root,
            "stage_a",
            seed,
        )
        for seed in seeds
    }
    teachers = {seed: load(lambda: build_teacher(psp), run_root, "teacher", seed) for seed in seeds}
    students = {seed: load(lambda: build_student(len(channels), psp), run_root, "student", seed) for seed in seeds}
    grus = {
        seed: load(lambda: GruForceRegressor(len(channels), psp), run_root, "gru", seed)
        for seed in seeds
    }

    def stage_b_for(seed: int):
        """Stage B seed ``k`` is the adapter distilled from Stage A seed ``k``."""
        scaffold = build_stage_b(
            build_stage_a(
                StageACfg(latent_dim=summary["cfg"]["latent_dim"]),
                tuple(summary["force_range"]),
                TRAINING_XI_RANGES.as_dict(),
            ),
            len(channels),
            psp,
        )
        path = stage_b_root / f"seed{seed}" / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"no Stage B checkpoint for seed {seed} at {path}.")
        scaffold.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        return scaffold.eval()

    stage_b = {seed: stage_b_for(seed) for seed in seeds}

    selection = SelectionCfg(force_range=task.peak_force_range, step=0.05)
    region = OperatingRegionCfg()
    criteria = task.criteria
    condition = torch.tensor([[task.goal_displacement, task.duration]], dtype=torch.float32)

    print("\n" + "=" * 96)
    print(f"[rma2] stage A   : {stage_a_root}")
    print(f"[rma2] stage B   : {stage_b_root}")
    print(f"[rma2] reference : {run_root} (teacher, Direct GRU -- read only)")
    print(f"[rma2] test xi   : {len(test_ids)} in-distribution hidden states, unseen in any split")
    print(f"[rma2] seeds     : {list(seeds)}, all deployed in ONE session from shared snapshots")
    print(f"[rma2] task      : d_goal={task.goal_displacement * 1000:g} mm T={task.duration:g} s")

    num_envs = min(args_cli.num_envs, len(test_ids))
    system = PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=ProbeControllerCfg(**manifest["probe_cfg"]),
            execution=RECOMMENDED_EXECUTION_CFG,
        )
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    probe_parameters = manifest["probe_task"]
    rows: list[dict] = []

    try:
        for start in range(0, len(test_ids), num_envs):
            batch_ids = test_ids[start : start + num_envs]
            padded_ids = batch_ids + [batch_ids[-1]] * (num_envs - len(batch_ids))
            randomizer.apply(
                system.env,
                [
                    DynamicsParameters(
                        name=state_id[:8],
                        drawer_mass=xi_by_id[state_id]["mass"],
                        joint_static_friction=xi_by_id[state_id]["static_friction"],
                        joint_dynamic_friction=xi_by_id[state_id]["dynamic_friction"],
                        joint_damping=xi_by_id[state_id]["damping"],
                    )
                    for state_id in padded_ids
                ],
            )
            system.reset()

            task_start = system.reader.drawer_position.clone()
            probe = system.probe.run_fixed_budget(**probe_parameters)
            system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
            pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
            post_velocity = system.reader.drawer_velocity.cpu().numpy().copy()
            snapshot = capture_snapshot(system, label="rma2 stage A")

            history, lengths = probe_tensors(probe, num_envs, channels, scaler)
            batch = _Batch(
                history=history,
                lengths=lengths,
                post_probe=torch.from_numpy(
                    scaler.transform_post_probe(
                        np.stack([pre_execution, post_velocity], axis=1)
                    ).astype(np.float32)
                ),
                task_condition=condition.expand(num_envs, -1),
                xi=torch.tensor(
                    [[xi_by_id[s][name] for name in XI_DIMENSIONS] for s in padded_ids],
                    dtype=torch.float32,
                ),
            )

            choices: list[tuple[str, int, np.ndarray]] = []
            for seed in seeds:
                with torch.no_grad():
                    raw = stage_a[seed](batch).numpy()
                # Both Stage A and Stage B emit a force already inside the allowed range;
                # snapping it to the shared 0.05 N grid is what the other point regressors
                # get, so it is applied here too rather than giving them a continuous edge.
                choices.append(("RMA2 Stage A (xi -> point)", seed, select_nearest(raw, selection).force))
                with torch.no_grad():
                    distilled = stage_b[seed](batch).numpy()
                choices.append(("RMA2 Stage B (probe -> latent -> point)", seed, select_nearest(distilled, selection).force))
                with torch.no_grad():
                    teacher_context = teachers[seed].encoder(batch.xi)
                    student_context = students[seed].encoder(batch.history, batch.lengths)
                choices.append(
                    ("teacher (xi -> landscape)", seed,
                     landscape_choice(teachers[seed].head, teacher_context, batch, scaler, num_envs, selection))
                )
                choices.append(
                    ("ACE + PSP (probe -> landscape)", seed,
                     landscape_choice(students[seed].head, student_context, batch, scaler, num_envs, selection))
                )
                with torch.no_grad():
                    predicted = grus[seed](batch).numpy()
                choices.append(("D GRU (probe -> point)", seed, select_nearest(predicted, selection).force))

            for method, seed, forces in choices:
                restore_snapshot(system, snapshot)
                result = system.execution.run(
                    peak_force=[float(f) for f in forces], duration=task.duration
                )
                report = evaluate_execution(
                    result, criteria, region, pre_execution_displacement=pre_execution
                )
                for index, state_id in enumerate(batch_ids):
                    verdict = report.verdicts[index]
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "xi_id": state_id,
                            "chosen_force": float(forces[index]),
                            "reference_force": reference_by_state.get(state_id),
                            "total_displacement": verdict.total_displacement,
                            "final_velocity": verdict.terminal_velocity,
                            "reach_success": bool(verdict.reach_success),
                            "stable_success": bool(verdict.stable_success),
                            "valid": bool(verdict.valid),
                            "invalid_reasons": [r.value for r in verdict.invalid_reasons],
                        }
                    )
            print(f"[rma2] batch {start // num_envs + 1}/{-(-len(test_ids) // num_envs)} "
                  f"({time.perf_counter() - started:.0f} s)")
    finally:
        system.close()

    methods = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [r for r in rows if r["method"] == method]
        per_seed = {}
        for seed in sorted({r["seed"] for r in selected}):
            subset = [r for r in selected if r["seed"] == seed]
            per_seed[seed] = float(np.mean([r["reach_success"] for r in subset])) * 100
        errors = [abs(r["total_displacement"] - task.goal_displacement) * 1000 for r in selected]
        biases = [
            abs(r["chosen_force"] - r["reference_force"])
            for r in selected
            if r["reference_force"] is not None
        ]
        rates = list(per_seed.values())
        methods[method] = {
            "episodes": len(selected),
            "reach_pp": float(np.mean([r["reach_success"] for r in selected])) * 100,
            "reach_sd_across_seeds": float(np.std(rates)),
            "reach_per_seed": per_seed,
            "median_position_error_mm": float(np.median(errors)),
            "mean_position_error_mm": float(np.mean(errors)),
            "closed_loop_force_mae": float(np.mean(biases)) if biases else None,
            "median_chosen_force": float(np.median([r["chosen_force"] for r in selected])),
            "invalid_rate": 1.0 - float(np.mean([r["valid"] for r in selected])),
            "safety_abort_rate": float(
                np.mean(["safety_abort" in r["invalid_reasons"] for r in selected])
            ),
            "stable_pp": float(np.mean([r["stable_success"] for r in selected])) * 100,
            "invalid_reasons": dict(Counter(r for row in selected for r in row["invalid_reasons"])),
        }

    payload = {
        "population": "in-distribution test split",
        "num_test_states": len(test_ids),
        "seeds": list(seeds),
        "task": task.as_dict(),
        "stage_a_cfg": summary["cfg"],
        "stage_a_offline_test_force_mae": summary["test_force_mae"],
        "stage_b_cfg": stage_b_summary["cfg"],
        "stage_b_offline_test_force_mae": stage_b_summary["test_force_mae"],
        "stage_b_val_latent_mse": stage_b_summary["val_latent_mse"],
        "note": (
            "All methods deployed in one session from shared probe snapshots (D047). Absolute "
            "rates are not comparable to other sessions; differences within this table are."
        ),
        "git_commit": git_commit(),
        "environment": collect_environment_info().as_dict(),
        "methods": methods,
        "rows": rows,
    }
    output = absolute(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=float))

    print("[rma2]")
    print(f"[rma2] {'method':>38} {'reach':>8} {'+-sd':>6} {'|d-goal| med':>13} "
          f"{'|F-ref| MAE':>12} {'F med':>8} {'invalid':>8} {'abort':>7}")
    for method, v in sorted(methods.items(), key=lambda i: -i[1]["reach_pp"]):
        print(f"[rma2] {method:>38} {v['reach_pp']:7.1f}% {v['reach_sd_across_seeds']:5.1f} "
              f"{v['median_position_error_mm']:12.2f}mm {v['closed_loop_force_mae']:11.3f}N "
              f"{v['median_chosen_force']:7.2f}N {v['invalid_rate'] * 100:7.1f}% "
              f"{v['safety_abort_rate'] * 100:6.1f}%")
    print("[rma2]")
    for method, v in sorted(methods.items(), key=lambda i: -i[1]["reach_pp"]):
        print(f"[rma2] {method:>38}  " + ", ".join(
            f"seed {s} {r:.1f}%" for s, r in sorted(v["reach_per_seed"].items())
        ))
    print(f"[rma2]")
    print(f"[rma2] written : {output}")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
