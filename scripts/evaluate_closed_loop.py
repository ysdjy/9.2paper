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
parser.add_argument(
    "--seeds",
    type=int,
    nargs="+",
    default=(0,),
    help=(
        "Which trained seeds to deploy. All of them run inside one Isaac Sim session and from "
        "the same probe snapshots, so a difference between seeds is the seed rather than the "
        "drawer -- and three separate launches would cost three times as much for nothing."
    ),
)
parser.add_argument("--num-xi", type=int, default=64, help="Test hidden states to evaluate (0 = all).")
parser.add_argument("--num_envs", type=int, default=32, help="Drawers in parallel.")
parser.add_argument(
    "--comparison",
    type=str,
    default="comparison.json",
    help=(
        "Which comparison file in the run directory supplies the channel list, the fitted "
        "scaler and the fixed-force baseline. Overridable so a run can be deployed before "
        "every seed has finished, or from an alternative comparison."
    ),
)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=0,
    help=(
        "Discarded settle steps before each batch. Intended as the D047 fix -- so a batch that "
        "has just executed a dozen pulls does not leave contact state in the next batch's "
        "probe -- and **it does not work**: it halves the tail but leaves the median unchanged, "
        "because the dominant cause turned out not to be cross-batch history at all. See "
        "docs/DECISIONS.md D047, revised. Default 0, which is the behaviour the reported "
        "results in docs/TRAINING_V1.md were produced with; raising it changes them."
    ),
)
parser.add_argument(
    "--batch-order",
    choices=("sorted", "reversed", "within-batch-reversed"),
    default="sorted",
    help=(
        "Order the test hidden states are batched in. 'sorted' is the reporting default. "
        "'reversed' changes both which batch a drawer lands in and which environment slot it "
        "occupies. 'within-batch-reversed' changes only the slot, keeping batch membership "
        "identical -- the two together separate cross-batch history from per-environment "
        "variation, which 'reversed' alone confounds (docs/DECISIONS.md D047)."
    ),
)
parser.add_argument(
    "--slot-permutation",
    type=int,
    default=0,
    help=(
        "Deterministically permute which environment slot each test drawer occupies. 0 is the "
        "identity and reproduces the reported table; any other integer selects a different, "
        "content-addressed permutation. Used to turn D047's slot sensitivity into an error bar "
        "instead of a caveat -- see scripts/report_slot_robustness.py."
    ),
)
parser.add_argument(
    "--ood-report",
    type=str,
    default=None,
    help=(
        "Deploy on the hidden states from an OOD feasibility sweep instead of the dataset's "
        "test split (docs/OOD_FEASIBILITY.md). Everything else is unchanged -- the same "
        "checkpoints, the same frozen probe, the same scaler fitted on the training split -- "
        "so the only difference is the population. The sweep's per-state feasibility and "
        "breakaway flags are carried into the rows so the report can stratify on them."
    ),
)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from probe_drawer.analysis.probe_features import extract_features  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples, stable_permutation  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import (  # noqa: E402
    OperatingRegionCfg,
    SelectionCfg,
    evaluate_execution,
    select_forces,
    select_nearest,
)
from probe_drawer.controllers import ProbeControllerCfg  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MainTask,
    RECOMMENDED_EXECUTION_CFG,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.models import PspCfg, build_student, build_teacher  # noqa: E402
from probe_drawer.models.baselines import STRONGEST_FEATURE, FeatureRegression, GruForceRegressor  # noqa: E402
from probe_drawer.training.trainer import TrainCfg  # noqa: E402,F401  (kept for config symmetry)
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.dataset.schema import XI_DIMENSIONS, xi_id  # noqa: E402
from probe_drawer.training import FeatureScaler, reference_force_per_probe  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)


def task_from_manifest(manifest: dict) -> MainTask:
    """The task the models were trained for, read back rather than assumed.

    Taking it from the dataset removes a whole class of quiet mismatch: a model trained at
    ``d_goal`` = 0.04 m evaluated against a 0.10 m goal would report a failure that is the
    harness's, not the model's.
    """
    return MainTask(**{**manifest["main_task"], "peak_force_range": tuple(manifest["main_task"]["peak_force_range"])})


def run_probe_from_manifest(system: PullSystem, manifest: dict):
    """Run whichever probe the dataset was built with (``docs/DECISIONS.md`` D044).

    Datasets written before Phase 13 have no ``probe_mode``; they all used the ramp probe, so
    that is the default rather than an error.
    """
    parameters = manifest["probe_task"]
    if manifest.get("probe_mode", "ramp_response_terminated") == "fixed_budget":
        return system.probe.run_fixed_budget(**parameters)
    return system.probe.run(**parameters)


def build_system(num_envs: int, probe_cfg: ProbeControllerCfg) -> PullSystem:
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
            probe=probe_cfg,
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

    comparison = json.loads((run_root / args_cli.comparison).read_text())
    channels = tuple(comparison["channels"])
    scaler = FeatureScaler.from_dict(comparison["scaler"])
    psp = PspCfg()

    store = DatasetStore(dataset_root)
    split_cfg = SplitCfg(**store.read_splits()["cfg"])
    split = split_samples(store.load_samples(), split_cfg)

    # The training split is still needed either way: the two closed-form baselines are refitted
    # from it, so an OOD deployment is a change of *test* population and nothing else.
    oracle_by_id: dict[str, dict] = {}
    if args_cli.ood_report:
        ood_path = Path(args_cli.ood_report)
        if not ood_path.is_absolute():
            ood_path = project_root() / ood_path
        ood = json.loads(ood_path.read_text())
        # Content-addressed, so an OOD state's identity is its four values and joins back to
        # the sweep without depending on row order.
        xi_by_id = {xi_id(row["hidden_state"]): row["hidden_state"] for row in ood["rows"]}
        oracle_by_id = {
            xi_id(row["hidden_state"]): {
                "reach_any_force": row["reach_any_force"],
                "reach_within_task_range": row["reach_within_task_range"],
                "oracle_required_force": row["required_force"],
                "oracle_band_width": row["band_width"],
                "probe_moved_in_sweep": row["probe_moved"],
                "novel_axes": row["novel_axes"],
            }
            for row in ood["rows"]
        }
        population = f"out-of-distribution ({ood_path.name})"
        test_ids = list(xi_by_id)
    else:
        xi_by_id = {row["xi_id"]: row["xi"] for row in store.hidden_states}
        population = "in-distribution test split"
        test_ids = sorted({sample.xi_id for sample in split.test})
    if args_cli.num_xi:
        test_ids = test_ids[: args_cli.num_xi]
    # Truncated first, then reordered, so every order covers the same population.
    if args_cli.batch_order == "reversed":
        test_ids = list(reversed(test_ids))
    elif args_cli.batch_order == "within-batch-reversed":
        width = min(args_cli.num_envs, len(test_ids))
        test_ids = [
            state_id
            for start in range(0, len(test_ids), width)
            for state_id in reversed(test_ids[start : start + width])
        ]
    if args_cli.slot_permutation:
        # Content-addressed, so permutation k is the same list on any machine and in any
        # process -- a seeded PRNG would depend on how much had been drawn before it.
        order = stable_permutation("slot-permutation", args_cli.slot_permutation, len(test_ids))
        test_ids = [test_ids[index] for index in order]
    print("\n" + "=" * 78)
    print(f"[deploy] run     : {run_root}")
    print(f"[deploy] dataset : {dataset_root} ({store.manifest.get('dataset_version')})")
    print(f"[deploy] test xi : {len(test_ids)} hidden states -- {population}")
    print(f"[deploy] seeds   : {list(args_cli.seeds)}")
    print(f"[deploy] batching: {args_cli.batch_order} order, slot permutation "
          f"{args_cli.slot_permutation}, warm-up "
          f"{'system default' if args_cli.warmup_steps is None else args_cli.warmup_steps} steps")

    # --- the models, one set per seed ---
    def load(build, name: str, seed: int):
        path = run_root / f"{name}_seed{seed}" / "best.pt"
        if not path.is_file():
            # Named explicitly: a bare FileNotFoundError from torch.load after Isaac Sim has
            # already launched is an expensive way to learn that a seed was not trained.
            raise FileNotFoundError(
                f"no {name} checkpoint for seed {seed} at {path}. Train it first, or pass "
                f"--seeds without {seed}."
            )
        model = build()
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        return model.eval()

    seeds = tuple(args_cli.seeds)
    students = {seed: load(lambda: build_student(len(channels), psp), "student", seed) for seed in seeds}
    teachers = {seed: load(lambda: build_teacher(psp), "teacher", seed) for seed in seeds}

    # The two closed-form baselines are refitted here rather than checkpointed: they are a
    # handful of coefficients over the training split, so refitting is exact and cheaper than
    # serialising them.
    train_samples = split.train
    targets = reference_force_per_probe(train_samples)
    training_targets = [targets[sample.probe_id] for sample in train_samples]
    # The single feature the *run* selected on its training split, not Phase 10's hard-coded
    # one: the fixed-budget probe has a different strongest feature, and deploying the wrong
    # one here would make the closed loop disagree with the offline table it is compared to.
    strongest = comparison.get("strongest_feature", STRONGEST_FEATURE)
    linear = FeatureRegression(features=(strongest,)).fit(train_samples, training_targets)
    # The ridge on all nine summary features is the *strongest* scalar-feature baseline
    # offline (48.2 % against the linear fit's 31.5 %), so leaving it out of the closed loop
    # would compare the learned model against the weakest available alternative.
    summary_features = tuple(comparison.get("summary_features") or (strongest,))
    ridge = FeatureRegression(features=summary_features, alpha=10.0).fit(train_samples, training_targets)
    fixed_force = float(
        next(row for row in comparison["results"]["fixed force"] if row["split"] == "test")["force"]
    )

    grus = {
        seed: load(lambda: GruForceRegressor(len(channels), psp), "gru", seed)
        for seed in seeds
        if (run_root / f"gru_seed{seed}" / "best.pt").exists()
    }

    manifest = store.manifest
    task = task_from_manifest(manifest)
    probe_cfg = ProbeControllerCfg(**manifest["probe_cfg"])
    selection_cfg = SelectionCfg(force_range=task.peak_force_range, step=0.05)
    region = OperatingRegionCfg()
    criteria = task.criteria
    task_condition = torch.tensor(
        [[task.goal_displacement, task.duration]], dtype=torch.float32
    )

    print(f"[eval] setting  : {manifest.get('setting', 'v0')} -- "
          f"{manifest.get('probe_mode', 'ramp_response_terminated')} probe, "
          f"d_goal={task.goal_displacement * 1000:g} mm T_goal={task.duration:g} s")

    system = build_system(min(args_cli.num_envs, len(test_ids)), probe_cfg)
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
            # Randomise first, then warm up: the discarded settle has to happen under the
            # dynamics this batch will use, or it seats the contacts for the wrong drawer.
            system.warm_up(args_cli.warmup_steps)

            task_start = system.reader.drawer_position.clone()
            probe = run_probe_from_manifest(system, manifest)
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

            condition = task_condition.expand(num_envs, -1)

            # Seed-independent methods: a fixed force, and two closed-form fits over the
            # training split. Deployed once and recorded with ``seed: None`` rather than three
            # times, which would only add identical rows.
            choices: list[tuple[str, int | None, np.ndarray]] = [
                ("fixed force", None, np.full(num_envs, fixed_force)),
                (
                    "A linear (1 feature)",
                    None,
                    select_nearest(
                        [float(linear.predict([_FeatureRow(row)])[0]) for row in features], selection_cfg
                    ).force,
                ),
                (
                    "B ridge (summary)",
                    None,
                    select_nearest(
                        [float(ridge.predict([_FeatureRow(row)])[0]) for row in features], selection_cfg
                    ).force,
                ),
            ]
            for seed in seeds:
                choices.append(
                    (
                        "teacher (privileged)",
                        seed,
                        _landscape_choice(
                            teachers[seed], xi_tensor, post_probe, condition, scaler, num_envs,
                            selection_cfg, privileged=True,
                        ),
                    )
                )
                choices.append(
                    (
                        "ACE + PSP",
                        seed,
                        _landscape_choice(
                            students[seed], (history, lengths), post_probe, condition, scaler,
                            num_envs, selection_cfg,
                        ),
                    )
                )
                if seed in grus:
                    with torch.no_grad():
                        predicted = grus[seed](_Batch(history, lengths, post_probe, condition)).numpy()
                    choices.append(
                        ("D GRU (history)", seed, select_nearest(predicted, selection_cfg).force)
                    )

            for method, seed, forces in choices:
                restore_snapshot(system, snapshot)
                result = system.execution.run(peak_force=[float(f) for f in forces], duration=task.duration)
                evaluation = evaluate_execution(
                    result, criteria, region, pre_execution_displacement=pre_execution
                )
                for index, state_id in enumerate(batch_ids):
                    verdict = evaluation.verdicts[index]
                    rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "xi_id": state_id,
                            "xi": xi_by_id[state_id],
                            **oracle_by_id.get(state_id, {}),
                            "chosen_force": float(forces[index]),
                            "total_displacement": verdict.total_displacement,
                            "final_velocity": verdict.terminal_velocity,
                            "success": bool(verdict.success),
                            "reach_success": bool(verdict.reach_success),
                            "stable_success": bool(verdict.stable_success),
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

    report = _summarise(rows, len(test_ids), task.goal_displacement)
    report.update(
        {
            "run": str(run_root),
            "dataset": str(dataset_root),
            "seeds": list(seeds),
            "selection": {"grid_step": selection_cfg.step, "range": list(selection_cfg.force_range)},
            "task": task.as_dict(),
            "setting": manifest.get("setting", "v0"),
            "population": population,
            "ood_report": args_cli.ood_report,
            "batch_order": args_cli.batch_order,
            "warmup_steps": args_cli.warmup_steps,
            "slot_permutation": args_cli.slot_permutation,
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
    """The four fields the deployed models read. Deliberately not the whole ``ProbeBatch``:
    a deployed model has no labels and no ``xi``, and this makes that structural.

    ``task_condition`` is among them because a deployed robot *is* told the task -- how far to
    open the drawer and by when. It is a condition, not something the model chooses (D044)."""

    def __init__(self, history, lengths, post_probe, task_condition) -> None:
        self.history = history
        self.lengths = lengths
        self.post_probe = post_probe
        self.task_condition = task_condition


def _landscape_choice(
    model, inputs, post_probe, task_condition, scaler, num_envs, cfg, privileged: bool = False
):
    """Scan the force grid through a success-landscape model and take each drawer's argmax."""

    def score(forces: np.ndarray) -> np.ndarray:
        scaled = torch.tensor(
            [scaler.transform_force(float(value)) for value in forces], dtype=torch.float32
        )
        with torch.no_grad():
            context = model.encoder(inputs) if privileged else model.encoder(*inputs)
            logits = model.head(context, scaled, post_probe, task_condition)
        return torch.sigmoid(logits).numpy()

    return select_forces(score, num_envs, cfg).force


def _method_metrics(selected: list[dict], goal: float) -> dict:
    """Every number the physical closed loop is judged on, for one method's rows."""
    valid = [row for row in selected if row["valid"]]
    aborted = [row for row in selected if "safety_abort" in row["invalid_reasons"]]
    errors = [abs(row["total_displacement"] - goal) for row in selected]
    velocities = [abs(row["final_velocity"]) for row in selected]
    forces = [row["chosen_force"] for row in selected]
    return {
        "episodes": len(selected),
        # reach_success is the primary metric (D046); stable_success is reported beside it and
        # is expected to be near zero at this operating point, which is a property of the task
        # rather than of any method.
        "reach_success_rate": float(np.mean([row["reach_success"] for row in selected])),
        "stable_success_rate": float(np.mean([row["stable_success"] for row in selected])),
        "invalid_rate": 1.0 - len(valid) / len(selected),
        "safety_abort_rate": len(aborted) / len(selected),
        "median_position_error_mm": float(np.median(errors)) * 1000,
        "mean_position_error_mm": float(np.mean(errors)) * 1000,
        "p90_position_error_mm": float(np.percentile(errors, 90)) * 1000,
        "median_terminal_velocity": float(np.median(velocities)),
        "p90_terminal_velocity": float(np.percentile(velocities, 90)),
        "mean_chosen_force": float(np.mean(forces)),
        "chosen_force_spread": [float(np.min(forces)), float(np.max(forces))],
        "invalid_reasons": dict(
            Counter(reason for row in selected for reason in row["invalid_reasons"])
        ),
    }


def _summarise(rows: list[dict], num_states: int, goal: float) -> dict:
    """Per method: pooled over seeds, and the across-seed spread of the primary metric.

    Both are reported because they answer different questions. The pooled number is the
    method's performance; the spread says whether a gap between two methods survives the
    variation between training runs, which a single seed cannot tell you.
    """
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        seeds = sorted({row["seed"] for row in selected if row["seed"] is not None})
        per_seed = {
            seed: _method_metrics([row for row in selected if row["seed"] == seed], goal)
            for seed in seeds
        }
        rates = [values["reach_success_rate"] for values in per_seed.values()]
        summary[method] = {
            **_method_metrics(selected, goal),
            "seeds": seeds,
            "per_seed": per_seed,
            # Population sd over the seeds actually run, not an inference about a wider
            # population: with three seeds an unbiased estimate would be noise either way, and
            # what is wanted here is "how much did these runs differ".
            "reach_success_sd_across_seeds": float(np.std(rates)) if len(rates) > 1 else 0.0,
            "reach_success_range_across_seeds": (
                [float(min(rates)), float(max(rates))] if rates else None
            ),
        }
    return {"num_test_states": num_states, "methods": summary}


def _print(report: dict, output: Path) -> None:
    """Ranked by ``reach``, the primary metric, with the across-seed spread beside it (D046)."""
    print("[deploy]")
    print(
        f"[deploy] {'method':>22} {'reach':>8} {'+-sd':>7} {'stable':>8} {'invalid':>8} "
        f"{'abort':>7} {'|d-goal| med':>13} {'|v(T)| med':>11} {'F range':>14}"
    )
    for method, values in sorted(report["methods"].items(), key=lambda item: -item[1]["reach_success_rate"]):
        low, high = values["chosen_force_spread"]
        print(
            f"[deploy] {method:>22} {values['reach_success_rate'] * 100:7.1f}% "
            f"{values['reach_success_sd_across_seeds'] * 100:6.1f} "
            f"{values['stable_success_rate'] * 100:7.1f}% {values['invalid_rate'] * 100:7.1f}% "
            f"{values['safety_abort_rate'] * 100:6.1f}% "
            f"{values['median_position_error_mm']:12.2f}mm "
            f"{values['median_terminal_velocity']:10.3f} {low:6.2f}-{high:5.2f} N"
        )
    seeded = {name: v for name, v in report["methods"].items() if v["seeds"]}
    if seeded:
        print("[deploy]")
        print("[deploy] per-seed reach success:")
        for method, values in sorted(seeded.items(), key=lambda item: -item[1]["reach_success_rate"]):
            rates = ", ".join(
                f"seed {seed} {per['reach_success_rate'] * 100:.1f}%"
                for seed, per in sorted(values["per_seed"].items())
            )
            print(f"[deploy] {method:>22}  {rates}")
    print("[deploy]")
    print(f"[deploy] report written: {output}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
