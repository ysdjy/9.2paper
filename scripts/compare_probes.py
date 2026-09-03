"""Is the three-phase probe better than the 3 mm probe, and at what cost?

Runs both probes on the same hidden states, from the same reset, and compares what each one's
scalar features can tell you. The comparison that matters is not "does the new probe collect
more data" -- it plainly does -- but whether the extra data identifies anything the old probe
could not, and whether it costs displacement the task can spare.

Five questions, all measured:

1. **Identifiability of the hidden state.** A leave-one-out linear readout of each of
   ``m``, ``mu_s``, ``mu_d``, ``b`` from each probe's features. Damping is the one to watch --
   Phase 10 measured the old probe as blind to it.
2. **Predictive power for the answer.** The same readout of the ``(F_peak, T)`` the drawer
   actually needs, taken from the 2-D Oracle already on disk.
3. **Cost in displacement.** The probe must not quietly perform the task. Reported against
   the 10-15 % of ``d_goal`` the plan proposes as a bound.
4. **Cost in time.**
5. **Safety.** Aborts, drift, peak velocity.

RMSE is reported next to R-squared throughout, because the two probes produce different
distributions of hidden state per feature-bin and R-squared is scaled by target variance --
the Phase 12 probe-duration analysis was nearly misread on exactly that.

Usage::

    python scripts/compare_probes.py --headless
    python scripts/compare_probes.py --headless --alphas 0.05 0.10 0.15 --betas 0.1 0.2 0.3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=32)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--alphas", type=float, nargs="+", default=(0.05, 0.10, 0.15))
parser.add_argument("--betas", type=float, nargs="+", default=(0.10, 0.20, 0.30))
parser.add_argument("--repeats", type=int, default=2, help="Independent probes per configuration.")
parser.add_argument("--seed", type=int, default=20260902)
parser.add_argument("--oracle", type=str, default="outputs/logs/landscape_2d_fine.json")
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

from probe_drawer.dataset.sampling import representative_hidden_states, success_mask  # noqa: E402
from probe_drawer.analysis.probe_features import PROBE_FEATURES, extract_features  # noqa: E402
from probe_drawer.analysis.readout import RIDGE_PENALTY, leave_one_out  # noqa: E402
from probe_drawer.experimental.response_probe_features import (  # noqa: E402
    RESPONSE_PROBE_FEATURES,
    extract_response_features,
)
from probe_drawer.analysis.sweep import SweepDataset  # noqa: E402
from probe_drawer.experimental.response_probe import ResponseProbeCfg, ResponseProbeController  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.experiment_plan import MAIN_TASK, RECOMMENDED_PROBE_CFG, RECOMMENDED_PROBE_TASK  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)

XI_NAMES = ("mass", "static_friction", "dynamic_friction", "damping")


def oracle_targets(path: Path) -> dict[tuple[float, ...], dict]:
    """The ``(F, T)`` each hidden state needs, from the 2-D Oracle already on disk.

    Uses the max-margin point: the succeeding parameter furthest from any failure, which is
    the target Phase 12 selected as the fair one for a single-point regressor.
    """
    dataset = SweepDataset.load(path)
    criteria = MAIN_TASK.criteria
    targets: dict[tuple[float, ...], dict] = {}
    for key in dataset.xi_keys():
        masks = success_mask(dataset, key, criteria)
        rows, columns = np.nonzero(masks["success"])
        if not rows.size:
            continue
        targets[tuple(round(value, 6) for value in key)] = {
            "force": float(np.median(masks["forces"][columns])),
            "duration": float(np.median(masks["durations"][rows])),
        }
    return targets


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    states = representative_hidden_states(args_cli.num_xi, seed=args_cli.seed)
    num_envs = min(args_cli.num_envs, len(states))
    oracle_path = Path(args_cli.oracle)
    if not oracle_path.is_absolute():
        oracle_path = project_root() / oracle_path
    targets = oracle_targets(oracle_path) if oracle_path.exists() else {}

    configurations = [("old", None, None)] + [
        ("new", alpha, beta) for alpha in args_cli.alphas for beta in args_cli.betas
    ]
    print("\n" + "=" * 78)
    print(f"[cmp] hidden states : {len(states)} x {args_cli.repeats} repeats")
    print(f"[cmp] configurations: old probe + {len(configurations) - 1} (alpha, beta) combinations")
    print(f"[cmp] d_goal        : {MAIN_TASK.goal_displacement * 1000:g} mm -> "
          f"d_trigger {[f'{a * MAIN_TASK.goal_displacement * 1000:.1f} mm' for a in args_cli.alphas]}")
    print(f"[cmp] oracle targets: {len(targets)} hidden states from {oracle_path.name}")

    system = PullSystem.build(
        PullSystemCfg(num_envs=num_envs, device=args_cli.device, probe=RECOMMENDED_PROBE_CFG)
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    collected: dict[str, list[dict]] = {}

    try:
        for kind, alpha, beta in configurations:
            label = "old" if kind == "old" else f"new a={alpha:.2f} b={beta:.2f}"
            controller = (
                system.probe
                if kind == "old"
                else ResponseProbeController(
                    system.env,
                    system.osc,
                    system.reader,
                    ResponseProbeCfg(
                        trigger_fraction=alpha,
                        release_fraction=beta,
                        max_total_displacement=max(0.010, 3.0 * alpha * MAIN_TASK.goal_displacement),
                    ),
                )
            )
            rows: list[dict] = []
            for repeat in range(args_cli.repeats):
                for start in range(0, len(states), num_envs):
                    batch = states[start : start + num_envs]
                    padded = batch + [batch[-1]] * (num_envs - len(batch))
                    randomizer.apply(
                        system.env,
                        [
                            DynamicsParameters(
                                drawer_mass=state["mass"],
                                joint_static_friction=state["static_friction"],
                                joint_dynamic_friction=state["dynamic_friction"],
                                joint_damping=state["damping"],
                                name=f"xi{index:03d}",
                            )
                            for index, state in enumerate(padded)
                        ],
                    )
                    system.reset()
                    result = (
                        controller.run(**RECOMMENDED_PROBE_TASK.as_kwargs())
                        if kind == "old"
                        else controller.run(goal_displacement=MAIN_TASK.goal_displacement)
                    )
                    for env_index, state in enumerate(batch):
                        features = (
                            extract_features(result, env_index).as_dict()
                            if kind == "old"
                            else extract_response_features(result, env_index).as_dict()
                        )
                        rows.append(
                            {
                                "xi": dict(state),
                                "repeat": repeat,
                                "features": features,
                                "duration": float(result.duration[env_index]),
                                "displacement": float(result.final_displacement[env_index]),
                                "termination": result.termination_reason[env_index].value,
                                "peak_measured_force": float(result.peak_measured_force[env_index]),
                            }
                        )
            collected[label] = rows
            durations = np.array([row["duration"] for row in rows])
            displacements = np.array([row["displacement"] for row in rows])
            print(
                f"[cmp] {label:<18} {len(rows):4d} probes  duration {durations.mean():.3f}s  "
                f"displacement {displacements.mean() * 1000:5.2f} mm "
                f"({displacements.mean() / MAIN_TASK.goal_displacement * 100:4.1f} % of d_goal)  "
                f"({time.perf_counter() - started:.0f} s)"
            )
    finally:
        system.close()

    report = _analyse(collected, targets)
    report.update(
        {
            "git_commit": git_commit(),
            "goal_displacement": MAIN_TASK.goal_displacement,
            "num_hidden_states": len(states),
            "repeats": args_cli.repeats,
            "alphas": list(args_cli.alphas),
            "betas": list(args_cli.betas),
            "oracle": str(oracle_path),
            "environment": collect_environment_info().as_dict(),
        }
    )
    output = Path(args_cli.output) if args_cli.output else project_root() / "outputs" / "logs" / "probe_comparison.json"
    output.write_text(json.dumps(report, indent=2, default=float))
    _print(report)
    print(f"[cmp] report written: {output}")
    print("=" * 78 + "\n")


def _analyse(collected: dict[str, list[dict]], targets: dict) -> dict:
    """Identifiability, cost and safety per probe configuration."""
    results = {}
    for label, rows in collected.items():
        names = PROBE_FEATURES if label == "old" else RESPONSE_PROBE_FEATURES
        matrix = np.array([[row["features"][name] for name in names] for row in rows], dtype=float)

        readouts = {}
        for dimension in XI_NAMES:
            readouts[dimension] = leave_one_out(
                matrix, np.array([row["xi"][dimension] for row in rows], dtype=float)
            )
        for axis in ("force", "duration"):
            available = [
                (index, targets[tuple(round(row["xi"][name], 6) for name in XI_NAMES)][axis])
                for index, row in enumerate(rows)
                if tuple(round(row["xi"][name], 6) for name in XI_NAMES) in targets
            ]
            if len(available) > matrix.shape[1] + 3:
                indices = [index for index, _ in available]
                readouts[f"required_{axis}"] = leave_one_out(
                    matrix[indices], np.array([value for _, value in available], dtype=float)
                )
            else:
                readouts[f"required_{axis}"] = {"r2": float("nan"), "rmse": float("nan"), "n": len(available)}

        durations = np.array([row["duration"] for row in rows])
        displacements = np.array([row["displacement"] for row in rows])
        coast = [row["features"].get("coast_fit_r2") for row in rows if "coast_fit_r2" in row["features"]]
        results[label] = {
            "probes": len(rows),
            "features": list(names),
            "readouts": readouts,
            "duration": {"mean": float(durations.mean()), "max": float(durations.max())},
            "displacement": {
                "mean": float(displacements.mean()),
                "max": float(displacements.max()),
                "mean_fraction_of_goal": float(displacements.mean() / MAIN_TASK.goal_displacement),
                "max_fraction_of_goal": float(displacements.max() / MAIN_TASK.goal_displacement),
            },
            "peak_measured_force": float(np.max([row["peak_measured_force"] for row in rows])),
            "terminations": {
                reason: sum(1 for row in rows if row["termination"] == reason)
                for reason in {row["termination"] for row in rows}
            },
            "coast_fit_r2_median": float(np.nanmedian(coast)) if coast else None,
            "coasted_to_rest_fraction": (
                float(np.mean([row["features"]["coasted_to_rest"] for row in rows]))
                if rows and "coasted_to_rest" in rows[0]["features"]
                else None
            ),
        }
    return {"per_configuration": results}


def _print(report: dict) -> None:
    print("[cmp]")
    print("[cmp] IDENTIFIABILITY -- leave-one-out linear readout, R2 / RMSE")
    header = f"[cmp] {'configuration':<18}" + "".join(
        f"{name[:11]:>15}" for name in (*XI_NAMES, "required_force", "required_duration")
    )
    print(header)
    for label, values in report["per_configuration"].items():
        cells = "".join(
            f"{values['readouts'][name]['r2']:7.3f}/{values['readouts'][name]['rmse']:7.3f}"
            for name in (*XI_NAMES, "required_force", "required_duration")
        )
        print(f"[cmp] {label:<18}{cells}")
    print(
        f"[cmp] (ridge penalty {RIDGE_PENALTY}, applied to both probes: the old one has 9 "
        "features and the new one 18, and an unregularised fit on ~64 probes would compare "
        "conditioning rather than information. Read RMSE, not R2 alone.)"
    )

    print("[cmp]")
    print("[cmp] COST AND SAFETY")
    print(
        f"[cmp] {'configuration':<18} {'duration':>9} {'displacement':>16} {'%d_goal':>8} "
        f"{'peak wrist':>11} {'coast R2':>9} {'to rest':>8}"
    )
    for label, values in report["per_configuration"].items():
        coast = values["coast_fit_r2_median"]
        rest = values["coasted_to_rest_fraction"]
        print(
            f"[cmp] {label:<18} {values['duration']['mean']:8.3f}s "
            f"{values['displacement']['mean'] * 1000:9.2f} mm "
            f"{values['displacement']['mean_fraction_of_goal'] * 100:7.1f}% "
            f"{values['peak_measured_force']:10.2f}N "
            f"{(coast if coast is not None else float('nan')):9.3f} "
            f"{((rest * 100) if rest is not None else float('nan')):7.0f}%"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
