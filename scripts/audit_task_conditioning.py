r"""Is the task condition worth conditioning on? An Oracle audit, no training.

Setting V1 feeds the model ``(d_goal, T_goal)`` but holds both constant across Dataset v1, so
they carry no information yet. Before a multi-goal experiment is worth running, one thing has
to be true: the goal must move the required force by **more than the width of the success
band**. If the 80 mm and 120 mm bands still contain the 100 mm optimum, a single-goal model
transfers for free and a multi-goal study measures nothing.

**One sweep, three readings.** Neither controller reads the goal -- the execution takes a force
and a duration (D004), the fixed-budget probe an amplitude and a budget (D044) -- and validity
bounds travel and drift rather than the target. So the same episodes serve every goal and the
goals differ only in the scoring. The three numbers are therefore three views of identical
physics, not three experiments, which is both cheaper and free of any between-run confound.

Frozen throughout: the probe (3.5 N / 0.3 s), ``T_goal`` = 1.5 s, the controllers, and the
in-distribution hidden-state ranges. Nothing is trained and no dataset is written.

Usage::

    python scripts/audit_task_conditioning.py --headless
    python scripts/audit_task_conditioning.py --headless --goals 0.08 0.10 0.12 --num-xi 32
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=32, help="Representative in-distribution states.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--goals", type=float, nargs="+", default=(0.08, 0.10, 0.12), help="Goal displacements (m)."
)
parser.add_argument("--force-low", type=float, default=0.25)
parser.add_argument(
    "--force-high",
    type=float,
    default=9.0,
    help="Swept above the frozen 6.5 N ceiling, so truncation can be told from infeasibility.",
)
parser.add_argument("--force-step", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=20260902)
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

from probe_drawer.analysis.sweep import force_grid  # noqa: E402
from probe_drawer.analysis.task_conditioning import summarise_task_conditioning  # noqa: E402
from probe_drawer.dataset import branch_order  # noqa: E402
from probe_drawer.dataset.sampling import representative_hidden_states  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    RECOMMENDED_EXECUTION_CFG,
    SEQUENTIAL_TRANSITION_STEPS,
    SETTING_V1_PROBE,
    SETTING_V1_PROBE_CFG,
    SETTING_V1_TASK,
    TRAINING_XI_RANGES,
)
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    states = representative_hidden_states(args_cli.num_xi, seed=args_cli.seed)
    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    task = SETTING_V1_TASK
    goals = list(args_cli.goals)
    region = OperatingRegionCfg()
    num_envs = min(args_cli.num_envs, len(states))
    if num_envs < len(states):
        raise ValueError(
            f"this audit runs one batch so that every goal shares one probe; asked for "
            f"{len(states)} states with num_envs={num_envs}."
        )

    print("\n" + "=" * 90)
    print(f"[cond] states  : {len(states)} representative in-distribution hidden states")
    print(f"[cond] probe   : {SETTING_V1_PROBE.as_kwargs()} (frozen), T_goal={task.duration:g} s")
    print(f"[cond] goals   : {[f'{g * 1000:g} mm' for g in goals]}, eps_d={task.displacement_tolerance * 1000:g} mm")
    print(f"[cond] F_peak  : {forces[0]:.2f}..{forces[-1]:.2f} N step {args_cli.force_step:g} "
          f"({len(forces)} values); the frozen action range is {list(task.peak_force_range)}")
    print(f"[cond] note    : one sweep, scored three ways -- no controller reads d_goal")

    system = PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=SETTING_V1_PROBE_CFG,
            execution=RECOMMENDED_EXECUTION_CFG,
        )
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()

    displacement = np.zeros((len(states), len(forces)), dtype=float)
    valid = np.zeros((len(states), len(forces)), dtype=bool)
    velocity = np.zeros((len(states), len(forces)), dtype=float)

    try:
        randomizer.apply(
            system.env,
            [
                DynamicsParameters(
                    name=f"xi{index:03d}",
                    drawer_mass=state["mass"],
                    joint_static_friction=state["static_friction"],
                    joint_dynamic_friction=state["dynamic_friction"],
                    joint_damping=state["damping"],
                )
                for index, state in enumerate(states)
            ],
        )
        system.reset()

        task_start = system.reader.drawer_position.clone()
        probe = system.probe.run_fixed_budget(**SETTING_V1_PROBE.as_kwargs())
        system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
        pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
        snapshot = capture_snapshot(system, label="task conditioning")

        for position, index in enumerate(branch_order("task-conditioning", len(forces))):
            force = forces[index]
            restore_snapshot(system, snapshot)
            result = system.execution.run(peak_force=force, duration=task.duration)
            validity = assess_validity(result, region, pre_execution_displacement=pre_execution)
            for env in range(len(states)):
                displacement[env, index] = validity.verdicts[env].metrics["final_displacement"]
                valid[env, index] = validity.verdicts[env].valid
                velocity[env, index] = float(result.final_velocity[env])
            if position % 25 == 0:
                print(f"[cond] force {position + 1}/{len(forces)} F={force:.2f} N "
                      f"({time.perf_counter() - started:.0f} s)")
    finally:
        system.close()

    rows = [
        {
            "hidden_state": dict(state),
            "probe_displacement": float(pre_execution[index]),
            "forces": list(forces),
            "displacement": displacement[index].tolist(),
            "valid": valid[index].tolist(),
            "terminal_velocity": velocity[index].tolist(),
        }
        for index, state in enumerate(states)
    ]
    summary = summarise_task_conditioning(
        rows,
        goals,
        task.displacement_tolerance,
        args_cli.force_step,
        tuple(task.peak_force_range),
    )

    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "task_conditioning_audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "phase": "task-conditioning feasibility audit",
                "probe": SETTING_V1_PROBE.as_dict(),
                "task": task.as_dict(),
                "xi_ranges": TRAINING_XI_RANGES.as_dict(),
                "seed": args_cli.seed,
                "operating_region": region.as_dict(),
                "git_commit": git_commit(),
                "environment": collect_environment_info().as_dict(),
                "elapsed_s": time.perf_counter() - started,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )

    print("[cond]")
    print(f"[cond] {'goal':>8} {'solvable':>10} {'band width':>22} {'required F':>22} "
          f"{'disconn':>8} {'at ceiling':>11}")
    for goal in goals:
        v = summary["per_goal"][goal]
        band, need = v["band_width"], v["required_force"]
        print(f"[cond] {goal * 1000:6.0f}mm {v['solvable']:>4}/{v['states']:<4} "
              f"{band['min']:.2f}-{band['max']:.2f} med {band['median']:.2f} N   "
              f"{need['min']:.2f}-{need['max']:.2f} med {need['median']:.2f} N   "
              f"{v['disconnected']:>7} {v['at_action_ceiling']:>10}")

    print("[cond]")
    print("[cond] how far the optimum moves between goals, per hidden state:")
    for label, v in summary["shift"].items():
        delta, ratio = v["abs_delta_centre"], v["shift_over_band_width"]
        signed = v["delta_centre"]
        print(f"[cond]   {label:>16}: |dF*| med {delta['median']:.3f} N (mean {delta['mean']:.3f}, "
              f"max {delta['max']:.3f}); signed mean {signed['mean']:+.3f} N; "
              f"shift/band med {ratio['median']:.2f}x")

    print("[cond]")
    middle = summary["transfer"][goals[len(goals) // 2]]["source_goal"]
    print(f"[cond] transferring the {middle * 1000:g} mm optimum unchanged:")
    for goal in goals:
        t = summary["transfer"][goal]
        print(f"[cond]   -> {goal * 1000:6.0f}mm : {t['reached']:>3}/{t['states_solvable_at_both']} "
              f"= {t['success_rate'] * 100:5.1f} %")
    print(f"[cond]")
    print(f"[cond] written : {output}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
