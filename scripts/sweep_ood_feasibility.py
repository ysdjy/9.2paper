"""Is the out-of-distribution range a solvable test domain under Setting V1?

Before any model is asked to generalise out of distribution, the domain has to be worth
asking about: if a large share of ``OOD_XI_RANGES`` has no succeeding force at all, then a low
OOD score would be measuring the *task* rather than the adaptation, and no amount of model work
would move it.

This is an Oracle sweep, not an evaluation. For each out-of-distribution hidden state it runs
the frozen Setting V1 probe and then sweeps ``F_peak`` from the post-probe snapshot, recording
whether any force reaches the goal, how wide the band is, which force is needed, and what went
invalid.

Two things it does deliberately:

**It samples states that are genuinely novel.** The OOD box *contains* the training box, so
about 13 % of its volume is in-distribution -- and those are the easy states, so counting them
as OOD would flatter the answer. ``sample_ood_hidden_states`` keeps only draws with at least
one axis outside the training range, and each state's novel axes are recorded so an unsolvable
one can be located rather than merely counted.

**It sweeps past the task's own force range.** ``SETTING_V1_TASK.peak_force_range`` tops out at
6.5 N. Sweeping only that far cannot distinguish "no force works" from "no force *in the range
we allow* works", and those have different consequences: the first is a physically infeasible
drawer, the second is a truncated action range. So the grid runs well beyond it and the report
separates the two.

Nothing here changes the setting, the dataset or any model.

Usage::

    python scripts/sweep_ood_feasibility.py --headless
    python scripts/sweep_ood_feasibility.py --headless --num-xi 32 --force-high 12.0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-xi", type=int, default=64, help="Out-of-distribution states to test.")
parser.add_argument("--num_envs", type=int, default=32, help="States in parallel.")
parser.add_argument("--force-low", type=float, default=0.25, help="Lowest F_peak swept (N).")
parser.add_argument(
    "--force-high",
    type=float,
    default=10.0,
    help=(
        "Highest F_peak swept (N). Deliberately above the task's 6.5 N ceiling, so that "
        "'unsolvable' can be told apart from 'needs more force than the task allows'."
    ),
)
parser.add_argument(
    "--force-step",
    type=float,
    default=0.10,
    help="F_peak spacing (N). 0.10 is what the Setting V1 band was measured at.",
)
parser.add_argument("--seed", type=int, default=20260903)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.analysis.probe_features import extract_features  # noqa: E402
from probe_drawer.analysis.sweep import force_grid  # noqa: E402
from probe_drawer.dataset import branch_order  # noqa: E402
from probe_drawer.dataset.sampling import (  # noqa: E402
    axes_outside,
    sample_ood_hidden_states,
    sampled_axis_values,
)
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, evaluate_execution  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    OOD_XI_RANGES,
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


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def build_system(num_envs: int) -> PullSystem:
    """The frozen Setting V1 wiring, unchanged."""
    return PullSystem.build(
        PullSystemCfg(
            num_envs=num_envs,
            device=args_cli.device,
            probe=SETTING_V1_PROBE_CFG,
            execution=RECOMMENDED_EXECUTION_CFG,
        )
    )


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    training, ood = TRAINING_XI_RANGES.as_dict(), OOD_XI_RANGES.as_dict()
    states = sample_ood_hidden_states(args_cli.num_xi, training, ood, seed=args_cli.seed)
    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    task = SETTING_V1_TASK
    criteria, region = task.criteria, OperatingRegionCfg()
    allowed_low, allowed_high = task.peak_force_range

    num_envs = min(args_cli.num_envs, len(states))
    grouped = batches(states, num_envs)
    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / "ood_feasibility.json"
    )

    print("\n" + "=" * 84)
    print(f"[ood] states     : {len(states)} out-of-distribution, in {len(grouped)} batch(es) of <= {num_envs}")
    print(f"[ood] probe      : fixed budget {SETTING_V1_PROBE.as_kwargs()} (frozen)")
    print(f"[ood] task       : d_goal={task.goal_displacement * 1000:g} mm T={task.duration:g} s "
          f"eps_d={task.displacement_tolerance * 1000:g} mm")
    print(f"[ood] F_peak     : {forces[0]:.2f} .. {forces[-1]:.2f} N step {args_cli.force_step:g} "
          f"({len(forces)} values); the task allows {allowed_low}-{allowed_high} N")
    print(f"[ood] episodes   : {len(grouped) * (1 + len(forces))}")

    system = build_system(num_envs)
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    rows: list[dict] = []

    try:
        for number, batch in enumerate(grouped, start=1):
            padded = batch + [batch[-1]] * (num_envs - len(batch))
            parameters = [
                DynamicsParameters(
                    name=f"ood{index:03d}",
                    drawer_mass=state["mass"],
                    joint_static_friction=state["static_friction"],
                    joint_dynamic_friction=state["dynamic_friction"],
                    joint_damping=state["damping"],
                )
                for index, state in enumerate(padded)
            ]
            randomizer.apply(system.env, parameters)
            system.reset()

            task_start = system.reader.drawer_position.clone()
            probe = system.probe.run_fixed_budget(**SETTING_V1_PROBE.as_kwargs())
            system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
            pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
            post_velocity = system.reader.drawer_velocity.cpu().numpy().copy()
            snapshot = capture_snapshot(system, label=f"ood batch {number}")

            reaching: list[list[float]] = [[] for _ in batch]
            invalid_reasons: list[Counter] = [Counter() for _ in batch]
            aborts = [0] * len(batch)
            best: list[tuple[float, float] | None] = [None] * len(batch)

            for position, index in enumerate(branch_order(f"ood-batch{number}", len(forces))):
                force = forces[index]
                restore_snapshot(system, snapshot)
                result = system.execution.run(peak_force=force, duration=task.duration)
                report = evaluate_execution(
                    result, criteria, region, pre_execution_displacement=pre_execution
                )
                for env in range(len(batch)):
                    verdict = report.verdicts[env]
                    if verdict.reach_success:
                        reaching[env].append(force)
                    for reason in verdict.invalid_reasons:
                        invalid_reasons[env][reason.value] += 1
                    aborts[env] += int(verdict.safety_aborted)
                    error = abs(verdict.displacement_error)
                    if best[env] is None or error < best[env][0]:
                        best[env] = (error, force)
                if position % 30 == 0:
                    print(
                        f"[ood] batch {number}/{len(grouped)} point {position + 1}/{len(forces)} "
                        f"F={force:.2f} solved so far {sum(1 for r in reaching if r)}/{len(batch)} "
                        f"({time.perf_counter() - started:.0f} s)"
                    )

            for env, state in enumerate(batch):
                hits = sorted(reaching[env])
                allowed = [force for force in hits if allowed_low <= force <= allowed_high]
                closest_error, closest_force = best[env] or (float("nan"), float("nan"))
                rows.append(
                    {
                        "hidden_state": dict(state),
                        "sampled_axes": sampled_axis_values(state),
                        "novel_axes": list(axes_outside(state, training)),
                        "probe_displacement": float(pre_execution[env]),
                        "probe_velocity": float(post_velocity[env]),
                        "probe_moved": bool(extract_features(probe, env).moved),
                        "reach_any_force": bool(hits),
                        "reach_within_task_range": bool(allowed),
                        "required_force": hits[0] if hits else None,
                        "band_low": hits[0] if hits else None,
                        "band_high": hits[-1] if hits else None,
                        "band_width": (hits[-1] - hits[0] + args_cli.force_step) if hits else 0.0,
                        "num_reaching_forces": len(hits),
                        "closest_force": closest_force,
                        "closest_position_error": closest_error,
                        "invalid_reasons": dict(invalid_reasons[env]),
                        "invalid_fraction": sum(invalid_reasons[env].values()) / len(forces),
                        "safety_aborts": aborts[env],
                    }
                )
    finally:
        system.close()

    payload = {
        "phase": "OOD feasibility pilot",
        "training_ranges": training,
        "ood_ranges": ood,
        "task": task.as_dict(),
        "probe": SETTING_V1_PROBE.as_dict(),
        "force_grid": list(forces),
        "operating_region": region.as_dict(),
        "seed": args_cli.seed,
        "git_commit": git_commit(),
        "environment": collect_environment_info().as_dict(),
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"[ood] written    : {output}")
    print(f"[ood] next       : python scripts/analyze_ood_feasibility.py --report {output}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
