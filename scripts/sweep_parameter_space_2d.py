"""Phase 12B -- sweep the two-dimensional execution parameter space.

Phase 11 fixed ``T`` and swept ``F_peak`` alone. In one dimension a hidden state's succeeding
forces turn out to be a contiguous interval whose midpoint works for 104 of 105 solvable
states, so predicting a whole landscape has no *structural* advantage over predicting one
number. This sweep opens the second axis so that question becomes answerable.

What varies and what does not
-----------------------------
Only ``T``. The normalised force profile ``phi(tau)``, ``tau = t/T``, is unchanged: the same
``rise_fraction``, the same ``fall_fraction = 0.35``, the same smoothstep shape. So a longer
``T`` stretches the identical curve over more time rather than reshaping it, and ``T`` stays a
single interpretable parameter (D041). Everything else -- the drawer, the OSC, the probe, the
8-step inference gap, the four-dimensional hidden state -- is the Phase 11 baseline untouched.

Structure
---------
Environments are the hidden-state axis, as in every previous sweep. One probe per batch, then
the post-probe state is snapshotted and every ``(F, T)`` candidate branches from it. So all
candidates of a hidden state answer the same counterfactual, which is what
``docs/COUNTERFACTUAL_BRANCHING.md`` validated -- with one new wrinkle: candidates now differ
in **duration**, so branches consume different numbers of steps. The snapshot restores
``episode_length_buf``, which is what keeps a long-``T`` sweep from tripping the 30 s episode
limit partway through.

The branch order is shuffled over the *joint* grid, so drift cannot align with either axis.

Usage::

    python scripts/sweep_parameter_space_2d.py --headless --stage coarse
    python scripts/sweep_parameter_space_2d.py --headless --stage fine \\
        --force-low 0.25 --force-high 4.5 --force-step 0.10 \\
        --duration-low 0.6 --duration-high 2.2 --duration-step 0.10
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("coarse", "fine"), default="coarse")
parser.add_argument("--num-xi", type=int, default=48, help="Representative hidden states.")
parser.add_argument("--num_envs", type=int, default=24, help="Hidden states in parallel.")
parser.add_argument("--force-low", type=float, default=0.15)
parser.add_argument("--force-high", type=float, default=6.0)
parser.add_argument("--force-step", type=float, default=0.25)
parser.add_argument("--duration-low", type=float, default=0.4)
parser.add_argument("--duration-high", type=float, default=2.5)
parser.add_argument("--duration-step", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=20260902)
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import time  # noqa: E402
from pathlib import Path  # noqa: E402

from probe_drawer.analysis.sweep import SweepDataset, SweepRecord, force_grid  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.dataset import branch_order  # noqa: E402
from probe_drawer.envs import DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.protocols import capture_snapshot, restore_snapshot  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import (  # noqa: E402
    collect_environment_info,
    enable_unbuffered_stdout,
    git_commit,
    project_root,
)
from probe_drawer.analysis.landscape_2d import representative_hidden_states  # noqa: E402


def build_system(num_envs: int) -> PullSystem:
    """The Phase 11 system, unchanged: no settle, so the probe's state survives."""
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


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def main() -> None:
    enable_unbuffered_stdout()
    started = time.perf_counter()

    forces = force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)
    durations = force_grid(args_cli.duration_low, args_cli.duration_high, args_cli.duration_step)
    grid = [(force, duration) for duration in durations for force in forces]
    states = representative_hidden_states(args_cli.num_xi, seed=args_cli.seed)
    num_envs = min(args_cli.num_envs, len(states))
    grouped = batches(states, num_envs)
    region = OperatingRegionCfg()

    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / f"landscape_2d_{args_cli.stage}.json"
    )

    print("\n" + "=" * 78)
    print(f"[2d] stage      : {args_cli.stage}")
    print(f"[2d] hidden     : {len(states)} representative states in {len(grouped)} batch(es) of <= {num_envs}")
    print(f"[2d] F (N)      : {forces[0]:.2f} .. {forces[-1]:.2f} step {args_cli.force_step} ({len(forces)} values)")
    print(f"[2d] T (s)      : {durations[0]:.2f} .. {durations[-1]:.2f} step {args_cli.duration_step} "
          f"({len(durations)} values)")
    print(f"[2d] grid points: {len(grid)} per hidden state -> {len(grid) * len(states)} episodes")
    print(f"[2d] task       : d_goal={MAIN_TASK.goal_displacement * 1000:g} mm "
          f"eps_d={MAIN_TASK.displacement_tolerance * 1000:g} mm eps_v={MAIN_TASK.velocity_tolerance:g} m/s")
    print(f"[2d] profile    : fall_fraction={RECOMMENDED_EXECUTION_CFG.fall_fraction} fixed; only T scales phi(t/T)")
    print(f"[2d] simulated  : {sum(d for _, d in grid) * len(states):.0f} s of execution")

    dataset = SweepDataset(
        metadata={
            "protocol": "sequential",
            "parameter_space": "2d: [F_peak, T]",
            "stage": args_cli.stage,
            "forces": list(forces),
            "durations": list(durations),
            "fall_fraction": RECOMMENDED_EXECUTION_CFG.fall_fraction,
            "transition_steps": SEQUENTIAL_TRANSITION_STEPS,
            "num_hidden_states": len(states),
            "hidden_state_seed": args_cli.seed,
            "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
            "probe_cfg": RECOMMENDED_PROBE_CFG.as_dict(),
            "operating_region": region.as_dict(),
            "main_task": MAIN_TASK.as_dict(),
            "git_commit": git_commit(),
            "environment": collect_environment_info().as_dict(),
        }
    )

    system = build_system(num_envs)
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    probe_task = RECOMMENDED_PROBE_TASK.as_kwargs()

    try:
        for batch_number, batch in enumerate(grouped, start=1):
            padded = batch + [batch[-1]] * (num_envs - len(batch))
            parameters = [
                DynamicsParameters(
                    drawer_mass=state["mass"],
                    joint_static_friction=state["static_friction"],
                    joint_dynamic_friction=state["dynamic_friction"],
                    joint_damping=state["damping"],
                    name=f"xi{index:03d}",
                )
                for index, state in enumerate(padded)
            ]
            randomizer.apply(system.env, parameters)
            system.reset()

            task_start = system.reader.drawer_position.clone()
            probe = system.probe.run(**probe_task)
            system.osc.coast(SEQUENTIAL_TRANSITION_STEPS)
            pre_execution = (system.reader.drawer_position - task_start).cpu().numpy().copy()
            snapshot = capture_snapshot(system, label=f"batch {batch_number}")

            # Shuffled over the joint grid, so branch drift aligns with neither F nor T.
            order = branch_order(f"{args_cli.stage}-batch{batch_number}", len(grid))
            for position, index in enumerate(order):
                force, duration = grid[index]
                restore_snapshot(system, snapshot)
                result = system.execution.run(peak_force=force, duration=duration)
                validity = assess_validity(result, region, pre_execution_displacement=pre_execution)
                dataset.extend(
                    SweepRecord.from_sequential_episode(
                        parameters[env_index],
                        duration,
                        _Episode(result, pre_execution, probe, force),
                        validity,
                        env_index,
                    )
                    for env_index in range(len(batch))
                )
                if position % 40 == 0:
                    print(
                        f"[2d] batch {batch_number}/{len(grouped)} point {position + 1}/{len(grid)} "
                        f"F={force:.2f} T={duration:.2f} "
                        f"valid={int(validity.valid[: len(batch)].sum())}/{len(batch)} "
                        f"({time.perf_counter() - started:.0f} s)"
                    )
    finally:
        system.close()

    path = dataset.save(output)
    elapsed = time.perf_counter() - started
    print("[2d]")
    print(f"[2d] episodes   : {len(dataset)} in {elapsed:.0f} s")
    print(f"[2d] valid      : {len(dataset.valid_records)} ({dataset.validity_rate() * 100:.1f} %)")
    print(f"[2d] invalid    : {dataset.invalid_reason_counts()}")
    print(f"[2d] written    : {path}")
    print(f"[2d] next       : python scripts/analyze_landscape_2d.py --dataset {path}")
    print("=" * 78 + "\n")


class _Episode:
    """The three fields ``SweepRecord.from_sequential_episode`` reads, for one branch.

    The real :class:`SequentialEpisode` describes a probe plus *one* execution. A branch is a
    probe plus one of many, so rather than fabricate a full episode per candidate this adapts
    the parts that exist: the shared probe, the shared pre-execution displacement, and this
    branch's own execution and force.
    """

    def __init__(self, execution, pre_execution, probe, force: float) -> None:
        import numpy as np  # noqa: PLC0415

        self.execution = execution
        self.probe = probe
        self.pre_execution_displacement = pre_execution
        self.probe_displacement = pre_execution
        self.total_displacement = pre_execution + execution.final_displacement
        self.peak_force = np.full(len(pre_execution), float(force))


if __name__ == "__main__":
    main()
    simulation_app.close()
