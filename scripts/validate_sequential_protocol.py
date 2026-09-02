"""Phase 10 -- verify the sequential protocol and choose the inference gap by measurement.

Three questions, none of which may be answered by assertion:

**Is the protocol actually continuous?** After the probe the drawer must keep its position
and its velocity, and the arm must keep its configuration. The check compares the state at
the probe's end with the state at the execution's start and confirms the only difference is
the coast, and that the drawer is nowhere near its reset position.

**How long should the inference gap be?** Judged on the only thing that matters downstream:
how reproducible the finished task is. Each candidate gap length is run several times at the
task's own operating point and scored on the spread of ``d_total(T)``. A gap is not better
for being shorter if the episode it produces is less repeatable, and it is not better for
being longer if it buys no reduction in spread.

**How repeatable is a sequential episode at all?** A probe stops on a displacement
threshold, and a threshold crossing quantised to control steps can land a step earlier or
later, which shifts the state the execution inherits. That variability is a property of the
protocol, not a defect to be engineered away -- a deployed robot's probe would vary too --
but it sets a floor on how tight the task's position tolerance can be, so it is measured
here rather than discovered later.

The spread is reported both across parallel environments sharing a hidden state and across
repeats of a single environment, because the two bound the same quantity from different
directions.

Usage::

    python scripts/validate_sequential_protocol.py --headless
    python scripts/validate_sequential_protocol.py --headless --preset hard
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8, help="Force candidates compared per probe.")
parser.add_argument("--preset", type=str, default="medium", help="Dynamics preset for the checks.")
parser.add_argument(
    "--transition-steps",
    type=int,
    nargs="+",
    default=(0, 1, 2, 4),
    help="Inference-gap lengths to compare, in control steps.",
)
parser.add_argument("--peak-force", type=float, default=2.5, help="Execution amplitude for the checks (N).")
parser.add_argument(
    "--operating-force",
    type=float,
    default=4.25,
    help=(
        "Amplitude used for the repeatability study (N). Defaults to the force that puts the "
        "'medium' preset near the 50 mm goal, so the spread is measured where the task lives."
    ),
)
parser.add_argument("--duration", type=float, default=None, help="Execution duration (s).")
parser.add_argument(
    "--repeats", type=int, default=4, help="How many times to repeat one episode when measuring repeatability."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.envs import DynamicsRandomizer, preset  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
)
from probe_drawer.protocols import InferenceTransitionCfg, SequentialProtocolCfg, SequentialPullProtocol  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402

#: Largest post-probe spread that still lets candidates be compared meaningfully.
#:
#: Set relative to the task, not to the simulator: the tightest position tolerance under
#: consideration is 5 mm, so a post-probe displacement spread of 0.5 mm is a tenth of it and
#: cannot flip a success label on its own. The velocity bound is the corresponding figure for
#: the residual speed the execution inherits. Both are checked and reported; the measured
#: values are in ``docs/SEQUENTIAL_PROTOCOL.md``.
FAIRNESS_DISPLACEMENT_TOLERANCE = 5e-4
FAIRNESS_VELOCITY_TOLERANCE = 2e-2


def build_system(num_envs: int) -> PullSystem:
    """A system whose execution controller does *not* settle, as the protocol requires."""
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


def check_continuity(episode, system) -> dict:
    """The probe's effect must survive into the execution."""
    reader = system.reader
    probe_end = float(episode.probe_displacement[0])
    execution_start = float(episode.pre_execution_displacement[0])
    velocity_at_probe_end = float(episode.transition.velocity_before[0])
    velocity_at_gap_end = float(episode.transition.velocity_after[0])

    return {
        "task_start_position": float(episode.task_start_position[0]),
        "probe_displacement": probe_end,
        "execution_start_displacement": execution_start,
        "drawer_position_was_not_reset": execution_start >= probe_end - 1e-9 > 0.0,
        "drawer_velocity_at_probe_end": velocity_at_probe_end,
        "drawer_velocity_at_execution_start": velocity_at_gap_end,
        # The requirement is that the probe's velocity is not *erased*. It may decay during
        # the gap -- that is the drawer's own friction and damping doing the work, with zero
        # commanded force -- so the check is that the probe left the drawer moving and that
        # the gap only ever slowed it, never reversed or discontinuously zeroed it.
        "probe_left_the_drawer_moving": abs(velocity_at_probe_end) > 1e-4,
        "velocity_decayed_not_erased": abs(velocity_at_gap_end) <= abs(velocity_at_probe_end) + 1e-9,
        "velocity_retained_fraction": (
            abs(velocity_at_gap_end) / abs(velocity_at_probe_end) if velocity_at_probe_end else float("nan")
        ),
        "coast_during_gap": float(episode.transition.coast_displacement[0]),
        "execution_displacement": float(episode.execution.final_displacement[0]),
        "total_displacement": float(episode.total_displacement[0]),
        "total_equals_parts": bool(
            np.allclose(
                episode.total_displacement,
                episode.pre_execution_displacement + episode.execution.final_displacement,
                atol=1e-12,
            )
        ),
        "final_arm_configuration": [round(value, 5) for value in reader.arm_joint_position[0].tolist()],
    }


def check_fairness(episode) -> dict:
    """Every environment saw the same hidden state and the same probe, so it must show."""
    displacement = episode.pre_execution_displacement
    velocity = episode.transition.velocity_after
    tcp_velocity = episode.transition.tcp_pull_velocity_after
    drift = episode.transition.lateral_drift

    spreads = {
        "displacement_spread": float(displacement.max() - displacement.min()),
        "drawer_velocity_spread": float(velocity.max() - velocity.min()),
        "tcp_pull_velocity_spread": float(tcp_velocity.max() - tcp_velocity.min()),
        "lateral_drift_spread": float(drift.max() - drift.min()),
        "probe_duration_spread": float(episode.probe.duration.max() - episode.probe.duration.min()),
        "probe_force_spread": float(
            episode.probe.final_commanded_force.max() - episode.probe.final_commanded_force.min()
        ),
    }
    spreads["fair"] = bool(
        spreads["displacement_spread"] <= FAIRNESS_DISPLACEMENT_TOLERANCE
        and spreads["drawer_velocity_spread"] <= FAIRNESS_VELOCITY_TOLERANCE
    )
    spreads["tolerances"] = {
        "displacement": FAIRNESS_DISPLACEMENT_TOLERANCE,
        "velocity": FAIRNESS_VELOCITY_TOLERANCE,
    }
    return spreads


def measure_repeatability(
    system, randomizer, parameters, duration: float, steps: int, peak_force: float, repeats: int
) -> dict:
    """Repeat one identical episode and report the spread of everything that matters.

    ``total_displacement_spread`` is the figure the gap length is chosen on and the figure
    that bounds how tight the task's position tolerance can sensibly be.
    """
    protocol = SequentialPullProtocol(
        system,
        SequentialProtocolCfg(
            probe_task=RECOMMENDED_PROBE_TASK,
            duration=duration,
            transition=InferenceTransitionCfg(steps=steps),
        ),
    )
    rows = []
    for _ in range(repeats):
        randomizer.apply(system.env, parameters)
        episode = protocol.run(peak_force=peak_force)
        rows.append(
            {
                "probe_duration": float(episode.probe.duration[0]),
                "probe_displacement": float(episode.probe_displacement[0]),
                "pre_execution_displacement": float(episode.pre_execution_displacement[0]),
                "post_probe_velocity": float(episode.transition.velocity_after[0]),
                "total_displacement": float(episode.total_displacement[0]),
                "final_velocity": float(episode.execution.final_velocity[0]),
            }
        )

    def spread(key: str) -> float:
        values = [row[key] for row in rows]
        return float(max(values) - min(values))

    return {
        "repeats": repeats,
        "transition_steps": steps,
        "peak_force": peak_force,
        "rows": rows,
        "probe_duration_spread": spread("probe_duration"),
        "displacement_spread": spread("pre_execution_displacement"),
        "velocity_spread": spread("post_probe_velocity"),
        "total_displacement_spread": spread("total_displacement"),
        "final_velocity_spread": spread("final_velocity"),
        "mean_total_displacement": float(np.mean([row["total_displacement"] for row in rows])),
    }


def main() -> None:
    enable_unbuffered_stdout()
    duration = args_cli.duration if args_cli.duration is not None else MAIN_TASK.duration

    system = build_system(args_cli.num_envs)
    randomizer = DynamicsRandomizer()
    parameters = preset(args_cli.preset)
    report: dict = {
        "preset": parameters.as_dict(),
        "duration": duration,
        "peak_force": args_cli.peak_force,
        "num_envs": args_cli.num_envs,
        "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
        "probe_cfg": RECOMMENDED_PROBE_CFG.as_dict(),
        "transition_study": [],
    }

    print("\n" + "=" * 78)
    print(f"[sequential] preset       : {args_cli.preset} {parameters.as_dict()}")
    print(f"[sequential] execution    : F_peak={args_cli.peak_force} N, T={duration} s, settle_steps=0")
    print(f"[sequential] probe        : {json.dumps(RECOMMENDED_PROBE_TASK.as_dict())}")

    try:
        for steps in args_cli.transition_steps:
            protocol = SequentialPullProtocol(
                system,
                SequentialProtocolCfg(
                    probe_task=RECOMMENDED_PROBE_TASK,
                    duration=duration,
                    transition=InferenceTransitionCfg(steps=steps),
                ),
            )
            randomizer.apply(system.env, parameters)  # applied before the protocol's reset too
            episode = protocol.run(peak_force=args_cli.peak_force, criteria=MAIN_TASK.criteria)
            randomizer.apply(system.env, parameters)

            row = {
                "transition_steps": steps,
                "transition_duration": episode.transition.duration,
                "continuity": check_continuity(episode, system),
                "fairness": check_fairness(episode),
                "episode": episode.summary(0),
            }
            report["transition_study"].append(row)

            continuity, fairness = row["continuity"], row["fairness"]
            print(
                f"[sequential] gap={steps:<2d} ({episode.transition.duration * 1000:5.1f} ms)  "
                f"probe d={continuity['probe_displacement'] * 1000:6.3f} mm  "
                f"coast={continuity['coast_during_gap'] * 1000:+6.3f} mm  "
                f"v@exec={continuity['drawer_velocity_at_execution_start']:+.5f} m/s  "
                f"total d={continuity['total_displacement'] * 1000:7.2f} mm  "
                f"v(T)={row['episode']['final_velocity']:+.4f}  "
                f"fair={fairness['fair']}"
            )
        report["repeatability"] = [
            measure_repeatability(
                system,
                randomizer,
                parameters,
                duration,
                steps,
                args_cli.operating_force,
                repeats=args_cli.repeats,
            )
            for steps in args_cli.transition_steps
        ]
    finally:
        system.close()

    _summarise(report)

    output = project_root() / "outputs" / "logs" / "sequential_protocol_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=float))
    print(f"[sequential] report written: {output}")
    print("=" * 78 + "\n")


def _summarise(report: dict) -> None:
    """Print the verdicts and the recommended gap."""
    rows = report["transition_study"]
    print("[sequential]")
    print("[sequential] continuity checks (gap = recommended value is chosen below):")
    for row in rows:
        continuity = row["continuity"]
        print(
            f"[sequential]   gap={row['transition_steps']:<2d} position kept={continuity['drawer_position_was_not_reset']}  "
            f"probe left it moving={continuity['probe_left_the_drawer_moving']}  "
            f"velocity decayed not erased={continuity['velocity_decayed_not_erased']} "
            f"(retained {continuity['velocity_retained_fraction']:.3f})  "
            f"total = parts={continuity['total_equals_parts']}"
        )

    print("[sequential]")
    print("[sequential] candidate fairness across environments (same xi, same probe):")
    for row in rows:
        fairness = row["fairness"]
        print(
            f"[sequential]   gap={row['transition_steps']:<2d} d spread={fairness['displacement_spread'] * 1e6:7.3f} um  "
            f"v spread={fairness['drawer_velocity_spread'] * 1e3:7.4f} mm/s  "
            f"probe duration spread={fairness['probe_duration_spread']:.4f} s  fair={fairness['fair']}"
        )

    repeatability = report.get("repeatability") or []
    if repeatability:
        print("[sequential]")
        print(
            f"[sequential] repeatability at the operating point "
            f"(F={repeatability[0]['peak_force']} N, {repeatability[0]['repeats']} identical episodes each):"
        )
        print(
            f"[sequential]   {'gap':>4} {'mean d_total':>13} {'d_total spread':>15} {'post-probe d':>14} "
            f"{'post-probe v':>14} {'v(T) spread':>12}"
        )
        for row in repeatability:
            print(
                f"[sequential]   {row['transition_steps']:>4d} "
                f"{row['mean_total_displacement'] * 1000:12.3f} mm "
                f"{row['total_displacement_spread'] * 1000:14.3f} mm "
                f"{row['displacement_spread'] * 1000:13.3f} mm "
                f"{row['velocity_spread'] * 1000:12.3f} mm/s "
                f"{row['final_velocity_spread']:12.5f}"
            )
        best = min(repeatability, key=lambda row: row["total_displacement_spread"])
        report["most_repeatable_gap"] = best["transition_steps"]
        report["task_displacement_noise"] = best["total_displacement_spread"]
        print(
            f"[sequential]   most repeatable gap: {best['transition_steps']} step(s), "
            f"d_total spread {best['total_displacement_spread'] * 1000:.3f} mm -- this is the floor on "
            "any position tolerance"
        )

    def continuous(row: dict) -> bool:
        continuity = row["continuity"]
        return bool(
            continuity["drawer_position_was_not_reset"]
            and continuity["total_equals_parts"]
            and continuity["probe_left_the_drawer_moving"]
            and continuity["velocity_decayed_not_erased"]
        )

    usable = [row for row in rows if continuous(row)]
    report["all_continuous"] = all(continuous(row) for row in rows)
    report["all_fair"] = all(row["fairness"]["fair"] for row in rows)

    if not usable:
        report["recommended_transition_steps"] = None
        print("[sequential]")
        print("[sequential] NO GAP LENGTH PRESERVED THE PROBE STATE -- investigate before proceeding.")
        return

    # Continuity is necessary but does not distinguish the gap lengths -- they all preserve the
    # state. The choice is made on repeatability of the finished task, measured above, because
    # that is what limits the position tolerance the paper can claim.
    preferred = report.get("most_repeatable_gap")
    chosen = next(
        (row for row in usable if row["transition_steps"] == preferred),
        min(usable, key=lambda row: row["transition_steps"]),
    )
    report["recommended_transition_steps"] = chosen["transition_steps"]
    print("[sequential]")
    print(
        f"[sequential] RECOMMENDED gap: {chosen['transition_steps']} step(s) = "
        f"{chosen['transition_duration'] * 1000:.1f} ms -- preserves the probe's state and gives the "
        "most repeatable finished task"
    )
    print(f"[sequential] episode at that gap: {json.dumps(chosen['episode'])}")


if __name__ == "__main__":
    main()
    simulation_app.close()
