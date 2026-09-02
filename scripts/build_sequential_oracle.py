"""Phase 10 -- the authoritative Oracle: probe, then execute, with no reset.

Phase 9 measured the same landscape with a reset between the probe and the execution. That
was a preliminary physical verification and it is not the protocol the paper runs. Here every
row is a complete sequential episode:

* one reset, into the recorded grasp with the drawer closed;
* the standardised probe, which moves the drawer a little and leaves it moving;
* a fixed inference gap of zero pull force, no braking;
* the execution, ``F(t) = F_peak * phi(t/T)``, for the full duration;
* a label computed from ``d_total(T)``, measured from *before* the probe.

Structure. Environments are the hidden-state axis and the loop runs over ``F_peak``, so every
hidden state receives an identical command at each point. Each ``(xi, F_peak)`` point is its
own full episode, probe included: a candidate force is only meaningful together with the
probe that preceded it, and re-running the probe is what makes each row a genuine episode
rather than a shared starting state reused. The cost is about one extra second of simulated
time per row.

The force grid is finer than Phase 9's, because Phase 9's 0.25 N spacing was what forced the
position tolerance up to 15 mm rather than any physical limit.

Usage::

    python scripts/build_sequential_oracle.py --headless
    python scripts/build_sequential_oracle.py --headless --fall-fraction 0.30 \\
        --output outputs/logs/sequential_oracle_fall030.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--forces", type=float, nargs="*", default=None, help="Override the peak forces (N).")
parser.add_argument("--force-step", type=float, default=0.10, help="Peak-force grid spacing (N).")
parser.add_argument("--force-low", type=float, default=1.0, help="Lowest peak force (N).")
parser.add_argument("--force-high", type=float, default=5.0, help="Highest peak force (N).")
parser.add_argument("--duration", type=float, default=None, help="Execution duration (s).")
parser.add_argument("--fall-fraction", type=float, default=None, help="Execution ramp-down fraction.")
parser.add_argument(
    "--transition-steps", type=int, default=None, help="Inference-gap length. Defaults to the validated value."
)
parser.add_argument("--output", type=str, default=None, help="Override the dataset path.")
parser.add_argument("--max-envs", type=int, default=18, help="Hidden states per batch.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import time  # noqa: E402
from pathlib import Path  # noqa: E402

from probe_drawer.analysis.probe_features import extract_features  # noqa: E402
from probe_drawer.analysis.sweep import SweepDataset, SweepRecord, force_grid, xi_grid  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.envs import DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.experiment_plan import (  # noqa: E402
    MAIN_TASK,
    RECOMMENDED_EXECUTION_CFG,
    RECOMMENDED_PROBE_CFG,
    RECOMMENDED_PROBE_TASK,
    SEQUENTIAL_TRANSITION_STEPS,
)
from probe_drawer.protocols import InferenceTransitionCfg, SequentialProtocolCfg, SequentialPullProtocol  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import collect_environment_info, enable_unbuffered_stdout, project_root  # noqa: E402

#: The hidden-state grid, unchanged from Phase 9 so the two Oracles are directly comparable.
GRID = dict(
    masses=(4.0, 8.0, 12.0),
    static_frictions=(0.5, 1.25, 2.0, 3.0),
    friction_ratios=(0.3, 0.65, 1.0),
    dampings=(2.0, 6.0, 10.0),
)


def requested_forces() -> tuple[float, ...]:
    """The peak forces to sweep: the explicit list if given, otherwise the configured grid."""
    if args_cli.forces:
        return tuple(args_cli.forces)
    return force_grid(args_cli.force_low, args_cli.force_high, args_cli.force_step)


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def build_system(num_envs: int, fall_fraction: float) -> PullSystem:
    """A system whose execution does not settle -- the protocol requires it."""
    execution = ExecutionControllerCfg(
        rise_fraction=RECOMMENDED_EXECUTION_CFG.rise_fraction,
        fall_fraction=fall_fraction,
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


def main() -> None:
    enable_unbuffered_stdout()

    duration = args_cli.duration if args_cli.duration is not None else MAIN_TASK.duration
    fall_fraction = (
        args_cli.fall_fraction if args_cli.fall_fraction is not None else RECOMMENDED_EXECUTION_CFG.fall_fraction
    )
    transition_steps = (
        args_cli.transition_steps if args_cli.transition_steps is not None else SEQUENTIAL_TRANSITION_STEPS
    )
    forces = requested_forces()
    hidden_states = xi_grid(**GRID)
    grouped = batches(hidden_states, args_cli.max_envs)
    batch_size = len(grouped[0])
    region = OperatingRegionCfg()

    output = (
        Path(args_cli.output)
        if args_cli.output
        else project_root() / "outputs" / "logs" / f"sequential_oracle_fall{int(round(fall_fraction * 100)):03d}.json"
    )

    print("\n" + "=" * 78)
    print(f"[seq-oracle] protocol      : probe -> {transition_steps}-step gap -> execution, no reset")
    print(f"[seq-oracle] probe         : {RECOMMENDED_PROBE_TASK.as_dict()}")
    print(f"[seq-oracle] execution     : T={duration} s, fall_fraction={fall_fraction}, settle_steps=0")
    print(f"[seq-oracle] hidden states : {len(hidden_states)} in {len(grouped)} batch(es) of <= {batch_size}")
    print(f"[seq-oracle] forces (N)    : {forces[0]} .. {forces[-1]} step {args_cli.force_step} ({len(forces)} values)")
    print(f"[seq-oracle] rows to collect: {len(hidden_states) * len(forces)}")

    protocol_cfg = SequentialProtocolCfg(
        probe_task=RECOMMENDED_PROBE_TASK,
        duration=duration,
        transition=InferenceTransitionCfg(steps=transition_steps),
        operating_region=region,
    )
    dataset = SweepDataset(
        metadata={
            "protocol": "sequential",
            "forces": list(forces),
            "durations": [duration],
            "fall_fraction": fall_fraction,
            "transition_steps": transition_steps,
            "num_hidden_states": len(hidden_states),
            "probe_task": RECOMMENDED_PROBE_TASK.as_dict(),
            "probe_cfg": RECOMMENDED_PROBE_CFG.as_dict(),
            "operating_region": region.as_dict(),
            "environment": collect_environment_info().as_dict(),
        }
    )

    started = time.perf_counter()
    system = build_system(batch_size, fall_fraction)
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    protocol = SequentialPullProtocol(system, protocol_cfg)
    try:
        for batch_index, batch in enumerate(grouped):
            padded = batch + [batch[-1]] * (batch_size - len(batch))
            for peak_force in forces:
                randomizer.apply(system.env, padded)
                episode = protocol.run(peak_force=peak_force)
                validity = assess_validity(
                    episode.execution, region, pre_execution_displacement=episode.pre_execution_displacement
                )
                dataset.extend(
                    SweepRecord.from_sequential_episode(
                        parameters,
                        duration,
                        episode,
                        validity,
                        index,
                        probe_features=extract_features(episode.probe, index).as_dict(),
                    )
                    for index, parameters in enumerate(batch)
                )
                print(
                    f"[seq-oracle] batch {batch_index + 1}/{len(grouped)} F={peak_force:<5.2f} "
                    f"valid={int(validity.valid[: len(batch)].sum())}/{len(batch)} "
                    f"probe d={episode.probe_displacement[0] * 1000:5.2f} mm "
                    f"d_total mm={[round(v * 1000, 1) for v in episode.total_displacement[: len(batch)].tolist()]}"
                )
    finally:
        system.close()

    elapsed = time.perf_counter() - started
    path = dataset.save(output)
    print("[seq-oracle]")
    print(f"[seq-oracle] rows collected : {len(dataset)} in {elapsed:.1f} s")
    print(f"[seq-oracle] valid rows     : {len(dataset.valid_records)} ({dataset.validity_rate() * 100:.1f} %)")
    print(f"[seq-oracle] invalid reasons: {dataset.invalid_reason_counts()}")
    print(f"[seq-oracle] dataset written: {path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
