"""Phase 9H/9I -- sweep ``(xi, F_peak, T)`` and record what the drawer actually does.

No prediction, no model: this is the physics, measured. Hidden states occupy the parallel
environments and the loop runs over ``(F_peak, T)``, so every hidden state receives a
bit-identical command at every point and a difference between two rows can only come from
``xi``.

Each row records ``d(T)``, ``v(T)``, the peak velocity, the peak wrist force, the held-axis
drift, the termination reason, and whether the point is a usable operating point at all
(:mod:`probe_drawer.evaluation.operating_region`). Trajectories are not stored: any row is
reproducible from its own ``(xi, F_peak, T)``.

Two stages, coarse first:

``--stage coarse``
    A wide, sparse ``(F_peak, T)`` grid over a small hidden-state set, to find where the
    usable region is at all.
``--stage fine``
    A dense force grid at one duration over a full factorial hidden-state grid, which is
    the input to the Oracle landscape.

Usage::

    python scripts/sweep_execution_space.py --headless --stage coarse
    python scripts/sweep_execution_space.py --headless --stage fine --duration 2.0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("coarse", "fine"), default="coarse", help="Which sweep to run.")
parser.add_argument("--forces", type=float, nargs="*", default=None, help="Override the peak forces (N).")
parser.add_argument("--durations", type=float, nargs="*", default=None, help="Override the durations (s).")
parser.add_argument(
    "--duration", type=float, default=None, help="Shorthand for a single --durations value (s)."
)
parser.add_argument("--output", type=str, default=None, help="Override the dataset path.")
parser.add_argument(
    "--fall-fraction",
    type=float,
    default=None,
    help=(
        "Override the execution profile's ramp-down fraction. The default 0.1 leaves the "
        "drawer still moving at T for anything but a very short pull, so this is a design "
        "axis the sweep has to cover, not a tuning knob."
    ),
)
parser.add_argument(
    "--max-envs", type=int, default=18, help="Hidden states run per batch. Batches are sequential."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import time  # noqa: E402
from pathlib import Path  # noqa: E402

from probe_drawer.analysis.sweep import SweepDataset, SweepRecord, xi_grid  # noqa: E402
from probe_drawer.controllers import ExecutionControllerCfg  # noqa: E402
from probe_drawer.envs import PRESETS, DynamicsParameters, DynamicsRandomizer  # noqa: E402
from probe_drawer.evaluation import OperatingRegionCfg, assess_validity  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import collect_environment_info, enable_unbuffered_stdout, project_root  # noqa: E402

#: Coarse stage. Forces start below anything that moved a drawer in Phase 8 and stop where
#: the drawer was already saturating its travel; durations bracket the provisional 2 s.
COARSE_FORCES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
COARSE_DURATIONS = (0.5, 1.0, 1.5, 2.0, 3.0)

#: Fine stage, with every bound taken from the coarse sweep rather than guessed.
#:
#: Forces stop at 8 N because the coarse sweep found no usable point above it for any
#: duration of 1 s or more -- the drawer is already into its end stop. Static friction stops
#: at 3 N because only about 70 % of a commanded force reaches the drawer, so a 3 N breakaway
#: already needs more than 4 N of command; beyond that the low-force half of the grid is all
#: "no measurable motion". Durations are the two the coarse sweep found workable.
FINE_FORCES = tuple(round(1.0 + 0.5 * index, 2) for index in range(15))  # 1.0 .. 8.0 N
FINE_DURATIONS = (1.0, 1.5)
FINE_MASSES = (4.0, 8.0, 12.0)
FINE_STATIC_FRICTIONS = (0.5, 1.25, 2.0, 3.0)
FINE_FRICTION_RATIOS = (0.3, 0.65, 1.0)
FINE_DAMPINGS = (2.0, 6.0, 10.0)


def coarse_hidden_states() -> list[DynamicsParameters]:
    """The six presets plus corners that isolate one dimension at a time."""
    extras = [
        DynamicsParameters(4.0, 0.5, 0.5, 2.0, name="light_free"),
        DynamicsParameters(12.0, 0.5, 0.5, 2.0, name="heavy_free"),
        DynamicsParameters(8.0, 5.0, 5.0, 2.0, name="high_friction"),
        DynamicsParameters(8.0, 0.5, 0.5, 10.0, name="high_damping"),
        DynamicsParameters(8.0, 5.0, 1.0, 6.0, name="very_asymmetric"),
    ]
    return [*PRESETS.values(), *extras]


def batches(items: list, size: int) -> list[list]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def dataset_path(override: str | None, default_name: str) -> Path:
    """An explicit override, else ``outputs/logs/<default_name>``."""
    return Path(override) if override else project_root() / "outputs" / "logs" / default_name


def main() -> None:
    enable_unbuffered_stdout()

    if args_cli.stage == "coarse":
        hidden_states = coarse_hidden_states()
        forces = tuple(args_cli.forces) if args_cli.forces else COARSE_FORCES
        durations = tuple(args_cli.durations) if args_cli.durations else COARSE_DURATIONS
        default_name = "sweep_execution_coarse.json"
    else:
        hidden_states = xi_grid(FINE_MASSES, FINE_STATIC_FRICTIONS, FINE_FRICTION_RATIOS, FINE_DAMPINGS)
        forces = tuple(args_cli.forces) if args_cli.forces else FINE_FORCES
        durations = tuple(args_cli.durations) if args_cli.durations else FINE_DURATIONS
        default_name = "sweep_execution_fine.json"

    output = dataset_path(args_cli.output, default_name)
    region = OperatingRegionCfg()
    grouped = batches(hidden_states, args_cli.max_envs)
    total_points = len(hidden_states) * len(forces) * len(durations)

    if args_cli.duration is not None:
        durations = (args_cli.duration,)

    print("\n" + "=" * 78)
    print(f"[sweep] stage           : {args_cli.stage}")
    print(f"[sweep] hidden states   : {len(hidden_states)} in {len(grouped)} batch(es) of <= {args_cli.max_envs}")
    print(f"[sweep] peak forces (N) : {list(forces)}")
    print(f"[sweep] durations (s)   : {list(durations)}")
    print(f"[sweep] episodes to run : {len(forces) * len(durations) * len(grouped)} calls, {total_points} rows")
    print(f"[sweep] validity        : {region.as_dict()}")
    if args_cli.fall_fraction is not None:
        print(f"[sweep] fall fraction   : {args_cli.fall_fraction} (default is 0.1)")

    dataset = SweepDataset(
        metadata={
            "stage": args_cli.stage,
            "forces": list(forces),
            "durations": list(durations),
            "num_hidden_states": len(hidden_states),
            "operating_region": region.as_dict(),
            "fall_fraction": args_cli.fall_fraction,
            "environment": collect_environment_info().as_dict(),
        }
    )

    # The environment count is fixed when the scene is built, and building it costs tens of
    # seconds, so the system is built once at the batch size and short final batches are
    # padded with a repeat whose rows are then discarded.
    batch_size = len(grouped[0])
    started = time.perf_counter()
    execution_cfg = ExecutionControllerCfg()
    if args_cli.fall_fraction is not None:
        execution_cfg = ExecutionControllerCfg(fall_fraction=args_cli.fall_fraction)
    system = PullSystem.build(
        PullSystemCfg(num_envs=batch_size, device=args_cli.device, execution=execution_cfg)
    )
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer()
    try:
        for batch_index, batch in enumerate(grouped):
            padded = batch + [batch[-1]] * (batch_size - len(batch))
            for duration in durations:
                for peak_force in forces:
                    system.reset()
                    randomizer.apply(system.env, padded)
                    result = system.execution.run(peak_force=peak_force, duration=duration)
                    validity = assess_validity(result, region)
                    dataset.extend(
                        SweepRecord.from_execution(params, peak_force, duration, result, validity, index)
                        for index, params in enumerate(batch)
                    )
                    print(
                        f"[sweep] batch {batch_index + 1}/{len(grouped)}  T={duration:<4.2f} F={peak_force:<5.2f} "
                        f"valid={int(validity.valid[: len(batch)].sum())}/{len(batch)}  "
                        f"d(T) mm={[round(v * 1000, 1) for v in result.final_displacement[: len(batch)].tolist()]}"
                    )
    finally:
        system.close()

    elapsed = time.perf_counter() - started
    path = dataset.save(output)

    print("[sweep]")
    print(f"[sweep] rows collected  : {len(dataset)} in {elapsed:.1f} s")
    print(f"[sweep] valid rows      : {len(dataset.valid_records)} ({dataset.validity_rate() * 100:.1f} %)")
    print(f"[sweep] invalid reasons : {dataset.invalid_reason_counts()}")
    print(f"[sweep] dataset written : {path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
