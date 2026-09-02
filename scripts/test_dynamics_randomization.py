"""Phase 7/8 -- prove the hidden dynamics really are hidden dynamics.

Runs the ``easy``, ``medium`` and ``hard`` presets **side by side in one simulation**
(one environment each, the controllers are vectorised) so that the only difference between
the three trajectories is ``xi = [mass, friction, damping]``: same grasp, same force
profile, same solver, same step.

Two checks:

* the requested parameters read back correctly out of PhysX, and the values that read back
  are the drawer's -- not some unrelated material's;
* the same ``(peak_force, duration)`` produces clearly different ``d(T)``, and the same
  standardised probe produces clearly different probe responses.

Usage::

    python scripts/test_dynamics_randomization.py --headless
    python scripts/test_dynamics_randomization.py --headless --peak-force 5 --duration 2
    python scripts/test_dynamics_randomization.py --headless --sample 4 --seed 0
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--peak-force", type=float, default=None, help="Execution plateau force (N).")
parser.add_argument("--duration", type=float, default=None, help="Execution duration (s).")
parser.add_argument("--initial-force", type=float, default=2.0, help="Probe ramp start (N).")
parser.add_argument("--max-force", type=float, default=10.0, help="Probe ramp end (N).")
parser.add_argument("--target-displacement", type=float, default=0.005, help="Probe displacement stop (m).")
parser.add_argument("--max-velocity", type=float, default=0.05, help="Probe velocity stop (m/s).")
parser.add_argument(
    "--separation-ratio",
    type=float,
    default=1.5,
    help="Minimum d(T) ratio required between consecutive presets for the check to pass.",
)
parser.add_argument(
    "--sample", type=int, default=0, help="Additionally sample this many random parameter sets and report them."
)
parser.add_argument("--seed", type=int, default=0, help="Seed for --sample.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.envs import (  # noqa: E402
    REFERENCE_DURATION,
    REFERENCE_PEAK_FORCE,
    DynamicsRandomizer,
    preset,
)
from probe_drawer.logging import EpisodeLogger  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout  # noqa: E402

PRESET_ORDER = ("easy", "medium", "hard")


def main() -> None:
    enable_unbuffered_stdout()

    peak_force = args_cli.peak_force if args_cli.peak_force is not None else REFERENCE_PEAK_FORCE
    duration = args_cli.duration if args_cli.duration is not None else REFERENCE_DURATION

    system = PullSystem.build(PullSystemCfg(num_envs=len(PRESET_ORDER), device=args_cli.device))
    system.verify_measured_force_available()
    randomizer = DynamicsRandomizer(seed=args_cli.seed)
    logger = EpisodeLogger()

    parameters = [preset(name) for name in PRESET_ORDER]

    print("\n" + "=" * 78)
    print(f"[dynamics] presets (env order)  : {PRESET_ORDER}")

    # -- Check 1: the parameters land where they are supposed to.
    system.reset()
    applied = randomizer.apply(system.env, parameters)
    print(f"[dynamics] requested            : {json.dumps([p.as_dict() for p in applied.requested])}")
    print(f"[dynamics] read back from PhysX : {json.dumps(applied.readback)}")
    print(f"[dynamics] handle mass          : {applied.handle_mass:.4f} kg")
    print(f"[dynamics] total moving mass    : {[round(m, 4) for m in applied.total_moving_mass]} kg")
    print(f"[dynamics] readback consistent  : {applied.consistent}")
    print(f"[dynamics] targets              : {json.dumps(applied.notes)}")

    # -- Check 2a: the same execution gives clearly different final displacements.
    execution = system.execution.run(peak_force=peak_force, duration=duration)
    logger.save(
        "dynamics_execution_presets",
        execution,
        dynamics_parameters=applied.as_dict(),
        notes={"script": "test_dynamics_randomization.py", "preset_order": list(PRESET_ORDER)},
    )
    displacement = execution.final_displacement
    print(f"\n[dynamics] execution            : peak_force={peak_force} N  duration={duration} s")
    for name, env_index in zip(PRESET_ORDER, range(len(PRESET_ORDER)), strict=True):
        print(f"[dynamics]   {name:7s} -> {json.dumps(execution.summary(env_index))}")
    print(f"[dynamics] d(T) (mm)            : {np.round(displacement * 1000, 3).tolist()}")
    print(f"[dynamics] peak |velocity| (m/s): {np.round(np.abs(execution.history.drawer_velocity).max(axis=0), 4).tolist()}")

    ratios = [float(displacement[i] / displacement[i + 1]) for i in range(len(PRESET_ORDER) - 1)]
    ordered = bool(np.all(np.diff(displacement) < 0))
    separated = all(r >= args_cli.separation_ratio for r in ratios)
    print(f"[dynamics] consecutive ratios   : {[round(r, 3) for r in ratios]} (need >= {args_cli.separation_ratio})")
    print(f"[dynamics] monotone easy>med>hard: {ordered}")
    print(f"[dynamics] separation check     : {'PASS' if ordered and separated else 'FAIL'}")

    # -- Check 2b: the same standardised probe gives clearly different probe responses.
    system.reset()
    randomizer.apply(system.env, parameters)
    probe = system.probe.run(
        initial_force=args_cli.initial_force,
        max_force=args_cli.max_force,
        target_displacement=args_cli.target_displacement,
        max_velocity=args_cli.max_velocity,
    )
    logger.save(
        "dynamics_probe_presets",
        probe,
        dynamics_parameters=applied.as_dict(),
        notes={"script": "test_dynamics_randomization.py", "preset_order": list(PRESET_ORDER)},
    )
    print("\n[dynamics] probe                : "
          f"{args_cli.initial_force} -> {args_cli.max_force} N, stop at "
          f"{args_cli.target_displacement} m / {args_cli.max_velocity} m/s")
    for name, env_index in zip(PRESET_ORDER, range(len(PRESET_ORDER)), strict=True):
        print(f"[dynamics]   {name:7s} -> {json.dumps(probe.summary(env_index))}")
    probe_separated = bool(np.all(np.diff(probe.duration) > 0))
    print(f"[dynamics] probe durations (s)  : {np.round(probe.duration, 4).tolist()}")
    print(f"[dynamics] probe distinguishes  : {'PASS' if probe_separated else 'FAIL'}")

    # -- Optional: report what random sampling produces.
    if args_cli.sample > 0:
        print(f"\n[dynamics] sampled (seed={args_cli.seed}) :")
        for params in randomizer.sample(args_cli.sample):
            print(f"[dynamics]   {json.dumps(params.as_dict())}")

    print("=" * 78 + "\n")
    system.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
