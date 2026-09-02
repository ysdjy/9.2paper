"""Phase 5/6 -- run one full-duration force-driven execution and report what happened.

Usage::

    python scripts/test_execution_pull.py --headless
    python scripts/test_execution_pull.py --headless --peak-force 3 --preset hard
    python scripts/test_execution_pull.py --headless --peak-force 5 --duration 2 --d-goal 0.15
    python scripts/test_execution_pull.py --video --experiment-id execution_easy_video

``--d-goal`` is evaluated by this *script*, after the controller has returned.  The
controller never sees it: see ``docs/DECISIONS.md`` D004.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to execute simultaneously.")
parser.add_argument("--peak-force", type=float, default=5.0, help="Plateau pull force (N).")
parser.add_argument("--duration", type=float, default=2.0, help="Total execution time (s).")
parser.add_argument("--preset", type=str, default=None, help="Dynamics preset: easy, medium, hard, or nominal.")
parser.add_argument(
    "--d-goal",
    type=float,
    default=None,
    help="Optional task goal displacement (m), evaluated after the run purely for reporting.",
)
parser.add_argument("--epsilon", type=float, default=0.02, help="Tolerance used with --d-goal (m).")
parser.add_argument("--experiment-id", type=str, default="execution_default", help="Episode directory name.")
parser.add_argument("--video", action="store_true", help="Record the episode to outputs/videos/.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

import numpy as np  # noqa: E402

from probe_drawer.envs import DynamicsRandomizer, preset  # noqa: E402
from probe_drawer.logging import EpisodeLogger  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402


def main() -> None:
    enable_unbuffered_stdout()

    system = PullSystem.build(
        PullSystemCfg(
            num_envs=args_cli.num_envs,
            device=args_cli.device,
            video_folder=(project_root() / "outputs" / "videos") if args_cli.video else None,
            video_name_prefix=args_cli.experiment_id,
        )
    )
    system.verify_measured_force_available()

    randomizer = DynamicsRandomizer()
    system.reset()
    dynamics = randomizer.apply(system.env, preset(args_cli.preset or "nominal"))

    system.start_recording()
    result = system.execution.run(peak_force=args_cli.peak_force, duration=args_cli.duration)
    system.stop_recording()

    directory = EpisodeLogger().save(
        args_cli.experiment_id,
        result,
        dynamics_parameters=dynamics.as_dict(),
        notes={"script": "test_execution_pull.py", "d_goal": args_cli.d_goal, "epsilon": args_cli.epsilon},
    )

    history = result.history
    normalised = history.commanded_force[:, 0] / args_cli.peak_force
    print("\n" + "=" * 78)
    print(f"[execution] experiment          : {args_cli.experiment_id}")
    print(f"[execution] dynamics preset     : {dynamics.preset_name}")
    print(f"[execution] commanded           : peak_force={args_cli.peak_force} N  duration={args_cli.duration} s")
    print(f"[execution] steps executed      : {history.num_steps} at step_dt={system.step_dt:.6f} s")
    for env_index in range(result.num_envs):
        print(f"[execution] env {env_index} -> {json.dumps(result.summary(env_index))}")
    print(f"[execution] profile start/end   : {normalised[0]:.4f} -> {normalised[-1]:.4f} (normalised)")
    print(f"[execution] profile plateau max : {normalised.max():.4f}")
    print(f"[execution] peak lateral error  : {history.tcp_lateral_error.max():.5f} m")
    print(f"[execution] peak orient. error  : {np.degrees(history.tcp_orientation_error.max()):.3f} deg")
    if args_cli.d_goal is not None:
        error = np.abs(result.final_displacement - args_cli.d_goal)
        print(f"[execution] |d(T) - d_goal|      : {np.round(error, 5).tolist()} m (epsilon={args_cli.epsilon})")
        print(f"[execution] task success         : {(error <= args_cli.epsilon).tolist()}")
    print(f"[execution] episode written     : {directory}")
    print("=" * 78 + "\n")

    system.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
