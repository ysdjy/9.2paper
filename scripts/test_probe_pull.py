"""Phase 4/6 -- run one standardised probe and report what the drawer actually did.

Usage::

    python scripts/test_probe_pull.py --headless
    python scripts/test_probe_pull.py --headless --max-velocity 0.005 --experiment-id probe_velocity_stop
    python scripts/test_probe_pull.py --headless --preset hard
    python scripts/test_probe_pull.py --video --experiment-id probe_easy_video

The episode is written to ``outputs/logs/<experiment-id>/`` as ``metadata.json`` plus
``trajectory.npz``; plot it with ``scripts/visualize_response.py``.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to probe simultaneously.")
parser.add_argument("--initial-force", type=float, default=2.0, help="Pull force at the start of the ramp (N).")
parser.add_argument("--max-force", type=float, default=10.0, help="Pull force at the end of the ramp (N).")
parser.add_argument("--target-displacement", type=float, default=0.005, help="Displacement that stops the probe (m).")
parser.add_argument("--max-velocity", type=float, default=0.05, help="Drawer speed that stops the probe (m/s).")
parser.add_argument("--ramp-duration", type=float, default=None, help="Override the probe ramp duration (s).")
parser.add_argument(
    "--max-probe-duration", type=float, default=None, help="Override the probe time budget (s)."
)
parser.add_argument("--preset", type=str, default=None, help="Dynamics preset: easy, medium, hard, or nominal.")
parser.add_argument("--experiment-id", type=str, default="probe_default", help="Episode directory name.")
parser.add_argument("--video", action="store_true", help="Record the episode to outputs/videos/.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402

from probe_drawer.controllers import ProbeControllerCfg  # noqa: E402
from probe_drawer.envs.dynamics_randomization import DynamicsRandomizer, preset  # noqa: E402
from probe_drawer.logging import EpisodeLogger  # noqa: E402
from probe_drawer.pull_system import PullSystem, PullSystemCfg  # noqa: E402
from probe_drawer.utils import enable_unbuffered_stdout, project_root  # noqa: E402


def build_probe_cfg() -> ProbeControllerCfg:
    """Probe character, with the two CLI overrides applied."""
    cfg = ProbeControllerCfg()
    if args_cli.ramp_duration is not None:
        cfg.ramp_duration = args_cli.ramp_duration
    if args_cli.max_probe_duration is not None:
        cfg.max_probe_duration = args_cli.max_probe_duration
    cfg.__post_init__()
    return cfg


def main() -> None:
    enable_unbuffered_stdout()

    system_cfg = PullSystemCfg(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        probe=build_probe_cfg(),
        video_folder=(project_root() / "outputs" / "videos") if args_cli.video else None,
        video_name_prefix=args_cli.experiment_id,
    )
    system = PullSystem.build(system_cfg)
    system.verify_measured_force_available()

    randomizer = DynamicsRandomizer()
    system.reset()
    dynamics = randomizer.apply(system.env, preset(args_cli.preset or "nominal"))

    system.start_recording()
    result = system.probe.run(
        initial_force=args_cli.initial_force,
        max_force=args_cli.max_force,
        target_displacement=args_cli.target_displacement,
        max_velocity=args_cli.max_velocity,
    )
    system.stop_recording()

    directory = EpisodeLogger().save(
        args_cli.experiment_id, result, dynamics_parameters=dynamics.as_dict(), notes={"script": "test_probe_pull.py"}
    )

    history = result.history
    print("\n" + "=" * 78)
    print(f"[probe] experiment          : {args_cli.experiment_id}")
    print(f"[probe] dynamics preset     : {dynamics.preset_name}")
    print(f"[probe] pull axis           : {system.pull_axis.name}")
    print(f"[probe] step_dt             : {system.step_dt:.6f} s")
    print(f"[probe] recorded steps      : {history.num_steps}")
    for env_index in range(result.num_envs):
        print(f"[probe] env {env_index} -> {json.dumps(result.summary(env_index))}")
    print(
        f"[probe] commanded force     : {history.commanded_force[0, 0]:.3f} N -> "
        f"{history.commanded_force[-1, 0]:.3f} N"
    )
    print(f"[probe] peak lateral error  : {history.tcp_lateral_error.max():.5f} m")
    print(f"[probe] peak orient. error  : {history.tcp_orientation_error.max() * 57.29578:.3f} deg")
    print(f"[probe] episode written     : {directory}")
    print("=" * 78 + "\n")

    system.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
