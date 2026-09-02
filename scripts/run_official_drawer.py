"""Phase 2 -- validate the official Franka drawer environment on this machine.

Runs ``Isaac-Open-Drawer-Franka-IK-Abs-v0`` through the full motion-driven sequence
(rest -> approach -> grasp -> settle -> open) using this project's
:class:`~probe_drawer.state_machines.DrawerGraspStateMachine`, and records what actually
happened: the phase timeline, the drawer joint trajectory, the empirically measured drawer
travel direction, and the arm configuration at the moment of grasp.

Two artefacts come out of this:

* ``outputs/logs/official_drawer_validation.json`` -- the validation record.
* ``configs/grasp_pose.yaml`` (with ``--export-grasp-pose``) -- the canonical grasped arm
  configuration, which the research environment resets straight into so that probe data is
  never polluted by approach/grasp variability (see spec section 39).

Usage::

    python scripts/run_official_drawer.py --num_envs 4 --headless
    python scripts/run_official_drawer.py --num_envs 1 --headless --deterministic-init --export-grasp-pose
    python scripts/run_official_drawer.py --num_envs 1 --video
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

OFFICIAL_ENV_ID = "Isaac-Open-Drawer-Franka-IK-Abs-v0"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--seconds", type=float, default=9.0, help="Simulated seconds to run.")
parser.add_argument(
    "--deterministic-init",
    action="store_true",
    help="Disable the official reset joint randomisation so the grasp configuration is reproducible.",
)
parser.add_argument(
    "--export-grasp-pose",
    action="store_true",
    help="Write configs/grasp_pose.yaml from the arm configuration at grasp. Requires --deterministic-init.",
)
parser.add_argument(
    "--grasp-offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Override the grasp waypoint offset from the handle frame (m). Used to calibrate the "
        "hand onto the handle centre: an off-centre hand makes the two fingers squeeze with "
        "different forces, which leaks a bias force onto the pull axis."
    ),
)
parser.add_argument("--video", action="store_true", help="Record outputs/videos/official_open_drawer.mp4.")
parser.add_argument(
    "--cabinet-x-offset",
    type=float,
    default=0.0,
    help=(
        "Shift the cabinet along the robot's +x (m) before grasping. Positive moves it away. "
        "Used to record a grasp for a repositioned cabinet, since the arm's shoulder-lift "
        "joint runs toward its stop as the drawer is pulled and the placement decides how "
        "much of that range a long pull consumes."
    ),
)
parser.add_argument(
    "--grasp-pose-output",
    type=str,
    default=None,
    help="Where to write the grasp record. Defaults to configs/grasp_pose.yaml.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.export_grasp_pose and not args_cli.deterministic_init:
    parser.error("--export-grasp-pose requires --deterministic-init, otherwise the exported pose is not reproducible.")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (import registers the environments)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

from probe_drawer.sensors import DrawerStateCfg, DrawerStateReader, PullAxis  # noqa: E402
from probe_drawer.state_machines import DrawerGraspStateMachine, GraspPhase, GraspStateMachineCfg  # noqa: E402
from probe_drawer.utils import collect_environment_info, enable_unbuffered_stdout, project_root  # noqa: E402

# The official cabinet sits at +0.8 m on the base x axis and opens back towards the robot,
# so the expected pull axis is -x. This is a *hypothesis* that the run below measures.
EXPECTED_PULL_AXIS = PullAxis(index=0, sign=-1.0)


def build_env() -> gym.Env:
    """Create the official IK-absolute drawer environment, lightly reconfigured."""
    env_cfg = parse_env_cfg(OFFICIAL_ENV_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    # The official episode length (8 s) is shorter than rest+approach+grasp+settle+open, so
    # the environment would reset mid-measurement.
    env_cfg.episode_length_s = max(args_cli.seconds + 2.0, env_cfg.episode_length_s)
    if args_cli.deterministic_init:
        env_cfg.events.reset_robot_joints = None
        env_cfg.observations.policy.enable_corruption = False
    if args_cli.cabinet_x_offset:
        position = env_cfg.scene.cabinet.init_state.pos
        env_cfg.scene.cabinet.init_state.pos = (
            position[0] + args_cli.cabinet_x_offset,
            position[1],
            position[2],
        )

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(OFFICIAL_ENV_ID, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        total_steps = int(args_cli.seconds / (env_cfg.sim.dt * env_cfg.decimation))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(project_root() / "outputs" / "videos"),
            name_prefix="official_open_drawer",
            step_trigger=lambda step: step == 0,
            video_length=total_steps,
            disable_logger=True,
        )
    return env


def main() -> None:
    enable_unbuffered_stdout()
    env = build_env()
    unwrapped = env.unwrapped
    step_dt = unwrapped.step_dt
    num_steps = int(args_cli.seconds / step_dt)

    env.reset()

    reader = DrawerStateReader(
        unwrapped, EXPECTED_PULL_AXIS, DrawerStateCfg(handle_contact_sensor=None)
    )
    sm_cfg = GraspStateMachineCfg(replicate_official_open=True)
    if args_cli.grasp_offset is not None:
        sm_cfg.grasp_offset = tuple(args_cli.grasp_offset)
    sm = DrawerGraspStateMachine(sm_cfg, step_dt, unwrapped.num_envs, unwrapped.device)

    reference_handle_pos_w = reader.handle_position_w.clone()
    reference_drawer_pos = reader.drawer_position.clone()

    timeline: list[dict] = []
    grasp_snapshot: dict | None = None
    actions = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
    actions[:, 3] = 1.0

    with torch.inference_mode():
        for step in range(num_steps):
            env.step(actions)

            tcp_pose = reader.tcp_pose
            handle_pose = reader.handle_pose
            actions = sm.compute(tcp_pose, handle_pose)

            timeline.append({
                "step": step,
                "t": round(step * step_dt, 4),
                "phase": [GraspPhase(int(p)).name for p in sm.phase],
                "drawer_position": [round(v, 6) for v in reader.drawer_position.tolist()],
                "drawer_velocity": [round(v, 6) for v in reader.drawer_velocity.tolist()],
                "tcp_position": [[round(c, 5) for c in row] for row in tcp_pose[:, :3].tolist()],
            })

            # Capture the arm configuration the instant every environment is done settling.
            if grasp_snapshot is None and bool((sm.phase >= int(GraspPhase.OPEN_DRAWER)).all()):
                robot = unwrapped.scene["robot"]
                fingers = reader.finger_joint_position[0].tolist()
                finger_names = [robot.joint_names[i] for i in reader.finger_joint_ids]
                grasp_snapshot = {
                    "captured_at_t": round(step * step_dt, 4),
                    "grasp_offset": list(sm_cfg.grasp_offset),
                    "finger_joint_names": finger_names,
                    "finger_joint_position": [round(v, 8) for v in fingers],
                    "finger_asymmetry": round(abs(fingers[0] - fingers[1]), 8),
                    "joint_names": list(robot.joint_names),
                    "joint_pos": [round(v, 8) for v in robot.data.joint_pos[0].tolist()],
                    "tcp_pose_env": [round(v, 6) for v in tcp_pose[0].tolist()],
                    "handle_pose_env": [round(v, 6) for v in handle_pose[0].tolist()],
                    "tcp_minus_handle_position": [
                        round(v, 6) for v in (tcp_pose[0, :3] - handle_pose[0, :3]).tolist()
                    ],
                    "drawer_position_at_grasp": round(float(reader.drawer_position[0]), 6),
                }

        displacement = reader.drawer_position - reference_drawer_pos
        measured_axis = reader.drawer_axis_world(displacement, reference_handle_pos_w)
        axis_ok, worst_angle = reader.verify_pull_axis(measured_axis)
        final_drawer = reader.drawer_position.clone()

    report = {
        "environment_id": OFFICIAL_ENV_ID,
        "environment_info": collect_environment_info().as_dict(),
        "run": {
            "num_envs": unwrapped.num_envs,
            "step_dt": step_dt,
            "num_steps": num_steps,
            "deterministic_init": args_cli.deterministic_init,
        },
        "geometry": {
            "expected_pull_axis": EXPECTED_PULL_AXIS.as_dict(),
            "measured_drawer_axis_world": [[round(c, 5) for c in row] for row in measured_axis.tolist()],
            "pull_axis_verified": axis_ok,
            "worst_angle_deg": round(worst_angle, 4),
        },
        "result": {
            "drawer_position_initial": [round(v, 6) for v in reference_drawer_pos.tolist()],
            "drawer_position_final": [round(v, 6) for v in final_drawer.tolist()],
            "drawer_displacement": [round(v, 6) for v in displacement.tolist()],
            "final_phase": [GraspPhase(int(p)).name for p in sm.phase],
        },
        "grasp_snapshot": grasp_snapshot,
        "timeline": timeline,
    }

    log_path = project_root() / "outputs" / "logs" / "official_drawer_validation.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 78)
    print(f"[official-drawer] env                : {OFFICIAL_ENV_ID}")
    print(f"[official-drawer] num_envs           : {unwrapped.num_envs}   step_dt={step_dt:.6f}s")
    print(f"[official-drawer] final phases       : {report['result']['final_phase']}")
    print(f"[official-drawer] drawer displacement: {report['result']['drawer_displacement']} m")
    print(f"[official-drawer] measured pull axis : {report['geometry']['measured_drawer_axis_world'][0]}")
    print(
        f"[official-drawer] pull axis {EXPECTED_PULL_AXIS.name} verified: "
        f"{axis_ok} (worst angle {worst_angle:.3f} deg)"
    )
    if grasp_snapshot is not None:
        print(f"[official-drawer] grasp captured at t = {grasp_snapshot['captured_at_t']}s")
        print(f"[official-drawer] grasp offset       : {grasp_snapshot['grasp_offset']}")
        print(f"[official-drawer] tcp - handle       : {grasp_snapshot['tcp_minus_handle_position']} m")
        print(f"[official-drawer] finger positions   : {grasp_snapshot['finger_joint_position']} m")
        print(f"[official-drawer] finger asymmetry   : {grasp_snapshot['finger_asymmetry']:.6f} m")
    print(f"[official-drawer] report             : {log_path}")
    print("=" * 78 + "\n")

    if args_cli.export_grasp_pose:
        if grasp_snapshot is None:
            raise RuntimeError("No grasp was completed within the run, so no grasp pose could be exported.")
        if not axis_ok:
            raise RuntimeError(
                f"Refusing to export a grasp pose: measured drawer axis is {worst_angle:.2f} deg away from "
                f"the configured pull axis {EXPECTED_PULL_AXIS.name}."
            )
        export_grasp_pose(grasp_snapshot, measured_axis)

    env.close()


def export_grasp_pose(snapshot: dict, measured_axis: torch.Tensor) -> None:
    """Persist the grasped arm configuration for the research environment to reset into."""
    path = (
        Path(args_cli.grasp_pose_output)
        if args_cli.grasp_pose_output
        else project_root() / "configs" / "grasp_pose.yaml"
    )
    if not path.is_absolute():
        path = project_root() / path
    payload = {
        "_generated_by": "scripts/run_official_drawer.py --deterministic-init --export-grasp-pose",
        "_source_environment": OFFICIAL_ENV_ID,
        "_cabinet_x_offset": args_cli.cabinet_x_offset,
        "_note": (
            "Arm configuration with the gripper closed on the top drawer handle, reached by the "
            "motion-driven approach state machine. The research environment resets into this "
            "configuration directly so probe data is not polluted by approach variability. "
            "finger_joint_position is the contact equilibrium of each finger; the research "
            "environment derives per-finger closed commands from it so the two fingers squeeze "
            "the handle equally (see ProbeDrawerEnvCfg.grip_squeeze)."
        ),
        "captured_at_t": snapshot["captured_at_t"],
        "grasp_offset": snapshot["grasp_offset"],
        "finger_joint_names": snapshot["finger_joint_names"],
        "finger_joint_position": snapshot["finger_joint_position"],
        "finger_asymmetry": snapshot["finger_asymmetry"],
        "joint_pos": dict(zip(snapshot["joint_names"], snapshot["joint_pos"], strict=True)),
        "tcp_pose_env": snapshot["tcp_pose_env"],
        "handle_pose_env": snapshot["handle_pose_env"],
        "tcp_minus_handle_position": snapshot["tcp_minus_handle_position"],
        "drawer_position_at_grasp": snapshot["drawer_position_at_grasp"],
        "measured_drawer_axis_world": [round(c, 6) for c in measured_axis[0].tolist()],
        "pull_axis": EXPECTED_PULL_AXIS.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    print(f"[official-drawer] grasp pose written : {path}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
