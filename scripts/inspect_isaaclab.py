"""Phase 1 -- report what Isaac Sim / Isaac Lab is *actually* installed on this machine.

Launches Isaac Sim headless (required before ``isaaclab_tasks`` can be imported), then
reports versions, the resolved Isaac Lab source paths, and which of the official drawer
environments are registered.  The report is printed and written to
``outputs/logs/isaaclab_inspection.json``.

Usage::

    python scripts/inspect_isaaclab.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=str, default=None, help="Path of the JSON report. Defaults to outputs/logs/.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import inspect  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402  (import registers the environments)

from probe_drawer.utils import collect_environment_info, enable_unbuffered_stdout, project_root  # noqa: E402
from probe_drawer.utils.isaaclab_compat import OFFICIAL_DRAWER_ENV_IDS  # noqa: E402


def _source_file(obj) -> str | None:
    try:
        return inspect.getsourcefile(obj)
    except (TypeError, OSError):
        return None


def collect_official_baseline() -> dict:
    """Resolve the on-disk locations and key config values of the official drawer stack."""
    from isaaclab_tasks.manager_based.manipulation.cabinet import cabinet_env_cfg
    from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka import ik_abs_env_cfg, joint_pos_env_cfg

    cfg = ik_abs_env_cfg.FrankaCabinetEnvCfg()
    scene = cfg.scene
    baseline: dict = {
        "source_files": {
            "cabinet_env_cfg": _source_file(cabinet_env_cfg),
            "ik_abs_env_cfg": _source_file(ik_abs_env_cfg),
            "joint_pos_env_cfg": _source_file(joint_pos_env_cfg),
        },
        "cabinet_usd": scene.cabinet.spawn.usd_path,
        "cabinet_init_pos": scene.cabinet.init_state.pos,
        "cabinet_init_rot_wxyz": scene.cabinet.init_state.rot,
        "cabinet_joint_names": sorted(scene.cabinet.init_state.joint_pos),
        "cabinet_actuators": {
            name: {
                "joint_names_expr": act.joint_names_expr,
                "stiffness": act.stiffness,
                "damping": act.damping,
                "effort_limit_sim": act.effort_limit_sim,
            }
            for name, act in scene.cabinet.actuators.items()
        },
        "cabinet_frame_source": scene.cabinet_frame.prim_path,
        "cabinet_frame_targets": [
            {"name": f.name, "prim_path": f.prim_path, "offset_pos": f.offset.pos, "offset_rot_wxyz": f.offset.rot}
            for f in scene.cabinet_frame.target_frames
        ],
        "ee_frame_source": scene.ee_frame.prim_path,
        "ee_frame_targets": [
            {"name": f.name, "prim_path": f.prim_path, "offset_pos": f.offset.pos} for f in scene.ee_frame.target_frames
        ],
        "robot_usd": scene.robot.spawn.usd_path,
        "robot_actuators": {
            name: {"joint_names_expr": act.joint_names_expr, "stiffness": act.stiffness, "damping": act.damping}
            for name, act in scene.robot.actuators.items()
        },
        "arm_action": {
            "class": type(cfg.actions.arm_action).__name__,
            "body_name": cfg.actions.arm_action.body_name,
            "joint_names": cfg.actions.arm_action.joint_names,
            "controller": repr(cfg.actions.arm_action.controller),
            "body_offset_pos": cfg.actions.arm_action.body_offset.pos,
        },
        "gripper_action": {
            "class": type(cfg.actions.gripper_action).__name__,
            "joint_names": cfg.actions.gripper_action.joint_names,
            "open_command_expr": cfg.actions.gripper_action.open_command_expr,
            "close_command_expr": cfg.actions.gripper_action.close_command_expr,
        },
        "sim": {"dt": cfg.sim.dt, "decimation": cfg.decimation, "episode_length_s": cfg.episode_length_s},
    }
    return baseline


def main() -> None:
    enable_unbuffered_stdout()
    info = collect_environment_info(check_gym_registry=False)
    known = set(gym.registry.keys())
    info.registered_drawer_envs = {env_id: env_id in known for env_id in OFFICIAL_DRAWER_ENV_IDS}

    report = {"environment": info.as_dict(), "official_baseline": collect_official_baseline()}

    output = Path(args_cli.output) if args_cli.output else project_root() / "outputs" / "logs" / "isaaclab_inspection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 78)
    print(json.dumps(report, indent=2, default=str))
    print("=" * 78)
    print(f"\n[inspect_isaaclab] report written to: {output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
