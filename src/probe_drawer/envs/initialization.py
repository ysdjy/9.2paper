"""Research initialisation: start every episode from a settled grasp on the handle.

The research question is about *pulling*, not about reaching and grasping.  Letting the
approach run every episode would inject grasp-pose variability into probe measurements, so
this project instead resets the arm directly into a known grasped configuration.

That configuration is not invented: it is recorded from the official motion-driven
approach by ``scripts/run_official_drawer.py --deterministic-init --export-grasp-pose``
and stored in ``configs/grasp_pose.yaml``.  This module loads it and applies it to an
``ArticulationCfg`` so Isaac Lab's own ``reset_scene_to_default`` event puts the arm there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from pathlib import Path

import yaml

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.assets import ArticulationCfg

from probe_drawer.utils import project_root

__all__ = ["GraspConfiguration", "default_grasp_pose_path", "load_grasp_configuration"]


def default_grasp_pose_path() -> Path:
    """Location of the recorded grasp configuration inside this repository."""
    return project_root() / "configs" / "grasp_pose.yaml"


@dataclass(frozen=True)
class GraspConfiguration:
    """A recorded arm configuration with the gripper closed on the drawer handle.

    Attributes:
        joint_pos: Joint name -> position (rad for the arm, m for the fingers).
        finger_equilibrium: Finger joint name -> contact equilibrium position (m). The two
            values differ because the hand does not sit exactly on the handle centre, and
            that difference is what :meth:`closed_gripper_command` corrects for.
        tcp_pose_env: TCP pose in the environment frame when the record was taken,
            ``[x, y, z, qw, qx, qy, qz]``.
        handle_pose_env: Handle frame pose in the environment frame, same layout.
        source_environment: The environment ID the configuration was recorded in.
        path: Where it was loaded from.
    """

    joint_pos: dict[str, float]
    finger_equilibrium: dict[str, float]
    tcp_pose_env: list[float]
    handle_pose_env: list[float]
    source_environment: str
    path: Path

    def closed_gripper_command(self, squeeze: float) -> dict[str, float]:
        """Per-finger closed position command that loads both fingers equally.

        The official environment commands both fingers to 0 m, so each pushes with
        ``stiffness * equilibrium``.  Because the two equilibria differ (measured: 7.5 mm
        and 15.2 mm), so do the two forces -- by roughly 15 N -- and the imbalance leaks a
        steady bias force of about 0.7 N onto the pull axis, which is the same order as the
        force a 2 N probe actually delivers to the drawer.  Commanding each finger
        ``squeeze`` metres *inside* its own equilibrium instead makes both deflections, and
        hence both forces, equal, and makes the grip force an explicit parameter:
        ``force_per_finger = finger_stiffness * squeeze``.

        Args:
            squeeze: Deflection commanded into each finger (m). Must be positive and no
                larger than the smaller equilibrium, so neither command opens the gripper.

        Raises:
            ValueError: If ``squeeze`` is not positive, or exceeds the smallest equilibrium.
        """
        smallest = min(self.finger_equilibrium.values())
        if squeeze <= 0.0:
            raise ValueError(f"squeeze must be > 0 m, got {squeeze}.")
        if squeeze > smallest:
            raise ValueError(
                f"squeeze ({squeeze} m) exceeds the smallest recorded finger equilibrium "
                f"({smallest} m); the command would try to open that finger."
            )
        return {name: value - squeeze for name, value in self.finger_equilibrium.items()}

    def apply_to(self, robot_cfg: ArticulationCfg) -> None:
        """Make this configuration the articulation's default (and hence reset) state.

        Mutates ``robot_cfg.init_state.joint_pos`` in place.  Joint names are used
        verbatim, so a mismatch with the robot's actual joints surfaces as an Isaac Lab
        error at scene construction rather than as a silently wrong pose.
        """
        robot_cfg.init_state.joint_pos = dict(self.joint_pos)


def load_grasp_configuration(path: Path | str | None = None) -> GraspConfiguration:
    """Load the recorded grasp configuration.

    Args:
        path: File to read. Defaults to :func:`default_grasp_pose_path`.

    Raises:
        FileNotFoundError: If the file does not exist, with the command that regenerates it.
    """
    path = Path(path) if path is not None else default_grasp_pose_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"No grasp configuration at {path}. Regenerate it with:\n"
            "    python scripts/run_official_drawer.py --num_envs 1 --headless "
            "--deterministic-init --export-grasp-pose"
        )
    payload = yaml.safe_load(path.read_text())
    finger_names = [str(n) for n in payload["finger_joint_names"]]
    finger_values = [float(v) for v in payload["finger_joint_position"]]
    return GraspConfiguration(
        joint_pos={str(k): float(v) for k, v in payload["joint_pos"].items()},
        finger_equilibrium=dict(zip(finger_names, finger_values, strict=True)),
        tcp_pose_env=[float(v) for v in payload["tcp_pose_env"]],
        handle_pose_env=[float(v) for v in payload["handle_pose_env"]],
        source_environment=str(payload.get("_source_environment", "unknown")),
        path=path,
    )
