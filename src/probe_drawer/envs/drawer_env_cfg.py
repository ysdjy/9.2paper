"""The research environment: the official Franka cabinet scene, force-controlled.

This configuration *inherits* from Isaac Lab's official cabinet environment
(``isaaclab_tasks.manager_based.manipulation.cabinet``) and changes exactly four things:

1. the arm action becomes an :class:`~isaaclab.envs.mdp.OperationalSpaceControllerAction`
   in **hybrid mode** -- open-loop force control on the drawer's pull axis, pose/impedance
   hold on the remaining five task-space degrees of freedom;
2. the arm actuators are switched to pure effort control (zero PD, gravity disabled),
   which is what the official OSC reach environment does too;
3. a :class:`~isaaclab.sensors.ContactSensor` is added to the drawer handle, as a witness of
   grip load (the *measured* pull force itself comes from the wrist reaction wrench -- see
   :mod:`probe_drawer.sensors.drawer_state`);
4. the reset state is the recorded grasp configuration with a balanced grip, and every
   remaining source of episode-to-episode randomness is switched off, so that a difference
   between two episodes can only come from the hidden dynamics.

Nothing in Isaac Lab's own source tree is modified.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.cabinet.config.franka.joint_pos_env_cfg import (
    FrankaCabinetEnvCfg as OfficialFrankaCabinetEnvCfg,
)

from probe_drawer.envs.hybrid_pull_cfg import HybridPullControlCfg
from probe_drawer.envs.initialization import GraspConfiguration, load_grasp_configuration

__all__ = ["ProbeDrawerEnvCfg"]

#: TCP offset from ``panda_hand``. Matches the official ``ee_frame`` sensor's ``ee_tcp``
#: target (0.1034 m), so the pose the OSC controls is the same point the sensors report.
TCP_OFFSET = (0.0, 0.0, 0.1034)


@configclass
class ProbeDrawerEnvCfg(OfficialFrankaCabinetEnvCfg):
    """Force-driven drawer-pulling environment used by every experiment in this project.

    Action layout, in the order the action manager concatenates the terms::

        [0:3]   TCP position target in the robot base frame (m)
        [3:7]   TCP orientation target, quaternion (w, x, y, z)
        [7:10]  task-frame force target (N)   -- only the pull-axis element has any effect
        [10:13] task-frame torque target (N m) -- masked out entirely, always pass zeros
        [13]    binary gripper command (-1 closed, +1 open)

    Build the action vector with :class:`~probe_drawer.controllers.HybridPullOSC` rather
    than by hand.
    """

    hybrid_pull: HybridPullControlCfg = HybridPullControlCfg()

    #: Episode length. Long enough for the longest probe or execution plus settling; the
    #: pull controllers manage their own durations and never rely on the time-out.
    research_episode_length_s: float = 30.0

    #: Set to ``False`` to keep the official randomised reset pose, e.g. when comparing
    #: against the motion-driven baseline.
    reset_to_grasp_configuration: bool = True

    #: Shift of the cabinet along the robot's ``+x`` (m), added to the official ``0.8``.
    #:
    #: Positive moves the cabinet **away** from the robot. It exists because the arm's
    #: shoulder-lift joint runs monotonically toward its lower limit as the drawer is pulled
    #: -- ``panda_joint2`` reaches -1.76 rad, its stop, past 300 mm -- so where the cabinet
    #: stands decides how much of that joint's range a long pull consumes. Zero reproduces the
    #: official scene exactly, and every result before this knob existed used zero.
    #:
    #: Changing it invalidates the recorded grasp in ``configs/grasp_pose.yaml``, which is
    #: joint angles for the official placement. A non-zero offset therefore requires either
    #: a grasp recorded at *that* placement (:attr:`grasp_pose_path`) or
    #: ``reset_to_grasp_configuration = False``.
    cabinet_x_offset: float = 0.0

    #: Where to read the grasped arm configuration from. ``None`` uses the canonical
    #: ``configs/grasp_pose.yaml``, which was recorded at the official cabinet placement.
    grasp_pose_path: str | None = None

    #: Drawer drive stiffness (N/m). The official cabinet uses 10 N/m, which acts as a
    #: spring pulling the drawer shut and would be a fourth hidden parameter on top of
    #: xi = [mass, friction, damping]. Removed by default; see docs/DECISIONS.md D008.
    drawer_drive_stiffness: float = 0.0

    #: Drawer drive damping (N s/m). Matches the official cabinet, and is the nominal value
    #: DynamicsRandomizer varies around.
    drawer_drive_damping: float = 1.0

    #: Deflection commanded into each gripper finger, inside its recorded contact
    #: equilibrium (m). With the official finger stiffness of 2000 N/m, 0.006 m gives 12 N
    #: of grip force per finger, i.e. 24 N of normal load, which keeps the handle from
    #: slipping under the largest pull forces this study uses. Balanced by construction;
    #: see :meth:`~probe_drawer.envs.GraspConfiguration.closed_gripper_command`.
    grip_squeeze: float = 0.006

    #: Friction coefficients the official environment *randomises* at startup, pinned here
    #: to the midpoints of its ranges. Randomising them would make the contact conditions a
    #: fourth hidden variable, so that two episodes with identical xi would not be
    #: comparable. Order: (static, dynamic).
    robot_friction: tuple[float, float] = (1.025, 1.025)
    handle_friction: tuple[float, float] = (1.125, 1.375)

    def __post_init__(self) -> None:
        super().__post_init__()

        self._configure_robot_for_effort_control()
        self._configure_cabinet_placement()
        self._configure_drawer_drive()
        self._pin_contact_friction()
        self._configure_hybrid_pull_action()
        self._add_handle_contact_sensor()
        if self.reset_to_grasp_configuration:
            # Read once: both the reset pose and the balanced grip come from the same record.
            self._configure_research_reset(load_grasp_configuration(self.grasp_pose_path))

        self.episode_length_s = self.research_episode_length_s
        # Rewards and observations belong to the RL formulation, which this project does not
        # use; the pull controllers read state through DrawerStateReader instead. They are
        # left in place so the config stays a drop-in relative of the official environment.

    def _configure_robot_for_effort_control(self) -> None:
        """Zero the arm PD gains and disable gravity, as required for OSC effort control."""
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for group in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[group].stiffness = 0.0
            self.scene.robot.actuators[group].damping = 0.0
        self.scene.robot.spawn.rigid_props.disable_gravity = True

    def _pin_contact_friction(self) -> None:
        """Replace the official startup friction randomisation with fixed values."""
        for event, friction in (
            (self.events.robot_physics_material, self.robot_friction),
            (self.events.cabinet_physics_material, self.handle_friction),
        ):
            event.params["static_friction_range"] = (friction[0], friction[0])
            event.params["dynamic_friction_range"] = (friction[1], friction[1])
            event.params["num_buckets"] = 1

    def _configure_cabinet_placement(self) -> None:
        """Move the cabinet along the robot's ``x`` axis, if asked.

        Raises:
            ValueError: If a non-zero offset is combined with the *canonical* recorded grasp,
                which is joint angles measured at the official placement.
        """
        if self.cabinet_x_offset == 0.0:
            return
        if self.reset_to_grasp_configuration and self.grasp_pose_path is None:
            raise ValueError(
                f"cabinet_x_offset = {self.cabinet_x_offset} m needs either a grasp recorded at "
                "that placement (grasp_pose_path) or reset_to_grasp_configuration = False. The "
                "canonical grasp is joint angles measured at the official placement and would "
                "put the gripper somewhere other than the handle."
            )
        position = self.scene.cabinet.init_state.pos
        self.scene.cabinet.init_state.pos = (
            position[0] + self.cabinet_x_offset,
            position[1],
            position[2],
        )

    def _configure_drawer_drive(self) -> None:
        """Make the drawer a pure mass-friction-damper system by default."""
        drawers = self.scene.cabinet.actuators["drawers"]
        drawers.stiffness = self.drawer_drive_stiffness
        drawers.damping = self.drawer_drive_damping

    def _configure_hybrid_pull_action(self) -> None:
        """Replace the official joint-position arm action with the hybrid OSC action."""
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            body_offset=OperationalSpaceControllerActionCfg.OffsetCfg(pos=TCP_OFFSET),
            # task_frame_rel_path stays None, so the OSC task frame is the robot base
            # frame. The drawer's travel direction is axis-aligned with that frame (verified
            # empirically, see docs/OFFICIAL_BASELINE.md), which is what lets a single
            # coordinate axis be force-controlled.
            task_frame_rel_path=None,
            controller_cfg=self.hybrid_pull.to_osc_cfg(),
            nullspace_joint_pos_target="default",
        )

    def _add_handle_contact_sensor(self) -> None:
        """Add the drawer-handle contact sensor.

        It witnesses the grip load; it is *not* where the measured pull force comes from --
        net contact force excludes tangential friction, so it cannot see a pull transmitted
        through a friction grip (``docs/DECISIONS.md`` D006).
        """
        self.scene.cabinet.spawn.activate_contact_sensors = True
        self.scene.handle_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
        )

    def _configure_research_reset(self, grasp: GraspConfiguration) -> None:
        """Reset into the recorded grasp, with a balanced grip and no randomisation.

        The per-finger close command is only meaningful relative to this record's contact
        equilibrium, so the reset pose and the grip are configured together.
        """
        grasp.apply_to(self.scene.robot)
        self.actions.gripper_action.close_command_expr = grasp.closed_gripper_command(self.grip_squeeze)
        # Randomising the reset pose would move the fingers off the handle.
        self.events.reset_robot_joints = None
        self.observations.policy.enable_corruption = False
