"""Read-only views onto the quantities a pull controller needs from the simulation.

This module owns two things and nothing else:

* :class:`PullAxis` -- the definition of *which direction "opening the drawer" is*, as a
  signed coordinate axis of the robot base frame.  It lives here rather than in
  :mod:`probe_drawer.envs` so that both the environment configuration and the controllers
  can depend on it without depending on each other.
* :class:`DrawerStateReader` -- a thin accessor that pulls drawer, end-effector and joint
  state out of a running ``ManagerBasedRLEnv``.  It never writes to the simulation and
  never decides anything.

Force provenance
----------------
:attr:`DrawerStateReader.measured_pull_force` is the pull-axis component of the **joint
reaction wrench at the robot's wrist** -- the wrench PhysX reports transmitted from
``panda_link7`` into ``panda_hand``.  That is the same quantity a real Franka's wrist
force/torque sensor measures, and it is a genuine measurement: it is never derived from the
commanded force.

The obvious alternative, a :class:`~isaaclab.sensors.ContactSensor` on the drawer handle,
was measured and rejected: its ``net_forces_w`` tracks only the grip's normal load and
stays near 0.23 N whether the commanded pull is 4 N or 12 N, because the pull is
transmitted through *tangential* finger friction that the net-contact-force report does not
include.  The handle contact force is still recorded as
:attr:`DrawerStateReader.handle_contact_force_w` because it is a useful witness of grip
load, but nothing decides anything from it.  See ``docs/DECISIONS.md`` D006.

The wrist wrench includes the small force needed to accelerate the hand and fingers
(roughly 0.9 kg, so about 0.2 N at typical pull accelerations); it is therefore an estimate
of the force delivered to the drawer, not an exact figure.

Velocity provenance
-------------------
The gripper's contact with the handle chatters at roughly half the 60 Hz control rate, and
sampling ``Articulation.data.joint_vel`` once per control step aliases that chatter badly
enough to invert the sign of the mean (measured: reported mean -0.0076 m/s while the drawer
was demonstrably opening at +0.0073 m/s).  :attr:`DrawerStateReader.drawer_velocity`
therefore reports a short moving average of the position finite difference across control
steps, which is unbiased at this rate.  The raw PhysX reading stays available as
:attr:`DrawerStateReader.drawer_joint_velocity_raw` and is logged alongside it, so the
substitution is visible rather than hidden.  See ``docs/DECISIONS.md`` D009.

Because the finite difference needs one sample per control step,
:meth:`DrawerStateReader.update` must be called exactly once after every ``env.step``.
Everything in this project steps through :meth:`~probe_drawer.controllers.HybridPullOSC.step`,
which does that for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

from probe_drawer.sensors.causal_derivative import CausalDerivative
from probe_drawer.sensors.pull_axis import PullAxis

if TYPE_CHECKING:  # pragma: no cover - these need the Isaac Sim app at runtime
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer

__all__ = ["DrawerStateCfg", "DrawerStateReader"]


@dataclass
class DrawerStateCfg:
    """Names of the scene entities a :class:`DrawerStateReader` reads from.

    The defaults match this project's environment (and, for the shared entities, Isaac
    Lab's official cabinet scene).
    """

    robot_asset: str = "robot"
    cabinet_asset: str = "cabinet"
    ee_frame_sensor: str = "ee_frame"
    cabinet_frame_sensor: str = "cabinet_frame"
    handle_contact_sensor: str | None = "handle_contact"
    drawer_joint_name: str = "drawer_top_joint"
    handle_body_name: str = "drawer_handle_top"
    arm_joint_expr: str = "panda_joint.*"
    finger_joint_expr: str = "panda_finger_joint.*"
    ee_body_name: str = "panda_hand"
    ee_parent_body_name: str = "panda_link7"
    """Parent of :attr:`ee_body_name`. The wrist reaction wrench is reported in its frame."""
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    """TCP offset from :attr:`ee_body_name`, matching the official ``ee_frame`` sensor."""
    drawer_body_name: str = "drawer_top"
    """Body whose mass, together with the handle's, is the drawer's moving mass."""
    velocity_filter_window: int = 2
    """Control steps averaged when estimating the drawer velocity.

    Two exactly cancels the two-step contact chatter observed in this scene; see this
    module's *Velocity provenance* note.
    """
    acceleration_filter_window: int = 4
    """Control steps averaged when estimating any acceleration.

    An acceleration is a second difference of position and therefore noisier than a
    velocity, so it gets a longer window: four steps, i.e. 25 ms of lag at 60 Hz.
    """


class DrawerStateReader:
    """Accessor for drawer / end-effector / joint state of a running cabinet environment.

    Args:
        env: A running manager-based environment containing a Franka and the cabinet.
        pull_axis: The drawer's opening direction in the robot base frame.
        cfg: Scene entity names. Defaults are correct for this project's environment.

    Raises:
        ValueError: If the robot base frame is rotated with respect to the environment
            frame, which would invalidate the base-frame pull-axis convention.
    """

    def __init__(self, env: ManagerBasedRLEnv, pull_axis: PullAxis, cfg: DrawerStateCfg | None = None) -> None:
        self.env = env
        self.pull_axis = pull_axis
        self.cfg = cfg or DrawerStateCfg()

        self._robot: Articulation = env.scene[self.cfg.robot_asset]
        self._cabinet: Articulation = env.scene[self.cfg.cabinet_asset]
        self._ee_frame: FrameTransformer = env.scene[self.cfg.ee_frame_sensor]
        self._cabinet_frame: FrameTransformer = env.scene[self.cfg.cabinet_frame_sensor]

        self._contact: ContactSensor | None = None
        if self.cfg.handle_contact_sensor is not None and self.cfg.handle_contact_sensor in env.scene.sensors:
            self._contact = env.scene[self.cfg.handle_contact_sensor]

        self._drawer_joint_idx = self._cabinet.find_joints(self.cfg.drawer_joint_name)[0][0]
        self._handle_body_idx = self._cabinet.find_bodies(self.cfg.handle_body_name)[0][0]
        self._drawer_body_idx = self._cabinet.find_bodies(self.cfg.drawer_body_name)[0][0]
        self._arm_joint_ids = self._robot.find_joints(self.cfg.arm_joint_expr)[0]
        self._finger_joint_ids = self._robot.find_joints(self.cfg.finger_joint_expr)[0]
        self._ee_body_idx = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self._ee_parent_body_idx = self._robot.find_bodies(self.cfg.ee_parent_body_name)[0][0]

        self._device = env.device
        self._direction_w = pull_axis.direction(self._device)
        self._ee_offset = torch.tensor(self.cfg.ee_offset, device=self._device).repeat(env.num_envs, 1)

        self._step_dt = float(env.step_dt)
        # One estimator per differentiated signal. Constructing them validates the windows.
        self._drawer_velocity_estimator = CausalDerivative(self._step_dt, self.cfg.velocity_filter_window)
        self._drawer_acceleration_estimator = CausalDerivative(self._step_dt, self.cfg.acceleration_filter_window)
        self._tcp_acceleration_estimator = CausalDerivative(self._step_dt, self.cfg.acceleration_filter_window)
        self._joint_acceleration_estimator = CausalDerivative(self._step_dt, self.cfg.acceleration_filter_window)
        self._has_samples = False

        self._check_base_frame_alignment()

    def _check_base_frame_alignment(self) -> None:
        """Fail loudly if the base frame is rotated relative to the environment frame.

        The whole project expresses the pull axis in the robot base frame and reads contact
        forces in the world frame.  Those two only coincide when the base is unrotated, so
        the assumption is checked once at construction instead of being trusted.
        """
        quat = self._robot.data.root_quat_w
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device)
        if not torch.allclose(quat.abs(), identity.abs().expand_as(quat), atol=1e-4):
            raise ValueError(
                "DrawerStateReader assumes the robot base frame is axis-aligned with the "
                f"environment frame, but root_quat_w is {quat[0].tolist()}. Either place the "
                "robot without rotation or extend the reader to rotate world-frame forces."
            )

    """
    Properties -- drawer.
    """

    @property
    def drawer_joint_name(self) -> str:
        return self.cfg.drawer_joint_name

    @property
    def drawer_position(self) -> torch.Tensor:
        """Drawer opening along its prismatic joint (m), shape ``(num_envs,)``."""
        return self._cabinet.data.joint_pos[:, self._drawer_joint_idx]

    @property
    def drawer_joint_velocity_raw(self) -> torch.Tensor:
        """PhysX's own drawer joint velocity reading (m/s), shape ``(num_envs,)``.

        Recorded for transparency but **not** used for any decision: at this control rate it
        is aliased by gripper contact chatter (see this module's *Velocity provenance* note).
        """
        return self._cabinet.data.joint_vel[:, self._drawer_joint_idx]

    @property
    def drawer_velocity(self) -> torch.Tensor:
        """Drawer opening rate (m/s), shape ``(num_envs,)``.

        Causal moving average of the position finite difference. Zero until :meth:`update`
        has been called after a reset.
        """
        return self._zeros() if not self._has_samples else self._drawer_velocity_estimator.filtered

    @property
    def drawer_acceleration(self) -> torch.Tensor:
        """Drawer acceleration (m/s^2), shape ``(num_envs,)``.

        Causal moving average of the difference of :attr:`drawer_velocity`, i.e. a smoothed
        second difference of position.
        """
        return self._zeros() if not self._has_samples else self._drawer_acceleration_estimator.filtered

    @property
    def drawer_acceleration_raw(self) -> torch.Tensor:
        """Unsmoothed drawer acceleration (m/s^2), shape ``(num_envs,)``."""
        return self._zeros() if not self._has_samples else self._drawer_acceleration_estimator.raw

    @property
    def drawer_moving_mass(self) -> torch.Tensor:
        """Total mass the drawer joint moves (kg), shape ``(num_envs,)``.

        The ``drawer_top`` body plus the rigidly attached handle, read out of PhysX so it
        always reflects whatever mass was last applied.
        """
        masses = self._cabinet.root_physx_view.get_masses()
        total = masses[:, self._drawer_body_idx] + masses[:, self._handle_body_idx]
        return total.to(self._device)

    """
    Per-step bookkeeping.
    """

    def update(self) -> None:
        """Advance every causal derivative estimate. Call once per ``env.step``.

        Order matters: the drawer's acceleration is the derivative of the *filtered*
        velocity, so the velocity estimator has to take its sample first.
        """
        self._drawer_velocity_estimator.update(self.drawer_position)
        self._has_samples = True
        self._drawer_acceleration_estimator.update(self._drawer_velocity_estimator.filtered)
        self._tcp_acceleration_estimator.update(self.tcp_pull_axis_velocity)
        self._joint_acceleration_estimator.update(self.arm_joint_velocity)

    def reset_history(self) -> None:
        """Discard every derivative estimate. Call after ``env.reset``."""
        for estimator in (
            self._drawer_velocity_estimator,
            self._drawer_acceleration_estimator,
            self._tcp_acceleration_estimator,
            self._joint_acceleration_estimator,
        ):
            estimator.reset()
        self._has_samples = False

    def filter_description(self) -> dict:
        """How each derived channel was filtered, recorded with every episode."""
        return {
            "drawer_velocity": self._drawer_velocity_estimator.describe(),
            "drawer_acceleration": self._drawer_acceleration_estimator.describe(),
            "tcp_pull_axis_acceleration": self._tcp_acceleration_estimator.describe(),
            "joint_acceleration": self._joint_acceleration_estimator.describe(),
        }

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.env.num_envs, device=self._device)

    @property
    def handle_pose(self) -> torch.Tensor:
        """Handle frame pose in the environment frame, shape ``(num_envs, 7)``.

        Uses the official ``cabinet_frame`` sensor, whose offset already aligns the frame
        with the end-effector convention.
        """
        pos = self._cabinet_frame.data.target_pos_w[:, 0, :] - self.env.scene.env_origins
        quat = self._cabinet_frame.data.target_quat_w[:, 0, :]
        return torch.cat([pos, quat], dim=-1)

    @property
    def handle_position_w(self) -> torch.Tensor:
        """Handle *body* position in the world frame, shape ``(num_envs, 3)``."""
        return self._cabinet.data.body_pos_w[:, self._handle_body_idx, :]

    """
    Properties -- end-effector.
    """

    @property
    def tcp_pose(self) -> torch.Tensor:
        """TCP pose in the environment frame, shape ``(num_envs, 7)`` as ``[pos, quat]``."""
        pos = self._ee_frame.data.target_pos_w[:, 0, :] - self.env.scene.env_origins
        quat = self._ee_frame.data.target_quat_w[:, 0, :]
        return torch.cat([pos, quat], dim=-1)

    @property
    def tcp_linear_velocity(self) -> torch.Tensor:
        """TCP linear velocity in the world frame (m/s), shape ``(num_envs, 3)``.

        ``FrameTransformer`` reports pose only, so the velocity is transported from the
        parent body: ``v_tcp = v_body + omega x r``, with ``r`` the TCP offset rotated into
        the world frame.
        """
        lin = self._robot.data.body_lin_vel_w[:, self._ee_body_idx, :]
        ang = self._robot.data.body_ang_vel_w[:, self._ee_body_idx, :]
        quat = self._robot.data.body_quat_w[:, self._ee_body_idx, :]
        offset_w = quat_apply(quat, self._ee_offset)
        return lin + torch.cross(ang, offset_w, dim=-1)

    @property
    def tcp_orientation(self) -> torch.Tensor:
        """TCP orientation in the environment frame, shape ``(num_envs, 4)`` as ``(w, x, y, z)``."""
        return self._ee_frame.data.target_quat_w[:, 0, :]

    @property
    def tcp_angular_velocity(self) -> torch.Tensor:
        """TCP angular velocity in the world frame (rad/s), shape ``(num_envs, 3)``."""
        return self._robot.data.body_ang_vel_w[:, self._ee_body_idx, :]

    @property
    def tcp_pull_axis_velocity(self) -> torch.Tensor:
        """TCP speed along the drawer's opening direction (m/s), shape ``(num_envs,)``.

        Taken straight from PhysX rather than differentiated: the *robot's* own end-effector
        velocity is reliable at this control rate, unlike the drawer joint's (D009).
        """
        return self.tcp_linear_velocity @ self._direction_w

    @property
    def tcp_pull_axis_acceleration(self) -> torch.Tensor:
        """TCP acceleration along the pull axis (m/s^2), shape ``(num_envs,)``."""
        return self._zeros() if not self._has_samples else self._tcp_acceleration_estimator.filtered

    @property
    def tcp_pull_axis_acceleration_raw(self) -> torch.Tensor:
        """Unsmoothed TCP acceleration along the pull axis (m/s^2), shape ``(num_envs,)``."""
        return self._zeros() if not self._has_samples else self._tcp_acceleration_estimator.raw

    """
    Properties -- arm joints.
    """

    @property
    def arm_joint_position(self) -> torch.Tensor:
        """Arm joint positions (rad), shape ``(num_envs, num_arm_joints)``."""
        return self._robot.data.joint_pos[:, self._arm_joint_ids]

    @property
    def arm_joint_velocity(self) -> torch.Tensor:
        """Arm joint velocities (rad/s), shape ``(num_envs, num_arm_joints)``."""
        return self._robot.data.joint_vel[:, self._arm_joint_ids]

    @property
    def arm_joint_acceleration(self) -> torch.Tensor:
        """Arm joint accelerations (rad/s^2), shape ``(num_envs, num_arm_joints)``.

        Differentiated here rather than read from ``Articulation.data.joint_acc`` so that
        the filter and its causality are explicit and recorded.
        """
        if not self._has_samples:
            return torch.zeros_like(self.arm_joint_velocity)
        return self._joint_acceleration_estimator.filtered

    @property
    def finger_joint_ids(self) -> list[int]:
        """Articulation joint indices of the two gripper fingers."""
        return list(self._finger_joint_ids)

    @property
    def finger_joint_position(self) -> torch.Tensor:
        """Gripper finger joint positions (m), shape ``(num_envs, 2)``.

        Their difference is how far off-centre the hand is on the handle, which is what
        makes the grasp squeeze leak a bias force onto the pull axis.
        """
        return self._robot.data.joint_pos[:, self._finger_joint_ids]

    @property
    def arm_joint_applied_effort(self) -> torch.Tensor:
        """Joint efforts PhysX was *asked* to apply (N m), shape ``(num_envs, num_arm_joints)``.

        This is a command, not a torque-sensor reading: Isaac Lab exposes
        ``applied_torque``, the actuator output after clipping.  Named accordingly so it is
        never mistaken for a measurement.
        """
        return self._robot.data.applied_torque[:, self._arm_joint_ids]

    """
    Properties -- forces.
    """

    @property
    def has_force_measurement(self) -> bool:
        """Whether a handle contact sensor is present in the scene."""
        return self._contact is not None

    @property
    def handle_contact_force_w(self) -> torch.Tensor:
        """Net contact force on the handle body in the world frame (N), ``(num_envs, 3)``.

        Zeros when the scene has no handle contact sensor.
        """
        if self._contact is None:
            return torch.zeros(self.env.num_envs, 3, device=self._device)
        return self._contact.data.net_forces_w[:, 0, :]

    @property
    def wrist_reaction_force_w(self) -> torch.Tensor:
        """Wrist joint reaction force in the world frame (N), shape ``(num_envs, 3)``.

        PhysX reports the wrench transmitted from a body's parent to the body in the
        *parent* body frame, so it is rotated out of ``panda_link7`` into the world frame
        here.
        """
        wrench_parent = self._robot.data.body_incoming_joint_wrench_b[:, self._ee_body_idx, 0:3]
        parent_quat = self._robot.data.body_quat_w[:, self._ee_parent_body_idx, :]
        return quat_apply(parent_quat, wrench_parent)

    @property
    def measured_pull_force(self) -> torch.Tensor:
        """Measured pull-axis force at the wrist (N), shape ``(num_envs,)``.

        Positive means the arm is pulling in the drawer-*opening* direction.  See this
        module's *Force provenance* note for what this is and what it is not.
        """
        return self.wrist_reaction_force_w @ self._direction_w

    @property
    def drawer_resistance_force(self) -> torch.Tensor:
        """The drawer's own internal resistance along its joint (N), shape ``(num_envs,)``.

        ``ArticulationView.get_dof_projected_joint_forces`` on ``drawer_top_joint``. Measured
        to be exactly ``-(mu_dynamic * sign(v) + b * v)``: with ``b = 8`` and ``v = 0.0951``
        it reported -0.7680 against -0.7611 predicted, and with ``mu_d = 3`` while sliding it
        reported exactly -3.0000. Negative while the drawer opens, because it opposes motion.

        **Simulator-only.** A real drawer does not publish its own friction. See
        ``docs/FORCE_CHANNEL_AUDIT.md``.
        """
        return self._cabinet.root_physx_view.get_dof_projected_joint_forces()[:, self._drawer_joint_idx].to(
            self._device
        )

    @property
    def drawer_external_force(self) -> torch.Tensor:
        """The axial force actually delivered to the drawer (N), shape ``(num_envs,)``.

        From the drawer's equation of motion along its one degree of freedom::

            m_total * a = F_external + F_resistance   =>   F_external = m_total * a - F_resistance

        Every term on the right is available: the mass from PhysX, the acceleration from
        :attr:`drawer_acceleration`, and the resistance from
        :attr:`drawer_resistance_force`. Validated against the wrist channel, which it
        matches to within the hand-and-finger inertial term of about 0.05-0.10 N
        (``docs/FORCE_CHANNEL_AUDIT.md``).

        **Simulator-only and privileged**: it needs the drawer's mass, which is part of the
        hidden state the robot is supposed to be inferring.
        """
        return self.drawer_moving_mass * self.drawer_acceleration - self.drawer_resistance_force

    """
    Verification helpers.
    """

    def drawer_axis_world(self, displacement: torch.Tensor, reference_handle_pos_w: torch.Tensor) -> torch.Tensor:
        """Empirical unit vector of the drawer's travel direction in the world frame.

        Args:
            displacement: Drawer joint displacement since the reference (m), ``(num_envs,)``.
            reference_handle_pos_w: Handle body world position at zero displacement,
                shape ``(num_envs, 3)``.

        Returns:
            Unit vectors of shape ``(num_envs, 3)``; rows with negligible displacement are
            returned as zeros.
        """
        delta = self.handle_position_w - reference_handle_pos_w
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        moved = norm.squeeze(-1) > 1e-4
        unit = torch.where(moved.unsqueeze(-1), delta / norm.clamp(min=1e-12), torch.zeros_like(delta))
        # Orient along increasing joint displacement.
        return unit * torch.sign(displacement).clamp(min=-1.0, max=1.0).unsqueeze(-1)

    def verify_pull_axis(
        self, measured_axis_w: torch.Tensor, tolerance_deg: float = 5.0
    ) -> tuple[bool, float]:
        """Check that a measured travel direction matches the configured :class:`PullAxis`.

        Args:
            measured_axis_w: Unit vectors from :meth:`drawer_axis_world`, ``(num_envs, 3)``.
            tolerance_deg: Maximum allowed angle between measured and configured direction.

        Returns:
            ``(ok, worst_angle_deg)``.
        """
        cos = (measured_axis_w @ self._direction_w).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.arccos(cos))
        worst = float(angles.max())
        return worst <= tolerance_deg, worst
