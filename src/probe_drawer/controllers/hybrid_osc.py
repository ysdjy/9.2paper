"""The one low-level controller both the Probe and the Execution controller drive.

Neither :class:`~probe_drawer.controllers.ProbePullController` nor
:class:`~probe_drawer.controllers.ExecutionPullController` implements robot control.  Both
own a :class:`HybridPullOSC`, which is a thin wrapper around Isaac Lab's official
:class:`~isaaclab.controllers.OperationalSpaceController` (configured by
:class:`~probe_drawer.envs.HybridPullControlCfg` and instantiated by Isaac Lab's own
``OperationalSpaceControllerAction`` term).  There is exactly one OSC implementation in
this project, and it is Isaac Lab's.

What this wrapper adds:

* it assembles the environment's 14-element action vector, so callers never index into
  raw action layouts;
* it owns the *pose reference* -- the TCP pose captured at the start of a pull, which the
  five non-pull degrees of freedom are held at;
* it reports how far the held degrees of freedom have drifted, which is the correctness
  criterion for hybrid control (spec section 26, Probe Test 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.envs import ManagerBasedRLEnv

from probe_drawer.sensors import DrawerStateReader, PullAxis

__all__ = ["HybridPullOSC", "HybridPullOSCCfg"]

#: Index layout of the arm portion of the action vector (see ``ProbeDrawerEnvCfg``).
_POSE_SLICE = slice(0, 7)
_FORCE_SLICE = slice(7, 10)
_TORQUE_SLICE = slice(10, 13)
_ARM_ACTION_DIM = 13


@dataclass
class HybridPullOSCCfg:
    """Configuration of the hybrid pull command assembler.

    Args:
        gripper_command: Binary gripper action held throughout a pull. ``-1`` closes the
            fingers, matching the official ``BinaryJointPositionActionCfg``.
        settle_brake_gain: Damping gain used **only** by :meth:`HybridPullOSC.settle`
            (N s/m). The pull axis is force-controlled and therefore has no position or
            velocity feedback of its own, so residual momentum left over from the grasp
            would otherwise persist for tens of seconds and make every probe start from a
            different, non-zero initial velocity. The brake is initialisation-only and is
            never active during a probe or an execution.
        settle_brake_limit: Absolute cap on the braking force (N).
    """

    gripper_command: float = -1.0
    settle_brake_gain: float = 200.0
    settle_brake_limit: float = 15.0


class HybridPullOSC:
    """Assembles hybrid force/pose actions and tracks how well the held axes hold.

    Args:
        env: The running research environment.
        reader: State accessor for the same environment.
        pull_axis: The drawer's opening direction. Must match the one the environment's
            operational-space controller was configured with.
        cfg: Command-assembly options.
        stepper: The object whose ``step`` actually advances the simulation. Defaults to
            ``env``. :class:`~probe_drawer.pull_system.PullSystem` passes the *wrapped*
            environment instead, so that wrappers -- video recording in particular -- see
            every step. Stepping the unwrapped environment would silently bypass them.

    Raises:
        ValueError: If the environment's action space is not the expected 14 elements,
            which means the environment was not configured for hybrid pulling.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        reader: DrawerStateReader,
        pull_axis: PullAxis,
        cfg: HybridPullOSCCfg | None = None,
        stepper: Any | None = None,
    ) -> None:
        self.env = env
        self.reader = reader
        self.pull_axis = pull_axis
        self.cfg = cfg or HybridPullOSCCfg()
        self._stepper = stepper if stepper is not None else env

        expected = _ARM_ACTION_DIM + 1
        actual = int(env.action_manager.total_action_dim)
        if actual != expected:
            raise ValueError(
                f"Expected an action dimension of {expected} "
                f"({_ARM_ACTION_DIM} hybrid OSC + 1 gripper) but the environment reports {actual}. "
                "Use probe_drawer.envs.ProbeDrawerEnvCfg."
            )

        self._device = env.device
        self._num_envs = env.num_envs
        self._direction = pull_axis.direction(self._device)
        self._pose_reference = torch.zeros(self._num_envs, 7, device=self._device)
        self._pose_reference[:, 3] = 1.0
        self._has_reference = False

    @property
    def action_dim(self) -> int:
        """Total environment action dimension (hybrid OSC plus gripper)."""
        return _ARM_ACTION_DIM + 1

    @property
    def pose_reference(self) -> torch.Tensor:
        """The held TCP pose, shape ``(num_envs, 7)`` as ``[position, quaternion]``."""
        return self._pose_reference

    @property
    def has_reference(self) -> bool:
        """Whether a pose reference has been captured since the last reset."""
        return self._has_reference

    def load_pose_reference(self, reference: torch.Tensor, has_reference: bool = True) -> None:
        """Set the held pose reference directly, instead of reading it from the robot.

        Used only by dataset generation, to put the controller back to a captured instant so
        that several candidate forces can be compared from one probe
        (``docs/COUNTERFACTUAL_BRANCHING.md``). Deployment always uses
        :meth:`capture_pose_reference`, which reads where the TCP actually is.

        Args:
            reference: Pose reference, shape ``(num_envs, 7)``.
            has_reference: Whether the controller should consider itself referenced.

        Raises:
            ValueError: On a wrongly shaped reference.
        """
        expected = (self._num_envs, 7)
        if tuple(reference.shape) != expected:
            raise ValueError(f"reference must have shape {expected}, got {tuple(reference.shape)}.")
        self._pose_reference = reference.clone().to(self._device)
        self._has_reference = bool(has_reference)

    def capture_pose_reference(self) -> torch.Tensor:
        """Latch the current TCP pose as the pose the held axes are servoed to.

        Called once at the start of every probe or execution, so that the five held degrees
        of freedom hold the pose the grasp actually settled at rather than a nominal one.
        """
        self._pose_reference = self.reader.tcp_pose.clone()
        self._has_reference = True
        return self._pose_reference

    def action(self, pull_force: torch.Tensor) -> torch.Tensor:
        """Build the environment action for a commanded pull-axis force.

        Args:
            pull_force: Commanded force along the drawer's opening direction (N), shape
                ``(num_envs,)``. Positive opens the drawer.

        Returns:
            Action tensor of shape ``(num_envs, 14)``.

        Raises:
            RuntimeError: If no pose reference has been captured yet.
        """
        if not self._has_reference:
            raise RuntimeError("capture_pose_reference() must be called before action().")
        if pull_force.shape != (self._num_envs,):
            raise ValueError(f"pull_force must have shape ({self._num_envs},), got {tuple(pull_force.shape)}.")

        action = torch.zeros(self._num_envs, self.action_dim, device=self._device)
        action[:, _POSE_SLICE] = self._pose_reference
        action[:, _FORCE_SLICE] = pull_force.unsqueeze(-1) * self._direction
        # The torque channel is masked out by contact_wrench_control_axes_task; it is written
        # explicitly as zero so the intent is visible rather than implied.
        action[:, _TORQUE_SLICE] = 0.0
        action[:, -1] = self.cfg.gripper_command
        return action

    def step(self, action: torch.Tensor) -> None:
        """Advance the environment one control step and keep the reader in sync.

        Every step in this project goes through here, for two reasons:
        :meth:`DrawerStateReader.update` must run exactly once per step for the
        finite-difference drawer velocity to be correct, and the step must reach the
        environment through its wrappers so that video recording sees it.
        """
        self._stepper.step(action)
        self.reader.update()

    def reset(self) -> None:
        """Clear the pose reference and the reader's velocity history after an env reset."""
        self.reader.reset_history()
        self._has_reference = False

    def settle(self, steps: int) -> None:
        """Bring the grasped system to rest before a pull starts.

        Holds the pose on the five motion-controlled axes and applies a pure velocity brake
        on the pull axis, so the grasp transient is damped out instead of coasting.  See
        :attr:`HybridPullOSCCfg.settle_brake_gain` for why the brake is necessary and why it
        exists only here.

        Args:
            steps: Number of environment steps to settle for. Zero or fewer is a no-op
                apart from latching the pose reference.
        """
        if not self._has_reference:
            self.capture_pose_reference()
        if steps <= 0:
            return
        limit = self.cfg.settle_brake_limit
        for _ in range(steps):
            velocity = self.reader.tcp_linear_velocity @ self._direction
            brake = (-self.cfg.settle_brake_gain * velocity).clamp(-limit, limit)
            self.step(self.action(brake))
            # The brake lets the TCP drift along the pull axis; keep the held-axis reference
            # attached to where the TCP actually is, so it never fights that drift.
            self.capture_pose_reference()

    def coast(self, steps: int) -> None:
        """Hold the five motion axes and command **zero** pull force, without braking.

        This is the physics of the inference transition between a probe and an execution:
        the pull axis is released and whatever momentum the probe left is allowed to persist,
        because that momentum is part of the state the probe produced and erasing it would
        make the sequential protocol a fiction (``docs/DECISIONS.md`` D029).

        Contrast with :meth:`settle`, which *does* brake the pull axis and is therefore only
        appropriate at the start of an episode, before any measurement has been taken.

        The pose reference is left where it is: the held axes have not moved, and re-latching
        each step would remove the very restoring force that keeps them still.

        Args:
            steps: Number of environment steps to coast for. Zero is a no-op.

        Raises:
            RuntimeError: If no pose reference has been captured yet.
        """
        if steps <= 0:
            return
        if not self._has_reference:
            raise RuntimeError("capture_pose_reference() must be called before coast().")
        zero = torch.zeros(self._num_envs, device=self._device)
        for _ in range(steps):
            self.step(self.action(zero))

    def residual_pull_velocity(self) -> torch.Tensor:
        """TCP speed along the pull axis (m/s), shape ``(num_envs,)``.

        Recorded at the start of every pull so that a poorly settled initial condition is
        visible in the episode metadata rather than silently biasing the measurement.
        """
        return self.reader.tcp_linear_velocity @ self._direction

    """
    Held-axis drift diagnostics.
    """

    def lateral_error(self) -> torch.Tensor:
        """TCP drift orthogonal to the pull axis (m), shape ``(num_envs,)``.

        This is the quantity that must stay small for hybrid control to be correct: the
        pull axis is free to move, the other two translational axes are not.
        """
        delta = self.reader.tcp_pose[:, :3] - self._pose_reference[:, :3]
        along = (delta @ self._direction).unsqueeze(-1) * self._direction
        return torch.linalg.norm(delta - along, dim=-1)

    def pull_axis_displacement(self) -> torch.Tensor:
        """TCP travel along the pull axis since the reference (m), shape ``(num_envs,)``.

        Positive means the TCP moved in the drawer-opening direction.
        """
        return (self.reader.tcp_pose[:, :3] - self._pose_reference[:, :3]) @ self._direction

    def orientation_error(self) -> torch.Tensor:
        """TCP orientation drift from the reference (rad), shape ``(num_envs,)``."""
        return quat_error_magnitude(self.reader.tcp_pose[:, 3:7], self._pose_reference[:, 3:7])
