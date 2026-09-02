"""Shared data contracts for every pull controller in this project.

Every public controller returns one of the dataclasses defined here.  Controllers must
never invent their own ad-hoc ``dict`` return format -- downstream training code and the
episode logger both rely on these types.

Conventions used throughout the project
---------------------------------------
* forces are newtons (N), positions metres (m), velocities m/s, times seconds (s)
* the *pull axis* is the drawer's opening direction; a **positive** force and a
  **positive** displacement both mean "drawer opening"
* every per-environment quantity is a ``numpy`` array whose leading axis is the
  environment index, so a controller driving ``num_envs`` environments returns
  ``num_envs`` results at once
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = [
    "ExecutionResult",
    "PullHistory",
    "TerminationReason",
    "ProbeResult",
]


class TerminationReason(str, Enum):
    """Why a pull episode stopped.

    ``DISPLACEMENT_REACHED``/``VELOCITY_LIMIT``/``MAX_FORCE_REACHED``/``TIMEOUT`` are Probe
    stop conditions (see :mod:`probe_drawer.controllers.probe_pull_controller`).
    ``DURATION_COMPLETED`` is the *only* nominal Execution outcome -- the execution
    controller runs the full commanded duration and never stops on task progress.
    ``SAFETY_ABORT`` may end either.
    """

    DISPLACEMENT_REACHED = "displacement_reached"
    VELOCITY_LIMIT = "velocity_limit"
    MAX_FORCE_REACHED = "max_force_reached"
    TIMEOUT = "timeout"
    DURATION_COMPLETED = "duration_completed"
    SAFETY_ABORT = "safety_abort"


@dataclass
class PullHistory:
    """Time series recorded during a pull episode.

    ``time`` has shape ``(T,)``.  Scalar per-environment signals have shape
    ``(T, num_envs)``; Cartesian signals ``(T, num_envs, 3)``; joint signals
    ``(T, num_envs, num_joints)``.

    ``active[k, e]`` says whether environment ``e`` was still being driven at step ``k``.
    A controller keeps stepping until *every* environment has stopped, and zeroes the force
    command of the ones that already have, so the tail of an early-stopping environment's
    row is padding rather than measurement.  Mask with :meth:`active_steps` before
    analysing or plotting a single environment.

    The distinction between :attr:`commanded_force` and :attr:`measured_force` is
    load-bearing for this project and must not be blurred:

    ``commanded_force``
        The pull-axis force the controller *asked* the operational-space controller for.
        Open-loop: what the force profile produced at that instant.
    ``measured_force``
        The pull-axis component of the contact force PhysX actually reports on the drawer
        handle body, from a :class:`~isaaclab.sensors.ContactSensor`.  This is a *measured*
        physical quantity, not a copy of the command.

    Likewise :attr:`drawer_velocity` is the finite-difference estimate every decision is
    based on, while :attr:`drawer_velocity_raw` is PhysX's own aliased reading, kept so the
    substitution stays auditable.  :attr:`drawer_position` is the drawer opening
    *relative to the start of the pull*, not the absolute joint coordinate; the absolute
    reference is recorded in the result's ``parameters["reference_drawer_position"]``.
    """

    time: np.ndarray
    active: np.ndarray
    commanded_force: np.ndarray
    measured_force: np.ndarray
    drawer_position: np.ndarray
    drawer_velocity: np.ndarray
    drawer_velocity_raw: np.ndarray
    tcp_position: np.ndarray
    tcp_linear_velocity: np.ndarray
    tcp_angular_velocity: np.ndarray
    tcp_pull_axis_position: np.ndarray
    tcp_lateral_error: np.ndarray
    tcp_orientation_error: np.ndarray
    handle_contact_force_w: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_applied_effort: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.commanded_force.shape[1])

    def active_steps(self, env_index: int) -> np.ndarray:
        """Boolean mask of the steps environment ``env_index`` was actually driven for."""
        return self.active[:, env_index]

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Flat ``{name: array}`` view, used by the episode logger to write an ``.npz``."""
        return {
            "time": self.time,
            "active": self.active,
            "commanded_force": self.commanded_force,
            "measured_force": self.measured_force,
            "drawer_position": self.drawer_position,
            "drawer_velocity": self.drawer_velocity,
            "drawer_velocity_raw": self.drawer_velocity_raw,
            "tcp_position": self.tcp_position,
            "tcp_linear_velocity": self.tcp_linear_velocity,
            "tcp_angular_velocity": self.tcp_angular_velocity,
            "tcp_pull_axis_position": self.tcp_pull_axis_position,
            "tcp_lateral_error": self.tcp_lateral_error,
            "tcp_orientation_error": self.tcp_orientation_error,
            "handle_contact_force_w": self.handle_contact_force_w,
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
            "joint_applied_effort": self.joint_applied_effort,
        }


@dataclass
class ProbeResult:
    """Outcome of one standardised probe, for every environment that was probed.

    Attributes:
        termination_reason: One :class:`TerminationReason` per environment.
        duration: Probe wall-clock-equivalent simulated duration per environment (s).
        final_displacement: Drawer opening at the probe's stop instant (m).
        final_velocity: Drawer opening velocity at the stop instant (m/s).
        final_commanded_force: Pull-axis force command at the stop instant (N).
        peak_measured_force: Largest measured pull-axis contact force seen (N).
        reached_target: Whether ``final_displacement >= target_displacement``.
        history: Full time series, including the steps of environments that stopped early.
    """

    termination_reason: list[TerminationReason]
    duration: np.ndarray
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    final_commanded_force: np.ndarray
    peak_measured_force: np.ndarray
    reached_target: np.ndarray
    history: PullHistory
    parameters: dict = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return len(self.termination_reason)

    def summary(self, env_index: int = 0) -> dict:
        """Human-readable one-environment summary, for logs and test assertions."""
        return {
            "termination_reason": self.termination_reason[env_index].value,
            "duration": float(self.duration[env_index]),
            "final_displacement": float(self.final_displacement[env_index]),
            "final_velocity": float(self.final_velocity[env_index]),
            "final_commanded_force": float(self.final_commanded_force[env_index]),
            "peak_measured_force": float(self.peak_measured_force[env_index]),
            "reached_target": bool(self.reached_target[env_index]),
        }


@dataclass
class ExecutionResult:
    """Outcome of one full-duration force-driven execution.

    Deliberately absent: any notion of task success.  Whether ``final_displacement`` is
    close enough to a goal is an *evaluation* question and belongs to the caller, never to
    the execution controller (see ``docs/DECISIONS.md``, D004).

    Attributes:
        duration: Simulated duration actually executed per environment (s).
        final_displacement: Drawer opening when the force went back to zero (m).
        final_velocity: Drawer opening velocity at that instant (m/s).
        peak_commanded_force: Largest pull-axis force command issued (N).
        peak_measured_force: Largest measured pull-axis contact force (N).
        safety_aborted: Whether an absolute safety limit cut the episode short.
        termination_reason: ``DURATION_COMPLETED`` nominally, ``SAFETY_ABORT`` otherwise.
        history: Full time series.
    """

    termination_reason: list[TerminationReason]
    duration: np.ndarray
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    peak_commanded_force: np.ndarray
    peak_measured_force: np.ndarray
    safety_aborted: np.ndarray
    history: PullHistory
    parameters: dict = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return len(self.termination_reason)

    def summary(self, env_index: int = 0) -> dict:
        """Human-readable one-environment summary, for logs and test assertions."""
        return {
            "termination_reason": self.termination_reason[env_index].value,
            "duration": float(self.duration[env_index]),
            "final_displacement": float(self.final_displacement[env_index]),
            "final_velocity": float(self.final_velocity[env_index]),
            "peak_commanded_force": float(self.peak_commanded_force[env_index]),
            "peak_measured_force": float(self.peak_measured_force[env_index]),
            "safety_aborted": bool(self.safety_aborted[env_index]),
        }
