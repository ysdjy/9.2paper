"""Machinery shared by the Probe and Execution controllers.

Both public controllers are the *same* control loop with a different force profile and a
different set of stop conditions.  That loop lives here exactly once:

* :class:`SafetyLimits` -- the absolute limits that may abort any pull;
* :class:`_HistoryRecorder` -- accumulates the time series into a
  :class:`~probe_drawer.controllers.types.PullHistory`;
* :class:`BasePullController` -- steps the environment, applies the force profile,
  evaluates stop conditions per environment, and snapshots each environment's state at the
  instant it stopped.

A subclass contributes only two things: a force profile and an ordered list of stop
conditions.  It must not step the environment or record history itself.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.envs import ManagerBasedRLEnv

from probe_drawer.controllers.force_profiles import ForceProfile
from probe_drawer.controllers.hybrid_osc import HybridPullOSC
from probe_drawer.controllers.types import HISTORY_CHANNELS, PullHistory, TerminationReason
from probe_drawer.sensors import DrawerStateReader

__all__ = ["BasePullController", "PullOutcome", "SafetyLimits", "StopCondition"]

#: An ordered stop condition: a reason and the per-environment mask that triggers it.
StopCondition = tuple[TerminationReason, torch.Tensor]


@dataclass
class SafetyLimits:
    """Absolute limits that abort a pull regardless of what the task wants.

    These are *not* task parameters.  They exist to stop the simulation from diverging or
    the arm from doing something violent, and they are the **only** reason the Execution
    controller may stop before its commanded duration elapses (see ``docs/DECISIONS.md``
    D004).  Every limit is generous relative to normal operation, so a trip means something
    is genuinely wrong.

    Args:
        max_commanded_force: Hard cap on the pull-axis force command (N). A profile asking
            for more than this is a configuration error and raises rather than clipping.
        max_drawer_velocity: Drawer opening speed above which the pull is aborted (m/s).
        max_tcp_speed: TCP speed above which the pull is aborted (m/s).
        max_lateral_error: TCP drift orthogonal to the pull axis above which the pull is
            aborted (m) -- i.e. the hybrid controller has lost the held axes.
        max_orientation_error_deg: TCP orientation drift above which the pull is aborted.
        max_arm_joint_velocity: Arm joint speed above which the pull is aborted (rad/s).
    """

    max_commanded_force: float = 60.0
    max_drawer_velocity: float = 1.0
    max_tcp_speed: float = 2.0
    max_lateral_error: float = 0.05
    max_orientation_error_deg: float = 30.0
    max_arm_joint_velocity: float = 6.0

    def as_dict(self) -> dict:
        return {
            "max_commanded_force": self.max_commanded_force,
            "max_drawer_velocity": self.max_drawer_velocity,
            "max_tcp_speed": self.max_tcp_speed,
            "max_lateral_error": self.max_lateral_error,
            "max_orientation_error_deg": self.max_orientation_error_deg,
            "max_arm_joint_velocity": self.max_arm_joint_velocity,
        }


@dataclass
class PullOutcome:
    """Raw per-environment outcome of :meth:`BasePullController.run_profile`.

    The public controllers turn this into a
    :class:`~probe_drawer.controllers.types.ProbeResult` or
    :class:`~probe_drawer.controllers.types.ExecutionResult`; nothing outside
    :mod:`probe_drawer.controllers` should need it.
    """

    termination_reason: list[TerminationReason]
    duration: np.ndarray
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    final_commanded_force: np.ndarray
    peak_commanded_force: np.ndarray
    peak_measured_force: np.ndarray
    peak_drawer_velocity: np.ndarray
    history: PullHistory
    reference_drawer_position: np.ndarray
    initial_drawer_velocity: np.ndarray
    initial_tcp_pull_velocity: np.ndarray


class _HistoryRecorder:
    """Accumulates one time step per :meth:`record` call.

    Generic over :data:`~probe_drawer.controllers.types.HISTORY_CHANNELS` rather than
    listing twenty-five fields twice: a channel added to ``PullHistory`` and to
    :meth:`BasePullController._sample` is recorded without touching this class.
    """

    def __init__(self) -> None:
        self.time: list[float] = []
        self.channels: dict[str, list[np.ndarray]] = {name: [] for name in HISTORY_CHANNELS}

    def record(self, elapsed: float, sample: dict[str, np.ndarray]) -> None:
        missing = set(HISTORY_CHANNELS) - set(sample)
        if missing:
            raise KeyError(f"Missing channels in a recorded step: {sorted(missing)}.")
        unexpected = set(sample) - set(HISTORY_CHANNELS)
        if unexpected:
            raise KeyError(f"Unknown channels in a recorded step: {sorted(unexpected)}.")
        self.time.append(elapsed)
        for name, value in sample.items():
            self.channels[name].append(value)

    def build(self) -> PullHistory:
        return PullHistory(
            time=np.asarray(self.time, dtype=np.float64),
            **{name: np.stack(rows, axis=0) for name, rows in self.channels.items()},
        )


class _StopAccumulator:
    """Per-environment bookkeeping for a pull: peaks, stop instants, stop reasons.

    A controller keeps stepping until every environment has stopped, so each environment's
    "final" values have to be snapshotted at the instant *it* stopped rather than read at
    the end of the loop. That bookkeeping lives here so that
    :meth:`BasePullController.run_profile` reads as the control loop it is.
    """

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self.active = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.reason: list[TerminationReason | None] = [None] * num_envs
        self.stop_time = torch.zeros(num_envs, device=device)
        self.final_displacement = torch.zeros(num_envs, device=device)
        self.final_velocity = torch.zeros(num_envs, device=device)
        self.final_commanded = torch.zeros(num_envs, device=device)
        self.peak_commanded = torch.zeros(num_envs, device=device)
        self.peak_measured = torch.zeros(num_envs, device=device)
        self.peak_velocity = torch.zeros(num_envs, device=device)

    @property
    def any_active(self) -> bool:
        return bool(self.active.any())

    def note_peaks(self, commanded: torch.Tensor, measured: torch.Tensor, velocity: torch.Tensor) -> None:
        """Update the running peaks, ignoring environments that have already stopped."""
        for name, value in (
            ("peak_commanded", commanded),
            ("peak_measured", measured),
            ("peak_velocity", velocity.abs()),
        ):
            running = getattr(self, name)
            setattr(self, name, torch.maximum(running, torch.where(self.active, value, running)))

    def stop(
        self,
        mask: torch.Tensor,
        reason: TerminationReason,
        elapsed: float,
        displacement: torch.Tensor,
        velocity: torch.Tensor,
        commanded: torch.Tensor,
    ) -> None:
        """Retire the still-active environments selected by ``mask``, snapshotting their state."""
        newly = self.active & mask
        if not bool(newly.any()):
            return
        self.stop_time = torch.where(newly, torch.full_like(self.stop_time, elapsed), self.stop_time)
        self.final_displacement = torch.where(newly, displacement, self.final_displacement)
        self.final_velocity = torch.where(newly, velocity, self.final_velocity)
        self.final_commanded = torch.where(newly, commanded, self.final_commanded)
        for env_index in newly.nonzero(as_tuple=False).flatten().tolist():
            self.reason[env_index] = reason
        self.active = self.active & ~newly

    def reasons(self) -> list[TerminationReason]:
        """The stop reasons, once every environment has one.

        Raises:
            RuntimeError: If any environment ended without a recorded reason, which would
                mean a stop condition is unreachable.
            """
        missing = [i for i, reason in enumerate(self.reason) if reason is None]
        if missing:
            raise RuntimeError(
                f"environments {missing} finished without a termination reason; a stop "
                "condition is unreachable"
            )
        return [reason for reason in self.reason if reason is not None]


class BasePullController(ABC):
    """Steps the environment under a force profile until every environment has stopped.

    Args:
        env: The running research environment.
        osc: The shared hybrid operational-space controller wrapper.
        reader: State accessor for the same environment.
        safety: Absolute limits. The project-wide defaults are used when omitted.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        osc: HybridPullOSC,
        reader: DrawerStateReader,
        safety: SafetyLimits | None = None,
    ) -> None:
        self.env = env
        self.osc = osc
        self.reader = reader
        self.safety = safety or SafetyLimits()
        self._device = env.device
        self._num_envs = env.num_envs
        self._reference_drawer_position = torch.zeros(self._num_envs, device=self._device)
        self._timeout_reason = TerminationReason.TIMEOUT

    @property
    def step_dt(self) -> float:
        """Environment step time (``sim.dt * decimation``) in seconds."""
        return float(self.env.step_dt)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def drawer_displacement(self) -> torch.Tensor:
        """Drawer opening since the current pull started (m), shape ``(num_envs,)``.

        The reference is latched by :meth:`run_profile` after settling, so subclasses'
        stop conditions and the base loop always measure displacement from the same zero.
        """
        return self.reader.drawer_position - self._reference_drawer_position

    def steps_for(self, duration: float) -> int:
        """Number of environment steps that best represents ``duration`` seconds."""
        return max(1, int(round(duration / self.step_dt)))

    @abstractmethod
    def _stop_conditions(self, elapsed: float, commanded_force: torch.Tensor) -> Sequence[StopCondition]:
        """Task stop conditions, highest priority first.

        Evaluated once per step *after* the environment has advanced, for the environments
        that are still active.  Safety limits are checked by the base class before these,
        so a subclass never needs to repeat them.

        Args:
            elapsed: Simulated time since the pull started (s).
            commanded_force: The force command just applied, shape ``(num_envs,)``.
        """

    def run_profile(
        self,
        profile: ForceProfile,
        max_steps: int,
        timeout_reason: TerminationReason,
        settle_steps: int = 0,
        force_scale: torch.Tensor | None = None,
        on_step: Callable[[int, float, torch.Tensor], None] | None = None,
    ) -> PullOutcome:
        """Drive the environment with ``profile`` until every environment stops.

        Args:
            profile: Supplies the commanded pull-axis force at each instant.
            max_steps: Hard step budget. An environment still active when it runs out
                terminates with ``timeout_reason``.
            timeout_reason: Reason recorded when the step budget is exhausted --
                ``TIMEOUT`` for a probe, ``DURATION_COMPLETED`` for an execution.
            settle_steps: Steps of zero-force pose holding before the pull, so grasp
                contact transients do not contaminate the measurement.
            on_step: Called after every step as ``(step, elapsed, commanded)``. For observing
                a run without reimplementing its loop -- the diagnostic video recorder uses it
                to capture a frame per step. It must not modify anything; it is given the
                commanded force by value and the controller does not read it back.
            force_scale: Per-environment multiplier on the profile, shape ``(num_envs,)``.
                Defaults to ones. This is how several force candidates are compared from the
                same starting state: every environment follows the *same* normalised shape
                ``phi(t/T)`` and differs only in amplitude, which is exactly the invariance
                the execution profile is designed around.

        Raises:
            ValueError: If the profile, after scaling, would command more than
                :attr:`SafetyLimits.max_commanded_force`.
        """
        if force_scale is None:
            force_scale = torch.ones(self._num_envs, device=self._device)
        if force_scale.shape != (self._num_envs,):
            raise ValueError(
                f"force_scale must have shape ({self._num_envs},), got {tuple(force_scale.shape)}."
            )
        self._validate_profile(profile, max_steps, float(force_scale.abs().max()))
        self._timeout_reason = timeout_reason

        self.osc.settle(settle_steps)
        self.osc.capture_pose_reference()

        self._reference_drawer_position = self.reader.drawer_position.clone()
        initial_drawer_velocity = self.reader.drawer_velocity.clone()
        initial_tcp_pull_velocity = self.osc.residual_pull_velocity().clone()

        recorder = _HistoryRecorder()
        accumulator = _StopAccumulator(self._num_envs, self._device)

        elapsed = 0.0
        for step in range(max_steps):
            # Environments that have already stopped are commanded zero force, so they stop
            # being driven without interrupting the ones still running.
            commanded = force_scale * float(profile.force(elapsed))
            commanded = torch.where(accumulator.active, commanded, torch.zeros_like(commanded))

            self.osc.step(self.osc.action(commanded))
            elapsed = (step + 1) * self.step_dt

            displacement = self.drawer_displacement
            measured = self.reader.measured_pull_force
            self._record_step(recorder, elapsed, accumulator.active, commanded, displacement, measured)
            accumulator.note_peaks(commanded, measured, self.reader.drawer_velocity)

            if on_step is not None:
                on_step(step, elapsed, commanded)

            for reason, mask in self._all_stop_conditions(step, max_steps, elapsed, commanded):
                accumulator.stop(
                    mask, reason, elapsed, displacement, self.reader.drawer_velocity, commanded
                )
            if not accumulator.any_active:
                break

        return PullOutcome(
            termination_reason=accumulator.reasons(),
            duration=accumulator.stop_time.cpu().numpy(),
            final_displacement=accumulator.final_displacement.cpu().numpy(),
            final_velocity=accumulator.final_velocity.cpu().numpy(),
            final_commanded_force=accumulator.final_commanded.cpu().numpy(),
            peak_commanded_force=accumulator.peak_commanded.cpu().numpy(),
            peak_measured_force=accumulator.peak_measured.cpu().numpy(),
            peak_drawer_velocity=accumulator.peak_velocity.cpu().numpy(),
            history=recorder.build(),
            reference_drawer_position=self._reference_drawer_position.cpu().numpy(),
            initial_drawer_velocity=initial_drawer_velocity.cpu().numpy(),
            initial_tcp_pull_velocity=initial_tcp_pull_velocity.cpu().numpy(),
        )

    def _all_stop_conditions(
        self, step: int, max_steps: int, elapsed: float, commanded: torch.Tensor
    ) -> list[StopCondition]:
        """Safety first, then the subclass's task conditions, then the step budget.

        The order is the priority order: when two conditions fire on the same step, the one
        listed first is the reason that gets recorded.
        """
        conditions: list[StopCondition] = [
            (TerminationReason.SAFETY_ABORT, self._safety_violation()),
            *self._stop_conditions(elapsed, commanded),
        ]
        if step == max_steps - 1:
            conditions.append(
                (self._timeout_reason, torch.ones(self._num_envs, dtype=torch.bool, device=self._device))
            )
        return conditions

    def _validate_profile(self, profile: ForceProfile, max_steps: int, scale: float = 1.0) -> None:
        """Reject a profile that would exceed the absolute force limit once scaled."""
        samples = np.linspace(0.0, max_steps * self.step_dt, 512)
        peak = scale * float(np.max(np.abs(np.asarray(profile.force(samples)))))
        if peak > self.safety.max_commanded_force:
            raise ValueError(
                f"{type(profile).__name__} peaks at {peak:.2f} N, above the absolute safety limit of "
                f"{self.safety.max_commanded_force:.2f} N. Raise SafetyLimits.max_commanded_force "
                "deliberately if this experiment really needs it."
            )
        if not math.isfinite(peak):
            raise ValueError(f"{type(profile).__name__} produced a non-finite force command.")

    def _record_step(
        self,
        recorder: _HistoryRecorder,
        elapsed: float,
        active: torch.Tensor,
        commanded: torch.Tensor,
        displacement: torch.Tensor,
        measured: torch.Tensor,
    ) -> None:
        """Append the post-step state of every environment to the history."""
        recorder.record(elapsed, self._sample(active, commanded, displacement, measured))

    def _sample(
        self,
        active: torch.Tensor,
        commanded: torch.Tensor,
        displacement: torch.Tensor,
        measured: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """One step of every channel in :data:`HISTORY_CHANNELS`, as numpy arrays.

        Adding a channel means adding it here, to ``PullHistory`` and to
        ``OBSERVATION_SPECS``; ``tests/unit/test_observation_spec.py`` fails if the three
        ever disagree.
        """
        reader, osc = self.reader, self.osc

        def to_numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.detach().cpu().numpy()

        return {
            "active": to_numpy(active),
            "commanded_force": to_numpy(commanded),
            "drawer_position": to_numpy(displacement),
            "drawer_velocity": to_numpy(reader.drawer_velocity),
            "drawer_velocity_raw": to_numpy(reader.drawer_joint_velocity_raw),
            "drawer_acceleration": to_numpy(reader.drawer_acceleration),
            "drawer_acceleration_raw": to_numpy(reader.drawer_acceleration_raw),
            "tcp_pull_axis_position": to_numpy(osc.pull_axis_displacement()),
            "tcp_pull_axis_velocity": to_numpy(reader.tcp_pull_axis_velocity),
            "tcp_pull_axis_acceleration": to_numpy(reader.tcp_pull_axis_acceleration),
            "tcp_pull_axis_acceleration_raw": to_numpy(reader.tcp_pull_axis_acceleration_raw),
            "tcp_position": to_numpy(reader.tcp_pose[:, :3]),
            "tcp_orientation": to_numpy(reader.tcp_orientation),
            "tcp_linear_velocity": to_numpy(reader.tcp_linear_velocity),
            "tcp_angular_velocity": to_numpy(reader.tcp_angular_velocity),
            "tcp_lateral_error": to_numpy(osc.lateral_error()),
            "tcp_orientation_error": to_numpy(osc.orientation_error()),
            "joint_position": to_numpy(reader.arm_joint_position),
            "joint_velocity": to_numpy(reader.arm_joint_velocity),
            "joint_acceleration": to_numpy(reader.arm_joint_acceleration),
            "joint_applied_effort": to_numpy(reader.arm_joint_applied_effort),
            "measured_force": to_numpy(measured),
            "handle_contact_force_w": to_numpy(reader.handle_contact_force_w),
            "drawer_resistance_force": to_numpy(reader.drawer_resistance_force),
            "drawer_external_force": to_numpy(reader.drawer_external_force),
        }

    def _safety_violation(self) -> torch.Tensor:
        """Per-environment mask of environments that have tripped an absolute limit."""
        limits = self.safety
        tcp_speed = torch.linalg.norm(self.reader.tcp_linear_velocity, dim=-1)
        joint_speed = self.reader.arm_joint_velocity.abs().amax(dim=-1)
        return (
            (self.reader.drawer_velocity.abs() > limits.max_drawer_velocity)
            | (tcp_speed > limits.max_tcp_speed)
            | (self.osc.lateral_error() > limits.max_lateral_error)
            | (torch.rad2deg(self.osc.orientation_error()) > limits.max_orientation_error_deg)
            | (joint_speed > limits.max_arm_joint_velocity)
            | ~torch.isfinite(self.reader.drawer_position)
        )
