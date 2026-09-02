r"""A probe that stops when the drawer *responds*, then lets go and watches it coast.

The standardised probe of Phases 8-11 ramps the force and stops the moment the drawer has
moved 3 mm. That makes it short, cheap and blind to one thing: it never observes the drawer
moving under **no** applied force, so it cannot see how the drawer slows down. Phase 10
measured the consequence -- sweeping damping from 2 to 11 N*s/m leaves the probe's duration
and breakaway force essentially unchanged.

This probe adds the missing observation. Three phases, per environment:

``RAMP_UP``
    Force rises from near zero at a fixed rate until the drawer has travelled
    :math:`d_\text{trigger} = \alpha\, d_\text{goal}`. A stiff drawer keeps getting more force
    until it moves -- or until ``max_force``, ``max_ramp_duration``, ``max_velocity`` or the
    safety limit intervenes, whichever comes first.
``RAMP_DOWN``
    The force falls smoothly to zero over :math:`T_\text{release} = \beta\, T_\text{ramp}`,
    where :math:`T_\text{ramp}` is however long *that* environment's ramp took. Only the
    commanded force is driven to zero -- the drawer's velocity is not controlled, deliberately.
``COAST``
    Zero commanded force. The drawer decelerates under its own dynamic friction, damping and
    inertia until :math:`|v| \le v_\text{end}`, or a timeout, or a displacement bound.

Why the coast is the point
--------------------------
With no applied force the drawer obeys

.. math:: m\,a = -(\mu_d \operatorname{sign}(v) + b\,v)

so regressing the measured deceleration on velocity during the coast gives
:math:`\mu_d/m` as the intercept and :math:`b/m` as the slope. Damping enters the
*slope* of that line, which is a quantity the old probe never measured, because it never let
go. Whether that translates into better identifiability is an empirical question and
``scripts/compare_probes.py`` is what answers it -- this module only makes the measurement
possible.

The phases are recorded per environment per step, because environments reach
:math:`d_\text{trigger}` at different times and a single scalar phase would be a fiction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

from probe_drawer.controllers.base_pull_controller import (
    BasePullController,
    SafetyLimits,
    _HistoryRecorder,
)
from probe_drawer.controllers.force_profiles import smoothstep
from probe_drawer.controllers.types import PullHistory, TerminationReason

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

    from probe_drawer.controllers.hybrid_osc import HybridPullOSC
    from probe_drawer.sensors import DrawerStateReader

__all__ = ["ProbePhase", "ResponseProbeCfg", "ResponseProbeController", "ResponseProbeResult"]


class ProbePhase(IntEnum):
    """Which segment an environment is in. Recorded per step, per environment."""

    RAMP_UP = 0
    RAMP_DOWN = 1
    COAST = 2
    DONE = 3


@dataclass(frozen=True)
class ResponseProbeCfg:
    r"""The new probe's character.

    Args:
        force_rate: Ramp slope (N/s). 5 N/s reproduces the old probe's 1 s climb to 6 N, so
            the two probes push equally hard and differ only in when they stop.
        initial_force: Where the ramp starts (N). Near zero rather than the old probe's 1 N,
            because the point is to observe the drawer from rest.
        trigger_fraction: :math:`\alpha`. The drawer must travel
            ``trigger_fraction * goal_displacement`` before the ramp ends.
        release_fraction: :math:`\beta`. The ramp-down lasts ``release_fraction`` of however
            long that environment's ramp-up took, so a stiff drawer that took 1.5 s to move
            gets a proportionally longer release than a slippery one that moved in 0.2 s.
        max_force: Hard ceiling on the commanded force (N).
        max_ramp_duration: Hard ceiling on the ramp-up (s).
        max_velocity: Ends the ramp-up if the drawer is already moving this fast (m/s).
        coast_end_velocity: The coast ends below this speed (m/s). Small enough to be
            "stopped" against the task's 0.03 m/s tolerance, large enough that the causal
            velocity filter can resolve it.
        max_coast_duration: Hard ceiling on the coast (s).
        max_total_displacement: Aborts the probe if it has moved the drawer this far (m). The
            probe must not quietly perform the task.
        capture_terminal_frame: Run one extra zero-force step after the last environment
            finishes, so the terminal near-rest observation is in the history rather than
            inferred. Flagged in the result when it happens.
    """

    force_rate: float = 5.0
    initial_force: float = 0.1
    trigger_fraction: float = 0.10
    release_fraction: float = 0.20
    max_force: float = 6.0
    max_ramp_duration: float = 2.0
    max_velocity: float = 0.08
    coast_end_velocity: float = 0.002
    max_coast_duration: float = 1.5
    max_total_displacement: float = 0.010
    capture_terminal_frame: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.trigger_fraction <= 0.5:
            raise ValueError(f"trigger_fraction must be in (0, 0.5], got {self.trigger_fraction}.")
        if not 0.0 < self.release_fraction <= 1.0:
            raise ValueError(f"release_fraction must be in (0, 1], got {self.release_fraction}.")
        if self.force_rate <= 0.0:
            raise ValueError(f"force_rate must be > 0 N/s, got {self.force_rate}.")
        if self.initial_force < 0.0 or self.initial_force >= self.max_force:
            raise ValueError(f"initial_force must be in [0, max_force), got {self.initial_force}.")
        if self.coast_end_velocity <= 0.0:
            raise ValueError(f"coast_end_velocity must be > 0 m/s, got {self.coast_end_velocity}.")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResponseProbeResult:
    """What the three-phase probe measured.

    Attributes:
        history: One continuous recording across all three phases.
        phase: ``(steps, num_envs)`` of :class:`ProbePhase` values.
        termination_reason: Why each environment's probe ended.
        duration: Total probe time per environment (s).
        ramp_duration, release_duration, coast_duration: Per-phase times (s).
        trigger_displacement: Where the ramp-up ended (m).
        final_displacement: Total probe travel (m).
        final_velocity: Speed when the probe ended (m/s).
        release_force: The force being commanded when the ramp-down began (N) -- the old
            probe's ``final_commanded_force``, and the closest analogue of a breakaway force.
        peak_measured_force: Largest wrist pull force (N).
        reached_trigger: Whether the drawer actually travelled ``d_trigger``.
        coasted_to_rest: Whether the coast ended on the velocity condition rather than a
            timeout. False means the drawer was still moving when the probe gave up, and the
            coast-decay features are then extrapolations.
        terminal_frame_added: Whether the extra zero-force step was appended.
        parameters: Everything needed to reproduce the run.
    """

    history: PullHistory
    phase: np.ndarray
    termination_reason: list[TerminationReason]
    duration: np.ndarray
    ramp_duration: np.ndarray
    release_duration: np.ndarray
    coast_duration: np.ndarray
    trigger_displacement: np.ndarray
    final_displacement: np.ndarray
    final_velocity: np.ndarray
    release_force: np.ndarray
    peak_measured_force: np.ndarray
    reached_trigger: np.ndarray
    coasted_to_rest: np.ndarray
    terminal_frame_added: bool = False
    parameters: dict = field(default_factory=dict)

    @property
    def num_envs(self) -> int:
        return int(self.final_displacement.shape[0])

    def phase_steps(self, env_index: int, phase: ProbePhase) -> np.ndarray:
        """Indices of the steps one environment spent in one phase."""
        return np.nonzero(self.phase[:, env_index] == int(phase))[0]

    def summary(self, env_index: int = 0) -> dict:
        return {
            "termination_reason": self.termination_reason[env_index].value,
            "duration": float(self.duration[env_index]),
            "ramp_duration": float(self.ramp_duration[env_index]),
            "release_duration": float(self.release_duration[env_index]),
            "coast_duration": float(self.coast_duration[env_index]),
            "trigger_displacement": float(self.trigger_displacement[env_index]),
            "final_displacement": float(self.final_displacement[env_index]),
            "final_velocity": float(self.final_velocity[env_index]),
            "release_force": float(self.release_force[env_index]),
            "reached_trigger": bool(self.reached_trigger[env_index]),
            "coasted_to_rest": bool(self.coasted_to_rest[env_index]),
        }


class ResponseProbeController(BasePullController):
    """Runs the ramp-up / ramp-down / coast probe.

    Subclasses :class:`BasePullController` for the sampling, safety and displacement
    machinery, but not for :meth:`run_profile`: that drives one profile to a stop condition,
    and this needs three segments whose transitions are per-environment and data-dependent.
    The loop below is therefore its own, and shares every channel definition with the old
    probe so the two are directly comparable.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        osc: HybridPullOSC,
        reader: DrawerStateReader,
        cfg: ResponseProbeCfg | None = None,
        safety: SafetyLimits | None = None,
    ) -> None:
        super().__init__(env, osc, reader, safety)
        self.cfg = cfg or ResponseProbeCfg()

    def _stop_conditions(self, elapsed: float, commanded_force: torch.Tensor) -> tuple:
        """Unused: this controller does not go through ``run_profile``."""
        return ()

    def run(
        self,
        goal_displacement: float,
        on_step: Callable[[int, float, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> ResponseProbeResult:
        r"""Probe the drawer and let it coast to a stop.

        Args:
            goal_displacement: The task's :math:`d_\text{goal}` (m). Only used to set
                :math:`d_\text{trigger} = \alpha\, d_\text{goal}`; the probe never learns what
                the goal *is* and never tries to reach it (D004).
            on_step: Called after each step as ``(step, elapsed, commanded, phase)``. An
                observer -- its return value is discarded -- so the diagnostic video recorder
                can render the three phases without reimplementing this loop.

        Returns:
            A :class:`ResponseProbeResult`.
        """
        if goal_displacement <= 0.0:
            raise ValueError(f"goal_displacement must be > 0 m, got {goal_displacement}.")
        cfg = self.cfg
        trigger = cfg.trigger_fraction * goal_displacement
        if trigger >= cfg.max_total_displacement:
            raise ValueError(
                f"d_trigger = {trigger * 1000:.1f} mm is at or beyond the probe's own "
                f"displacement bound of {cfg.max_total_displacement * 1000:.1f} mm; the probe "
                "would abort before it could trigger."
            )

        device = self._device
        count = self._num_envs
        zeros = torch.zeros(count, device=device)

        self.osc.settle(0)
        self.osc.capture_pose_reference()
        self._reference_drawer_position = self.reader.drawer_position.clone()

        phase = torch.zeros(count, dtype=torch.long, device=device)
        phase_started = torch.zeros(count, device=device)
        ramp_duration = torch.zeros(count, device=device)
        release_duration = torch.zeros(count, device=device)
        coast_duration = torch.zeros(count, device=device)
        release_force = torch.zeros(count, device=device)
        trigger_displacement = torch.zeros(count, device=device)
        peak_measured = torch.zeros(count, device=device)
        reached_trigger = torch.zeros(count, dtype=torch.bool, device=device)
        coasted_to_rest = torch.zeros(count, dtype=torch.bool, device=device)
        stop_time = torch.zeros(count, device=device)
        final_displacement = torch.zeros(count, device=device)
        final_velocity = torch.zeros(count, device=device)
        reasons: list[TerminationReason | None] = [None] * count

        recorder = _HistoryRecorder()
        phase_log: list[np.ndarray] = []
        max_steps = self.steps_for(cfg.max_ramp_duration + cfg.max_coast_duration + 1.0)
        elapsed = 0.0

        for step in range(max_steps):
            active = phase != int(ProbePhase.DONE)
            if not bool(active.any()):
                break

            in_ramp = phase == int(ProbePhase.RAMP_UP)
            in_release = phase == int(ProbePhase.RAMP_DOWN)
            phase_elapsed = elapsed - phase_started

            ramp_force = (cfg.initial_force + cfg.force_rate * phase_elapsed).clamp(max=cfg.max_force)
            # Smooth descent from whatever force this environment had reached, over its own
            # release window. ``clamp_min`` guards the zero-length window a first-step trigger
            # would otherwise create.
            fraction = (phase_elapsed / release_duration.clamp_min(self.step_dt)).clamp(0.0, 1.0)
            release_shape = torch.as_tensor(
                smoothstep((1.0 - fraction).cpu().numpy(), "smoothstep"), device=device, dtype=zeros.dtype
            )
            commanded = torch.where(in_ramp, ramp_force, torch.where(in_release, release_force * release_shape, zeros))
            commanded = torch.where(active, commanded, zeros)

            self.osc.step(self.osc.action(commanded))
            elapsed = (step + 1) * self.step_dt

            displacement = self.drawer_displacement
            velocity = self.reader.drawer_velocity
            measured = self.reader.measured_pull_force
            self._record_step(recorder, elapsed, active, commanded, displacement, measured)
            phase_log.append(phase.detach().cpu().numpy().copy())
            peak_measured = torch.maximum(peak_measured, measured.abs())
            if on_step is not None:
                on_step(step, elapsed, commanded, phase)

            unsafe = self._safety_violation()
            too_far = displacement.abs() >= cfg.max_total_displacement

            # --- RAMP_UP -> RAMP_DOWN, or straight to DONE on a limit ---
            triggered = in_ramp & (displacement >= trigger)
            ramp_expired = in_ramp & (
                (phase_elapsed >= cfg.max_ramp_duration)
                | (commanded >= cfg.max_force)
                | (velocity.abs() >= cfg.max_velocity)
            )
            leaving_ramp = in_ramp & (triggered | ramp_expired)
            if bool(leaving_ramp.any()):
                ramp_duration = torch.where(leaving_ramp, phase_elapsed + self.step_dt, ramp_duration)
                release_duration = torch.where(
                    leaving_ramp,
                    (cfg.release_fraction * ramp_duration).clamp_min(self.step_dt),
                    release_duration,
                )
                release_force = torch.where(leaving_ramp, commanded, release_force)
                trigger_displacement = torch.where(leaving_ramp, displacement, trigger_displacement)
                reached_trigger = reached_trigger | triggered
                phase = torch.where(leaving_ramp, torch.full_like(phase, int(ProbePhase.RAMP_DOWN)), phase)
                phase_started = torch.where(leaving_ramp, torch.full_like(phase_started, elapsed), phase_started)

            # --- RAMP_DOWN -> COAST ---
            released = (phase == int(ProbePhase.RAMP_DOWN)) & (phase_elapsed + self.step_dt >= release_duration)
            if bool(released.any()):
                coast_start = phase_elapsed + self.step_dt
                release_duration = torch.where(released, coast_start, release_duration)
                phase = torch.where(released, torch.full_like(phase, int(ProbePhase.COAST)), phase)
                phase_started = torch.where(released, torch.full_like(phase_started, elapsed), phase_started)

            # --- COAST -> DONE ---
            coasting = phase == int(ProbePhase.COAST)
            at_rest = coasting & (velocity.abs() <= cfg.coast_end_velocity)
            coast_expired = coasting & ((elapsed - phase_started) >= cfg.max_coast_duration)
            finishing = coasting & (at_rest | coast_expired | too_far | unsafe)
            # A limit hit during any phase ends the probe immediately.
            aborting = active & (too_far | unsafe) & ~finishing
            ending = finishing | aborting

            if bool(ending.any()):
                coast_duration = torch.where(coasting & ending, elapsed - phase_started, coast_duration)
                stop_time = torch.where(ending, torch.full_like(stop_time, elapsed), stop_time)
                final_displacement = torch.where(ending, displacement, final_displacement)
                final_velocity = torch.where(ending, velocity, final_velocity)
                coasted_to_rest = coasted_to_rest | at_rest
                for index in torch.nonzero(ending).flatten().tolist():
                    if reasons[index] is None:
                        reasons[index] = (
                            TerminationReason.SAFETY_ABORT
                            if bool(unsafe[index])
                            else TerminationReason.DISPLACEMENT_REACHED
                            if bool(too_far[index])
                            else TerminationReason.AT_REST
                            if bool(at_rest[index])
                            else TerminationReason.TIMEOUT
                        )
                phase = torch.where(ending, torch.full_like(phase, int(ProbePhase.DONE)), phase)

        terminal_frame = False
        if cfg.capture_terminal_frame:
            # The stop conditions fire *after* a step, so the last recorded sample is the one
            # that triggered them rather than the state that resulted. One extra zero-force
            # step puts the settled terminal observation in the history instead of leaving it
            # to be inferred. Flagged, because it makes the history one step longer than the
            # phase timings account for.
            self.osc.step(self.osc.action(zeros))
            elapsed += self.step_dt
            self._record_step(
                recorder,
                elapsed,
                torch.zeros(count, dtype=torch.bool, device=device),
                zeros,
                self.drawer_displacement,
                self.reader.measured_pull_force,
            )
            phase_log.append(np.full(count, int(ProbePhase.DONE)))
            final_velocity = self.reader.drawer_velocity.clone()
            final_displacement = self.drawer_displacement.clone()
            terminal_frame = True

        for index, reason in enumerate(reasons):
            if reason is None:
                reasons[index] = TerminationReason.TIMEOUT
                stop_time[index] = elapsed
                final_displacement[index] = self.drawer_displacement[index]
                final_velocity[index] = self.reader.drawer_velocity[index]

        return ResponseProbeResult(
            history=recorder.build(),
            phase=np.stack(phase_log) if phase_log else np.zeros((0, count), dtype=int),
            termination_reason=[reason for reason in reasons if reason is not None],
            duration=stop_time.cpu().numpy(),
            ramp_duration=ramp_duration.cpu().numpy(),
            release_duration=release_duration.cpu().numpy(),
            coast_duration=coast_duration.cpu().numpy(),
            trigger_displacement=trigger_displacement.cpu().numpy(),
            final_displacement=final_displacement.cpu().numpy(),
            final_velocity=final_velocity.cpu().numpy(),
            release_force=release_force.cpu().numpy(),
            peak_measured_force=peak_measured.cpu().numpy(),
            reached_trigger=reached_trigger.cpu().numpy(),
            coasted_to_rest=coasted_to_rest.cpu().numpy(),
            terminal_frame_added=terminal_frame,
            parameters={
                "controller": "ResponseProbeController",
                "goal_displacement": goal_displacement,
                "trigger_displacement": trigger,
                "config": cfg.as_dict(),
                "safety": self.safety.as_dict(),
            },
        )
