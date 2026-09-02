"""Public API 1 -- the standardised physical probe.

The probe is the robot's one chance to *feel* an unknown drawer before committing to a
pull.  It applies a known, reproducible, monotonically increasing pull force and records
the drawer's response until one of its stop conditions fires.

What is a task parameter and what is not
----------------------------------------
:meth:`ProbePullController.run` takes the four task parameters the research protocol
varies -- ``initial_force``, ``max_force``, ``target_displacement``, ``max_velocity``.
Everything that defines the probe's *character* -- the ramp shape, the ramp duration, the
timeout, the settling time -- lives in :class:`ProbeControllerCfg` and is fixed for a whole
study.  If callers could reshape the ramp per episode the probe would no longer be a
standardised measurement and probe histories would not be comparable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from isaaclab.envs import ManagerBasedRLEnv

from probe_drawer.controllers.base_pull_controller import BasePullController, SafetyLimits, StopCondition
from probe_drawer.controllers.force_profiles import RampForceProfile, RampShape
from probe_drawer.controllers.hybrid_osc import HybridPullOSC
from probe_drawer.controllers.types import ProbeResult, TerminationReason
from probe_drawer.sensors import DrawerStateReader

__all__ = ["ProbeControllerCfg", "ProbePullController"]

#: Slack when comparing a float32 force command against ``max_force`` (N).
_FORCE_TOLERANCE = 1e-4


@dataclass
class ProbeControllerCfg:
    """Fixed character of the probe. Not per-episode task parameters.

    Args:
        ramp_duration: Time the force takes to go from ``initial_force`` to ``max_force``
            (s). Together with the two force levels this fixes the force *rate*.
        ramp_shape: Interpolation between the two force levels. ``"linear"`` gives a
            constant, easily interpreted force rate.
        hold_after_max_force: Seconds to hold ``max_force`` before stopping. ``0.0`` means
            the probe stops the moment the command first reaches ``max_force`` -- the
            simple, deterministic first-version behaviour (``docs/DECISIONS.md`` D007).
        max_probe_duration: Hard time budget (s). A probe that has not stopped for any
            other reason terminates with :attr:`~TerminationReason.TIMEOUT`. With the
            default configuration the force-limit condition fires first; this is the
            backstop that keeps a misconfigured or immovable case from running forever.
        settle_steps: Zero-force pose-holding steps before the ramp starts, so grasp
            contact transients are not measured as probe response.
    """

    ramp_duration: float = 1.0
    ramp_shape: RampShape = "linear"
    hold_after_max_force: float = 0.0
    max_probe_duration: float = 2.5
    settle_steps: int = 30

    def __post_init__(self) -> None:
        if self.ramp_duration <= 0.0:
            raise ValueError(f"ramp_duration must be > 0, got {self.ramp_duration}.")
        if self.hold_after_max_force < 0.0:
            raise ValueError(f"hold_after_max_force must be >= 0, got {self.hold_after_max_force}.")
        if self.max_probe_duration <= 0.0:
            raise ValueError(f"max_probe_duration must be > 0, got {self.max_probe_duration}.")
        if self.settle_steps < 0:
            raise ValueError(f"settle_steps must be >= 0, got {self.settle_steps}.")

    def as_dict(self) -> dict:
        return {
            "ramp_duration": self.ramp_duration,
            "ramp_shape": self.ramp_shape,
            "hold_after_max_force": self.hold_after_max_force,
            "max_probe_duration": self.max_probe_duration,
            "settle_steps": self.settle_steps,
        }


class ProbePullController(BasePullController):
    """Runs one standardised probe and returns the full response history.

    Args:
        env: The running research environment.
        osc: The shared hybrid operational-space controller wrapper.
        reader: State accessor for the same environment.
        cfg: Fixed probe character. Project defaults when omitted.
        safety: Absolute limits. Project defaults when omitted.

    Example:
        >>> probe = ProbePullController(env, osc, reader)
        >>> result = probe.run(
        ...     initial_force=2.0, max_force=10.0, target_displacement=0.005, max_velocity=0.05
        ... )
        >>> result.summary()
        {'termination_reason': 'displacement_reached', ...}
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        osc: HybridPullOSC,
        reader: DrawerStateReader,
        cfg: ProbeControllerCfg | None = None,
        safety: SafetyLimits | None = None,
    ) -> None:
        super().__init__(env, osc, reader, safety)
        self.cfg = cfg or ProbeControllerCfg()
        self._target_displacement = 0.0
        self._max_force = 0.0
        self._max_velocity = 0.0
        self._stop_force_time = 0.0

    def run(
        self,
        initial_force: float,
        max_force: float,
        target_displacement: float,
        max_velocity: float,
    ) -> ProbeResult:
        """Apply the standardised force ramp and stop at the first stop condition.

        Args:
            initial_force: Pull-axis force at the start of the probe (N).
            max_force: Pull-axis force at the end of the ramp (N).
            target_displacement: Drawer opening at which the probe has learned enough and
                stops (m).
            max_velocity: Drawer opening speed at which the probe stops (m/s).

        Returns:
            A :class:`~probe_drawer.controllers.types.ProbeResult` with one entry per
            environment plus the complete time series.

        Raises:
            ValueError: On non-physical arguments, or if the ramp would exceed the absolute
                force safety limit.

        Stop conditions, in priority order after the absolute safety limits:

        1. ``displacement >= target_displacement`` -> ``DISPLACEMENT_REACHED``
        2. ``|velocity| >= max_velocity`` -> ``VELOCITY_LIMIT``
        3. the command has reached ``max_force`` (and held it for
           :attr:`ProbeControllerCfg.hold_after_max_force`) -> ``MAX_FORCE_REACHED``
        4. :attr:`ProbeControllerCfg.max_probe_duration` elapsed -> ``TIMEOUT``
        """
        self._validate_task_parameters(initial_force, max_force, target_displacement, max_velocity)

        self._target_displacement = target_displacement
        self._max_force = max_force
        self._max_velocity = max_velocity
        self._stop_force_time = self.cfg.ramp_duration + self.cfg.hold_after_max_force

        profile = RampForceProfile(
            initial_force=initial_force,
            max_force=max_force,
            ramp_duration=self.cfg.ramp_duration,
            shape=self.cfg.ramp_shape,
        )
        outcome = self.run_profile(
            profile=profile,
            max_steps=self.steps_for(self.cfg.max_probe_duration),
            timeout_reason=TerminationReason.TIMEOUT,
            settle_steps=self.cfg.settle_steps,
        )

        return ProbeResult(
            termination_reason=outcome.termination_reason,
            duration=outcome.duration,
            final_displacement=outcome.final_displacement,
            final_velocity=outcome.final_velocity,
            final_commanded_force=outcome.final_commanded_force,
            peak_measured_force=outcome.peak_measured_force,
            reached_target=outcome.final_displacement >= target_displacement,
            history=outcome.history,
            parameters={
                "controller": "ProbePullController",
                "initial_force": initial_force,
                "max_force": max_force,
                "target_displacement": target_displacement,
                "max_velocity": max_velocity,
                "profile": profile.describe(),
                "config": self.cfg.as_dict(),
                "safety": self.safety.as_dict(),
                "step_dt": self.step_dt,
                "reference_drawer_position": np.asarray(outcome.reference_drawer_position).tolist(),
                "initial_drawer_velocity": np.asarray(outcome.initial_drawer_velocity).tolist(),
                "initial_tcp_pull_velocity": np.asarray(outcome.initial_tcp_pull_velocity).tolist(),
            },
        )

    def _stop_conditions(self, elapsed: float, commanded_force: torch.Tensor) -> Sequence[StopCondition]:
        displacement = self.drawer_displacement
        velocity = self.reader.drawer_velocity
        # `commanded_force` is the command that was issued at the *start* of the step that
        # has just been simulated, so the instant it was issued is one step back. Comparing
        # `elapsed` directly would stop the probe one step early, at 9.87 N rather than the
        # 10 N the caller asked for.
        issued_at = elapsed - self.step_dt
        force_limit_reached = commanded_force >= self._max_force - _FORCE_TOLERANCE
        if issued_at < self._stop_force_time - 1e-9:
            force_limit_reached = torch.zeros_like(force_limit_reached)
        return (
            (TerminationReason.DISPLACEMENT_REACHED, displacement >= self._target_displacement),
            (TerminationReason.VELOCITY_LIMIT, velocity.abs() >= self._max_velocity),
            (TerminationReason.MAX_FORCE_REACHED, force_limit_reached),
        )

    @staticmethod
    def _validate_task_parameters(
        initial_force: float, max_force: float, target_displacement: float, max_velocity: float
    ) -> None:
        if initial_force < 0.0:
            raise ValueError(f"initial_force must be >= 0 N, got {initial_force}.")
        if max_force < initial_force:
            raise ValueError(
                f"max_force ({max_force} N) must be >= initial_force ({initial_force} N): the probe input "
                "is monotonically non-decreasing."
            )
        if target_displacement <= 0.0:
            raise ValueError(f"target_displacement must be > 0 m, got {target_displacement}.")
        if max_velocity <= 0.0:
            raise ValueError(f"max_velocity must be > 0 m/s, got {max_velocity}.")
