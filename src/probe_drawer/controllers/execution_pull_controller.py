"""Public API 2 -- the full-duration force-driven execution.

The execution controller answers one question: *given a peak force and a duration, what
does the drawer do?*  It applies ``F(t) = peak_force * phi(t / duration)`` with a fixed
normalised shape and runs for the whole duration.

The goal displacement is deliberately absent
--------------------------------------------
``d_goal`` is **not** an input, and no stop condition anywhere in this class refers to
drawer displacement.  Letting the controller stop when it reached a goal would turn an
open-loop force-execution study into a closed-loop position-control study and would change
the research question outright (``docs/DECISIONS.md`` D004).  Evaluating
``|d(T) - d_goal| <= epsilon`` is the caller's job.

The only thing that may cut an execution short is an absolute safety violation, and that is
implemented in :class:`~probe_drawer.controllers.base_pull_controller.BasePullController`,
not here: :meth:`ExecutionPullController._stop_conditions` returns nothing at all, which
makes the guarantee structural rather than a matter of review discipline.
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
from probe_drawer.controllers.force_profiles import RampShape, TrapezoidForceProfile
from probe_drawer.controllers.hybrid_osc import HybridPullOSC
from probe_drawer.controllers.types import ExecutionResult, TerminationReason
from probe_drawer.sensors import DrawerStateReader

__all__ = ["ExecutionControllerCfg", "ExecutionPullController"]


@dataclass
class ExecutionControllerCfg:
    """Fixed shape of the execution force profile. Not per-episode task parameters.

    Args:
        rise_fraction: Fraction of the duration spent rising smoothly from 0 to the peak.
        fall_fraction: Fraction of the duration spent falling smoothly back to 0.
        shape: Interpolation of the rise and fall segments. ``"smoothstep"`` is C1, so the
            command never steps.
        settle_steps: Zero-force pose-holding steps before the profile starts.
        zero_force_cleanup_steps: Steps of explicit zero pull force sent *after* ``T``.

            The profile satisfies ``phi(1) == 0``, but commands are held for a whole control
            step, so the last command of the episode is the one issued at ``T - dt``: about
            2 % of the peak with a 10 % smoothstep fall. That command is correct for the
            interval it covers, and ``d(T)`` is right. What would be wrong is leaving it
            standing after ``T``, because the environment holds the last action it was given
            until something replaces it. These steps replace it with zero.

            They are **not** part of ``T``: they do not appear in the returned history, do
            not change ``duration``, ``final_displacement`` or ``final_velocity``, and take
            no part in success evaluation (``docs/DECISIONS.md`` D022).
        post_execution_settle_steps: Further zero-force pose-holding steps after the
            cleanup, for callers that want the system quiet before the next episode. Also
            outside ``T`` and also absent from the result.
    """

    rise_fraction: float = 0.1
    fall_fraction: float = 0.1
    shape: RampShape = "smoothstep"
    settle_steps: int = 30
    zero_force_cleanup_steps: int = 2
    post_execution_settle_steps: int = 0

    def __post_init__(self) -> None:
        for name in ("settle_steps", "zero_force_cleanup_steps", "post_execution_settle_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}.")

    def as_dict(self) -> dict:
        return {
            "rise_fraction": self.rise_fraction,
            "fall_fraction": self.fall_fraction,
            "shape": self.shape,
            "settle_steps": self.settle_steps,
            "zero_force_cleanup_steps": self.zero_force_cleanup_steps,
            "post_execution_settle_steps": self.post_execution_settle_steps,
        }


class ExecutionPullController(BasePullController):
    """Applies a peak-scaled force profile for a fixed duration and reports what happened.

    Args:
        env: The running research environment.
        osc: The shared hybrid operational-space controller wrapper.
        reader: State accessor for the same environment.
        cfg: Fixed profile shape. Project defaults when omitted.
        safety: Absolute limits. Project defaults when omitted.

    Example:
        >>> execution = ExecutionPullController(env, osc, reader)
        >>> result = execution.run(peak_force=16.0, duration=2.0)
        >>> success = abs(result.final_displacement[0] - d_goal) <= epsilon  # caller's job
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        osc: HybridPullOSC,
        reader: DrawerStateReader,
        cfg: ExecutionControllerCfg | None = None,
        safety: SafetyLimits | None = None,
    ) -> None:
        super().__init__(env, osc, reader, safety)
        self.cfg = cfg or ExecutionControllerCfg()

    def run(self, peak_force: float | Sequence[float], duration: float) -> ExecutionResult:
        """Execute the full force profile.

        Args:
            peak_force: Plateau pull-axis force (N). A single value is applied to every
                environment; a sequence of length ``num_envs`` gives each environment its own
                amplitude. The normalised shape ``phi(t/T)`` is identical either way, so a
                per-environment sequence compares force candidates from one starting state
                without changing anything else (``docs/DECISIONS.md`` D027).
            duration: Total execution time (s). The controller runs for this long, then the
                force is back at zero.

        Returns:
            An :class:`~probe_drawer.controllers.types.ExecutionResult` with one entry per
            environment plus the complete time series. It carries no notion of success.

        Raises:
            ValueError: On non-physical arguments, or if the profile would exceed the
                absolute force safety limit.
        """
        peaks = self._resolve_peak_forces(peak_force)
        if duration <= 0.0:
            raise ValueError(f"duration must be > 0 s, got {duration}.")

        # A unit-amplitude profile scaled per environment, so every environment follows the
        # same normalised curve and only the amplitude differs.
        profile = TrapezoidForceProfile(
            peak_force=1.0,
            duration=duration,
            rise_fraction=self.cfg.rise_fraction,
            fall_fraction=self.cfg.fall_fraction,
            shape=self.cfg.shape,
        )
        outcome = self.run_profile(
            profile=profile,
            max_steps=self.steps_for(duration),
            timeout_reason=TerminationReason.DURATION_COMPLETED,
            settle_steps=self.cfg.settle_steps,
            force_scale=peaks,
        )
        # `outcome` is already a set of numpy snapshots taken at t = T, so nothing below can
        # alter what this call returns.
        cleanup_steps = self._release_pull_force()

        safety_aborted = np.asarray(
            [reason is TerminationReason.SAFETY_ABORT for reason in outcome.termination_reason]
        )
        return ExecutionResult(
            termination_reason=outcome.termination_reason,
            duration=outcome.duration,
            final_displacement=outcome.final_displacement,
            final_velocity=outcome.final_velocity,
            peak_velocity=outcome.peak_drawer_velocity,
            peak_commanded_force=outcome.peak_commanded_force,
            peak_measured_force=outcome.peak_measured_force,
            safety_aborted=safety_aborted,
            history=outcome.history,
            parameters={
                "controller": "ExecutionPullController",
                "peak_force": peaks.tolist() if peaks.numel() > 1 else float(peaks[0]),
                "duration": duration,
                "profile": profile.describe(),
                "config": self.cfg.as_dict(),
                "safety": self.safety.as_dict(),
                "step_dt": self.step_dt,
                "commanded_steps": self.steps_for(duration),
                "post_execution_steps_excluded_from_result": cleanup_steps,
                "reference_drawer_position": np.asarray(outcome.reference_drawer_position).tolist(),
                "initial_drawer_velocity": np.asarray(outcome.initial_drawer_velocity).tolist(),
                "initial_tcp_pull_velocity": np.asarray(outcome.initial_tcp_pull_velocity).tolist(),
            },
        )

    def _resolve_peak_forces(self, peak_force: float | Sequence[float]) -> torch.Tensor:
        """Normalise the ``peak_force`` argument to a per-environment tensor.

        Raises:
            ValueError: If any value is non-positive, or a sequence has the wrong length.
        """
        if isinstance(peak_force, (int, float)):
            values = [float(peak_force)] * self.num_envs
        else:
            values = [float(value) for value in peak_force]
            if len(values) != self.num_envs:
                raise ValueError(f"peak_force must have 1 or {self.num_envs} values, got {len(values)}.")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"peak_force must be > 0 N everywhere, got {values}.")
        return torch.tensor(values, device=self._device)

    def _release_pull_force(self) -> int:
        """Command zero pull force after ``T``, so no residual command is left standing.

        Runs after the result has been snapshotted, and its steps are deliberately not
        recorded: they belong to episode teardown, not to the commanded duration.

        Returns:
            How many steps were stepped, for the record.
        """
        steps = self.cfg.zero_force_cleanup_steps + self.cfg.post_execution_settle_steps
        if steps == 0:
            return 0
        zero = torch.zeros(self.num_envs, device=self._device)
        for _ in range(steps):
            self.osc.step(self.osc.action(zero))
        return steps

    def _stop_conditions(self, elapsed: float, commanded_force: torch.Tensor) -> Sequence[StopCondition]:
        """No task stop conditions. See this module's docstring and DECISIONS D004.

        Returning an empty sequence is the enforcement mechanism: there is nowhere in this
        class for a displacement-triggered stop to live.
        """
        return ()
