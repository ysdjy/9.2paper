r"""The protocol the paper's experiments actually run: probe, then execute, without a reset.

Phase 9 measured the physics with a reset between the probe and the execution, which made
each execution start from a clean, identical state. That is convenient and it is not what a
robot does. A real robot probes the drawer it is about to pull, and by probing it has already
moved it and left it moving.

This module orchestrates the real thing::

    reset                 the arm into the recorded grasp, the drawer closed
      |                   -- the only reset in the whole protocol
    PROBE                 the standardised probe; its own settle brakes the pull axis here,
      |                   before any measurement, which is legitimate
    TRANSITION            zero pull force, five axes held, no braking, fixed length
      |                   -- stands in for the time an adaptation model would need
    EXECUTION             F(t) = F_peak * phi(t/T) for the full T, no settle
      |
    EVALUATE              d_total(T) vs d_goal, |v(T)| vs eps_v, validity

Three properties are the point of the module, and each is enforced rather than intended:

**No reset after the probe.** The drawer keeps its position and its velocity, the arm keeps
its configuration, the grasp is never re-established. :meth:`SequentialPullProtocol.run`
resets once, at the top.

**No artificial quieting.** The transition uses
:meth:`~probe_drawer.controllers.HybridPullOSC.coast`, which commands zero pull force and
does *not* brake, and the execution runs with ``settle_steps = 0``. The protocol refuses to
run with a settling execution configuration, because that would brake the pull axis and
erase the probe's effect (``docs/DECISIONS.md`` D029).

**The task is measured from before the probe.** ``d_goal`` is relative to the drawer's
position at the start of the *task*, so a probe that moved the drawer 3 mm followed by an
execution that moved it 47 mm has reached a 50 mm goal (D027). The execution controller still
knows nothing about any of this: it is handed a force and a duration, and the arithmetic
happens in :mod:`probe_drawer.evaluation` afterwards (D004).

This module contains no control or physics of its own. It calls the existing probe and
execution controllers and the existing evaluator; if you find yourself writing a force
profile or a stop condition here, it belongs in ``controllers/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from probe_drawer.controllers.types import ExecutionResult, ProbeResult
from probe_drawer.evaluation.operating_region import OperatingRegionCfg
from probe_drawer.evaluation.task_evaluator import EvaluationReport, SuccessCriteria, evaluate_execution
from probe_drawer.experiment_plan import ProbeTask

if TYPE_CHECKING:  # pragma: no cover - needs the Isaac Sim app at runtime
    from probe_drawer.pull_system import PullSystem

__all__ = [
    "InferenceTransitionCfg",
    "SequentialEpisode",
    "SequentialProtocolCfg",
    "SequentialPullProtocol",
    "TransitionRecord",
]


@dataclass(frozen=True)
class InferenceTransitionCfg:
    """The fixed gap between the probe ending and the execution starting.

    A deployed system needs some wall-clock time after the probe to run its adaptation model
    and choose a force. The protocol reserves that time explicitly and identically in every
    episode, so no episode gets a shorter or longer coast than another.

    Args:
        steps: Control steps of zero pull force between probe and execution. Chosen by
            measurement, not preference -- see ``scripts/validate_sequential_protocol.py``
            and ``docs/SEQUENTIAL_PROTOCOL.md``.
    """

    steps: int = 2

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError(f"transition steps must be >= 0, got {self.steps}.")

    def duration(self, step_dt: float) -> float:
        """How long the transition lasts (s)."""
        return self.steps * step_dt


@dataclass
class SequentialProtocolCfg:
    """What one sequential episode consists of.

    Args:
        probe_task: The standardised probe's four task parameters. The same probe runs in
            every episode; that is what makes probe histories comparable.
        duration: :math:`T_\\text{goal}`, the execution's fixed length (s).
        transition: The inference gap.
        operating_region: Validity thresholds used when the episode is evaluated.
    """

    probe_task: ProbeTask
    duration: float
    transition: InferenceTransitionCfg = field(default_factory=InferenceTransitionCfg)
    operating_region: OperatingRegionCfg = field(default_factory=OperatingRegionCfg)

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError(f"duration must be > 0 s, got {self.duration}.")

    def as_dict(self) -> dict:
        return {
            "probe_task": self.probe_task.as_dict(),
            "duration": self.duration,
            "transition_steps": self.transition.steps,
            "operating_region": self.operating_region.as_dict(),
        }


@dataclass
class TransitionRecord:
    """What the drawer and the arm did during the inference gap.

    Kept separate from both histories on purpose: the transition is neither part of the
    probe the adaptation model sees nor part of the commanded duration the task is judged
    over (``docs/DECISIONS.md`` D030). It is recorded so that the gap's effect can be
    inspected rather than assumed negligible.
    """

    steps: int
    duration: float
    displacement_before: np.ndarray
    displacement_after: np.ndarray
    velocity_before: np.ndarray
    velocity_after: np.ndarray
    tcp_pull_velocity_before: np.ndarray
    tcp_pull_velocity_after: np.ndarray
    lateral_drift: np.ndarray

    @property
    def coast_displacement(self) -> np.ndarray:
        """How far the drawer coasted during the gap (m), per environment."""
        return self.displacement_after - self.displacement_before

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "duration": self.duration,
            "displacement_before": self.displacement_before.tolist(),
            "displacement_after": self.displacement_after.tolist(),
            "coast_displacement": self.coast_displacement.tolist(),
            "velocity_before": self.velocity_before.tolist(),
            "velocity_after": self.velocity_after.tolist(),
            "tcp_pull_velocity_before": self.tcp_pull_velocity_before.tolist(),
            "tcp_pull_velocity_after": self.tcp_pull_velocity_after.tolist(),
            "lateral_drift": self.lateral_drift.tolist(),
        }


@dataclass
class SequentialEpisode:
    """One complete probe-then-execute episode.

    Attributes:
        task_start_position: Absolute drawer joint coordinate before the probe (m). The
            origin ``d_goal`` is measured from.
        probe: The probe's own result, exactly as the controller returned it.
        probe_displacement: What the probe moved the drawer, from the task start (m).
        transition: The inference gap.
        pre_execution_displacement: Drawer displacement from the task start at the instant
            the execution began (m) -- the probe plus whatever coasted during the gap.
        execution: The execution's own result. Its ``final_displacement`` is relative to the
            *execution's* start, not the task's.
        peak_force: The commanded amplitude per environment (N).
        evaluation: The success labels, if the episode was evaluated.
    """

    task_start_position: np.ndarray
    probe: ProbeResult
    probe_displacement: np.ndarray
    transition: TransitionRecord
    pre_execution_displacement: np.ndarray
    execution: ExecutionResult
    peak_force: np.ndarray
    evaluation: EvaluationReport | None = None

    @property
    def num_envs(self) -> int:
        return self.execution.num_envs

    @property
    def total_displacement(self) -> np.ndarray:
        """:math:`d_\\text{total}(T)`, measured from the task's start (m)."""
        return self.pre_execution_displacement + self.execution.final_displacement

    def post_probe_state(self, env_index: int) -> dict:
        """The state the execution inherited, for verifying candidate fairness."""
        return {
            "displacement": float(self.pre_execution_displacement[env_index]),
            "drawer_velocity": float(self.transition.velocity_after[env_index]),
            "tcp_pull_velocity": float(self.transition.tcp_pull_velocity_after[env_index]),
            "lateral_drift": float(self.transition.lateral_drift[env_index]),
        }

    def summary(self, env_index: int = 0) -> dict:
        """Human-readable one-environment summary, for logs and test assertions."""
        payload = {
            "peak_force": float(self.peak_force[env_index]),
            "probe_displacement": float(self.probe_displacement[env_index]),
            "probe_termination": self.probe.termination_reason[env_index].value,
            "transition_coast": float(self.transition.coast_displacement[env_index]),
            "pre_execution_displacement": float(self.pre_execution_displacement[env_index]),
            "execution_displacement": float(self.execution.final_displacement[env_index]),
            "total_displacement": float(self.total_displacement[env_index]),
            "final_velocity": float(self.execution.final_velocity[env_index]),
            "peak_velocity": float(self.execution.peak_velocity[env_index]),
            "execution_termination": self.execution.termination_reason[env_index].value,
        }
        if self.evaluation is not None:
            payload["success"] = bool(self.evaluation.success[env_index])
            payload["invalid_reasons"] = [
                reason.value for reason in self.evaluation.verdicts[env_index].invalid_reasons
            ]
        return payload


class SequentialPullProtocol:
    """Runs probe, transition and execution back to back, with one reset at the top.

    Args:
        system: A built pull system. Its execution controller **must** be configured with
            ``settle_steps = 0``: a settle brakes the pull axis, which would erase exactly
            the state the probe produced.
        cfg: What the episode consists of.

    Raises:
        ValueError: If the execution controller would settle before executing.

    Example:
        >>> protocol = SequentialPullProtocol(system, SequentialProtocolCfg(probe_task, 1.5))
        >>> episode = protocol.run(peak_force=[1.5, 2.0, 2.5], criteria=MAIN_TASK.criteria)
        >>> episode.total_displacement
    """

    def __init__(self, system: PullSystem, cfg: SequentialProtocolCfg) -> None:
        if system.execution.cfg.settle_steps != 0:
            raise ValueError(
                "The sequential protocol needs an execution controller with settle_steps = 0. "
                f"This one settles for {system.execution.cfg.settle_steps} steps, which brakes the "
                "pull axis and would erase the velocity the probe left behind (docs/DECISIONS.md D029)."
            )
        self.system = system
        self.cfg = cfg

    @property
    def step_dt(self) -> float:
        return self.system.step_dt

    def run(
        self,
        peak_force: float | Sequence[float],
        criteria: SuccessCriteria | None = None,
    ) -> SequentialEpisode:
        """Run one episode: reset, probe, transition, execute, and optionally evaluate.

        Args:
            peak_force: Execution amplitude (N). A sequence of length ``num_envs`` compares
                that many force candidates from the same probe, one per environment.
            criteria: When given, the episode is labelled. Passing it here does not let the
                execution controller see it: the labelling happens after the fact.

        Returns:
            A :class:`SequentialEpisode`.
        """
        system = self.system
        system.reset()  # the only reset in the protocol
        task_start_position = system.reader.drawer_position.clone()

        probe = system.probe.run(**self.cfg.probe_task.as_kwargs())
        probe_displacement = (system.reader.drawer_position - task_start_position).cpu().numpy()

        transition = self._run_transition(task_start_position)
        pre_execution_displacement = transition.displacement_after.copy()

        execution = system.execution.run(peak_force=peak_force, duration=self.cfg.duration)

        evaluation = None
        if criteria is not None:
            evaluation = evaluate_execution(
                execution,
                criteria,
                self.cfg.operating_region,
                pre_execution_displacement=pre_execution_displacement,
            )

        peaks = execution.parameters["peak_force"]
        return SequentialEpisode(
            task_start_position=task_start_position.cpu().numpy(),
            probe=probe,
            probe_displacement=probe_displacement,
            transition=transition,
            pre_execution_displacement=pre_execution_displacement,
            execution=execution,
            peak_force=np.atleast_1d(np.asarray(peaks, dtype=float)),
            evaluation=evaluation,
        )

    def _run_transition(self, task_start_position) -> TransitionRecord:
        """Coast for the configured number of steps, recording the state on both sides."""
        system = self.system
        reader, osc = system.reader, system.osc

        def displacement() -> np.ndarray:
            return (reader.drawer_position - task_start_position).cpu().numpy()

        before = (displacement(), reader.drawer_velocity.cpu().numpy(), osc.residual_pull_velocity().cpu().numpy())
        osc.coast(self.cfg.transition.steps)
        after = (displacement(), reader.drawer_velocity.cpu().numpy(), osc.residual_pull_velocity().cpu().numpy())

        return TransitionRecord(
            steps=self.cfg.transition.steps,
            duration=self.cfg.transition.duration(self.step_dt),
            displacement_before=before[0],
            displacement_after=after[0],
            velocity_before=before[1],
            velocity_after=after[1],
            tcp_pull_velocity_before=before[2],
            tcp_pull_velocity_after=after[2],
            lateral_drift=osc.lateral_error().cpu().numpy(),
        )
