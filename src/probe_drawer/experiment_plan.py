r"""The experiment parameters, and where each number came from.

Every value here is the output of a sweep, not a preference. The provenance is recorded
alongside it so the next phase can cite it and a reviewer can re-derive it:

* the task and the execution profile come from ``scripts/refine_task_space.py`` scoring the
  **sequential** Oracle (report: ``outputs/logs/task_refinement.json``);
* the inference gap comes from ``scripts/validate_sequential_protocol.py``
  (report: ``outputs/logs/sequential_protocol_validation.json``);
* the probe parameters come from ``scripts/calibrate_probe.py``
  (report: ``outputs/logs/probe_calibration.json``);
* the hidden-state ranges come from the sweeps in ``outputs/logs/sequential_oracle_fall*.json``.

The Phase 9 figures these superseded are kept in ``docs/EXPERIMENT_SPACE.md`` as
``PHASE9_RESET_TASK`` below, because the comparison between the two protocols is itself a
result (``docs/DECISIONS.md`` D026).

Nothing in this module is loaded by the controllers. They keep taking their parameters as
arguments; this is the *experiment* definition, which is a separate thing (and the reason
``d_goal`` still does not appear anywhere near the execution controller -- see
``docs/DECISIONS.md`` D004).

Full reasoning, including the candidates that were rejected and why, is in
``docs/EXPERIMENT_SPACE.md`` and ``docs/ORACLE_LANDSCAPE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from probe_drawer.controllers.execution_pull_controller import ExecutionControllerCfg
from probe_drawer.controllers.probe_pull_controller import ProbeControllerCfg
from probe_drawer.envs.dynamics_randomization import DynamicsRandomizerCfg
from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = [
    "MAIN_TASK",
    "OOD_XI_RANGES",
    "PHASE9_RESET_TASK",
    "SEQUENTIAL_TRANSITION_STEPS",
    "RECOMMENDED_EXECUTION_CFG",
    "RECOMMENDED_PROBE_CFG",
    "RECOMMENDED_PROBE_TASK",
    "TRAINING_XI_RANGES",
    "MainTask",
    "ProbeTask",
    "XiRanges",
]


@dataclass(frozen=True)
class MainTask:
    """The task the paper's main experiment poses.

    Args:
        duration: :math:`T_\\text{goal}`, the fixed execution time (s).
        goal_displacement: :math:`d_\\text{goal}` (m).
        displacement_tolerance: :math:`\\epsilon_d` (m).
        velocity_tolerance: :math:`\\epsilon_v` (m/s).
        peak_force_range: Envelope of every hidden state's success band (N) -- the union of
            all bands, not the range of best forces. A predictor's output should be clipped
            to this.
    """

    duration: float
    goal_displacement: float
    displacement_tolerance: float
    velocity_tolerance: float
    peak_force_range: tuple[float, float]

    @property
    def criteria(self) -> SuccessCriteria:
        return SuccessCriteria(
            goal_displacement=self.goal_displacement,
            displacement_tolerance=self.displacement_tolerance,
            velocity_tolerance=self.velocity_tolerance,
        )

    def as_dict(self) -> dict:
        return {
            "duration": self.duration,
            "goal_displacement": self.goal_displacement,
            "displacement_tolerance": self.displacement_tolerance,
            "velocity_tolerance": self.velocity_tolerance,
            "peak_force_range": list(self.peak_force_range),
        }


@dataclass(frozen=True)
class ProbeTask:
    """The standardised probe's four task parameters."""

    initial_force: float
    max_force: float
    target_displacement: float
    max_velocity: float

    def as_dict(self) -> dict:
        return {
            "initial_force": self.initial_force,
            "max_force": self.max_force,
            "target_displacement": self.target_displacement,
            "max_velocity": self.max_velocity,
        }

    def as_kwargs(self) -> dict:
        """Ready to splat into :meth:`ProbePullController.run`."""
        return self.as_dict()


@dataclass(frozen=True)
class XiRanges:
    """Sampling ranges for the four hidden dimensions.

    ``dynamic_friction_ratio`` rather than an absolute range, because PhysX requires
    ``mu_d <= mu_s`` and a ratio keeps every draw inside that region by construction.
    """

    mass: tuple[float, float]
    static_friction: tuple[float, float]
    dynamic_friction_ratio: tuple[float, float]
    damping: tuple[float, float]

    def as_randomizer_cfg(self) -> DynamicsRandomizerCfg:
        return DynamicsRandomizerCfg(
            mass_range=self.mass,
            static_friction_range=self.static_friction,
            dynamic_friction_ratio_range=self.dynamic_friction_ratio,
            damping_range=self.damping,
        )

    def as_dict(self) -> dict:
        return {
            "mass": list(self.mass),
            "static_friction": list(self.static_friction),
            "dynamic_friction_ratio": list(self.dynamic_friction_ratio),
            "damping": list(self.damping),
        }


#: The main experiment's task, selected against the **sequential** Oracle.
#:
#: ``d_goal`` is measured from the drawer's position at the start of the task, *before* the
#: probe, so the quantity judged is ``d_probe + d_execution`` (``docs/DECISIONS.md`` D027).
#:
#: Chosen by the priority order in ``scripts/refine_task_space.py`` -- coverage, then
#: position precision, then terminal-velocity precision, then the remaining acceptance
#: conditions, and only then the spread of required force. Measured at this operating point
#: over the 108-point hidden-state grid: **105 of 108** hidden states have a succeeding
#: force, the best force per hidden state spans **0.20-4.30 N, a 21.5x range** (median
#: 1.50 N), the median success band is 0.20 N wide (0.14 relative), 100 of 105 bands are
#: contiguous, and no succeeding episode exceeds 12 % of the drawer's travel.
#:
#: ``eps_d`` is 7.5 mm rather than 5 mm for a reason that is not coverage: at 5 mm the
#: coverage is actually higher (0.981), but the success band collapses to 0.10 N -- one force
#: grid step, and 7 % of the required force. That is a knife edge no regression could be
#: expected to hit, so it fails the project's own learnability floor.
MAIN_TASK = MainTask(
    duration=1.5,
    goal_displacement=0.04,
    displacement_tolerance=0.0075,
    velocity_tolerance=0.03,
    peak_force_range=(0.15, 4.5),
)

#: The Phase 9 task, selected against the reset Oracle. Kept for the protocol comparison in
#: ``docs/ORACLE_LANDSCAPE.md``; **not** the paper's task.
PHASE9_RESET_TASK = MainTask(
    duration=1.5,
    goal_displacement=0.05,
    displacement_tolerance=0.015,
    velocity_tolerance=0.08,
    peak_force_range=(1.0, 5.0),
)

#: The execution profile the task was selected with.
#:
#: ``fall_fraction`` is 0.35, up from Phase 9's 0.20 and the original 0.10. The terminal
#: velocity requirement is what drives it: a drawer still moving when the force reaches zero
#: has not been placed at the goal, and a short ramp-down leaves a low-resistance drawer no
#: time to decelerate. Measured over the grid, the largest ``d(T)`` reachable with
#: ``|v(T)| <= 0.05 m/s`` at ``T = 1.5 s`` grew from 49 mm at ``fall = 0.10`` to 83 mm at 0.35,
#: and the selected task's coverage at ``eps_v = 0.03`` is 0.972 at 0.35 against 0.954 at 0.30
#: (``docs/DECISIONS.md`` D023, revised).
RECOMMENDED_EXECUTION_CFG = ExecutionControllerCfg(fall_fraction=0.35, settle_steps=0)

#: The standardised probe's task parameters.
#:
#: Selected as the least intrusive of seven candidates whose best feature correlated with
#: the required force within 0.02 of the ceiling (|rho| = 0.978, against the reset Oracle;
#: |rho| = 0.910 against the sequential one). Measured: every one of the 108 hidden states
#: breaks away, every probe terminates on displacement, the median probe lasts 0.467 s, and
#: the probe travels 3.3-3.7 mm -- 8-9 % of the 40 mm goal, which now counts towards it
#: (``docs/DECISIONS.md`` D027).
#:
#: Unchanged from Phase 9 on purpose, so that the protocol is the only thing that differs
#: between the two Oracles. It does not identify damping; that is recorded as a limitation
#: rather than fixed with a second probe segment (D032).
RECOMMENDED_PROBE_TASK = ProbeTask(
    initial_force=1.0,
    max_force=6.0,
    target_displacement=0.003,
    max_velocity=0.08,
)

#: The probe's fixed character, unchanged from Phase 8 except that it is now confirmed
#: rather than assumed: a 1 s linear ramp inside a 1.5 s budget.
RECOMMENDED_PROBE_CFG = ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)

#: Control steps of zero pull force between the probe ending and the execution starting.
#:
#: A deployed system needs wall-clock time to run its adaptation model, and the gap is
#: reserved explicitly and identically in every episode. Eight steps is 133 ms at 60 Hz.
#:
#: Chosen on repeatability, not on being short. Measured over six identical episodes at the
#: task's operating point (F = 4.25 N), the spread of ``d_total(T)`` was 3.58 mm with no gap,
#: 2.61 mm at 2 steps, 3.29 mm at 4, **0.90 mm at 8** and 1.40 mm at 12 -- a clear minimum.
#: A second run over 4, 8 and 12 steps gave 1.66, 1.14 and 1.40 mm, so the floor at 8 steps is
#: about 0.9-1.1 mm and the ordering is reproducible. The mechanism is
#: that ``dd/dF`` reaches about 40 mm/N just above breakaway, so the residual velocity the
#: probe leaves is amplified into the finished task; letting the drawer coast to a near-stop
#: under its own friction removes that amplification. Nothing is written to the simulation --
#: the velocity decays by physics, retaining under 0.2 % of its probe-end value at 8 steps
#: (``docs/DECISIONS.md`` D028).
SEQUENTIAL_TRANSITION_STEPS = 8

#: Training distribution for the hidden state.
#:
#: The swept grid's bounds. Its upper friction bound is set by the operating point rather
#: than by the simulator: only about 60-80 % of a commanded force reaches the drawer, so a
#: static friction much above 3 N cannot be broken away inside the task's force range.
TRAINING_XI_RANGES = XiRanges(
    mass=(4.0, 12.0),
    static_friction=(0.5, 3.0),
    dynamic_friction_ratio=(0.3, 1.0),
    damping=(2.0, 10.0),
)

#: Out-of-distribution ranges, one step beyond training on every axis.
#:
#: Chosen to stay inside the *physically valid* region: the coarse sweep reached 12 N and
#: 3 s without the simulation misbehaving, so these are extrapolations of the task, not of
#: the simulator. A drawer at the top of this friction range needs a force near the top of
#: the task's range, which is the point.
OOD_XI_RANGES = XiRanges(
    mass=(2.0, 18.0),
    static_friction=(0.25, 4.5),
    dynamic_friction_ratio=(0.15, 1.0),
    damping=(1.0, 16.0),
)
