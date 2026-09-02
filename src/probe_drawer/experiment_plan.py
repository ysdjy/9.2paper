r"""The experiment parameters Phase 9 selected, and where each number came from.

Every value here is the output of a sweep, not a preference. The provenance is recorded
alongside it so the next phase can cite it and a reviewer can re-derive it:

* the execution operating point and the success tolerances come from
  ``scripts/build_oracle_landscape.py`` (report: ``outputs/logs/oracle_landscape.json``);
* the probe parameters come from ``scripts/calibrate_probe.py``
  (report: ``outputs/logs/probe_calibration.json``);
* the hidden-state ranges come from the sweeps in ``outputs/logs/sweep_fine_fall*.json``.

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
        peak_force_range: The span of :math:`F_\\text{peak}` the training hidden states
            require (N). A predictor's output should be clipped to this.
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


#: The main experiment's task.
#:
#: Selected as the accepted candidate with the greatest spread of required force across the
#: 108-point hidden-state grid. Measured at this operating point: 106 of 108 hidden states
#: have a succeeding force, the required force spans 1.00-4.50 N (a 4.5x range), the median
#: success band is 0.50 N wide (0.16 relative), 99 % of bands are contiguous, and no
#: succeeding episode exceeds 16 % of the drawer's travel.
MAIN_TASK = MainTask(
    duration=1.5,
    goal_displacement=0.05,
    displacement_tolerance=0.015,
    velocity_tolerance=0.08,
    peak_force_range=(1.0, 5.0),
)

#: The execution profile the task was selected with.
#:
#: ``fall_fraction`` is 0.20, not the earlier 0.10. With a 10 % ramp-down a drawer that
#: travels 50 mm in 1.5 s is still moving at roughly 0.16 m/s when the force reaches zero,
#: so "reach the goal *and* come to rest" was unreachable for most hidden states: the
#: largest displacement achievable with ``|v(T)| <= 0.05 m/s`` was 49 mm at ``fall = 0.10``
#: against 79 mm at ``fall = 0.35``. 0.20 was the best-scoring value in the 0.15-0.30 band.
RECOMMENDED_EXECUTION_CFG = ExecutionControllerCfg(fall_fraction=0.20)

#: The standardised probe's task parameters.
#:
#: Selected as the least intrusive of seven candidates whose best feature correlated with
#: the required force within 0.02 of the ceiling (|rho| = 0.978). Measured: every one of the
#: 108 hidden states breaks away, every probe terminates on displacement, the median probe
#: lasts 0.467 s, and the probe travels 3.5 mm -- 6.9 % of the 50 mm goal.
RECOMMENDED_PROBE_TASK = ProbeTask(
    initial_force=1.0,
    max_force=6.0,
    target_displacement=0.003,
    max_velocity=0.08,
)

#: The probe's fixed character, unchanged from Phase 8 except that it is now confirmed
#: rather than assumed: a 1 s linear ramp inside a 1.5 s budget.
RECOMMENDED_PROBE_CFG = ProbeControllerCfg(ramp_duration=1.0, max_probe_duration=1.5)

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
