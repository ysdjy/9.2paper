r"""How far can this drawer actually be pulled, and what breaks first?

The task has used :math:`d_\text{goal} = 40` mm since Phase 9, chosen from a sweep that never
looked past 100 mm. The drawer's travel is 400 mm, so most of its range has never been tested,
and "long pulls are probably a problem" has been an assumption rather than a measurement.

This module holds the measurement. For each candidate goal distance it asks not only *whether*
some execution lands there, but **what the binding constraint is** when it does not. Four
candidate explanations, and they are distinguishable in the data:

``drawer_limit``
    The drawer reaches its own end stop. A property of the cabinet.
``posture``
    The arm runs out of reach or manipulability along the pull axis, or approaches a joint
    limit. A property of the robot's configuration, and the one that would argue for
    repositioning the base rather than shortening the task.
``control``
    The five pose-held degrees of freedom drift, or the OSC destabilises, while the drawer and
    the joints are both fine. A property of the controller.
``task``
    Nothing breaks; no ``(F, T)`` in the swept box simply happens to land there with a small
    enough terminal velocity. A property of the parameter range, fixable by widening it.

Telling these apart is the point. "Performance degrades past 250 mm" is not actionable;
"performance degrades past 250 mm because the wrist passes within 8 % of a joint limit" is.

Nothing here touches a simulator -- it consumes records produced by
``scripts/sweep_goal_distance.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "GoalFeasibility",
    "LongPullRecord",
    "attribute_binding_constraint",
    "feasibility_by_goal",
]

#: Goal distances to test, in metres.
#:
#: 40 mm is the incumbent task; 400 mm is the drawer's full travel, so the top of the range is
#: deliberately at the mechanical limit rather than short of it -- the question is where
#: feasibility ends, and that cannot be answered by a sweep that stops before it.
CANDIDATE_GOALS = (0.040, 0.060, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.390)

#: Fraction of a joint's range, measured from either end, inside which it counts as "near its
#: limit".
#:
#: 10 % of the range is far enough from the stop that the actuator is still linear and the
#: soft limits Isaac Lab imposes are not yet active, and close enough that a controller error
#: could reach it. Reported as a continuous margin too, so the threshold can be revisited
#: without re-running anything.
JOINT_LIMIT_MARGIN = 0.10

#: Fraction of the drawer's travel above which a pull counts as "near the drawer's limit".
#:
#: Matches ``OperatingRegionCfg.mechanical_margin_fraction`` so this analysis and the validity
#: check agree on what "close to the end stop" means.
DRAWER_LIMIT_MARGIN = 0.80


@dataclass
class LongPullRecord:
    r"""One ``(xi, F, T)`` episode, with every diagnostic the attribution needs.

    Attributes:
        xi: The four hidden values.
        peak_force, duration: The execution parameters (N, s).
        final_displacement: :math:`d_\text{total}(T)` from the task's start (m).
        final_velocity: :math:`v(T)` (m/s).
        peak_velocity: Largest drawer speed during the episode (m/s).
        peak_measured_force: Largest wrist pull force (N).
        wrist_force_spike: Largest single-step jump in the wrist force (N). An impact against
            the end stop shows here far more clearly than in the peak, which a slow hard pull
            also raises.
        peak_lateral_drift, peak_orientation_drift_deg: Held-axis error (m, degrees).
        travel_fraction: ``final_displacement`` over the drawer's 400 mm travel.
        joint_position: Arm joint angles at ``T`` (rad).
        min_joint_margin: Smallest distance to a joint limit at ``T``, as a fraction of that
            joint's range. Small means the arm is running out of configuration space.
        limiting_joint: Which joint that was.
        manipulability: :math:`\sqrt{\det(JJ^\top)}` at ``T``. Zero at a singularity.
        pull_axis_transmission: :math:`1/\sqrt{u^\top (JJ^\top)^{-1} u}` along the pull
            direction -- how much end-effector velocity a unit of joint velocity buys *in the
            direction that matters*. More informative than the determinant, which can stay
            healthy while the one useful direction collapses.
        jacobian_condition: Condition number of ``J``. Large means ill-conditioned.
        safety_aborted, valid, invalid_reasons: From the controller and the operating region.
    """

    xi: dict
    peak_force: float
    duration: float
    final_displacement: float
    final_velocity: float
    peak_velocity: float
    peak_measured_force: float
    wrist_force_spike: float
    peak_lateral_drift: float
    peak_orientation_drift_deg: float
    travel_fraction: float
    joint_position: list[float]
    min_joint_margin: float
    limiting_joint: int
    manipulability: float
    pull_axis_transmission: float
    jacobian_condition: float
    safety_aborted: bool
    valid: bool
    invalid_reasons: list[str] = field(default_factory=list)

    @property
    def xi_key(self) -> tuple[float, ...]:
        return tuple(self.xi[name] for name in ("mass", "static_friction", "dynamic_friction", "damping"))

    @property
    def near_joint_limit(self) -> bool:
        return self.min_joint_margin < JOINT_LIMIT_MARGIN

    @property
    def near_drawer_limit(self) -> bool:
        return self.travel_fraction > DRAWER_LIMIT_MARGIN

    def reaches(self, goal: float, tolerance: float) -> bool:
        return abs(self.final_displacement - goal) <= tolerance

    def succeeds(self, goal: float, tolerance: float, velocity_tolerance: float) -> bool:
        return (
            self.valid
            and not self.safety_aborted
            and self.reaches(goal, tolerance)
            and abs(self.final_velocity) <= velocity_tolerance
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> LongPullRecord:
        return cls(**payload)


def attribute_binding_constraint(
    records: list[LongPullRecord], goal: float, tolerance: float, velocity_tolerance: float
) -> dict:
    r"""Why this hidden state cannot reach this goal, when it cannot.

    Looks only at the episodes that got *closest* to the goal, because those are the ones
    whose failure is informative: an episode that undershot by 200 mm tells you nothing about
    what stops a 300 mm pull.

    Returns the attribution and the evidence behind it, so a reader can disagree with the
    label without re-running the sweep.
    """
    if not records:
        return {"attribution": "no_data", "evidence": {}}

    reached = [record for record in records if record.reaches(goal, tolerance)]
    if any(record.succeeds(goal, tolerance, velocity_tolerance) for record in records):
        return {"attribution": "feasible", "evidence": {}}

    # Nothing succeeded. The candidates that at least arrived at the right distance are the
    # ones that say why it did not count.
    pool = reached or sorted(records, key=lambda record: abs(record.final_displacement - goal))[:5]
    furthest = max(record.final_displacement for record in records)

    evidence = {
        "reached_the_distance": bool(reached),
        "closest_displacement": float(min(abs(r.final_displacement - goal) for r in records)),
        "furthest_reached": float(furthest),
        "near_drawer_limit_fraction": float(np.mean([r.near_drawer_limit for r in pool])),
        "near_joint_limit_fraction": float(np.mean([r.near_joint_limit for r in pool])),
        "min_joint_margin": float(min(r.min_joint_margin for r in pool)),
        "median_lateral_drift": float(np.median([r.peak_lateral_drift for r in pool])),
        "median_terminal_velocity": float(np.median([abs(r.final_velocity) for r in pool])),
        "invalid_reasons": dict(Counter(reason for r in pool for reason in r.invalid_reasons)),
        "safety_aborts": int(sum(r.safety_aborted for r in pool)),
    }

    # Ordered by how fundamental the constraint is. The drawer's end stop cannot be engineered
    # around; a posture problem can be, by moving the base; a control problem can be, by
    # tuning; and "the box did not contain a suitable (F, T)" is not a constraint at all.
    if furthest < goal - tolerance:
        attribution = "unreachable_in_swept_box" if not evidence["near_drawer_limit_fraction"] else "drawer_limit"
    elif evidence["near_drawer_limit_fraction"] > 0.5:
        attribution = "drawer_limit"
    elif evidence["near_joint_limit_fraction"] > 0.5 or evidence["min_joint_margin"] < JOINT_LIMIT_MARGIN:
        attribution = "posture"
    elif any(
        reason in evidence["invalid_reasons"]
        for reason in ("excessive_lateral_drift", "excessive_orientation_drift")
    ) or evidence["safety_aborts"]:
        attribution = "control"
    else:
        attribution = "task"
    return {"attribution": attribution, "evidence": evidence}


@dataclass
class GoalFeasibility:
    """What one candidate goal distance costs, over all hidden states."""

    goal: float
    tolerance: float
    velocity_tolerance: float
    hidden_states: int
    feasible_states: int
    feasible_fraction: float
    stable_control_fraction: float
    near_joint_limit_fraction: float
    near_drawer_limit_fraction: float
    median_lateral_drift: float
    worst_lateral_drift: float
    median_orientation_drift_deg: float
    worst_orientation_drift_deg: float
    median_wrist_force: float
    worst_wrist_force: float
    worst_wrist_spike: float
    median_manipulability: float
    worst_pull_axis_transmission: float
    median_peak_velocity: float
    attributions: dict
    recommended: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def feasibility_by_goal(
    records: list[LongPullRecord],
    goals: tuple[float, ...] = CANDIDATE_GOALS,
    tolerance: float = 0.0075,
    velocity_tolerance: float = 0.03,
    min_feasible_fraction: float = 0.90,
    max_worst_lateral_drift: float = 0.005,
) -> list[GoalFeasibility]:
    """Per goal distance: can it be hit, how safely, and should it be a candidate task?

    Args:
        records: Every swept episode.
        goals: Candidate goal distances (m).
        tolerance: Position tolerance to judge with -- the task's ``eps_d``.
        velocity_tolerance: The task's ``eps_v``.
        min_feasible_fraction: Fraction of hidden states that must be reachable for the goal
            to be recommended. 0.90 matches the coverage floor the task selection has used
            since Phase 9.
        max_worst_lateral_drift: The operating region's own drift bound. A goal whose *worst*
            episode exceeds it is recommended against even if the median is fine, because the
            worst case is what a deployed system meets.

    Returns:
        One :class:`GoalFeasibility` per goal, in the order given.
    """
    by_state: dict[tuple[float, ...], list[LongPullRecord]] = {}
    for record in records:
        by_state.setdefault(record.xi_key, []).append(record)

    results = []
    for goal in goals:
        feasible_states = []
        attributions: Counter = Counter()
        # Only the episodes that land near this goal describe what reaching *it* costs;
        # averaging over the whole sweep would mix in gentle 40 mm pulls.
        relevant: list[LongPullRecord] = []

        for key, rows in by_state.items():
            near = [row for row in rows if row.reaches(goal, tolerance)]
            relevant.extend(near)
            verdict = attribute_binding_constraint(rows, goal, tolerance, velocity_tolerance)
            attributions[verdict["attribution"]] += 1
            if verdict["attribution"] == "feasible":
                feasible_states.append(key)

        pool = relevant or [
            min(rows, key=lambda row: abs(row.final_displacement - goal)) for rows in by_state.values()
        ]
        stable = [row for row in pool if row.valid and not row.safety_aborted]

        feasible_fraction = len(feasible_states) / len(by_state) if by_state else 0.0
        worst_lateral = float(max(row.peak_lateral_drift for row in pool))
        reasons = []
        if feasible_fraction < min_feasible_fraction:
            reasons.append(f"only {feasible_fraction * 100:.0f}% of hidden states reachable")
        if worst_lateral > max_worst_lateral_drift:
            reasons.append(f"worst lateral drift {worst_lateral * 1000:.1f} mm")
        results.append(
            GoalFeasibility(
                goal=goal,
                tolerance=tolerance,
                velocity_tolerance=velocity_tolerance,
                hidden_states=len(by_state),
                feasible_states=len(feasible_states),
                feasible_fraction=feasible_fraction,
                stable_control_fraction=len(stable) / len(pool) if pool else float("nan"),
                near_joint_limit_fraction=float(np.mean([row.near_joint_limit for row in pool])),
                near_drawer_limit_fraction=float(np.mean([row.near_drawer_limit for row in pool])),
                median_lateral_drift=float(np.median([row.peak_lateral_drift for row in pool])),
                worst_lateral_drift=worst_lateral,
                median_orientation_drift_deg=float(
                    np.median([row.peak_orientation_drift_deg for row in pool])
                ),
                worst_orientation_drift_deg=float(max(row.peak_orientation_drift_deg for row in pool)),
                median_wrist_force=float(np.median([row.peak_measured_force for row in pool])),
                worst_wrist_force=float(max(row.peak_measured_force for row in pool)),
                worst_wrist_spike=float(max(row.wrist_force_spike for row in pool)),
                median_manipulability=float(np.median([row.manipulability for row in pool])),
                worst_pull_axis_transmission=float(min(row.pull_axis_transmission for row in pool)),
                median_peak_velocity=float(np.median([row.peak_velocity for row in pool])),
                attributions=dict(attributions),
                recommended=not reasons,
                reason="; ".join(reasons) if reasons else "meets the coverage and drift bounds",
            )
        )
    return results
