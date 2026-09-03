r"""Scoring the candidate fixed-budget probes, by a rule written down before the data existed.

Setting V1's probe is one force profile :math:`F_\text{probe}\,\phi(t/H)` applied to every
hidden state (``docs/DECISIONS.md`` D044). Two numbers pick it out -- the amplitude
:math:`F_\text{probe}` and the budget :math:`H` -- and this module says how they are chosen,
separately from the script that gathers the measurements, so the rule can be read and
unit-tested without running a simulation.

The tension the rule has to resolve
-----------------------------------
One fixed force cannot be gentle with every drawer. Breakaway is a *force* threshold, so an
amplitude below roughly ``mu_s / 0.7`` leaves the stiffest hidden states motionless and the
probe learns nothing about exactly the cases that most need identifying. But the same
amplitude applied to the softest state accelerates it freely, and over a long budget it
travels a large fraction of the goal -- at which point the probe has performed the task
rather than measured it. The lever that resolves this is the *budget*, not the amplitude:
displacement grows roughly with :math:`H^2` while breakaway does not depend on :math:`H` at
all. So the rule below is willing to pay amplitude and unwilling to pay time.

The rule
--------
Four gates, then one score. Gates first because each is a property the probe must have to be
usable at all, and no amount of predictive power compensates for lacking one:

1. **Safe.** No hidden state may trip a safety limit during the probe.
2. **Responsive.** Every hidden state must break away. A probe that cannot move the stiffest
   drawer returns a constant for it, and a constant identifies nothing.
3. **Non-intrusive.** The largest post-probe displacement, across hidden states, must stay
   within :data:`MAX_INTRUSION` of the goal. A probe that has already travelled a third of
   the way has begun the task.
4. **Leaves the task doable.** At least :data:`MIN_REACH_COVERAGE` of hidden states must have
   some candidate force that reaches the goal afterwards. This can fail even when the first
   three pass -- an intrusive probe can overshoot a near hidden state past every tolerance.

Among the candidates that pass all four, the score is the **leave-one-out RMSE of the
required peak force** predicted from the probe's own features: the smaller, the more the
probe has actually told us about what this drawer will need. RMSE rather than
:math:`R^2` because the candidates have different required-force spreads and
:math:`R^2` would reward the spread (``probe_drawer.analysis.readout``). Ties within
:data:`TIE_FRACTION` go to the shorter probe, because probe time is cost and nothing else.

If no candidate passes, that is reported as such. Widening the candidate set is a decision
for a person, not a fallback for this function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probe_drawer.analysis.readout import leave_one_out

__all__ = [
    "MAX_INTRUSION",
    "MIN_REACH_COVERAGE",
    "TIE_FRACTION",
    "CandidateOutcome",
    "FixedProbeCandidate",
    "XiOutcome",
    "score_candidate",
    "select_candidate",
]

#: Largest allowed post-probe displacement, as a fraction of the goal.
#:
#: A third of the way is where "measuring" stops being an honest description. The Phase 8-11
#: ramp probe sat at 8-9 % of a 40 mm goal, so this is loose by comparison and is a ceiling
#: rather than a target.
MAX_INTRUSION = 0.30

#: Fraction of hidden states that must remain solvable after the probe.
#:
#: Not 1.0: Dataset v0 found 0.98 % of hidden states with no succeeding force even before any
#: probe, and demanding perfection here would reject a candidate for the task's own edges.
MIN_REACH_COVERAGE = 0.90

#: Two candidates whose RMSE differs by less than this fraction are treated as tied.
TIE_FRACTION = 0.05


@dataclass(frozen=True)
class FixedProbeCandidate:
    """One candidate excitation: an amplitude and a time budget.

    Args:
        peak_force: :math:`F_\\text{probe}`, the plateau force (N).
        duration: :math:`H`, the total probe time including rise and release (s).
    """

    peak_force: float
    duration: float

    def __post_init__(self) -> None:
        if self.peak_force <= 0.0:
            raise ValueError(f"peak_force must be > 0 N, got {self.peak_force}.")
        if self.duration <= 0.0:
            raise ValueError(f"duration must be > 0 s, got {self.duration}.")

    @property
    def label(self) -> str:
        return f"F{self.peak_force:g}N_H{self.duration:g}s"

    def as_dict(self) -> dict:
        return {"peak_force": self.peak_force, "duration": self.duration, "label": self.label}


@dataclass
class XiOutcome:
    """What one candidate probe did to one hidden state, and what the task then needed.

    Args:
        hidden_state: The four sampled dimensions. Privileged; recorded for the report only
            and never handed to a model.
        moved: Whether the drawer broke away during the probe.
        post_probe_displacement: Drawer opening when the execution starts (m), i.e. after the
            probe *and* the inference gap. This is the quantity gate 3 reads.
        post_probe_velocity: Drawer speed at that instant (m/s). Carried into the execution
            rather than zeroed (``docs/DECISIONS.md`` D029).
        safety_aborted: Whether a safety limit ended the probe.
        features: The probe's deployable feature vector, in ``PROBE_FEATURES`` order.
        required_force: Smallest candidate force that achieved ``reach_success``, or ``None``
            if no candidate force did.
    """

    hidden_state: dict
    moved: bool
    post_probe_displacement: float
    post_probe_velocity: float
    safety_aborted: bool
    features: tuple[float, ...]
    required_force: float | None


@dataclass
class CandidateOutcome:
    """A candidate's gate results and score, with every input a reader would want to check."""

    candidate: FixedProbeCandidate
    gates: dict[str, bool]
    metrics: dict
    readout: dict
    rows: list[XiOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    @property
    def score(self) -> float:
        """Leave-one-out RMSE of the required force (N). ``inf`` when it could not be fit."""
        rmse = self.readout.get("rmse", float("nan"))
        return float(rmse) if np.isfinite(rmse) else float("inf")

    def as_dict(self) -> dict:
        return {
            "candidate": self.candidate.as_dict(),
            "gates": dict(self.gates),
            "passed": self.passed,
            "metrics": dict(self.metrics),
            "readout": dict(self.readout),
            "score_rmse": self.score,
        }


def score_candidate(
    candidate: FixedProbeCandidate, rows: list[XiOutcome], goal_displacement: float
) -> CandidateOutcome:
    """Apply the four gates and compute the score for one candidate.

    Args:
        candidate: The excitation that produced ``rows``.
        rows: One entry per hidden state, all probed with ``candidate``.
        goal_displacement: :math:`d_\\text{goal}` (m), which the intrusion gate is relative to.

    Returns:
        A :class:`CandidateOutcome`. Gates are reported individually rather than as a single
        boolean, so a rejected candidate says *which* property it lacked.

    Raises:
        ValueError: If ``rows`` is empty or ``goal_displacement`` is not positive.
    """
    if not rows:
        raise ValueError("score_candidate needs at least one hidden state.")
    if goal_displacement <= 0.0:
        raise ValueError(f"goal_displacement must be > 0 m, got {goal_displacement}.")

    solved = [row for row in rows if row.required_force is not None]
    intrusion = max(row.post_probe_displacement for row in rows) / goal_displacement
    coverage = len(solved) / len(rows)

    gates = {
        "safe": not any(row.safety_aborted for row in rows),
        "responsive": all(row.moved for row in rows),
        "non_intrusive": intrusion <= MAX_INTRUSION,
        "task_remains_solvable": coverage >= MIN_REACH_COVERAGE,
    }

    # Fit only on hidden states that have a required force: there is no target for the rest,
    # and substituting the range's ceiling would invent a label the sweep never produced.
    if len(solved) >= 2:
        readout = leave_one_out(
            np.asarray([row.features for row in solved], dtype=float),
            np.asarray([row.required_force for row in solved], dtype=float),
        )
    else:
        readout = {"r2": float("nan"), "rmse": float("nan"), "n": len(solved), "target_sd": float("nan")}

    forces = [row.required_force for row in solved]
    metrics = {
        "num_hidden_states": len(rows),
        "moved_fraction": sum(row.moved for row in rows) / len(rows),
        "safety_aborts": sum(row.safety_aborted for row in rows),
        "max_post_probe_displacement": max(row.post_probe_displacement for row in rows),
        "median_post_probe_displacement": float(np.median([row.post_probe_displacement for row in rows])),
        "max_intrusion_fraction": intrusion,
        "max_post_probe_velocity": max(abs(row.post_probe_velocity) for row in rows),
        "reach_coverage": coverage,
        "required_force_min": min(forces) if forces else None,
        "required_force_max": max(forces) if forces else None,
        "required_force_ratio": (max(forces) / min(forces)) if forces and min(forces) > 0 else None,
    }
    return CandidateOutcome(candidate=candidate, gates=gates, metrics=metrics, readout=readout, rows=rows)


def select_candidate(outcomes: list[CandidateOutcome]) -> CandidateOutcome | None:
    """The winner under the documented rule, or ``None`` if nothing passed the gates.

    Lowest leave-one-out RMSE among gate-passing candidates; among candidates within
    :data:`TIE_FRACTION` of the best, the shortest probe. Returning ``None`` rather than the
    least-bad candidate is deliberate: a probe that fails a gate is not a probe with a worse
    score, and quietly adopting one would hide that from the report.
    """
    passing = [outcome for outcome in outcomes if outcome.passed and np.isfinite(outcome.score)]
    if not passing:
        return None
    best = min(outcome.score for outcome in passing)
    tied = [outcome for outcome in passing if outcome.score <= best * (1.0 + TIE_FRACTION)]
    return min(tied, key=lambda outcome: (outcome.candidate.duration, outcome.score))
