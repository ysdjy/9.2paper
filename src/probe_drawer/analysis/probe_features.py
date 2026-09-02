r"""Turning a probe history into a handful of numbers, and asking whether they identify xi.

A probe is only worth running if what it produces separates the hidden states it is meant
to distinguish -- and, more sharply, if it correlates with the answer the robot needs: the
peak force that will land the drawer on the goal.

Two groups of features come out of a probe:

*Coulomb / breakaway*, which the static friction should dominate
    when the drawer first moves, and how much force it took.
*Post-breakaway motion*, which dynamic friction, damping and mass should dominate
    the speed and acceleration reached once sliding, and how long the probe ran.

Every feature is computed from deployable channels only (``commanded_force``,
``drawer_position``, ``drawer_velocity``, ``drawer_acceleration``, and the TCP pull-axis
channels), so a feature that turns out to be predictive is one a real robot could also
compute. Nothing here reads the privileged force channels or the hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probe_drawer.controllers.types import ProbeResult
from probe_drawer.observations import OBSERVATION_SPECS, Deployability

__all__ = [
    "BREAKAWAY_SPEED",
    "PROBE_FEATURES",
    "ProbeFeatures",
    "assert_features_are_deployable",
    "extract_features",
    "rank_correlation",
]

#: Speed at which the drawer is considered to have broken away (m/s).
#:
#: Set an order of magnitude above the residual zero-command creep of roughly 1.3 mm/s
#: (``docs/DECISIONS.md`` D010) so the bias cannot be mistaken for motion, and far below the
#: speeds a probe reaches, so the instant is well defined.
BREAKAWAY_SPEED = 0.005

#: Feature names, in the order :meth:`ProbeFeatures.as_vector` returns them.
PROBE_FEATURES: tuple[str, ...] = (
    "breakaway_time",
    "breakaway_force",
    "duration",
    "final_commanded_force",
    "final_displacement",
    "final_velocity",
    "mean_speed_after_breakaway",
    "peak_acceleration",
    "displacement_per_newton",
)


@dataclass(frozen=True)
class ProbeFeatures:
    """Summary of one environment's probe response.

    Attributes:
        moved: Whether the drawer ever broke away. Everything else is meaningless if not.
        breakaway_time: When the drawer first exceeded :data:`BREAKAWAY_SPEED` (s).
        breakaway_force: The commanded force at that instant (N) -- the probe's estimate of
            what it takes to start this drawer moving.
        duration: How long the probe ran before its stop condition fired (s).
        final_commanded_force: Command at the stop instant (N).
        final_displacement: Drawer opening at the stop instant (m).
        final_velocity: Drawer speed at the stop instant (m/s).
        mean_speed_after_breakaway: Mean speed over the sliding part of the probe (m/s).
        peak_acceleration: Largest drawer acceleration seen (m/s^2).
        displacement_per_newton: Displacement divided by the force-time integral
            (m per N s) -- a compliance-like summary that mixes mass and resistance.
        termination_reason: Which stop condition fired.
    """

    moved: bool
    breakaway_time: float
    breakaway_force: float
    duration: float
    final_commanded_force: float
    final_displacement: float
    final_velocity: float
    mean_speed_after_breakaway: float
    peak_acceleration: float
    displacement_per_newton: float
    termination_reason: str

    def as_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in PROBE_FEATURES)

    def as_dict(self) -> dict:
        payload = {name: float(getattr(self, name)) for name in PROBE_FEATURES}
        payload["moved"] = self.moved
        payload["termination_reason"] = self.termination_reason
        return payload


def extract_features(result: ProbeResult, env_index: int) -> ProbeFeatures:
    """Summarise one environment's probe, using deployable channels only."""
    history = result.history
    driven = history.active_steps(env_index)
    time = history.time[driven]
    command = history.commanded_force[driven, env_index]
    displacement = history.drawer_position[driven, env_index]
    velocity = history.drawer_velocity[driven, env_index]
    acceleration = history.drawer_acceleration[driven, env_index]

    moving = np.abs(velocity) > BREAKAWAY_SPEED
    moved = bool(moving.any())
    first = int(np.argmax(moving)) if moved else len(time) - 1

    # numpy 1.26 here; `trapezoid` is the numpy 2 name for the same function.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    impulse = float(integrate(command, time)) if len(time) > 1 else 0.0
    duration = float(result.duration[env_index])

    return ProbeFeatures(
        moved=moved,
        breakaway_time=float(time[first]) if moved else duration,
        breakaway_force=float(command[first]) if moved else float(command[-1]),
        duration=duration,
        final_commanded_force=float(result.final_commanded_force[env_index]),
        final_displacement=float(result.final_displacement[env_index]),
        final_velocity=float(result.final_velocity[env_index]),
        mean_speed_after_breakaway=float(np.abs(velocity[first:]).mean()) if moved else 0.0,
        peak_acceleration=float(np.abs(acceleration).max()) if len(acceleration) else 0.0,
        displacement_per_newton=float(displacement[-1] / impulse) if impulse > 1e-9 else 0.0,
        termination_reason=result.termination_reason[env_index].value,
    )


def rank_correlation(left: list[float], right: list[float]) -> float:
    """Spearman rank correlation, computed without a SciPy dependency.

    Rank correlation rather than Pearson because the relationship a probe feature has with
    the required force is expected to be monotone but not linear, and because ranks are not
    thrown off by the one or two hidden states that sit far from the rest.

    Returns ``nan`` when either input is constant, since a constant carries no information.
    """
    if len(left) != len(right):
        raise ValueError(f"inputs must be the same length, got {len(left)} and {len(right)}.")
    if len(left) < 3:
        return float("nan")

    ranked_left, ranked_right = _ranks(left), _ranks(right)
    if np.std(ranked_left) == 0 or np.std(ranked_right) == 0:
        return float("nan")
    return float(np.corrcoef(ranked_left, ranked_right)[0, 1])


def _ranks(values: list[float]) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation.

    Ties are common here: probe durations are multiples of the control step, so many hidden
    states share one. Leaving them as arbitrary ordinal ranks would invent an ordering the
    data does not contain.
    """
    array = np.asarray(values, dtype=float)
    order = array.argsort()
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)

    _, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    for index in np.flatnonzero(counts > 1):
        tied = inverse == index
        ranks[tied] = ranks[tied].mean()
    return ranks


def assert_features_are_deployable() -> None:
    """Fail if any channel the feature extraction reads is not deployable.

    Called by the calibration script so that a probe feature can never quietly depend on
    something a real robot cannot measure.
    """
    required = (
        "commanded_force",
        "drawer_position",
        "drawer_velocity",
        "drawer_acceleration",
    )
    undeployable = [
        name for name in required if OBSERVATION_SPECS[name].deployability is not Deployability.DEPLOYABLE
    ]
    if undeployable:
        raise ValueError(f"Probe features read non-deployable channels: {undeployable}.")
