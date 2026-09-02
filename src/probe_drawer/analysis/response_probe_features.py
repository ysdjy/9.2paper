r"""Scalar features of the three-phase probe, including the ones only a coast can give.

The old probe's features all describe a drawer being *pushed*: how long the ramp took, what
force broke it loose, how far it went per newton. They are informative about friction and
almost silent about damping, because a drawer under a rising force is dominated by the
friction it has to overcome.

The coast phase changes what is measurable. With no applied force,

.. math:: m\,a = -(\mu_d \operatorname{sign}(v) + b\,v)

so for :math:`v > 0` the deceleration is affine in velocity,

.. math:: a = -\frac{\mu_d}{m} - \frac{b}{m} v,

and a straight-line fit of measured acceleration against measured velocity recovers
:math:`\mu_d/m` as its intercept and :math:`b/m` as its **slope**. Damping enters the slope,
which is a quantity nothing in the old probe measured.

That is the hypothesis. Whether it survives contact with a 60 Hz causal velocity filter and a
coast that lasts a few hundred milliseconds is what ``scripts/compare_probes.py`` measures,
and it is measured rather than assumed -- the fit's own :math:`R^2` is returned alongside
every coefficient so a bad fit is visible instead of silently producing a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from probe_drawer.controllers.response_probe import ProbePhase, ResponseProbeResult

__all__ = ["RESPONSE_PROBE_FEATURES", "ResponseProbeFeatures", "extract_response_features"]

#: Smallest coast segment worth fitting a line to.
#:
#: Four samples: three is the minimum for a two-parameter fit to have any residual at all,
#: and four leaves one degree of freedom to notice a bad one. Below this the fit is reported
#: as ``nan`` rather than as a number the data cannot support.
MIN_COAST_SAMPLES = 4

#: Speed below which a coast sample carries no information about damping.
#:
#: The Coulomb term dominates as ``v`` approaches zero and the causal velocity estimate is
#: quantised by the 60 Hz position difference, so near-stationary samples are mostly filter
#: noise. Excluding them is what makes the slope identifiable rather than dominated by the
#: cluster at the origin.
MIN_COAST_VELOCITY = 1e-3


@dataclass(frozen=True)
class ResponseProbeFeatures:
    r"""One environment's probe, as scalars.

    Attributes:
        ramp_duration: How long the force had to rise before the drawer moved (s).
        release_force: The force being commanded when the ramp ended (N).
        trigger_displacement: Displacement at the end of the ramp (m).
        breakaway_force, breakaway_time: The commanded force and time at the first sample
            where the drawer exceeded ``MIN_COAST_VELOCITY``. The old probe's analogues.
        displacement_per_newton: ``trigger_displacement / release_force`` (m/N).
        peak_velocity: Largest speed during the whole probe (m/s).
        peak_acceleration: Largest magnitude of acceleration during the ramp (m/s^2).
        release_velocity: Speed at the moment the commanded force reached zero (m/s).
        coast_duration: Time spent with zero commanded force before near-rest (s).
        coast_distance: Distance travelled during the coast (m).
        coast_velocity_ratio: Fraction of the release velocity still present at the end of
            the coast. A drawer with high damping loses it faster.
        coast_friction_over_mass: :math:`\mu_d/m` from the coast fit's intercept (m/s^2).
        coast_damping_over_mass: :math:`b/m` from the coast fit's slope (1/s). **The feature
            this probe exists to produce.**
        coast_fit_r2: The fit's coefficient of determination. Low means the affine model did
            not describe this coast and the two coefficients above should not be trusted.
        coast_samples: How many samples the fit used.
        total_duration, total_displacement: The whole probe's cost.
        reached_trigger, coasted_to_rest: Whether the probe did what it set out to.
    """

    ramp_duration: float
    release_force: float
    trigger_displacement: float
    breakaway_force: float
    breakaway_time: float
    displacement_per_newton: float
    peak_velocity: float
    peak_acceleration: float
    release_velocity: float
    coast_duration: float
    coast_distance: float
    coast_velocity_ratio: float
    coast_friction_over_mass: float
    coast_damping_over_mass: float
    coast_fit_r2: float
    coast_samples: int
    total_duration: float
    total_displacement: float
    reached_trigger: bool
    coasted_to_rest: bool

    def as_dict(self) -> dict:
        return asdict(self)


#: Feature names in a stable order, for building design matrices.
RESPONSE_PROBE_FEATURES: tuple[str, ...] = tuple(
    name for name in ResponseProbeFeatures.__dataclass_fields__ if name not in ("reached_trigger", "coasted_to_rest")
)


def _coast_fit(velocity: np.ndarray, acceleration: np.ndarray) -> dict:
    r"""Fit :math:`a = -\mu_d/m - (b/m)\,v` to a coast segment.

    Signs are handled by working with :math:`|v|` and the deceleration along the direction of
    travel, so a drawer coasting in either direction gives positive coefficients.
    """
    speed = np.abs(velocity)
    keep = speed > MIN_COAST_VELOCITY
    if int(keep.sum()) < MIN_COAST_SAMPLES:
        return {
            "friction_over_mass": float("nan"),
            "damping_over_mass": float("nan"),
            "r2": float("nan"),
            "samples": int(keep.sum()),
        }

    # Deceleration along the direction of motion: positive when slowing down.
    deceleration = -acceleration[keep] * np.sign(velocity[keep])
    design = np.column_stack([np.ones(int(keep.sum())), speed[keep]])
    solution, *_ = np.linalg.lstsq(design, deceleration, rcond=None)
    predicted = design @ solution
    variance = float(np.var(deceleration))
    return {
        "friction_over_mass": float(solution[0]),
        "damping_over_mass": float(solution[1]),
        "r2": float(1.0 - np.mean((predicted - deceleration) ** 2) / variance) if variance > 0 else float("nan"),
        "samples": int(keep.sum()),
    }


def extract_response_features(result: ResponseProbeResult, env_index: int = 0) -> ResponseProbeFeatures:
    """Summarise one environment's three-phase probe."""
    history = result.history
    phase = result.phase[:, env_index]
    time = history.time
    commanded = history.commanded_force[:, env_index]
    displacement = history.drawer_position[:, env_index]
    velocity = history.drawer_velocity[:, env_index]
    acceleration = history.drawer_acceleration[:, env_index]

    ramp = phase == int(ProbePhase.RAMP_UP)
    coast = phase == int(ProbePhase.COAST)

    moving = np.abs(velocity) > MIN_COAST_VELOCITY
    first_move = int(np.argmax(moving)) if moving.any() else -1
    release_index = int(np.nonzero(coast)[0][0]) - 1 if coast.any() else len(time) - 1
    release_index = max(0, min(release_index, len(time) - 1))

    fit = _coast_fit(velocity[coast], acceleration[coast]) if coast.any() else _coast_fit(
        np.zeros(0), np.zeros(0)
    )
    release_velocity = float(abs(velocity[release_index]))
    coast_indices = np.nonzero(coast)[0]

    trigger = float(result.trigger_displacement[env_index])
    release_force = float(result.release_force[env_index])
    return ResponseProbeFeatures(
        ramp_duration=float(result.ramp_duration[env_index]),
        release_force=release_force,
        trigger_displacement=trigger,
        breakaway_force=float(commanded[first_move]) if first_move >= 0 else float("nan"),
        breakaway_time=float(time[first_move]) if first_move >= 0 else float("nan"),
        # Guarded: a probe that aborted before applying force would divide by zero here, and
        # the honest answer is "unmeasured" rather than an infinity.
        displacement_per_newton=trigger / release_force if release_force > 1e-6 else float("nan"),
        peak_velocity=float(np.abs(velocity).max()) if velocity.size else float("nan"),
        peak_acceleration=float(np.abs(acceleration[ramp]).max()) if ramp.any() else float("nan"),
        release_velocity=release_velocity,
        coast_duration=float(result.coast_duration[env_index]),
        coast_distance=(
            float(displacement[coast_indices[-1]] - displacement[coast_indices[0]])
            if coast_indices.size
            else 0.0
        ),
        coast_velocity_ratio=(
            float(abs(velocity[coast_indices[-1]]) / release_velocity)
            if coast_indices.size and release_velocity > MIN_COAST_VELOCITY
            else float("nan")
        ),
        coast_friction_over_mass=fit["friction_over_mass"],
        coast_damping_over_mass=fit["damping_over_mass"],
        coast_fit_r2=fit["r2"],
        coast_samples=fit["samples"],
        total_duration=float(result.duration[env_index]),
        total_displacement=float(result.final_displacement[env_index]),
        reached_trigger=bool(result.reached_trigger[env_index]),
        coasted_to_rest=bool(result.coasted_to_rest[env_index]),
    )
