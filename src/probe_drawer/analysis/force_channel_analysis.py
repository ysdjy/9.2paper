r"""Comparing the force channels against the drawer's equation of motion.

Pure analysis: everything here works on a recorded
:class:`~probe_drawer.controllers.types.ExecutionResult` plus the hidden state that produced
it, so it needs no simulator and is unit-testable.

Along the drawer's single degree of freedom, with :math:`m` the total moving mass,
:math:`\mu_d` the dynamic friction effort and :math:`b` the viscous damping,

.. math::

    m\,a = F_\text{external} + F_\text{resistance},
    \qquad
    F_\text{resistance} = -\bigl(\mu_d\,\operatorname{sign}(v) + b\,v\bigr).

The resistance term is read directly out of PhysX, so the first identity above is a genuine
check of that channel against the hidden state rather than a restatement of it. The
delivered force follows from the same equation and is then compared against the wrist
sensor, which brackets the same interaction from the robot's side of the grasp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from probe_drawer.controllers.types import ExecutionResult
from probe_drawer.envs.dynamics_randomization import AppliedDynamics, DynamicsParameters

__all__ = [
    "AUDIT_CASES",
    "END_STOP_CASE",
    "ForceAuditCase",
    "analyse_end_stop_episode",
    "analyse_force_channels",
    "sliding_window",
]

#: Speed above which Coulomb friction is unambiguously at its dynamic value and its sign is
#: well defined (m/s). Below this the drawer may be in the stick regime, where the
#: resistance identity does not hold.
SLIDING_SPEED_THRESHOLD = 0.01

#: Fraction of the peak the command must have reached for a step to count as plateau.
PLATEAU_COMMAND_FRACTION = 0.95

#: Tolerance on the resistance identity, applied to the *window mean* (N).
#:
#: The identity is checked on means rather than per step because the two sides are sampled
#: differently: PhysX applies ``b * v`` with its own instantaneous velocity, while the
#: prediction uses this project's causally filtered velocity, which lags by a step and
#: carries a small ripple. With ``b = 8`` a 0.02 m/s ripple is already 0.16 N per step, so a
#: per-step bound would measure the filter, not the channel. Over a window the ripple
#: cancels: measured mean residuals are 0.000-0.010 N against forces of 2-3 N.
RESISTANCE_TOLERANCE = 0.05


@dataclass(frozen=True)
class ForceAuditCase:
    """One hidden state to audit the force channels at.

    The cases deliberately isolate the terms: no resistance at all, damping only, friction
    only, both, and a heavy drawer, so a channel that silently conflates two of them shows
    up as a residual in exactly one row.
    """

    name: str
    parameters: DynamicsParameters
    peak_force: float = 4.0
    duration: float = 1.5


AUDIT_CASES: tuple[ForceAuditCase, ...] = (
    ForceAuditCase("free (mu=0, b=0)", DynamicsParameters(8.0, 0.0, 0.0, 0.0, name="free")),
    ForceAuditCase("damping only (b=8)", DynamicsParameters(8.0, 0.0, 0.0, 8.0, name="damping_only")),
    ForceAuditCase("friction only (mu=2)", DynamicsParameters(8.0, 2.0, 2.0, 0.0, name="friction_only")),
    ForceAuditCase("friction + damping", DynamicsParameters(8.0, 2.0, 2.0, 6.0, name="friction_damping")),
    # mu_s above mu_d, but low enough that the audit's plateau force still breaks it away:
    # otherwise the drawer sticks and there is no sliding window to check.
    ForceAuditCase("asymmetric friction", DynamicsParameters(8.0, 2.5, 0.5, 4.0, name="asymmetric")),
    ForceAuditCase("heavy (m=14)", DynamicsParameters(14.0, 2.0, 2.0, 6.0, name="heavy")),
)

#: A deliberately under-damped drawer driven hard enough to reach its mechanical end stop,
#: to show where a wrist force far above the command comes from.
END_STOP_CASE = ForceAuditCase(
    "end-stop impact",
    DynamicsParameters(5.0, 1.5, 1.5, 4.0, name="end_stop"),
    peak_force=6.0,
    duration=2.5,
)


def sliding_window(result: ExecutionResult, env_index: int) -> np.ndarray:
    """Boolean mask of the steps where the resistance identity is meaningful.

    Requires the environment to be driven, the command to be on its plateau, and the drawer
    to be sliding rather than sticking.
    """
    history = result.history
    peak = float(result.peak_commanded_force[env_index])
    return (
        history.active_steps(env_index)
        & (history.commanded_force[:, env_index] >= PLATEAU_COMMAND_FRACTION * peak)
        & (np.abs(history.drawer_velocity[:, env_index]) > SLIDING_SPEED_THRESHOLD)
    )


def analyse_force_channels(
    result: ExecutionResult,
    parameters: list[DynamicsParameters],
    applied: AppliedDynamics,
    hand_mass: float | None = None,
) -> dict:
    """Compare all four force channels, per environment, over the sliding window.

    Args:
        result: The execution to analyse. One environment per entry of ``parameters``.
        parameters: The hidden state each environment ran under.
        applied: The randomiser's record, used for the drawer's total moving mass.
        hand_mass: Mass of the hand and fingers (kg), used to bound the expected gap between
            the delivered force and the wrist reading. ``None`` skips that bound.

    Returns:
        A serialisable report, including a verdict per check.
    """
    history = result.history
    rows: list[dict] = []

    for index, params in enumerate(parameters):
        window = sliding_window(result, index)
        if not window.any():
            rows.append({"name": params.name, "error": "no sliding plateau steps -- drawer never slid"})
            continue

        velocity = history.drawer_velocity[window, index]
        acceleration = history.drawer_acceleration[window, index]
        resistance = history.drawer_resistance_force[window, index]
        external = history.drawer_external_force[window, index]
        wrist = history.measured_force[window, index]
        commanded = history.commanded_force[window, index]

        predicted_resistance = -(params.joint_dynamic_friction * np.sign(velocity) + params.joint_damping * velocity)
        mass = applied.total_moving_mass[index]

        rows.append(
            {
                "name": params.name,
                "xi": params.as_dict(),
                "total_moving_mass": mass,
                "steps": int(window.sum()),
                "mean_velocity": float(velocity.mean()),
                "mean_acceleration": float(acceleration.mean()),
                "commanded_force": float(commanded.mean()),
                "measured_force": float(wrist.mean()),
                "drawer_resistance_force": float(resistance.mean()),
                "drawer_external_force": float(external.mean()),
                "predicted_resistance": float(predicted_resistance.mean()),
                "resistance_residual": float(abs(resistance.mean() - predicted_resistance.mean())),
                "resistance_residual_max_per_step": float(np.abs(resistance - predicted_resistance).max()),
                "external_minus_wrist": float(abs(external.mean() - wrist.mean())),
                "command_share": float(external.mean() / commanded.mean()) if commanded.mean() else None,
            }
        )

    measured_rows = [row for row in rows if "error" not in row]
    max_resistance_residual = max((row["resistance_residual"] for row in measured_rows), default=float("nan"))
    max_residual_per_step = max(
        (row["resistance_residual_max_per_step"] for row in measured_rows), default=float("nan")
    )
    max_gap = max((row["external_minus_wrist"] for row in measured_rows), default=float("nan"))

    bound = float("nan")
    if hand_mass is not None and measured_rows:
        peak_acceleration = max(abs(row["mean_acceleration"]) for row in measured_rows)
        # The wrist carries the hand and fingers as well as the drawer, so the two channels
        # are expected to differ by their inertial term. A floor keeps the bound meaningful
        # when the acceleration is momentarily near zero.
        bound = hand_mass * max(peak_acceleration, 0.05) + 0.2

    return {
        "cases": rows,
        "hand_mass": hand_mass,
        "hand_inertia_bound": bound,
        "max_resistance_residual": max_resistance_residual,
        "max_resistance_residual_per_step": max_residual_per_step,
        "max_external_wrist_gap": max_gap,
        "resistance_tolerance": RESISTANCE_TOLERANCE,
        "command_share": [row.get("command_share") for row in rows],
        "resistance_verdict": (
            "PASS" if max_resistance_residual <= RESISTANCE_TOLERANCE else "FAIL"
        ),
        "delivered_verdict": ("PASS" if not np.isfinite(bound) or max_gap <= bound else "FAIL"),
        "window_definition": {
            "sliding_speed_threshold": SLIDING_SPEED_THRESHOLD,
            "plateau_command_fraction": PLATEAU_COMMAND_FRACTION,
        },
    }


def analyse_end_stop_episode(result: ExecutionResult, env_index: int = 0) -> dict:
    """Locate the wrist-force spike relative to the drawer reaching its end stop.

    A wrist force several times the commanded force is alarming until you notice when it
    happens. This reports the peak, the step it occurs at, and the drawer displacement at
    that moment, so the coincidence with the mechanical limit is on the record.
    """
    from probe_drawer.evaluation.operating_region import DRAWER_TRAVEL_LIMIT  # noqa: PLC0415

    history = result.history
    driven = history.active_steps(env_index)
    wrist = history.measured_force[:, env_index]
    displacement = history.drawer_position[:, env_index]

    spike_step = int(np.argmax(np.abs(np.where(driven, wrist, 0.0))))
    return {
        "peak_commanded_force": float(result.peak_commanded_force[env_index]),
        "peak_abs_wrist_force": float(np.abs(wrist[driven]).max()),
        "wrist_to_command_ratio": float(np.abs(wrist[driven]).max() / result.peak_commanded_force[env_index]),
        "spike_step": spike_step,
        "spike_time": float(history.time[spike_step]),
        "displacement_at_spike": float(displacement[spike_step]),
        "travel_fraction_at_spike": float(displacement[spike_step] / DRAWER_TRAVEL_LIMIT),
        "final_displacement": float(result.final_displacement[env_index]),
        "final_travel_fraction": float(result.final_displacement[env_index] / DRAWER_TRAVEL_LIMIT),
        "peak_velocity": float(result.peak_velocity[env_index]),
    }
