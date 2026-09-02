r"""Which ``(xi, F_peak, T)`` operating points are usable at all.

Separate from success. An episode can be perfectly *valid* and still miss the goal; what
this module rejects are episodes whose physics or control quality makes them unusable as
evidence either way -- the drawer slamming into its end stop, the hybrid controller losing
the held axes, the drawer not moving measurably, or the simulation going non-finite.

Thresholds
----------
None of the defaults is a round number picked for looks. Each is anchored to a measurement
already in ``docs/VALIDATION.md`` and re-checked against the Phase 9 sweep; the reasoning
per threshold is in the :class:`OperatingRegionCfg` field docs and in
``docs/EXPERIMENT_SPACE.md``. They are deliberately much tighter than
:class:`~probe_drawer.controllers.SafetyLimits`: safety stops the simulation from
diverging, validity decides whether an episode is good enough to build an Oracle label on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from probe_drawer.controllers.types import ExecutionResult, TerminationReason

__all__ = [
    "DRAWER_TRAVEL_LIMIT",
    "PROVISIONAL_VALIDATION_DURATION",
    "PROVISIONAL_VALIDATION_PEAK_FORCE",
    "InvalidReason",
    "OperatingRegionCfg",
    "ValidityReport",
    "ValidityVerdict",
    "assess_validity",
]

#: Travel limit of ``drawer_top_joint``, measured from the official cabinet asset (m).
#: See ``docs/OFFICIAL_BASELINE.md``.
DRAWER_TRAVEL_LIMIT = 0.4

#: The operating point Phases 6-8 used for validation. **Provisional**: it was chosen to
#: make the dynamics presets separate cleanly during development, not by any experiment
#: design criterion. The paper's operating point comes from the Phase 9 sweep -- see
#: ``docs/EXPERIMENT_SPACE.md`` and ``docs/DECISIONS.md`` D021.
PROVISIONAL_VALIDATION_PEAK_FORCE = 5.0
PROVISIONAL_VALIDATION_DURATION = 2.0


class InvalidReason(str, Enum):
    """Why an operating point is not usable."""

    SAFETY_ABORT = "safety_abort"
    MECHANICAL_LIMIT = "mechanical_limit"
    EXCESSIVE_VELOCITY = "excessive_velocity"
    EXCESSIVE_LATERAL_DRIFT = "excessive_lateral_drift"
    EXCESSIVE_ORIENTATION_DRIFT = "excessive_orientation_drift"
    NO_MEASURABLE_MOTION = "no_measurable_motion"
    NON_FINITE = "non_finite"


@dataclass
class OperatingRegionCfg:
    """Validity thresholds, each with the measurement it is anchored to.

    Args:
        mechanical_margin_fraction: Largest fraction of :data:`DRAWER_TRAVEL_LIMIT` the
            drawer may reach. Anchored to Phase 8: at 0.326 m (0.82 of travel) the TCP
            lateral drift was 14-15 mm, against 0.36-0.66 mm for runs that stayed below
            0.2 m, so behaviour near the end stop is qualitatively different and must not
            enter an Oracle label.
        max_peak_velocity: Largest drawer speed reached at any point (m/s). Anchored to the
            same comparison: clean Phase 8 runs peaked at 0.054-0.132 m/s, the drifting one
            at 0.418 m/s.
        max_lateral_drift: Largest TCP drift orthogonal to the pull axis (m). Clean Phase 8
            runs stayed under 0.7 mm; 5 mm leaves a 7x margin over that while staying 10x
            inside the 50 mm safety limit.
        max_orientation_drift_deg: Largest TCP orientation drift. Clean runs stayed under
            0.41 deg; 5 deg leaves a comparable margin inside the 30 deg safety limit.
        min_displacement: Smallest ``d(T)`` that counts as measurable motion (m). One
            millimetre is about 20x the residual zero-command creep of roughly 2.5 mm over
            2 s divided over a comparable window, and well above sensor-scale noise.
    """

    mechanical_margin_fraction: float = 0.8
    max_peak_velocity: float = 0.25
    max_lateral_drift: float = 0.005
    max_orientation_drift_deg: float = 5.0
    min_displacement: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 < self.mechanical_margin_fraction <= 1.0:
            raise ValueError(
                f"mechanical_margin_fraction must lie in (0, 1], got {self.mechanical_margin_fraction}."
            )
        for name in ("max_peak_velocity", "max_lateral_drift", "max_orientation_drift_deg", "min_displacement"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}.")

    @property
    def max_displacement(self) -> float:
        """Largest ``d(T)`` that keeps a safe margin from the mechanical end stop (m)."""
        return self.mechanical_margin_fraction * DRAWER_TRAVEL_LIMIT

    def as_dict(self) -> dict:
        return {
            "mechanical_margin_fraction": self.mechanical_margin_fraction,
            "max_displacement": self.max_displacement,
            "drawer_travel_limit": DRAWER_TRAVEL_LIMIT,
            "max_peak_velocity": self.max_peak_velocity,
            "max_lateral_drift": self.max_lateral_drift,
            "max_orientation_drift_deg": self.max_orientation_drift_deg,
            "min_displacement": self.min_displacement,
        }


@dataclass
class ValidityVerdict:
    """Whether one environment's episode is usable, and the metrics behind that.

    Attributes:
        valid: ``True`` only if no reason fired.
        reasons: Every reason that fired, in the order they are checked.
        metrics: The measured quantities the thresholds were applied to.
    """

    valid: bool
    reasons: list[InvalidReason]
    metrics: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reasons": [reason.value for reason in self.reasons],
            "metrics": self.metrics,
        }


@dataclass
class ValidityReport:
    """Per-environment validity verdicts for one execution."""

    verdicts: list[ValidityVerdict]
    cfg: OperatingRegionCfg = field(default_factory=OperatingRegionCfg)

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of usable environments, shape ``(num_envs,)``."""
        return np.asarray([verdict.valid for verdict in self.verdicts], dtype=bool)

    def as_dict(self) -> dict:
        return {
            "thresholds": self.cfg.as_dict(),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
        }


def assess_validity(result: ExecutionResult, cfg: OperatingRegionCfg | None = None) -> ValidityReport:
    """Decide, per environment, whether an execution is usable as Oracle evidence.

    Deterministic: the same :class:`~probe_drawer.controllers.types.ExecutionResult` always
    produces the same verdict, because every check reads only recorded values.

    Args:
        result: The execution to assess.
        cfg: Validity thresholds. Project defaults when omitted.
    """
    cfg = cfg or OperatingRegionCfg()
    history = result.history
    verdicts: list[ValidityVerdict] = []

    for index in range(result.num_envs):
        driven = history.active_steps(index)
        displacement = float(result.final_displacement[index])
        peak_velocity = float(np.abs(history.drawer_velocity[driven, index]).max())
        lateral_drift = float(history.tcp_lateral_error[driven, index].max())
        orientation_drift = float(np.degrees(history.tcp_orientation_error[driven, index].max()))

        finite = all(
            np.all(np.isfinite(array[driven, index] if array.ndim > 1 else array))
            for name, array in history.as_arrays().items()
            if name != "time"
        )

        reasons: list[InvalidReason] = []
        if result.termination_reason[index] is TerminationReason.SAFETY_ABORT:
            reasons.append(InvalidReason.SAFETY_ABORT)
        if not finite:
            reasons.append(InvalidReason.NON_FINITE)
        if displacement > cfg.max_displacement:
            reasons.append(InvalidReason.MECHANICAL_LIMIT)
        if peak_velocity > cfg.max_peak_velocity:
            reasons.append(InvalidReason.EXCESSIVE_VELOCITY)
        if lateral_drift > cfg.max_lateral_drift:
            reasons.append(InvalidReason.EXCESSIVE_LATERAL_DRIFT)
        if orientation_drift > cfg.max_orientation_drift_deg:
            reasons.append(InvalidReason.EXCESSIVE_ORIENTATION_DRIFT)
        if displacement < cfg.min_displacement:
            reasons.append(InvalidReason.NO_MEASURABLE_MOTION)

        verdicts.append(
            ValidityVerdict(
                valid=not reasons,
                reasons=reasons,
                metrics={
                    "final_displacement": displacement,
                    "peak_velocity": peak_velocity,
                    "peak_lateral_drift": lateral_drift,
                    "peak_orientation_drift_deg": orientation_drift,
                    "travel_fraction": displacement / DRAWER_TRAVEL_LIMIT,
                },
            )
        )

    return ValidityReport(verdicts=verdicts, cfg=cfg)
