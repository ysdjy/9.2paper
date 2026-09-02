r"""Sweeping ``(xi, F_peak, T)`` and reading the result back off disk.

An Oracle sweep is a table, not a trajectory archive: one row per executed episode,
carrying the hidden state it ran under, the command it was given, what the drawer did, and
whether the operating point was usable at all. Trajectories are not kept -- any row can be
reproduced exactly by re-running its ``(xi, F_peak, T)``, and keeping hundreds of full
histories would trade a large amount of disk for nothing.

The module is pure: it defines the grid, the record format, and the queries the parameter
selection needs. Collecting the rows is ``scripts/sweep_execution_space.py``'s job.

Vectorisation note. Environments are the ``xi`` axis and the loop is over ``(F_peak, T)``:
one ``execution.run`` call drives every hidden state under an identical command, which is
the only way to be certain that a difference between two rows came from ``xi`` and not from
anything else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from probe_drawer.controllers.types import ExecutionResult
from probe_drawer.envs.dynamics_randomization import XI_FIELDS, DynamicsParameters
from probe_drawer.evaluation.operating_region import DRAWER_TRAVEL_LIMIT, ValidityReport
from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = [
    "SweepDataset",
    "SweepRecord",
    "force_grid",
    "success_interval",
    "xi_grid",
]


@dataclass(frozen=True)
class SweepRecord:
    """One executed ``(xi, F_peak, T)`` point.

    Attributes:
        xi: The hidden state, as a :meth:`DynamicsParameters.as_dict` mapping.
        peak_force: Commanded plateau force (N).
        duration: Commanded duration (s).
        final_displacement: ``d(T)`` (m).
        final_velocity: ``v(T)`` (m/s).
        peak_velocity: Largest drawer speed during the episode (m/s).
        peak_measured_force: Largest wrist pull force (N).
        travel_fraction: ``d(T)`` as a fraction of the drawer's mechanical travel.
        peak_lateral_drift: Largest TCP drift orthogonal to the pull axis (m).
        peak_orientation_drift_deg: Largest TCP orientation drift.
        termination_reason: The controller's own reason string.
        valid: Whether the operating point is usable as Oracle evidence.
        invalid_reasons: Why not, if not.
        protocol: ``"reset"`` for a Phase 9 row (execution only, drawer starting closed) or
            ``"sequential"`` for a Phase 10 row (probe, gap, then execution with no reset).
        pre_execution_displacement: For a sequential row, how far the drawer had already
            travelled from the task's start when the execution began (m) -- the probe plus the
            coast during the inference gap. Zero for a reset row.
        probe_displacement: What the probe alone moved the drawer (m).
        probe_duration: How long the probe ran (s).
        probe_features: The probe's summary features, from
            :func:`~probe_drawer.analysis.probe_features.extract_features`.

    Note on ``final_displacement``: it is always the quantity the task is judged on, i.e.
    measured from the *task's* start. For a reset row that is the execution's own
    displacement; for a sequential row it is ``pre_execution_displacement`` plus the
    execution's (``docs/DECISIONS.md`` D027). Keeping one meaning for the field is what lets
    the same Oracle analysis read both protocols.
    """

    xi: dict
    peak_force: float
    duration: float
    final_displacement: float
    final_velocity: float
    peak_velocity: float
    peak_measured_force: float
    travel_fraction: float
    peak_lateral_drift: float
    peak_orientation_drift_deg: float
    termination_reason: str
    valid: bool
    invalid_reasons: list[str] = field(default_factory=list)
    protocol: str = "reset"
    pre_execution_displacement: float = 0.0
    probe_displacement: float = 0.0
    probe_duration: float = 0.0
    probe_features: dict = field(default_factory=dict)

    @property
    def execution_displacement(self) -> float:
        """What the execution segment contributed on its own (m)."""
        return self.final_displacement - self.pre_execution_displacement

    @property
    def xi_vector(self) -> tuple[float, ...]:
        """The hidden state in :data:`XI_FIELDS` order."""
        return tuple(float(self.xi[name]) for name in XI_FIELDS)

    @property
    def xi_key(self) -> tuple[float, ...]:
        """Hashable identity of the hidden state, rounded so float noise cannot split a group."""
        return tuple(round(value, 6) for value in self.xi_vector)

    def succeeds(self, criteria: SuccessCriteria) -> bool:
        """Whether this row meets a task definition. Mirrors ``evaluate_execution``."""
        return bool(
            self.valid
            and abs(self.final_displacement - criteria.goal_displacement) <= criteria.displacement_tolerance
            and abs(self.final_velocity) <= criteria.velocity_tolerance
        )

    def as_dict(self) -> dict:
        return {
            "xi": self.xi,
            "peak_force": self.peak_force,
            "duration": self.duration,
            "final_displacement": self.final_displacement,
            "final_velocity": self.final_velocity,
            "peak_velocity": self.peak_velocity,
            "peak_measured_force": self.peak_measured_force,
            "travel_fraction": self.travel_fraction,
            "peak_lateral_drift": self.peak_lateral_drift,
            "peak_orientation_drift_deg": self.peak_orientation_drift_deg,
            "termination_reason": self.termination_reason,
            "valid": self.valid,
            "invalid_reasons": self.invalid_reasons,
            "protocol": self.protocol,
            "pre_execution_displacement": self.pre_execution_displacement,
            "probe_displacement": self.probe_displacement,
            "probe_duration": self.probe_duration,
            "probe_features": self.probe_features,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> SweepRecord:
        return cls(**payload)

    @classmethod
    def from_sequential_episode(
        cls,
        parameters: DynamicsParameters,
        duration: float,
        episode,
        validity: ValidityReport,
        env_index: int,
        probe_features: dict | None = None,
    ) -> SweepRecord:
        """Extract one environment's row from a sequential probe-then-execute episode.

        ``final_displacement`` is the *total* displacement from the task's start, which is
        what the success definition compares against the goal.
        """
        verdict = validity.verdicts[env_index]
        result = episode.execution
        total = float(episode.total_displacement[env_index])
        return cls(
            xi=parameters.as_dict(),
            peak_force=float(episode.peak_force[env_index]),
            duration=float(duration),
            final_displacement=total,
            final_velocity=float(result.final_velocity[env_index]),
            peak_velocity=float(result.peak_velocity[env_index]),
            peak_measured_force=float(result.peak_measured_force[env_index]),
            travel_fraction=total / DRAWER_TRAVEL_LIMIT,
            peak_lateral_drift=float(verdict.metrics["peak_lateral_drift"]),
            peak_orientation_drift_deg=float(verdict.metrics["peak_orientation_drift_deg"]),
            termination_reason=result.termination_reason[env_index].value,
            valid=bool(verdict.valid),
            invalid_reasons=[reason.value for reason in verdict.reasons],
            protocol="sequential",
            pre_execution_displacement=float(episode.pre_execution_displacement[env_index]),
            probe_displacement=float(episode.probe_displacement[env_index]),
            probe_duration=float(episode.probe.duration[env_index]),
            probe_features=probe_features or {},
        )

    @classmethod
    def from_execution(
        cls,
        parameters: DynamicsParameters,
        peak_force: float,
        duration: float,
        result: ExecutionResult,
        validity: ValidityReport,
        env_index: int,
    ) -> SweepRecord:
        """Extract one environment's row from a multi-environment execution."""
        verdict = validity.verdicts[env_index]
        return cls(
            xi=parameters.as_dict(),
            peak_force=float(peak_force),
            duration=float(duration),
            final_displacement=float(result.final_displacement[env_index]),
            final_velocity=float(result.final_velocity[env_index]),
            peak_velocity=float(result.peak_velocity[env_index]),
            peak_measured_force=float(result.peak_measured_force[env_index]),
            travel_fraction=float(result.final_displacement[env_index] / DRAWER_TRAVEL_LIMIT),
            peak_lateral_drift=float(verdict.metrics["peak_lateral_drift"]),
            peak_orientation_drift_deg=float(verdict.metrics["peak_orientation_drift_deg"]),
            termination_reason=result.termination_reason[env_index].value,
            valid=bool(verdict.valid),
            invalid_reasons=[reason.value for reason in verdict.reasons],
        )


@dataclass
class SweepDataset:
    """A collection of :class:`SweepRecord` rows, with the queries the analysis needs."""

    records: list[SweepRecord] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    #: Lazily built ``(xi_key, duration) -> rows`` index. Task-parameter selection scores
    #: hundreds of candidates against every hidden state, so a linear scan per query turns
    #: the analysis from seconds into minutes.
    _index: dict[tuple[tuple[float, ...], float], list[SweepRecord]] | None = field(
        default=None, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, records: Iterable[SweepRecord]) -> None:
        self.records.extend(records)
        self._index = None

    def _grouped(self) -> dict[tuple[tuple[float, ...], float], list[SweepRecord]]:
        if self._index is None:
            index: dict[tuple[tuple[float, ...], float], list[SweepRecord]] = {}
            for record in self.records:
                index.setdefault((record.xi_key, record.duration), []).append(record)
            for rows in index.values():
                rows.sort(key=lambda row: row.peak_force)
            self._index = index
        return self._index

    @property
    def valid_records(self) -> list[SweepRecord]:
        return [record for record in self.records if record.valid]

    def forces(self) -> list[float]:
        """Distinct commanded peak forces, ascending."""
        return sorted({record.peak_force for record in self.records})

    def durations(self) -> list[float]:
        """Distinct commanded durations, ascending."""
        return sorted({record.duration for record in self.records})

    def xi_keys(self) -> list[tuple[float, ...]]:
        """Distinct hidden states, in first-seen order."""
        return list(dict.fromkeys(record.xi_key for record in self.records))

    def select(
        self,
        xi_key: tuple[float, ...] | None = None,
        duration: float | None = None,
        peak_force: float | None = None,
        valid_only: bool = False,
    ) -> list[SweepRecord]:
        """Rows matching every supplied constraint, sorted by force then duration."""
        if xi_key is not None and duration is not None:
            rows = list(self._grouped().get((xi_key, duration), []))
        else:
            rows = list(self.records)
            if xi_key is not None:
                rows = [row for row in rows if row.xi_key == xi_key]
            if duration is not None:
                rows = [row for row in rows if row.duration == duration]
        if valid_only:
            rows = [row for row in rows if row.valid]
        if peak_force is not None:
            rows = [row for row in rows if row.peak_force == peak_force]
        return sorted(rows, key=lambda row: (row.peak_force, row.duration))

    def surface(self, field_name: str, xi_key: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One hidden state's ``(F_peak, T) -> field`` surface.

        Returns ``(forces, durations, values)`` with ``values`` of shape
        ``(len(durations), len(forces))`` and ``nan`` where a point was not swept.
        """
        forces, durations = self.forces(), self.durations()
        values = np.full((len(durations), len(forces)), np.nan)
        for row in self.select(xi_key=xi_key):
            values[durations.index(row.duration), forces.index(row.peak_force)] = getattr(row, field_name)
        return np.asarray(forces), np.asarray(durations), values

    def validity_rate(self) -> float:
        """Fraction of swept points that were usable."""
        return len(self.valid_records) / len(self.records) if self.records else 0.0

    def invalid_reason_counts(self) -> dict[str, int]:
        """How often each invalidity reason fired, most common first."""
        counts: dict[str, int] = {}
        for record in self.records:
            for reason in record.invalid_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "metadata": self.metadata,
                    "xi_fields": list(XI_FIELDS),
                    "records": [record.as_dict() for record in self.records],
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> SweepDataset:
        payload = json.loads(Path(path).read_text())
        return cls(
            records=[SweepRecord.from_dict(row) for row in payload["records"]],
            metadata=payload.get("metadata", {}),
        )


def force_grid(low: float, high: float, step: float) -> tuple[float, ...]:
    """Inclusive peak-force grid, rounded so the values are exact on the requested spacing.

    The rounding matters: a grid built by repeated addition drifts, and then two sweeps that
    asked for the same forces produce keys that do not match and cannot be merged. The
    supplementary low-force passes of Phase 10 are merged into the main dataset by exactly
    this equality, so the values have to be reproducible rather than merely close.

    Args:
        low: First force (N).
        high: Last force (N), included when it lands on the grid.
        step: Spacing (N). Must be > 0.

    Raises:
        ValueError: If ``step <= 0`` or ``high < low``.
    """
    if step <= 0.0:
        raise ValueError(f"step must be > 0 N, not {step}")
    if high < low:
        raise ValueError(f"high ({high} N) must be >= low ({low} N)")
    count = int(round((high - low) / step)) + 1
    return tuple(round(low + index * step, 6) for index in range(count))


def xi_grid(
    masses: Sequence[float],
    static_frictions: Sequence[float],
    friction_ratios: Sequence[float],
    dampings: Sequence[float],
    name_prefix: str = "grid",
) -> list[DynamicsParameters]:
    """Full factorial grid over the four hidden dimensions.

    ``mu_d`` is specified as a *ratio* of ``mu_s`` so that every point satisfies the
    ``mu_s >= mu_d`` constraint PhysX imposes, and so that the friction asymmetry is an
    independent axis rather than something implied by two absolute levels.
    """
    for name, values in (
        ("masses", masses),
        ("static_frictions", static_frictions),
        ("friction_ratios", friction_ratios),
        ("dampings", dampings),
    ):
        if not values:
            raise ValueError(f"{name} must not be empty.")
    if any(not 0.0 <= ratio <= 1.0 for ratio in friction_ratios):
        raise ValueError(f"friction_ratios must lie inside [0, 1], got {list(friction_ratios)}.")

    grid: list[DynamicsParameters] = []
    for mass in masses:
        for static in static_frictions:
            for ratio in friction_ratios:
                for damping in dampings:
                    grid.append(
                        DynamicsParameters(
                            drawer_mass=float(mass),
                            joint_static_friction=float(static),
                            joint_dynamic_friction=float(static * ratio),
                            joint_damping=float(damping),
                            name=f"{name_prefix}_m{mass:g}_mus{static:g}_r{ratio:g}_b{damping:g}",
                        )
                    )
    return grid


def success_interval(
    dataset: SweepDataset,
    xi_key: tuple[float, ...],
    criteria: SuccessCriteria,
    duration: float,
) -> dict:
    """The band of peak forces that succeed for one hidden state at one duration.

    A contiguous band is what makes the task learnable: a model has to predict a number, and
    a band gives it tolerance. The report therefore includes whether the succeeding forces
    are contiguous on the swept grid, and how wide the band is relative to its centre.
    """
    rows = dataset.select(xi_key=xi_key, duration=duration)
    succeeding = [row.peak_force for row in rows if row.succeeds(criteria)]
    swept = [row.peak_force for row in rows]

    if not succeeding:
        return {
            "xi": list(xi_key),
            "duration": duration,
            "any_success": False,
            "swept_forces": swept,
            "success_forces": [],
        }

    low, high = min(succeeding), max(succeeding)
    inside = [force for force in swept if low <= force <= high]
    centre = 0.5 * (low + high)
    best = min(
        (row for row in rows if row.succeeds(criteria)),
        key=lambda row: abs(row.final_displacement - criteria.goal_displacement),
    )
    return {
        "xi": list(xi_key),
        "duration": duration,
        "any_success": True,
        "swept_forces": swept,
        "success_forces": succeeding,
        "force_low": low,
        "force_high": high,
        "force_centre": centre,
        "force_width": high - low,
        "relative_width": (high - low) / centre if centre else None,
        "contiguous": len(succeeding) == len(inside),
        "best_force": best.peak_force,
        "best_displacement": best.final_displacement,
        "best_velocity": best.final_velocity,
    }
