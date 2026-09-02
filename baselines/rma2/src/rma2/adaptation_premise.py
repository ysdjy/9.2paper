r"""Is the adaptation problem well posed, and how hard is it?

Four questions have to be answered from data *before* any adaptation model is trained. They
are cheap -- they read a finished Oracle sweep and never touch the simulator -- and each one
can invalidate the study on its own.

**Is adaptation necessary?** If one fixed :math:`F_\text{peak}` succeeded on most hidden
states, a constant would solve the task and no probe would be needed.
:func:`band_structure` reports the best constant's success rate.

**Is the answer a set or a point?** For each hidden state the succeeding forces form a set.
If those sets are contiguous intervals, their midpoint is always a valid answer and a
single-output regressor can in principle be optimal. If they are not, averaging is unsafe and
a model that predicts *where* the successes are has something a regressor cannot express.
:func:`band_structure` measures contiguity and whether the midpoint succeeds.

**Does the probe determine the answer?** Two drawers with indistinguishable probes but
different required forces make :math:`p^*(\tau_p)` multi-valued, and any regressor must then
average across the ambiguity. :func:`probe_ambiguity` measures how often that average leaves
the success band, as a function of how finely the probe is resolved.

**How precise must a predictor be?** :func:`identifiability` fits leave-one-out readouts from
the probe features and from the true :math:`\xi`, and reports both the error and the fraction
of predictions that actually land inside the band. A high :math:`R^2` that still misses a
0.20 N band is not a useful model, and the two numbers can disagree sharply.

The readouts here are deliberately simple -- a linear fit, a quadratic fit, and
:math:`k`-nearest neighbours. They are not competitors to a learned model; they establish
what is achievable *without* one, so that a learned model's number has something to beat.
Every estimate is leave-one-out, so none of them is a fit to its own data.

Nothing in this module imports Isaac Lab or runs control (``docs/ARCHITECTURE.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from probe_drawer.analysis.probe_features import PROBE_FEATURES
from probe_drawer.analysis.sweep import SweepDataset, SweepRecord
from probe_drawer.envs.dynamics_randomization import XI_FIELDS
from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = [
    "DEFAULT_AMBIGUITY_RADII",
    "AdaptationPremise",
    "HiddenStateBand",
    "audit",
    "band_structure",
    "collect_bands",
    "identifiability",
    "probe_ambiguity",
]

#: Radii, in standardised probe-feature units, at which :func:`probe_ambiguity` is evaluated.
#:
#: A radius stands in for how finely an encoder resolves the probe: everything inside it is
#: treated as indistinguishable. The three values bracket the interesting range -- at 0.25 the
#: hidden states are essentially separated, at 1.0 they are not.
DEFAULT_AMBIGUITY_RADII: tuple[float, ...] = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class HiddenStateBand:
    """One hidden state's succeeding forces at the task, plus the probe that preceded them.

    Attributes:
        xi: The hidden state, in :data:`~probe_drawer.analysis.sweep.XI_FIELDS` order.
        success_forces: Every swept ``F_peak`` that succeeded (N), ascending.
        swept_forces: Every valid ``F_peak`` that was tried (N), ascending.
        features: The probe's summary features, in :data:`PROBE_FEATURES` order.
    """

    xi: tuple[float, ...]
    success_forces: tuple[float, ...]
    swept_forces: tuple[float, ...]
    features: tuple[float, ...]

    @property
    def low(self) -> float:
        return self.success_forces[0]

    @property
    def high(self) -> float:
        return self.success_forces[-1]

    @property
    def centre(self) -> float:
        """The band's midpoint -- the maximum-margin answer, and the regression target."""
        return 0.5 * (self.low + self.high)

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def interior_failures(self) -> int:
        """Swept forces inside ``[low, high]`` that did **not** succeed.

        Non-zero means the success set is not an interval, so its midpoint is not
        guaranteed to succeed and averaging two valid answers can give an invalid one.
        """
        inside = [f for f in self.swept_forces if self.low <= f <= self.high]
        return len(inside) - len(self.success_forces)

    def contains(self, force: float) -> bool:
        """Whether ``force`` lies in ``[low, high]``.

        Interval membership only. For a band with an interior failure this is **not** the same
        as succeeding, which is why :meth:`succeeds_at` exists and is what the audits use.
        """
        return self.low <= force <= self.high

    def succeeds_at(self, force: float) -> bool:
        """Whether the swept force nearest ``force`` actually succeeded.

        Snapping to the grid is the honest test: success was only ever measured at swept
        points, so a claim about an unswept force would be an interpolation, not evidence. A
        prediction outside the swept range snaps to an endpoint and correctly fails there.
        """
        nearest = min(self.swept_forces, key=lambda swept: abs(swept - force))
        return nearest in self.success_forces


def collect_bands(
    dataset: SweepDataset, criteria: SuccessCriteria, duration: float
) -> tuple[list[HiddenStateBand], int]:
    """Every solvable hidden state's success band, and how many had none.

    Args:
        dataset: A finished sweep. Invalid rows are dropped -- they are evidence about the
            rig, not about the drawer.
        criteria: The success definition to label with.
        duration: The execution duration to select.

    Returns:
        ``(bands, unsolvable)``, where ``unsolvable`` counts hidden states with no succeeding
        force at all. Those states are real data and are reported rather than hidden, but they
        have no regression target and so cannot appear in ``bands``.
    """
    bands: list[HiddenStateBand] = []
    unsolvable = 0
    for xi_key in dataset.xi_keys():
        rows = [row for row in dataset.select(xi_key=xi_key, duration=duration) if row.valid]
        if not rows:
            continue
        succeeding = sorted(row.peak_force for row in rows if row.succeeds(criteria))
        if not succeeding:
            unsolvable += 1
            continue
        bands.append(
            HiddenStateBand(
                xi=xi_key,
                success_forces=tuple(succeeding),
                swept_forces=tuple(sorted(row.peak_force for row in rows)),
                features=_feature_vector(rows[0]),
            )
        )
    return bands, unsolvable


def _feature_vector(row: SweepRecord) -> tuple[float, ...]:
    """The row's probe features in :data:`PROBE_FEATURES` order, ``nan`` where absent."""
    return tuple(float(row.probe_features.get(name, np.nan)) for name in PROBE_FEATURES)


def band_structure(
    dataset: SweepDataset, criteria: SuccessCriteria, duration: float
) -> dict:
    """Whether adaptation is necessary, and whether the answer is a point or a set.

    Reports, in one pass: coverage; the spread of required force; band widths; how many bands
    have interior failures; how often the band midpoint succeeds; and the best constant force,
    which is the Fixed Conservative baseline's ceiling.
    """
    bands, unsolvable = collect_bands(dataset, criteria, duration)
    total = len(bands) + unsolvable
    if not bands:
        return {"solvable": 0, "total": total, "coverage": 0.0}

    centres = np.array([band.centre for band in bands])
    widths = np.array([band.width for band in bands])

    per_force: dict[float, int] = {}
    for band in bands:
        for force in band.success_forces:
            per_force[force] = per_force.get(force, 0) + 1
    ranked = sorted(per_force.items(), key=lambda item: (-item[1], item[0]))

    return {
        "total_hidden_states": total,
        "solvable": len(bands),
        "unsolvable": unsolvable,
        "coverage": len(bands) / total,
        "required_force": {
            "min": float(centres.min()),
            "max": float(centres.max()),
            "median": float(np.median(centres)),
            "ratio": float(centres.max() / centres.min()) if centres.min() > 0 else float("inf"),
        },
        "band_width": {
            "median": float(np.median(widths)),
            "min": float(widths.min()),
            "max": float(widths.max()),
            "median_half_width": float(np.median(widths) / 2.0),
            "median_relative_half_width": float(np.median(widths) / 2.0 / np.median(centres)),
        },
        "succeeding_forces_per_state": {
            "median": float(np.median([len(band.success_forces) for band in bands])),
            "min": int(min(len(band.success_forces) for band in bands)),
            "max": int(max(len(band.success_forces) for band in bands)),
        },
        "non_contiguous_bands": sum(1 for band in bands if band.interior_failures > 0),
        "largest_interior_gap": max(band.interior_failures for band in bands),
        "midpoint_succeeds": sum(1 for band in bands if band.succeeds_at(band.centre)),
        "best_constant_force": {
            "force": float(ranked[0][0]),
            "successes": int(ranked[0][1]),
            "success_rate": ranked[0][1] / total,
            "runners_up": [(float(force), int(count)) for force, count in ranked[1:5]],
        },
    }


def probe_ambiguity(
    dataset: SweepDataset,
    criteria: SuccessCriteria,
    duration: float,
    radii: tuple[float, ...] = DEFAULT_AMBIGUITY_RADII,
) -> dict:
    """How often averaging over probe-indistinguishable hidden states leaves the band.

    Hidden states are placed in standardised probe-feature space. For each radius, every state
    within it of a given state is treated as indistinguishable from it, and the mean required
    force over that neighbourhood is checked against the state's own band. A high failure rate
    means ``p*`` is effectively multi-valued given the probe, which is the condition under
    which predicting a success *landscape* has something a single-output regressor lacks.

    The radius is the free parameter and it stands for an encoder's resolution, which is why
    several are reported rather than one.
    """
    bands, _ = collect_bands(dataset, criteria, duration)
    if len(bands) < 3:
        return {"solvable": len(bands), "radii": {}}

    standardised = _standardise(np.array([band.features for band in bands]))
    centres = np.array([band.centre for band in bands])
    distances = np.linalg.norm(standardised[:, None, :] - standardised[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)

    nearest = distances.argmin(axis=1)
    report = {
        "solvable": len(bands),
        "nearest_neighbour": {
            "median_force_gap": float(np.median(np.abs(centres - centres[nearest]))),
            "max_force_gap": float(np.abs(centres - centres[nearest]).max()),
            "neighbour_misses_band": float(
                np.mean([not band.succeeds_at(centres[nearest[i]]) for i, band in enumerate(bands)])
            ),
        },
        "radii": {},
    }
    for radius in radii:
        sizes, spreads, misses = [], [], 0
        for index, band in enumerate(bands):
            group = np.append(np.where(distances[index] <= radius)[0], index)
            sizes.append(len(group))
            spreads.append(float(centres[group].max() - centres[group].min()))
            misses += not band.succeeds_at(float(centres[group].mean()))
        report["radii"][radius] = {
            "mean_cluster_size": float(np.mean(sizes)),
            "median_force_spread": float(np.median(spreads)),
            "max_force_spread": float(np.max(spreads)),
            "cluster_mean_misses_band": misses / len(bands),
        }
    return report


def identifiability(
    dataset: SweepDataset,
    criteria: SuccessCriteria,
    duration: float,
    neighbours: int = 3,
) -> dict:
    """What the probe determines, and what precision the task demands.

    Two things are reported and they must be read together:

    * how well simple leave-one-out readouts recover each ``xi`` dimension from the probe --
      a dimension no readout recovers is one the probe does not identify;
    * how well the same readouts predict the band centre, **and** how often that prediction
      lands inside the band. These diverge: a readout can explain 90 % of the variance in the
      required force and still be inside the band a fifth of the time, because the band is
      narrower than the residual.

    The same readouts are run from the true ``xi`` as an upper bound. It is an upper bound at
    *this dataset size*, not a property of the task, and it is a lower bound on what a model
    trained on more hidden states could reach.
    """
    bands, _ = collect_bands(dataset, criteria, duration)
    if len(bands) < 4:
        return {"solvable": len(bands)}

    features = np.array([band.features for band in bands])
    xi = np.array([list(band.xi) for band in bands])
    centres = np.array([band.centre for band in bands])

    report: dict = {
        "solvable": len(bands),
        "xi_from_probe": {
            name: _loo_scores(_linear_design(features), xi[:, index])
            for index, name in enumerate(XI_FIELDS)
        },
        "force_from_probe": {},
        "force_from_xi": {},
        "precision_required": {
            "median_half_width": float(np.median([band.width for band in bands]) / 2.0),
            "median_target": float(np.median(centres)),
        },
    }
    for label, source, target in (("probe", features, "force_from_probe"), ("xi", xi, "force_from_xi")):
        report[target] = {
            "linear": _band_scores(_loo_linear(_linear_design(source), centres), bands),
            "quadratic": _band_scores(_loo_linear(_quadratic_design(source), centres), bands),
            f"knn_{neighbours}": _band_scores(_loo_knn(source, centres, neighbours), bands),
        }
    return report


@dataclass(frozen=True)
class AdaptationPremise:
    """The three audits together, with the dataset and the settings they were computed from.

    ``settings`` is recorded rather than assumed so that a number in the report can always be
    traced back to how the question was asked -- the ambiguity radii in particular change the
    answer a great deal, and reporting one without the other would be misleading.
    """

    source: str
    duration: float
    criteria: dict
    settings: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)
    ambiguity: dict = field(default_factory=dict)
    identifiability: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "duration": self.duration,
            "criteria": self.criteria,
            "settings": self.settings,
            "structure": self.structure,
            "ambiguity": self.ambiguity,
            "identifiability": self.identifiability,
        }


def audit(
    dataset: SweepDataset,
    criteria: SuccessCriteria,
    duration: float,
    source: str = "",
    radii: tuple[float, ...] = DEFAULT_AMBIGUITY_RADII,
    neighbours: int = 3,
) -> AdaptationPremise:
    """Run all three audits against one sweep."""
    return AdaptationPremise(
        source=source,
        duration=duration,
        criteria=criteria.as_dict() if hasattr(criteria, "as_dict") else {},
        settings={"ambiguity_radii": list(radii), "knn_neighbours": neighbours},
        structure=band_structure(dataset, criteria, duration),
        ambiguity=probe_ambiguity(dataset, criteria, duration, radii),
        identifiability=identifiability(dataset, criteria, duration, neighbours),
    )


# -- readouts ----------------------------------------------------------------------------


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per column; constant columns are left alone."""
    spread = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(spread == 0.0, 1.0, spread)


def _linear_design(values: np.ndarray) -> np.ndarray:
    return np.hstack([_standardise(values), np.ones((len(values), 1))])


def _quadratic_design(values: np.ndarray) -> np.ndarray:
    standardised = _standardise(values)
    return np.hstack([standardised, standardised**2, np.ones((len(values), 1))])


def _loo_linear(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Leave-one-out least-squares predictions.

    Refitting per held-out point rather than fitting once is the whole point: with 105 points
    and 19 quadratic terms, an in-sample fit would report a precision the model does not have.
    """
    predictions = np.empty(len(target))
    for index in range(len(target)):
        mask = np.ones(len(target), dtype=bool)
        mask[index] = False
        weights, *_ = np.linalg.lstsq(design[mask], target[mask], rcond=None)
        predictions[index] = design[index] @ weights
    return predictions


def _loo_knn(features: np.ndarray, target: np.ndarray, neighbours: int) -> np.ndarray:
    """Leave-one-out k-nearest-neighbour predictions in standardised feature space."""
    standardised = _standardise(features)
    distances = np.linalg.norm(standardised[:, None, :] - standardised[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    chosen = np.argsort(distances, axis=1)[:, :neighbours]
    return target[chosen].mean(axis=1)


def _loo_scores(design: np.ndarray, target: np.ndarray) -> dict:
    """``R^2`` and RMSE of a leave-one-out linear readout of one quantity."""
    predictions = _loo_linear(design, target)
    residual = ((predictions - target) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    return {
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(((predictions - target) ** 2).mean())),
    }


def _band_scores(predictions: np.ndarray, bands: list[HiddenStateBand]) -> dict:
    """RMSE against the band centre, and how often the prediction is actually in the band.

    Both are reported because they answer different questions and can point in opposite
    directions -- which is the finding, not an inconvenience.
    """
    centres = np.array([band.centre for band in bands])
    residual = ((predictions - centres) ** 2).sum()
    total = ((centres - centres.mean()) ** 2).sum()
    inside = sum(band.succeeds_at(float(predictions[index])) for index, band in enumerate(bands))
    return {
        "r2": float(1.0 - residual / total) if total > 0 else float("nan"),
        "rmse": float(np.sqrt(((predictions - centres) ** 2).mean())),
        "in_band": inside,
        "in_band_rate": inside / len(bands),
    }
