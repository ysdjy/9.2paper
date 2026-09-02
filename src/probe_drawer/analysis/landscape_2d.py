r"""The shape of the two-dimensional success region, and whether that shape matters.

Phase 11 established that in one dimension a hidden state's succeeding forces form a
contiguous interval whose midpoint works for 104 of 105 solvable states. A model predicting
the whole landscape therefore has no *structural* advantage there -- averaging two good
answers is safe by construction, so a single-output regressor can in principle be optimal.

Opening ``T`` changes what is possible but not necessarily what is true, and this module is
what decides which. It measures, per hidden state, whether the success set

.. math:: S(\xi) = \{(F, T) : |d(T) - d_\text{goal}| \le \epsilon_d,\ |v(T)| \le \epsilon_v,\ \text{valid}\}

is one blob or several, convex or not, and whether the mean of two succeeding parameter pairs
succeeds. That last quantity -- the **midpoint failure rate** -- is the operational form of
the question. If it is near zero the region is effectively convex and direct regression
remains defensible; if it is large, regressing a single point is ill-posed no matter how
accurate the regressor, because the target it is asked to average toward can itself fail.

Everything here is offline: it reads a finished sweep and never touches a simulator. Pure
numpy, deliberately -- convexity is measured by the midpoint test rather than by a convex
hull, so the analysis needs no new dependency and reports the quantity that actually bears on
the decision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from probe_drawer.evaluation.task_evaluator import SuccessCriteria

__all__ = [
    "LandscapeMetrics",
    "analyse_landscape",
    "connected_components",
    "midpoint_failure_rate",
    "representative_hidden_states",
    "success_mask",
]


def representative_hidden_states(count: int = 48, seed: int = 20260902, ranges: dict | None = None) -> list[dict]:
    r"""Hidden states chosen to cover the box's corners *and* its interior.

    A sweep over ``easy``/``medium``/``hard`` presets walks one diagonal of the
    four-dimensional box and would miss, for instance, a light drawer with high static
    friction -- exactly the combination that turned out to be hardest in Phase 10. So the
    first :math:`2^4 = 16` states are the corners in
    :math:`[m, \mu_s, \text{ratio}, b]`, and the rest are a scrambled Sobol fill of the
    interior.

    Corners are pulled 5 % inside each bound. On the bound itself a coordinate is at the edge
    of what the randomiser accepts and of what Phase 10 validated; 5 % keeps every draw
    strictly inside a region already known to run.

    Args:
        count: Total states. Must be at least 16, so the corners always fit.
        seed: Sobol scrambling seed.
        ranges: ``{name: (low, high)}`` for ``mass``, ``static_friction``,
            ``dynamic_friction_ratio``, ``damping``. Defaults to the training ranges.

    Returns:
        ``count`` dicts with ``mass``, ``static_friction``, ``dynamic_friction``, ``damping``.
        ``dynamic_friction = ratio * static_friction``, so ``mu_d <= mu_s`` holds by
        construction (D016).
    """
    from probe_drawer.experiment_plan import TRAINING_XI_RANGES  # noqa: PLC0415 - avoids a cycle

    if count < 16:
        raise ValueError(f"count must be >= 16 so all 16 corners fit, got {count}.")
    bounds = ranges or {
        "mass": TRAINING_XI_RANGES.mass,
        "static_friction": TRAINING_XI_RANGES.static_friction,
        "dynamic_friction_ratio": TRAINING_XI_RANGES.dynamic_friction_ratio,
        "damping": TRAINING_XI_RANGES.damping,
    }
    order = ("mass", "static_friction", "dynamic_friction_ratio", "damping")
    lows = np.array([bounds[name][0] for name in order], dtype=float)
    highs = np.array([bounds[name][1] for name in order], dtype=float)
    inset = 0.05

    unit = np.array(
        [[(1.0 - inset) if (index >> axis) & 1 else inset for axis in range(4)] for index in range(16)]
    )
    remaining = count - 16
    if remaining:
        engine = SobolEngine(dimension=4, scramble=True, seed=seed)
        unit = np.vstack([unit, engine.draw(remaining).double().numpy()])

    scaled = unit * (highs - lows) + lows
    return [
        {
            "mass": float(mass),
            "static_friction": float(static),
            "dynamic_friction": float(ratio * static),
            "damping": float(damping),
        }
        for mass, static, ratio, damping in scaled
    ]


def success_mask(dataset, xi_key: tuple[float, ...], criteria: SuccessCriteria) -> dict:
    r"""One hidden state's ``(T, F)`` boolean grids.

    Returns ``forces``, ``durations``, and three masks of shape ``(len(durations),
    len(forces))``: ``swept`` (a point exists), ``valid`` (inside the operating region) and
    ``success``. Success requires validity, so ``success`` is a subset of ``valid``.

    Note the axis order: rows are ``T``, columns are ``F``, matching
    :meth:`SweepDataset.surface` so a mask and a surface can be indexed together.
    """
    forces, durations = dataset.forces(), dataset.durations()
    shape = (len(durations), len(forces))
    swept = np.zeros(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    success = np.zeros(shape, dtype=bool)

    for row in dataset.select(xi_key=xi_key):
        index = (durations.index(row.duration), forces.index(row.peak_force))
        swept[index] = True
        valid[index] = row.valid
        success[index] = row.succeeds(criteria)

    return {
        "forces": np.asarray(forces, dtype=float),
        "durations": np.asarray(durations, dtype=float),
        "swept": swept,
        "valid": valid,
        "success": success,
    }


def connected_components(mask: np.ndarray, diagonal: bool = False) -> tuple[np.ndarray, int]:
    """Label a boolean grid's connected regions by flood fill.

    Args:
        mask: 2-D boolean array.
        diagonal: 8-connectivity instead of 4. Off by default: two cells touching only at a
            corner are separated by a failing cell on both grid-aligned paths between them, so
            calling them one region would hide exactly the structure being looked for.

    Returns:
        ``(labels, count)`` with ``labels`` zero where ``mask`` is false and ``1..count``
        elsewhere.
    """
    labels = np.zeros(mask.shape, dtype=int)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    count = 0
    for start in zip(*np.nonzero(mask), strict=True):
        if labels[start]:
            continue
        count += 1
        queue = deque([start])
        labels[start] = count
        while queue:
            row, column = queue.popleft()
            for delta_row, delta_column in steps:
                neighbour = (row + delta_row, column + delta_column)
                if (
                    0 <= neighbour[0] < mask.shape[0]
                    and 0 <= neighbour[1] < mask.shape[1]
                    and mask[neighbour]
                    and not labels[neighbour]
                ):
                    labels[neighbour] = count
                    queue.append(neighbour)
    return labels, count


def midpoint_failure_rate(mask: np.ndarray, swept: np.ndarray) -> dict:
    """How often the mean of two succeeding parameter pairs fails.

    Only pairs whose midpoint lands **exactly** on a swept grid point are considered -- both
    index differences even -- so the answer is measured rather than interpolated. Snapping a
    midpoint to the nearest grid point would let the grid's own resolution decide the result.

    This is the operational test for convexity, and the quantity that bears on whether direct
    regression is well posed: if averaging two good answers can fail, then a regressor asked
    for one answer has no safe target to aim at, however accurate it is.

    Returns:
        The rate, the counts behind it, and up to five concrete examples as
        ``(index_a, index_b, midpoint_index)`` triples.
    """
    points = list(zip(*np.nonzero(mask), strict=True))
    checked = 0
    failed = 0
    examples: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for first in range(len(points)):
        row_a, column_a = points[first]
        for second in range(first + 1, len(points)):
            row_b, column_b = points[second]
            if (row_a - row_b) % 2 or (column_a - column_b) % 2:
                continue
            middle = ((row_a + row_b) // 2, (column_a + column_b) // 2)
            if middle in ((row_a, column_a), (row_b, column_b)) or not swept[middle]:
                continue
            checked += 1
            if not mask[middle]:
                failed += 1
                if len(examples) < 5:
                    examples.append(((row_a, column_a), (row_b, column_b), middle))

    return {
        "pairs_checked": checked,
        "pairs_whose_midpoint_fails": failed,
        "rate": failed / checked if checked else float("nan"),
        "examples": [
            {"a": list(a), "b": list(b), "midpoint": list(m)} for a, b, m in examples
        ],
    }


def _principal_axis(rows: np.ndarray, columns: np.ndarray, durations: np.ndarray, forces: np.ndarray) -> dict:
    """Orientation and extent of the success cloud in physical units.

    Computed in normalised coordinates -- each axis divided by its own span -- because ``F``
    is in newtons and ``T`` in seconds, and an angle between them is meaningless until both
    are dimensionless. The reported angle is then "degrees from the F axis in a unit box",
    which is comparable between hidden states.
    """
    if len(rows) < 3:
        return {"orientation_deg": float("nan"), "elongation": float("nan")}

    force_span = float(forces[-1] - forces[0]) or 1.0
    duration_span = float(durations[-1] - durations[0]) or 1.0
    x = (forces[columns] - forces[0]) / force_span
    y = (durations[rows] - durations[0]) / duration_span

    stacked = np.vstack([x - x.mean(), y - y.mean()])
    if np.allclose(stacked, 0.0):
        return {"orientation_deg": float("nan"), "elongation": float("nan")}
    values, vectors = np.linalg.eigh(np.cov(stacked))
    major = vectors[:, int(np.argmax(values))]
    minor_value, major_value = float(np.min(values)), float(np.max(values))
    return {
        "orientation_deg": float(np.degrees(np.arctan2(major[1], major[0])) % 180.0),
        "elongation": float(np.sqrt(major_value / minor_value)) if minor_value > 1e-12 else float("inf"),
    }


@dataclass
class LandscapeMetrics:
    """Everything measured about one hidden state's success region.

    Attributes:
        xi: The four hidden values.
        swept_points, valid_points, success_points: Grid-point counts.
        valid_fraction, success_fraction: Of the swept points.
        force_extent, duration_extent: ``(min, max)`` over succeeding points, in N and s.
        centroid: ``(F, T)`` mean of the succeeding points.
        components: Connected regions under 4-connectivity.
        largest_component_fraction: Of the succeeding points.
        orientation_deg, elongation: Principal axis in the normalised box.
        midpoint: :func:`midpoint_failure_rate` output.
        row_contiguity, column_contiguity: Fraction of ``T`` rows (resp. ``F`` columns) whose
            succeeding points form one unbroken run.
        boundary_fraction: Succeeding points with at least one non-succeeding swept
            neighbour, over all succeeding points. A blob has a low value; a filigree has a
            high one.
        best_margin: The succeeding point furthest (in grid steps) from any failure, and that
            distance. This is what a max-margin parameter target would pick.
        min_cost: The succeeding point minimising a normalised ``F + T`` cost.
    """

    xi: dict
    swept_points: int
    valid_points: int
    success_points: int
    valid_fraction: float
    success_fraction: float
    force_extent: tuple[float, float] | None
    duration_extent: tuple[float, float] | None
    centroid: tuple[float, float] | None
    components: int
    largest_component_fraction: float
    orientation_deg: float
    elongation: float
    midpoint: dict
    row_contiguity: float
    column_contiguity: float
    boundary_fraction: float
    best_margin: dict | None = None
    min_cost: dict | None = None
    per_component: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _contiguity(mask: np.ndarray, axis: int) -> float:
    """Fraction of lines along ``axis`` whose succeeding cells form one unbroken run."""
    lines = mask if axis == 0 else mask.T
    populated = [line for line in lines if line.any()]
    if not populated:
        return float("nan")
    unbroken = 0
    for line in populated:
        indices = np.nonzero(line)[0]
        if indices[-1] - indices[0] + 1 == len(indices):
            unbroken += 1
    return unbroken / len(populated)


def _grid_distance_to_failure(mask: np.ndarray, swept: np.ndarray) -> np.ndarray:
    """Chebyshev distance in grid steps from each succeeding cell to the nearest failure.

    A multi-source breadth-first search from every non-succeeding swept cell and from the
    grid's outside, so a region touching the sweep's edge is not credited with infinite
    margin -- beyond the edge is unmeasured, not safe.
    """
    distance = np.full(mask.shape, np.inf)
    queue: deque[tuple[int, int]] = deque()
    for index in zip(*np.nonzero(swept & ~mask), strict=True):
        distance[index] = 0.0
        queue.append(index)
    rows, columns = mask.shape
    # Treat the grid's border as a failure source: outside the sweep nothing is known.
    for row in range(rows):
        for column in (0, columns - 1):
            if mask[row, column] and distance[row, column] > 1.0:
                distance[row, column] = 1.0
                queue.append((row, column))
    for column in range(columns):
        for row in (0, rows - 1):
            if mask[row, column] and distance[row, column] > 1.0:
                distance[row, column] = 1.0
                queue.append((row, column))

    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    while queue:
        row, column = queue.popleft()
        for delta_row, delta_column in steps:
            neighbour = (row + delta_row, column + delta_column)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < columns):
                continue
            if mask[neighbour] and distance[neighbour] > distance[row, column] + 1:
                distance[neighbour] = distance[row, column] + 1
                queue.append(neighbour)
    return np.where(mask, distance, np.nan)


def analyse_landscape(dataset, xi_key: tuple[float, ...], criteria: SuccessCriteria) -> LandscapeMetrics:
    """Measure one hidden state's success region."""
    masks = success_mask(dataset, xi_key, criteria)
    forces, durations = masks["forces"], masks["durations"]
    swept, valid, success = masks["swept"], masks["valid"], masks["success"]

    rows, columns = np.nonzero(success)
    labels, components = connected_components(success)
    sizes = [int((labels == label).sum()) for label in range(1, components + 1)]

    metrics = LandscapeMetrics(
        xi=dict(zip(("mass", "static_friction", "dynamic_friction", "damping"), xi_key, strict=True)),
        swept_points=int(swept.sum()),
        valid_points=int(valid.sum()),
        success_points=int(success.sum()),
        valid_fraction=float(valid.sum() / swept.sum()) if swept.any() else float("nan"),
        success_fraction=float(success.sum() / swept.sum()) if swept.any() else float("nan"),
        force_extent=(float(forces[columns].min()), float(forces[columns].max())) if len(columns) else None,
        duration_extent=(float(durations[rows].min()), float(durations[rows].max())) if len(rows) else None,
        centroid=(float(forces[columns].mean()), float(durations[rows].mean())) if len(rows) else None,
        components=components,
        largest_component_fraction=float(max(sizes) / sum(sizes)) if sizes else float("nan"),
        midpoint=midpoint_failure_rate(success, swept),
        row_contiguity=_contiguity(success, axis=0),
        column_contiguity=_contiguity(success, axis=1),
        boundary_fraction=_boundary_fraction(success, swept),
        per_component=[
            {
                "label": label,
                "points": int((labels == label).sum()),
                "force_extent": [
                    float(forces[np.nonzero(labels == label)[1]].min()),
                    float(forces[np.nonzero(labels == label)[1]].max()),
                ],
                "duration_extent": [
                    float(durations[np.nonzero(labels == label)[0]].min()),
                    float(durations[np.nonzero(labels == label)[0]].max()),
                ],
            }
            for label in range(1, components + 1)
        ],
        **_principal_axis(rows, columns, durations, forces),
    )

    if success.any():
        distance = _grid_distance_to_failure(success, swept)
        best = np.unravel_index(np.nanargmax(distance), distance.shape)
        metrics.best_margin = {
            "force": float(forces[best[1]]),
            "duration": float(durations[best[0]]),
            "grid_steps_to_failure": float(distance[best]),
        }
        force_span = float(forces[-1] - forces[0]) or 1.0
        duration_span = float(durations[-1] - durations[0]) or 1.0
        cost = (forces[columns] - forces[0]) / force_span + (durations[rows] - durations[0]) / duration_span
        cheapest = int(np.argmin(cost))
        metrics.min_cost = {
            "force": float(forces[columns[cheapest]]),
            "duration": float(durations[rows[cheapest]]),
            "normalised_cost": float(cost[cheapest]),
        }
    return metrics


def _boundary_fraction(mask: np.ndarray, swept: np.ndarray) -> float:
    """Succeeding cells adjacent to a swept failure, over all succeeding cells."""
    if not mask.any():
        return float("nan")
    padded_mask = np.pad(mask, 1, constant_values=False)
    padded_swept = np.pad(swept, 1, constant_values=False)
    boundary = np.zeros(mask.shape, dtype=bool)
    for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_mask = padded_mask[
            1 + delta_row : 1 + delta_row + mask.shape[0], 1 + delta_column : 1 + delta_column + mask.shape[1]
        ]
        shifted_swept = padded_swept[
            1 + delta_row : 1 + delta_row + mask.shape[0], 1 + delta_column : 1 + delta_column + mask.shape[1]
        ]
        boundary |= mask & shifted_swept & ~shifted_mask
    return float(boundary.sum() / mask.sum())
