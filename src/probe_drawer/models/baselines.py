"""The baselines the learned model has to beat, and one it must not lose to.

Phase 10 measured a rank correlation of 0.91 between a single scalar probe feature
(``displacement_per_newton``) and the force a drawer needs. That is a strong predictor, and
reporting it is not a formality: if a one-feature linear fit selects forces as well as
ACE + PSP, the honest conclusion is that the probe history buys nothing beyond a scalar, and
the paper's claim would have to change.

Four baselines, in increasing capacity, all predicting a **force** directly rather than a
success landscape:

``A`` linear regression on the single strongest scalar feature;
``B`` ridge regression on all probe summary features;
``C`` a small MLP on the probe summary features;
``D`` a GRU on the full 7-channel history.

``D`` is the one that isolates the question. It sees exactly what ACE sees, so a gap between
``D`` and ACE + PSP is attributable to *modelling the landscape* rather than to having more
input. A gap between ``B`` and ``D`` is attributable to the time series.

Linear and ridge fits are closed-form via ``numpy.linalg.lstsq``, so there is no sklearn
dependency for three lines of algebra.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from probe_drawer.models.psp import AdaptationContextEncoder, PspCfg

__all__ = [
    "STRONGEST_FEATURE",
    "FeatureRegression",
    "FixedForceBaseline",
    "GruForceRegressor",
    "MlpForceRegressor",
    "summary_matrix",
]

#: The feature Phase 10 found strongest against the sequential Oracle: Spearman -0.910.
#: Baseline A uses this one alone, so "a single scalar" is a concrete claim rather than a
#: hand-wave (``docs/ORACLE_LANDSCAPE.md``).
STRONGEST_FEATURE = "displacement_per_newton"


def summary_matrix(samples: Sequence, features: Sequence[str]) -> np.ndarray:
    """``(n, len(features))`` of probe summary values, in the order given."""
    return np.array(
        [[float(sample.probe_summary[name]) for name in features] for sample in samples], dtype=np.float64
    )


@dataclass
class FeatureRegression:
    """Least squares with an optional ridge penalty, on standardised inputs.

    Standardising inside the model rather than relying on the caller keeps the baseline
    self-contained -- it has to be fitted on the training split and applied unchanged
    elsewhere, and coupling that to the neural pipeline's scaler would make the two easy to
    mix up.

    Args:
        features: Which summary features to use. One name gives baseline A; the full list
            gives baseline B.
        alpha: Ridge penalty. ``0`` is ordinary least squares.
    """

    features: tuple[str, ...]
    alpha: float = 0.0
    coefficients: np.ndarray | None = field(default=None, init=False)
    intercept: float = field(default=0.0, init=False)
    mean: np.ndarray | None = field(default=None, init=False)
    std: np.ndarray | None = field(default=None, init=False)

    def fit(self, samples: Sequence, targets: Sequence[float]) -> FeatureRegression:
        values = summary_matrix(samples, self.features)
        target = np.asarray(targets, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.std = np.maximum(values.std(axis=0), 1e-9)
        design = (values - self.mean) / self.std

        if self.alpha > 0.0:
            # Augmenting the design matrix rather than forming the normal equations: better
            # conditioned, and it keeps the intercept out of the penalty.
            padded = np.vstack([design, np.sqrt(self.alpha) * np.eye(design.shape[1])])
            padded_target = np.concatenate([target - target.mean(), np.zeros(design.shape[1])])
            self.coefficients, *_ = np.linalg.lstsq(padded, padded_target, rcond=None)
            self.intercept = float(target.mean())
        else:
            augmented = np.hstack([design, np.ones((len(design), 1))])
            solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
            self.coefficients, self.intercept = solution[:-1], float(solution[-1])
        return self

    def predict(self, samples: Sequence) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("fit() before predict().")
        design = (summary_matrix(samples, self.features) - self.mean) / self.std
        return design @ self.coefficients + self.intercept

    def describe(self) -> dict:
        return {
            "kind": "ridge" if self.alpha > 0 else "linear",
            "features": list(self.features),
            "alpha": self.alpha,
            "coefficients": None if self.coefficients is None else self.coefficients.tolist(),
            "intercept": self.intercept,
        }


class MlpForceRegressor(nn.Module):
    """Baseline C: a small MLP from probe summary features to a force."""

    def __init__(self, num_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.num_features = num_features
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(batch, num_features) -> (batch,)`` predicted force."""
        return self.net(features).squeeze(-1)


class GruForceRegressor(nn.Module):
    """Baseline D: the same encoder ACE uses, regressing a force directly.

    Deliberately built from :class:`AdaptationContextEncoder` rather than a second GRU, so
    that a difference between this and ACE + PSP cannot be an architecture difference. Only
    the output changes: one number instead of a landscape.
    """

    def __init__(self, num_channels: int, cfg: PspCfg | None = None) -> None:
        super().__init__()
        cfg = cfg or PspCfg()
        self.encoder = AdaptationContextEncoder(num_channels, cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.z_dim + 2, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, batch) -> torch.Tensor:
        """``batch -> (batch,)`` predicted force.

        The post-probe state is included because the student gets it too; withholding it
        would make the comparison unfair in the learned model's favour.
        """
        context = self.encoder(batch.history, batch.lengths)
        return self.head(torch.cat([context, batch.post_probe], dim=-1)).squeeze(-1)


@dataclass
class FixedForceBaseline:
    """The floor: one force for every drawer, chosen on the training split.

    If a learned model cannot beat this, adaptation is not happening at all -- so it is the
    first thing any result has to clear, and it is cheap to compute.
    """

    force: float = 0.0

    def fit(self, forces: Sequence[float], labels: Sequence[float], grid: Sequence[float] | None = None):
        """Pick the force with the highest success rate among nearby candidates.

        Args:
            forces: Candidate force of each training row.
            labels: Whether it succeeded.
            grid: Forces to consider. Defaults to a 0.05 N grid over the observed range.
        """
        forces = np.asarray(forces, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if grid is None:
            grid = np.arange(forces.min(), forces.max() + 1e-9, 0.05)
        # A tolerance window rather than exact matching: candidate forces are jittered, so no
        # two probes share a force and an exact-match rate would be computed from one row.
        window = 0.1
        rates = [
            labels[np.abs(forces - candidate) <= window].mean()
            if np.any(np.abs(forces - candidate) <= window)
            else 0.0
            for candidate in grid
        ]
        self.force = float(grid[int(np.argmax(rates))])
        return self

    def predict(self, count: int) -> np.ndarray:
        return np.full(count, self.force, dtype=float)
