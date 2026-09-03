r"""Leave-one-out ridge readout -- how much a set of features says about a target.

Used wherever the question is "could a simple model recover this from what the probe
measured?", which is the question a probe design has to answer before anything is trained on
it. Deliberately linear and deliberately leave-one-out: a nonlinear model would confound the
probe's information content with the model's capacity, and an in-sample fit would mostly
measure the number of features.

Two corrections are baked in, both of which reversed a conclusion when they were found.

**Ridge, always.** The first comparison between the Phase 8-11 probe (9 features) and the
Phase 12 probe (18) produced leave-one-out :math:`R^2` values of -82 and -50 for the wider
feature sets. That is a numerically exploded fit, not a bad probe: with a few dozen points
and 18 correlated columns the unregularised solve is measuring conditioning. A single fixed
penalty applied to every candidate makes the comparison about what each probe measured.
The penalty is not tuned per candidate -- that would be fitting the comparison.

**Both** :math:`R^2` **and RMSE, always.** :math:`R^2` is normalised by the target's own
variance, so it rises when the *subset* is more spread out even if the predictions are no
better. Splitting Dataset v0 by probe duration gave RMSE of 0.332, 0.293 and 0.330 N across
terciles -- flat -- while :math:`R^2` climbed 0.378 to 0.695, purely because the target's sd
went from 0.42 to 0.60 N. Reporting one without the other invites exactly that mistake, so
:func:`leave_one_out` returns ``target_sd`` next to both and callers print all three.

Nothing here touches the simulation, and nothing under ``controllers/`` may import it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RIDGE_PENALTY", "leave_one_out"]

#: Ridge penalty on the standardised coefficients, fixed for every comparison.
#:
#: One value, applied to every candidate, so that a candidate with more features is not
#: rewarded or punished for that alone. See the module docstring for what happened without
#: it.
RIDGE_PENALTY = 1.0

#: Fewest usable rows for a readout to be reported at all.
#:
#: Below this the leave-one-out estimate is dominated by which row was left out. Reported as
#: ``nan`` rather than as a number a reader might quote.
MIN_ROWS = 8


def leave_one_out(features: np.ndarray, target: np.ndarray, penalty: float = RIDGE_PENALTY) -> dict:
    r"""Predict ``target`` from ``features``, each point from a fit that excluded it.

    Rows with a non-finite feature or target are **dropped rather than imputed**: a probe
    that aborted before it could measure something has not measured it, and substituting a
    column mean would let the fit borrow information the probe never had. The count of rows
    actually used comes back as ``n``, so a silently decimated readout is visible.

    Args:
        features: Shape ``(n_points, n_features)``. Standardised internally, so the columns
            need not share units.
        target: Shape ``(n_points,)``.
        penalty: Ridge penalty on the standardised coefficients. The intercept is left
            unpenalised -- shrinking it would bias every prediction toward zero instead of
            toward the target's mean.

    Returns:
        ``{"r2", "rmse", "n", "target_sd"}``. ``r2`` and ``rmse`` are ``nan`` when fewer than
        :data:`MIN_ROWS` rows survive, or when the target has no variance to explain.
    """
    features = np.atleast_2d(np.asarray(features, dtype=float))
    target = np.asarray(target, dtype=float).ravel()
    if features.shape[0] != target.shape[0]:
        raise ValueError(f"features has {features.shape[0]} rows and target has {target.shape[0]}.")
    if penalty < 0.0:
        raise ValueError(f"penalty must be >= 0, got {penalty}.")

    finite = np.isfinite(features).all(axis=1) & np.isfinite(target)
    features, target = features[finite], target[finite]
    count = int(len(target))
    if count < MIN_ROWS:
        return {"r2": float("nan"), "rmse": float("nan"), "n": count, "target_sd": float("nan")}

    standardised = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1e-9)
    width = standardised.shape[1]
    penalty_rows = np.sqrt(penalty) * np.hstack([np.eye(width), np.zeros((width, 1))])

    predictions = np.empty(count)
    for index in range(count):
        keep = np.ones(count, dtype=bool)
        keep[index] = False
        # Centring on the training fold's mean, and adding it back, is what keeps the
        # penalised solve from pulling predictions toward zero.
        centre = float(target[keep].mean())
        design = np.hstack([standardised[keep], np.ones((int(keep.sum()), 1))])
        augmented = np.vstack([design, penalty_rows])
        augmented_target = np.concatenate([target[keep] - centre, np.zeros(width)])
        solution, *_ = np.linalg.lstsq(augmented, augmented_target, rcond=None)
        predictions[index] = np.hstack([standardised[index], 1.0]) @ solution + centre

    residual = predictions - target
    variance = float(np.var(target))
    return {
        "r2": float(1.0 - np.mean(residual**2) / variance) if variance > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "n": count,
        "target_sd": float(np.sqrt(variance)),
    }
