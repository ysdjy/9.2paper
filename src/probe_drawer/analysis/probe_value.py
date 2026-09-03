r"""Is a dedicated active probe worth it, or would any interaction of the same length do?

The whole method rests on one premise: that a short, deliberate excitation reveals the force a
drawer will need. That premise has never been tested against the obvious alternatives -- doing
*nothing* for the same 18 steps, or applying a much weaker generic force. This module scores
those three histories on the same readout so the premise becomes a measurement.

The three differ **only in the excitation amplitude**: the same smoothstep trapezoid over the
same 0.3 s budget at 3.5 N (the frozen probe), 1.0 N (weak generic) and 0.0 N (passive
observation). Same length, same seven deployable channels, same feature extractor, same ridge
readout. Nothing else varies, so a difference in identifiability is attributable to the
amplitude and not to the format.

Two targets, and they answer different questions
------------------------------------------------
**own** -- each history predicts the force required *from the state it left behind*. This is
the deployment-faithful question: a system that spent its budget doing nothing still starts its
execution from wherever the drawer is, and needs to predict the force *it* will need.

**common** -- each history predicts the force the *frozen 3.5 N probe's* outcome requires. This
isolates information about the hidden dynamics from information about one's own post-probe
state, which the "own" target conflates.

Both are reported, because a passive observation scores differently on them for a reason worth
seeing: it leaves the drawer untouched, so its own target is easier (a fixed starting point) even
though it has learned less.

``R^2`` is never reported alone. It is normalised by the target's own variance, and the three
targets have different spreads, so a rise in ``R^2`` can be a change in the target rather than in
what was learned (``docs/DECISIONS.md`` D043). Every row carries RMSE and the target's sd too.
"""

from __future__ import annotations

import numpy as np

from probe_drawer.analysis.probe_features import PROBE_FEATURES, rank_correlation
from probe_drawer.analysis.readout import leave_one_out

__all__ = ["summarise_probe_value"]


def summarise_probe_value(variants: list[dict], feature_names: tuple[str, ...] = PROBE_FEATURES) -> dict:
    r"""Score each interaction history's readout of the required force.

    Args:
        variants: One entry per history, each with ``name``, ``amplitude`` (N), ``features``
            (``(n, k)`` in ``feature_names`` order), ``moved`` (per state), ``own_target`` and
            ``common_target`` (each ``(n,)`` newtons, with ``nan`` where undefined).
        feature_names: Column order of ``features``, for the per-feature table.

    Returns:
        ``per_variant`` keyed by name, plus ``comparison`` against the first variant listed
        (which must be the frozen probe) and ``feature_names``.

    Raises:
        ValueError: If fewer than two variants are given, or a variant's arrays disagree.
    """
    if len(variants) < 2:
        raise ValueError(f"need at least two histories to compare, got {len(variants)}.")
    for variant in variants:
        count = len(variant["own_target"])
        if not (
            variant["features"].shape[0] == count
            and len(variant["common_target"]) == count
            and len(variant["moved"]) == count
        ):
            raise ValueError(f"variant {variant['name']!r} has inconsistent array lengths.")

    per_variant = {}
    for variant in variants:
        features = np.asarray(variant["features"], dtype=float)
        entry: dict = {
            "amplitude": variant["amplitude"],
            "states": int(features.shape[0]),
            "breakaway_fraction": float(np.mean(variant["moved"])),
            # A feature that is identical for every drawer carries nothing, whatever the fit
            # then does with it. Counting them says *why* a history is uninformative.
            "constant_features": [
                name
                for index, name in enumerate(feature_names)
                if np.nanstd(features[:, index]) < 1e-12
            ],
        }
        for label in ("own", "common"):
            target = np.asarray(variant[f"{label}_target"], dtype=float)
            usable = np.isfinite(target)
            entry[label] = leave_one_out(features[usable], target[usable])
            # Spearman of the single strongest feature, reported beside the readout so a
            # history that is informative through one channel is not hidden by a fit.
            best_name, best_rho = None, 0.0
            for index, name in enumerate(feature_names):
                column = features[usable, index]
                if np.nanstd(column) < 1e-12 or len(column) < 3:
                    continue
                rho = abs(rank_correlation(list(column), list(target[usable])))
                if np.isfinite(rho) and rho > best_rho:
                    best_name, best_rho = name, float(rho)
            entry[label]["best_feature"] = best_name
            entry[label]["best_feature_abs_spearman"] = best_rho if best_name else None
        per_variant[variant["name"]] = entry

    reference = variants[0]["name"]
    comparison = {}
    for variant in variants[1:]:
        name = variant["name"]
        row = {}
        for label in ("own", "common"):
            probe, other = per_variant[reference][label], per_variant[name][label]
            row[label] = {
                "rmse_probe": probe["rmse"],
                "rmse_other": other["rmse"],
                "rmse_ratio": other["rmse"] / probe["rmse"] if probe["rmse"] else None,
                "rmse_increase_n": other["rmse"] - probe["rmse"],
                "r2_drop": probe["r2"] - other["r2"],
                "target_sd_probe": probe["target_sd"],
                "target_sd_other": other["target_sd"],
            }
        comparison[name] = row
    return {
        "reference": reference,
        "feature_names": list(feature_names),
        "per_variant": per_variant,
        "comparison": comparison,
    }
