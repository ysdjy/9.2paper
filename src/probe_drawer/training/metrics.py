"""Metrics, and the grouping that keeps them honest.

Candidate-level classification numbers (AUROC, AUPRC, Brier) answer "can the model rank
candidates", which is necessary but not the task. The task is *choose one force and have the
drawer end up at the goal*, so the metric that decides anything is the selection metric in
:func:`selection_metrics`: scan a probe's candidates, take the best-scoring one, and ask
whether that one actually succeeded.

They can diverge sharply. With about 6 % positives, a model that ranks well overall can still
pick the wrong candidate for the drawers whose success band is narrow -- which is precisely
the population the paper is about.

Pure numpy and torch; no sklearn, so nothing new to install.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

__all__ = [
    "calibration_error",
    "classification_metrics",
    "roc_auc",
    "selection_metrics",
]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the rank identity (Mann-Whitney U).

    Returns ``nan`` when one class is absent, because the quantity is undefined then rather
    than perfect or chance.
    """
    labels = np.asarray(labels).astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # Average the ranks of ties, or equal scores would be scored as if ordered.
    sorted_scores = scores[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve, the step-wise (AP) definition.

    Reported next to AUROC because with ~6 % positives AUROC is optimistic: a model can rank
    most negatives below most positives and still put a handful of negatives above every
    positive, which is what ruins a selection.
    """
    labels = np.asarray(labels).astype(bool)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order].astype(float)
    cumulative = np.cumsum(hits)
    precision = cumulative / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict:
    """Expected calibration error, plus the reliability curve it summarises.

    Calibration matters here beyond the usual reasons: force selection takes an argmax over
    predicted probabilities, and a model whose probabilities are ordered correctly but scaled
    wrongly still selects correctly. So a large ECE with good selection is informative rather
    than alarming, and the curve is returned so that can be seen.
    """
    labels = np.asarray(labels).astype(float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    curve, error = [], 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= 1.0)
        count = int(inside.sum())
        if count == 0:
            curve.append({"low": float(low), "high": float(high), "count": 0})
            continue
        confidence = float(probabilities[inside].mean())
        accuracy = float(labels[inside].mean())
        error += count / len(labels) * abs(accuracy - confidence)
        curve.append(
            {"low": float(low), "high": float(high), "count": count, "confidence": confidence, "accuracy": accuracy}
        )
    return {"ece": float(error), "curve": curve}


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Candidate-level scores. Necessary, and on their own not the answer."""
    labels = np.asarray(labels, dtype=float)
    logits = np.asarray(logits, dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    epsilon = 1e-7
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return {
        "rows": int(len(labels)),
        "positive_fraction": float(labels.mean()) if len(labels) else float("nan"),
        "auroc": roc_auc(labels, probabilities),
        "auprc": average_precision(labels, probabilities),
        "bce": float(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        **calibration_error(labels, probabilities),
    }


def selection_metrics(
    probe_ids: Sequence[str],
    forces: Sequence[float],
    labels: Sequence[float],
    scores: Sequence[float],
    reference_forces: dict[str, float] | None = None,
) -> dict:
    """The metric that matters: pick one force per probe and see whether it succeeded.

    Args:
        probe_ids: Which probe each row belongs to.
        forces: The candidate force of each row (N).
        labels: Whether each candidate actually succeeded.
        scores: The model's score for each row; higher is a stronger pick.
        reference_forces: Optional ``probe_id -> reference force``, e.g. the candidate whose
            displacement landed closest to the goal. Used for the force error only.

    Returns:
        Rates over all probes, and over the *feasible* probes only. The distinction is
        essential: a probe with no succeeding candidate cannot be selected correctly, so
        including it measures the dataset's coverage rather than the model.
    """
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for probe, force, label, score in zip(probe_ids, forces, labels, scores, strict=True):
        grouped[probe].append((float(score), float(force), float(label)))

    chosen_success, feasible_success, errors, chosen = [], [], [], {}
    for probe, rows in grouped.items():
        best = max(rows, key=lambda row: row[0])
        chosen[probe] = best[1]
        chosen_success.append(best[2])
        if any(row[2] > 0.5 for row in rows):
            feasible_success.append(best[2])
        if reference_forces and probe in reference_forces:
            errors.append(abs(best[1] - reference_forces[probe]))

    return {
        "probes": len(grouped),
        "feasible_probes": len(feasible_success),
        "selected_success_rate": float(np.mean(chosen_success)) if chosen_success else float("nan"),
        "selected_success_rate_feasible_only": (
            float(np.mean(feasible_success)) if feasible_success else float("nan")
        ),
        "force_mae": float(np.mean(errors)) if errors else float("nan"),
        "force_median_error": float(np.median(errors)) if errors else float("nan"),
        "selected_force": chosen,
    }


def reference_force_per_probe(samples: Sequence) -> dict[str, float]:
    """The candidate whose displacement landed closest to the goal, per probe.

    A regression target and a force-error reference that is defined for *every* probe,
    including one with no succeeding candidate -- unlike "the middle of the success band",
    which would silently drop the hardest cases.
    """
    best: dict[str, tuple[float, float]] = {}
    for sample in samples:
        error = abs(sample.final_total_displacement - sample.goal_displacement)
        if sample.probe_id not in best or error < best[sample.probe_id][0]:
            best[sample.probe_id] = (error, sample.candidate_peak_force)
    return {probe: force for probe, (_, force) in best.items()}


def success_forces_per_probe(samples: Sequence) -> dict[str, list[float]]:
    """The forces that actually succeeded, per probe."""
    forces: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.success:
            forces[sample.probe_id].append(sample.candidate_peak_force)
    return dict(forces)


def empirical_success_probability(samples: Sequence) -> dict[tuple[str, float], float]:
    """``(xi_id, force) -> fraction of repeats that succeeded``.

    The dataset stores binary labels, deliberately: a row records what happened, not an
    average. This computes the probability *for analysis* -- label-noise audits and
    calibration -- without ever writing it back (D036).
    """
    counts: dict[tuple[str, float], list[int]] = defaultdict(list)
    for sample in samples:
        counts[(sample.xi_id, round(sample.candidate_peak_force, 6))].append(int(sample.success))
    return {key: float(np.mean(values)) for key, values in counts.items()}
