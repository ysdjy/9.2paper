"""Everything that must be true before a dataset is trained on.

The checks fall into three kinds, and only the first kind can be argued with:

**Distributional** — how many positives, how the forces and sequence lengths are spread, how
balanced the splits are. These are *reported*; a skewed distribution is information about the
task, not necessarily a defect.

**Structural** — every candidate points at a probe that exists, every probe belongs to one
hidden state, no identifier repeats, no NaN. These are *gates*: a dataset that fails one is
broken, not merely awkward.

**Leakage** — no hidden state, probe or candidate spans two splits, and no privileged field
sits where a model could read it. Also gates, and the ones worth the most scrutiny, because a
leak inflates every number downstream without failing anything.

Nothing here imports Isaac Lab, and nothing here modifies the dataset.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from probe_drawer.dataset.schema import XI_DIMENSIONS, model_input_fields
from probe_drawer.dataset.splits import NESTING, DataSplit
from probe_drawer.dataset.storage import DatasetStore

__all__ = ["audit_dataset"]

#: Checks whose failure blocks training. The rest are reported.
GATES = (
    "identifiers_unique",
    "probe_references_resolve",
    "probes_belong_to_one_hidden_state",
    "repeats_share_a_candidate_grid",
    "finite_values",
    "no_privileged_fields_in_model_input",
    "split_has_no_leakage",
    "counts_match_the_manifest",
    "branch_index_decorrelated_from_force",
)

#: Floor on the force/branch-position correlation the audit will call a failure.
#:
#: The threshold is a 3-sigma test rather than a constant, because the constant has to depend
#: on the dataset's size and a fixed one is wrong at both ends. Under the null hypothesis
#: that the branch order is a uniform random shuffle -- which is exactly what
#: :func:`~probe_drawer.dataset.sampling.branch_order` produces -- one probe's correlation
#: between force and position has variance about ``1 / (n - 1)`` for ``n`` candidates, so the
#: mean over ``P`` probes has standard error ``1 / sqrt((n - 1) * P)``.
#:
#: For Dataset v0 (``n = 24``, ``P = 1536``) that is 0.005, so the gate sits at 0.016 and
#: would catch any alignment above about 1.6 %. For a small smoke dataset (``n = 6``,
#: ``P = 16``) the same formula gives 0.112 and the gate sits at 0.34 -- correctly loose,
#: because at that size a shuffle simply cannot be shown to be biased.
#:
#: The floor keeps the gate from becoming absurdly tight on a very large dataset, where
#: 3-sigma would eventually fail on a correlation too small to matter.
MIN_FORCE_BRANCH_CORRELATION_FLOOR = 0.01


def _histogram(values: list[float], bins: int, span: tuple[float, float]) -> dict:
    counts, edges = np.histogram(values, bins=bins, range=span)
    return {"edges": [float(edge) for edge in edges], "counts": [int(count) for count in counts]}


def _check_identifiers(store: DatasetStore) -> dict:
    """No identifier may repeat, at any level."""
    states = [row["xi_id"] for row in store.hidden_states]
    probes = [row["probe_id"] for row in store.probes]
    candidates = [row["candidate_id"] for row in store.candidates]
    duplicates = {
        "xi_id": [key for key, count in Counter(states).items() if count > 1],
        "probe_id": [key for key, count in Counter(probes).items() if count > 1],
        "candidate_id": [key for key, count in Counter(candidates).items() if count > 1],
    }
    return {
        "unique_hidden_states": len(set(states)),
        "unique_probes": len(set(probes)),
        "unique_candidates": len(set(candidates)),
        "duplicates": {level: keys[:5] for level, keys in duplicates.items() if keys},
        "passes": not any(duplicates.values()),
    }


def _check_references(store: DatasetStore) -> dict:
    """Every candidate must resolve to a stored probe, and every probe to a history file."""
    probe_ids = {row["probe_id"] for row in store.probes}
    state_ids = {row["xi_id"] for row in store.hidden_states}
    dangling_probes = sorted({row["probe_id"] for row in store.candidates if row["probe_id"] not in probe_ids})
    dangling_states = sorted({row["xi_id"] for row in store.probes if row["xi_id"] not in state_ids})
    missing_files = [
        row["probe_id"]
        for row in store.probes
        if not (store.root / "probes" / f"{row['probe_id']}.npz").exists()
    ]
    return {
        "dangling_probe_references": dangling_probes[:5],
        "dangling_hidden_state_references": dangling_states[:5],
        "missing_history_files": missing_files[:5],
        "passes": not (dangling_probes or dangling_states or missing_files),
    }


def _check_probe_ownership(store: DatasetStore) -> dict:
    """A probe belongs to exactly one hidden state, and a candidate agrees with its probe."""
    owner = {row["probe_id"]: row["xi_id"] for row in store.probes}
    # A candidate whose probe is missing entirely is reported by
    # ``probe_references_resolve``; counting it here too would just double the noise, so an
    # unknown probe is skipped rather than treated as a disagreement.
    mismatched = [
        row["candidate_id"]
        for row in store.candidates
        if row["probe_id"] in owner and owner[row["probe_id"]] != row["xi_id"]
    ]
    per_state = Counter(row["xi_id"] for row in store.probes)
    return {
        "candidates_disagreeing_with_their_probe": mismatched[:5],
        "probes_per_hidden_state": dict(Counter(per_state.values())),
        "passes": not mismatched,
    }


def _check_candidate_grids(store: DatasetStore) -> dict:
    """Every repeat of a hidden state must have been asked the same candidate forces.

    This is what makes ``(xi, F)`` a repeated measurement and so lets an empirical success
    probability be computed (D036). If the grids differed, the three repeats would be three
    different questions.
    """
    by_probe: dict[str, list[float]] = {}
    probe_state = {row["probe_id"]: row["xi_id"] for row in store.probes}
    for row in store.candidates:
        by_probe.setdefault(row["probe_id"], []).append(row["candidate_peak_force"])

    grids: dict[str, set] = {}
    offenders = []
    for probe, forces in by_probe.items():
        state = probe_state.get(probe)
        if state is None:
            # Dangling reference; ``probe_references_resolve`` owns that failure.
            continue
        grids.setdefault(state, set()).add(tuple(sorted(forces)))
    for state, distinct in grids.items():
        if len(distinct) > 1:
            offenders.append(state)
    return {
        "hidden_states_checked": len(grids),
        "hidden_states_with_inconsistent_grids": offenders[:5],
        "passes": not offenders,
    }


def _check_finite(store: DatasetStore) -> dict:
    """No NaN or infinity, in the rows or in the recorded histories."""
    numeric = (
        "candidate_peak_force",
        "final_total_displacement",
        "final_velocity",
        "duration",
        "goal_displacement",
    )
    bad_rows = [
        row["candidate_id"]
        for row in store.candidates
        if any(not math.isfinite(float(row[name])) for name in numeric)
    ]
    bad_histories = []
    unreadable = []
    for row in store.probes:
        try:
            history = store.probe_history(row["probe_id"])
        except FileNotFoundError:
            # Reported by ``probe_references_resolve``. An audit must survive the breakage it
            # exists to detect, so this check reports what it could not read and moves on.
            unreadable.append(row["probe_id"])
            continue
        if any(not np.all(np.isfinite(values)) for values in history.values()):
            bad_histories.append(row["probe_id"])
    empty_histories = [row["probe_id"] for row in store.probes if row["num_steps"] < 2]
    return {
        "non_finite_rows": bad_rows[:5],
        "non_finite_histories": bad_histories[:5],
        "unreadable_histories": unreadable[:5],
        "histories_shorter_than_two_steps": empty_histories[:5],
        "passes": not (bad_rows or bad_histories or empty_histories),
    }


def _check_privileged(store: DatasetStore) -> dict:
    """The privileged fields must be present in the file and absent from the model's input.

    Both halves matter. ``xi`` has to be recorded -- the privileged teacher and the
    per-dimension analysis need it -- and it must never be reachable through
    ``model_input_fields()`` (D017).
    """
    allowed = set(model_input_fields())
    forbidden_in_input = sorted(allowed & {"xi", "xi_id", "branch_index", "success", "valid"})
    states = store.hidden_states
    complete = all(set(row["xi"]) == set(XI_DIMENSIONS) for row in states)
    channels = {channel for row in store.probes for channel in row["channels"]}
    return {
        "hidden_state_recorded": bool(states) and complete,
        "privileged_fields_reachable_from_model_input": forbidden_in_input,
        "history_channels": sorted(channels),
        "diagnostics_kept_out_of_history": not any(name.startswith("diagnostic/") for name in channels),
        "passes": bool(states) and complete and not forbidden_in_input,
    }


def _check_counts(store: DatasetStore) -> dict:
    """The manifest's promise must match what is on disk."""
    declared = store.manifest.get("counts", {})
    actual = {
        "hidden_states": len(store.hidden_states),
        "probes": len(store.probes),
        "candidates": len(store.candidates),
    }
    planned = store.manifest.get("sampling", {})
    expected = {
        "hidden_states": planned.get("num_hidden_states"),
        "probes": planned.get("num_probes"),
        "candidates": planned.get("num_candidates"),
    }
    return {
        "declared": declared,
        "actual": actual,
        "planned": expected,
        "passes": declared == actual and all(
            expected[name] in (None, actual[name]) for name in actual
        ),
    }


def _check_branch_decorrelation(store: DatasetStore) -> dict:
    """Force and branch position must be uncorrelated.

    The shuffle in :func:`~probe_drawer.dataset.sampling.branch_order` is what guarantees
    this, and branching's measured drift with sweep position is what makes it necessary. The
    correlation is computed per probe and then averaged, because within one probe the force
    set is fixed and the ordering is the only thing that varies.
    """
    by_probe: dict[str, list[tuple[int, float]]] = {}
    for row in store.candidates:
        by_probe.setdefault(row["probe_id"], []).append((row["branch_index"], row["candidate_peak_force"]))

    correlations = []
    for pairs in by_probe.values():
        if len(pairs) < 3:
            continue
        positions = np.array([position for position, _ in pairs], dtype=float)
        forces = np.array([force for _, force in pairs], dtype=float)
        if positions.std() == 0 or forces.std() == 0:
            continue
        correlations.append(float(np.corrcoef(positions, forces)[0, 1]))

    mean = float(np.mean(correlations)) if correlations else 0.0
    candidates_per_probe = int(np.median([len(pairs) for pairs in by_probe.values()])) if by_probe else 0
    degrees = max(candidates_per_probe - 1, 1)
    standard_error = 1.0 / math.sqrt(degrees * len(correlations)) if correlations else float("inf")
    tolerance = max(3.0 * standard_error, MIN_FORCE_BRANCH_CORRELATION_FLOOR)
    return {
        "probes_checked": len(correlations),
        "candidates_per_probe": candidates_per_probe,
        "mean_correlation": mean,
        # A single probe's shuffle can correlate strongly by chance -- with 24 candidates the
        # per-probe standard deviation is about 0.21 -- so the worst case is reported but the
        # *mean* is what is tested.
        "max_abs_correlation": float(np.max(np.abs(correlations))) if correlations else 0.0,
        "null_standard_error": standard_error,
        "tolerance": tolerance,
        "sigmas": abs(mean) / standard_error if standard_error > 0 else float("inf"),
        "passes": abs(mean) <= tolerance,
    }


def _check_split(split: DataSplit | None) -> dict:
    """No group at or below the split level may span two subsets."""
    if split is None:
        return {"skipped": "no split supplied", "passes": True}

    subsets = {"train": split.train, "val": split.val, "test": split.test}
    shared = {}
    for level in NESTING[: NESTING.index(split.level) + 1]:
        seen: dict[str, str] = {}
        overlaps = []
        for name, rows in subsets.items():
            for value in {getattr(sample, level) for sample in rows}:
                if value in seen and seen[value] != name:
                    overlaps.append(value)
                else:
                    seen[value] = name
        shared[level] = overlaps

    counts = split.counts()
    positives = {
        name: sum(sample.success for sample in rows) / len(rows) if rows else 0.0
        for name, rows in subsets.items()
    }
    empty = [name for name, rows in subsets.items() if not rows]
    return {
        "level": split.level,
        "counts": counts,
        "positive_fraction": positives,
        "empty_subsets": empty,
        # Reported, not gated: with few groups the hashed assignment can legitimately leave a
        # subset empty, and that is a fact about the dataset's size rather than a defect in
        # the split. It does make the corresponding metric meaningless, so it is surfaced.
        "overlapping_groups": {level: values[:5] for level, values in shared.items() if values},
        "passes": not any(shared.values()),
    }


def _distributions(store: DatasetStore) -> dict:
    """Reported, not gated: what the dataset actually looks like."""
    candidates = store.candidates
    forces = [row["candidate_peak_force"] for row in candidates]
    successes = [bool(row["success"]) for row in candidates]
    valid = [bool(row["valid"]) for row in candidates]
    lengths = store.sequence_lengths()
    span = store.manifest.get("main_task", {}).get("peak_force_range", [min(forces), max(forces)])

    # Positives per probe: the statistic that says whether the candidate budget is enough.
    per_probe: dict[str, int] = {}
    for row in candidates:
        per_probe[row["probe_id"]] = per_probe.get(row["probe_id"], 0) + int(bool(row["success"]))
    counts = Counter(per_probe.values())

    # Success against force, so an imbalance can be seen to be physical rather than a bug.
    edges = np.linspace(span[0], span[1], 13)
    binned = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = [row for row in candidates if low <= row["candidate_peak_force"] < high]
        binned.append(
            {
                "low": float(low),
                "high": float(high),
                "rows": len(inside),
                "positive_fraction": (
                    sum(bool(row["success"]) for row in inside) / len(inside) if inside else 0.0
                ),
            }
        )

    # An incomplete hidden state is a gate failure in ``no_privileged_fields_in_model_input``;
    # here it must not stop the rest of the report being produced, because a report is how
    # the failure gets diagnosed.
    complete = [row for row in store.hidden_states if set(row["xi"]) >= set(XI_DIMENSIONS)]
    xi_values = np.array(
        [[row["xi"][name] for name in XI_DIMENSIONS] for row in complete], dtype=float
    ) if complete else np.zeros((0, len(XI_DIMENSIONS)))
    return {
        "hidden_states_with_incomplete_xi": len(store.hidden_states) - len(complete),
        "rows": len(candidates),
        "probes": len(store.probes),
        "hidden_states": len(store.hidden_states),
        "positive_fraction": sum(successes) / len(candidates) if candidates else 0.0,
        "invalid_fraction": 1.0 - (sum(valid) / len(candidates)) if candidates else 0.0,
        "invalid_reasons": dict(
            Counter(reason for row in candidates for reason in row.get("invalid_reasons", []))
        ),
        "force_range": [min(forces), max(forces)],
        "force_histogram": _histogram(forces, 12, (span[0], span[1])),
        "success_vs_force": binned,
        "positives_per_probe": {str(key): value for key, value in sorted(counts.items())},
        "probes_with_no_positive": counts.get(0, 0),
        "probes_with_at_least_one": len(per_probe) - counts.get(0, 0),
        "probes_with_at_least_two": sum(value for key, value in counts.items() if key >= 2),
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "distinct": len(set(lengths)),
            "histogram": dict(sorted(Counter(lengths).items())),
        },
        "xi_ranges": {
            name: [float(xi_values[:, index].min()), float(xi_values[:, index].max())]
            for index, name in enumerate(XI_DIMENSIONS)
        }
        if len(xi_values)
        else {},
        "dynamic_friction_never_exceeds_static": bool(
            np.all(
                xi_values[:, XI_DIMENSIONS.index("dynamic_friction")]
                <= xi_values[:, XI_DIMENSIONS.index("static_friction")] + 1e-9
            )
        ),
    }


def audit_dataset(store: DatasetStore, split: DataSplit | None = None) -> dict:
    """Run every check. The verdict is ``all_gates_passed``."""
    checks = {
        "identifiers_unique": _check_identifiers(store),
        "probe_references_resolve": _check_references(store),
        "probes_belong_to_one_hidden_state": _check_probe_ownership(store),
        "repeats_share_a_candidate_grid": _check_candidate_grids(store),
        "finite_values": _check_finite(store),
        "no_privileged_fields_in_model_input": _check_privileged(store),
        "counts_match_the_manifest": _check_counts(store),
        "branch_index_decorrelated_from_force": _check_branch_decorrelation(store),
        "split_has_no_leakage": _check_split(split),
    }
    failed = [name for name in GATES if not checks[name]["passes"]]
    return {
        "dataset": store.describe(),
        "manifest": store.manifest,
        "checks": checks,
        "distributions": _distributions(store),
        "gates": list(GATES),
        "failed_gates": failed,
        "all_gates_passed": not failed,
    }
