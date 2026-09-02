"""Unit tests for the adaptation-premise audit. No Isaac Sim.

The audit's job is to say "this task does not need adaptation" or "averaging is unsafe here"
*before* a model is trained, so the tests are constructed landscapes where the right verdict
is known: a landscape one constant force solves must be reported as such, a landscape with a
hole in the middle of a band must be reported as non-contiguous, and a probe that carries no
information about the hidden state must not be reported as identifying it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rma2_direct.adaptation_premise import (
    audit,
    band_structure,
    collect_bands,
    identifiability,
    probe_ambiguity,
)
from probe_drawer.analysis.probe_features import PROBE_FEATURES
from probe_drawer.analysis.sweep import SweepDataset, SweepRecord
from probe_drawer.envs import XI_FIELDS
from probe_drawer.evaluation import SuccessCriteria

CRITERIA = SuccessCriteria(goal_displacement=0.05, displacement_tolerance=0.01, velocity_tolerance=0.08)
DURATION = 1.5
FORCES = tuple(round(1.0 + 0.25 * index, 2) for index in range(17))


def make_record(
    xi: tuple[float, float, float, float],
    peak_force: float,
    displacement: float,
    *,
    features: dict | None = None,
    valid: bool = True,
) -> SweepRecord:
    return SweepRecord(
        xi=dict(zip(XI_FIELDS, xi, strict=True)),
        peak_force=peak_force,
        duration=DURATION,
        final_displacement=displacement,
        final_velocity=0.02,
        peak_velocity=0.02,
        peak_measured_force=peak_force,
        travel_fraction=displacement / 0.4,
        peak_lateral_drift=0.0005,
        peak_orientation_drift_deg=0.3,
        termination_reason="duration_completed",
        valid=valid,
        protocol="sequential",
        probe_features=features or {},
    )


def features_for(required: float, noise: float = 0.0) -> dict:
    """A probe summary that is a deterministic, invertible function of the required force."""
    return {
        name: required * (1.0 + 0.1 * index) + noise
        for index, name in enumerate(PROBE_FEATURES)
    }


def marching_landscape(num_states: int = 12, slope: float = 0.02) -> SweepDataset:
    """Hidden state ``i`` needs ``1 + 0.25 i`` newtons, and its probe says so.

    ``d(F) = d_goal + slope * (F - required)``, so each state has a symmetric band around its
    own required force -- the shape a learnable, adaptation-requiring task has.
    """
    dataset = SweepDataset()
    for index in range(num_states):
        required = 1.0 + 0.25 * index
        xi = (4.0 + index, 0.5 + 0.1 * index, 0.3 + 0.05 * index, 2.0 + index)
        for force in FORCES:
            displacement = CRITERIA.goal_displacement + slope * (force - required)
            dataset.extend([make_record(xi, force, displacement, features=features_for(required))])
    return dataset


def universal_landscape(num_states: int = 12) -> SweepDataset:
    """Every hidden state succeeds at exactly the same force. No adaptation is needed."""
    dataset = SweepDataset()
    for index in range(num_states):
        xi = (4.0 + index, 0.5 + 0.1 * index, 0.3 + 0.05 * index, 2.0 + index)
        for force in FORCES:
            displacement = CRITERIA.goal_displacement + 0.02 * (force - 2.0)
            dataset.extend([make_record(xi, force, displacement, features=features_for(2.0))])
    return dataset


def test_a_landscape_one_constant_solves_is_reported_as_such() -> None:
    report = band_structure(universal_landscape(), CRITERIA, DURATION)
    assert report["coverage"] == 1.0
    assert report["best_constant_force"]["success_rate"] == 1.0


def test_a_marching_landscape_needs_adaptation() -> None:
    report = band_structure(marching_landscape(), CRITERIA, DURATION)
    assert report["coverage"] == 1.0
    # No constant can serve twelve states whose bands march across the axis.
    assert report["best_constant_force"]["success_rate"] < 0.5
    assert report["required_force"]["ratio"] > 2.0


def test_a_contiguous_band_has_a_succeeding_midpoint() -> None:
    report = band_structure(marching_landscape(), CRITERIA, DURATION)
    assert report["non_contiguous_bands"] == 0
    assert report["midpoint_succeeds"] == report["solvable"]


def test_a_hole_in_a_band_is_detected() -> None:
    """A band with an interior failure is exactly the case where averaging is unsafe."""
    dataset = SweepDataset()
    xi = (6.0, 1.0, 0.5, 4.0)
    for force in FORCES:
        # Succeeds at 1.0 and 2.0 N but not at the 1.5 N in between.
        on_goal = force in (1.0, 2.0)
        displacement = CRITERIA.goal_displacement if on_goal else CRITERIA.goal_displacement + 0.05
        dataset.extend([make_record(xi, force, displacement, features=features_for(1.5))])

    bands, unsolvable = collect_bands(dataset, CRITERIA, DURATION)
    assert unsolvable == 0
    assert bands[0].interior_failures > 0
    # The midpoint is inside the interval and still fails -- exactly the case `contains`
    # would wave through and `succeeds_at` catches.
    assert bands[0].contains(1.5) and not bands[0].succeeds_at(1.5)
    report = band_structure(dataset, CRITERIA, DURATION)
    assert report["non_contiguous_bands"] == 1
    assert report["midpoint_succeeds"] == 0


def test_unsolvable_states_are_counted_not_dropped_silently() -> None:
    dataset = marching_landscape(num_states=4)
    for force in FORCES:
        dataset.extend([make_record((99.0, 9.0, 9.0, 9.0), force, 0.5, features=features_for(9.0))])
    report = band_structure(dataset, CRITERIA, DURATION)
    assert report["unsolvable"] == 1
    assert report["total_hidden_states"] == 5
    assert report["coverage"] == pytest.approx(0.8)


def test_invalid_rows_never_reach_a_band() -> None:
    dataset = SweepDataset()
    xi = (6.0, 1.0, 0.5, 4.0)
    for force in FORCES:
        # On-goal but invalid: the rig misbehaved, so this is not evidence about the drawer.
        dataset.extend([make_record(xi, force, CRITERIA.goal_displacement, valid=False)])
    bands, unsolvable = collect_bands(dataset, CRITERIA, DURATION)
    assert bands == [] and unsolvable == 0


def test_an_informative_probe_identifies_the_hidden_state() -> None:
    report = identifiability(marching_landscape(num_states=16), CRITERIA, DURATION)
    for scores in report["xi_from_probe"].values():
        assert scores["r2"] > 0.9
    assert report["force_from_probe"]["linear"]["in_band_rate"] > 0.9


def test_an_uninformative_probe_is_not_reported_as_identifying() -> None:
    """Every state reports the same probe, so nothing about xi can be read from it."""
    dataset = SweepDataset()
    for index in range(16):
        required = 1.0 + 0.25 * index
        xi = (4.0 + index, 0.5 + 0.1 * index, 0.3 + 0.05 * index, 2.0 + index)
        for force in FORCES:
            displacement = CRITERIA.goal_displacement + 0.02 * (force - required)
            dataset.extend([make_record(xi, force, displacement, features=features_for(1.0))])

    report = identifiability(dataset, CRITERIA, DURATION)
    for scores in report["xi_from_probe"].values():
        assert scores["r2"] <= 0.0
    assert report["force_from_probe"]["linear"]["in_band_rate"] < 0.5


def test_ambiguity_is_zero_when_probes_separate_the_states() -> None:
    report = probe_ambiguity(marching_landscape(num_states=16), CRITERIA, DURATION, radii=(0.25,))
    assert report["radii"][0.25]["cluster_mean_misses_band"] == 0.0


def test_ambiguity_is_total_when_every_probe_is_identical() -> None:
    """Identical probes over states needing different forces: the mean must miss most bands."""
    dataset = SweepDataset()
    for index in range(16):
        required = 1.0 + 0.25 * index
        xi = (4.0 + index, 0.5 + 0.1 * index, 0.3 + 0.05 * index, 2.0 + index)
        for force in FORCES:
            displacement = CRITERIA.goal_displacement + 0.02 * (force - required)
            dataset.extend([make_record(xi, force, displacement, features=features_for(1.0))])

    report = probe_ambiguity(dataset, CRITERIA, DURATION, radii=(0.25,))
    assert report["radii"][0.25]["mean_cluster_size"] == 16.0
    assert report["radii"][0.25]["cluster_mean_misses_band"] > 0.5


def test_the_precision_requirement_is_the_band_half_width() -> None:
    report = identifiability(marching_landscape(), CRITERIA, DURATION)
    bands, _ = collect_bands(marching_landscape(), CRITERIA, DURATION)
    expected = float(np.median([band.width for band in bands]) / 2.0)
    assert report["precision_required"]["median_half_width"] == pytest.approx(expected)


def test_audit_bundles_all_three_and_serialises() -> None:
    report = audit(marching_landscape(), CRITERIA, DURATION, source="synthetic")
    payload = report.as_dict()
    assert payload["source"] == "synthetic"
    assert payload["criteria"]["goal_displacement"] == CRITERIA.goal_displacement
    assert set(payload) == {"source", "duration", "criteria", "structure", "ambiguity", "identifiability"}
