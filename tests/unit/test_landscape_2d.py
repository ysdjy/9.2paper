"""The 2-D topology detectors, checked against masks whose answers are known.

These metrics decide whether Phase 12 proceeds, so they are validated on hand-built shapes
before being pointed at a sweep. A midpoint-failure detector that fires on a convex blob, or
a component counter that splits a staircase, would manufacture exactly the structure the
phase is supposed to be testing for -- and the numbers would look like physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.analysis.landscape_2d import (
    connected_components,
    midpoint_failure_rate,
    representative_hidden_states,
)
from probe_drawer.experiment_plan import TRAINING_XI_RANGES

FULL = np.ones((9, 9), dtype=bool)


def blob(shape=(9, 9), rows=slice(2, 7), columns=slice(2, 7)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def staircase(shape=(9, 9)) -> np.ndarray:
    """A thin band stepping one column per row: 4-disconnected, 8-connected, one physical band."""
    mask = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        column = min(row, shape[1] - 1)
        mask[row, column] = True
    return mask


def two_islands(shape=(9, 9)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[1:4, 1:3] = True
    mask[5:8, 6:8] = True
    return mask


class TestConnectedComponents:
    def test_a_blob_is_one_region(self) -> None:
        assert connected_components(blob())[1] == 1

    def test_two_islands_are_two_regions(self) -> None:
        assert connected_components(two_islands())[1] == 2

    def test_an_empty_mask_has_no_regions(self) -> None:
        assert connected_components(np.zeros((5, 5), dtype=bool))[1] == 0

    def test_a_staircase_splits_under_four_connectivity(self) -> None:
        """The artefact the analysis has to know about: this is one band, not nine."""
        assert connected_components(staircase(), diagonal=False)[1] > 1

    def test_a_staircase_is_one_region_under_eight_connectivity(self) -> None:
        assert connected_components(staircase(), diagonal=True)[1] == 1

    def test_two_islands_stay_two_under_eight_connectivity(self) -> None:
        """The complementary check: 8-connectivity must not merge everything."""
        assert connected_components(two_islands(), diagonal=True)[1] == 2

    def test_labels_cover_exactly_the_mask(self) -> None:
        mask = two_islands()
        labels, count = connected_components(mask)
        assert count == 2
        assert np.array_equal(labels > 0, mask)
        assert sorted(np.unique(labels[mask]).tolist()) == [1, 2]


class TestMidpointFailure:
    def test_a_convex_blob_never_fails(self) -> None:
        assert midpoint_failure_rate(blob(), FULL)["rate"] == pytest.approx(0.0)

    def test_the_full_grid_never_fails(self) -> None:
        assert midpoint_failure_rate(FULL, FULL)["rate"] == pytest.approx(0.0)

    def test_two_islands_fail_often(self) -> None:
        result = midpoint_failure_rate(two_islands(), FULL)
        assert result["rate"] > 0.5
        assert result["examples"]

    def test_a_ring_fails_through_its_hole(self) -> None:
        """A non-convex shape whose failures are unambiguous."""
        ring = blob()
        ring[4, 4] = False
        result = midpoint_failure_rate(ring, FULL)
        assert result["pairs_whose_midpoint_fails"] > 0

    def test_only_midpoints_on_the_grid_are_counted(self) -> None:
        """A pair whose mean falls between grid points is skipped, not rounded.

        Rounding would let the grid's resolution decide the verdict, which is the one thing
        this metric must not do.
        """
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[0, 1] = True  # midpoint would be at column 0.5 -- not a grid point
        assert midpoint_failure_rate(mask, FULL[:3, :3])["pairs_checked"] == 0

    def test_unswept_midpoints_are_skipped(self) -> None:
        """An unswept midpoint is unknown, not a failure."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        mask[0, 2] = True
        swept = np.ones((3, 3), dtype=bool)
        swept[0, 1] = False
        assert midpoint_failure_rate(mask, swept)["pairs_checked"] == 0

    def test_a_single_point_has_nothing_to_check(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert np.isnan(midpoint_failure_rate(mask, FULL[:5, :5])["rate"])


class TestRepresentativeHiddenStates:
    def test_the_first_sixteen_are_the_corners(self) -> None:
        """A diagonal of presets would miss a light drawer with high static friction, which
        Phase 10 found hardest."""
        corners = representative_hidden_states(48)[:16]
        ratios = [state["dynamic_friction"] / state["static_friction"] for state in corners]
        for values in (
            [state["mass"] for state in corners],
            [state["static_friction"] for state in corners],
            ratios,
            [state["damping"] for state in corners],
        ):
            assert len(set(round(value, 6) for value in values)) >= 2

    def test_every_axis_pairing_appears_among_the_corners(self) -> None:
        """All 16 sign combinations, not just 16 arbitrary points."""
        corners = representative_hidden_states(16)
        low = {
            "mass": TRAINING_XI_RANGES.mass[0],
            "static_friction": TRAINING_XI_RANGES.static_friction[0],
            "damping": TRAINING_XI_RANGES.damping[0],
        }
        high = {
            "mass": TRAINING_XI_RANGES.mass[1],
            "static_friction": TRAINING_XI_RANGES.static_friction[1],
            "damping": TRAINING_XI_RANGES.damping[1],
        }
        signature = {
            tuple(
                state[name] > 0.5 * (low[name] + high[name]) for name in ("mass", "static_friction", "damping")
            )
            + (state["dynamic_friction"] / state["static_friction"] > 0.65,)
            for state in corners
        }
        assert len(signature) == 16

    def test_dynamic_friction_never_exceeds_static(self) -> None:
        for state in representative_hidden_states(64):
            assert state["dynamic_friction"] <= state["static_friction"] + 1e-12

    def test_every_value_is_inside_the_training_box(self) -> None:
        for state in representative_hidden_states(64):
            assert TRAINING_XI_RANGES.mass[0] <= state["mass"] <= TRAINING_XI_RANGES.mass[1]
            assert (
                TRAINING_XI_RANGES.static_friction[0]
                <= state["static_friction"]
                <= TRAINING_XI_RANGES.static_friction[1]
            )
            assert TRAINING_XI_RANGES.damping[0] <= state["damping"] <= TRAINING_XI_RANGES.damping[1]

    def test_it_is_reproducible_and_index_stable(self) -> None:
        assert representative_hidden_states(32) == representative_hidden_states(32)
        assert representative_hidden_states(64)[:32] == representative_hidden_states(32)

    def test_fewer_than_sixteen_is_refused(self) -> None:
        with pytest.raises(ValueError, match="16 corners"):
            representative_hidden_states(8)
