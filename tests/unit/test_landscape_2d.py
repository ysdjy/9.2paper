"""The Phase 12 topology detectors, checked against masks whose answers are known.

These live in :mod:`probe_drawer.experimental` and are not part of Setting V1. The tests stay
because the Phase 12 conclusions rest on them: a midpoint-failure detector that fired on a
convex blob, or a component counter that split a staircase, would have manufactured exactly
the structure the phase was testing for.

These metrics decide whether Phase 12 proceeds, so they are validated on hand-built shapes
before being pointed at a sweep. A midpoint-failure detector that fires on a convex blob, or
a component counter that splits a staircase, would manufacture exactly the structure the
phase is supposed to be testing for -- and the numbers would look like physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.experimental.landscape_2d import connected_components, midpoint_failure_rate

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
