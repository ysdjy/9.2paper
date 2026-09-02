"""The force search: deterministic, on the grid, and blind to anything privileged.

This is the component that stands between a model and the drawer at deployment, so its
failure modes are the quiet kind -- a search that silently prefers the grid's first entry, or
one whose result depends on iteration order, would look like a weak model rather than a bug.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from probe_drawer.evaluation import SelectionCfg, select_forces, select_nearest
from probe_drawer.evaluation import force_selection
from probe_drawer.experiment_plan import MAIN_TASK


def peaked_at(centres: list[float]):
    """A score function with a triangular peak per environment."""

    def score(forces: np.ndarray) -> np.ndarray:
        return np.array([1.0 - abs(force - centre) for force, centre in zip(forces, centres, strict=True)])

    return score


class TestGrid:
    def test_it_covers_the_task_force_range(self) -> None:
        grid = SelectionCfg(force_range=MAIN_TASK.peak_force_range, step=0.05).grid()
        assert grid[0] == pytest.approx(MAIN_TASK.peak_force_range[0])
        assert grid[-1] == pytest.approx(MAIN_TASK.peak_force_range[1])

    def test_the_step_matches_the_oracle_resolution(self) -> None:
        """0.05 N is what Phase 10 resolved the success band at; finer would imply a
        precision the labels do not have."""
        grid = SelectionCfg(step=0.05).grid()
        assert np.allclose(np.diff(grid), 0.05)


class TestSelection:
    def test_it_finds_each_environment_s_peak(self) -> None:
        selection = select_forces(peaked_at([1.0, 2.5, 4.0]), 3, SelectionCfg(step=0.05))
        assert selection.force.tolist() == pytest.approx([1.0, 2.5, 4.0])

    def test_it_is_deterministic(self) -> None:
        cfg = SelectionCfg(step=0.05)
        first = select_forces(peaked_at([2.0, 3.0]), 2, cfg)
        second = select_forces(peaked_at([2.0, 3.0]), 2, cfg)
        assert first.force.tolist() == second.force.tolist()

    def test_a_tie_resolves_to_the_lower_force(self) -> None:
        """The conservative choice for a drawer, and it makes the result reproducible."""
        selection = select_forces(lambda forces: np.ones(len(forces)), 1, SelectionCfg(step=0.05))
        assert selection.force[0] == pytest.approx(SelectionCfg().force_range[0])

    def test_it_keeps_the_whole_landscape(self) -> None:
        """So a landscape can be plotted, or an abstention rule fitted, without re-running
        the simulator."""
        cfg = SelectionCfg(step=0.05)
        selection = select_forces(peaked_at([2.0, 3.0]), 2, cfg)
        assert selection.landscape.shape == (2, len(cfg.grid()))
        assert selection.grid.tolist() == cfg.grid().tolist()

    def test_low_confidence_is_flagged_not_acted_on(self) -> None:
        """This round still executes the argmax; the flag is data for a future policy."""
        selection = select_forces(lambda forces: np.full(len(forces), 0.2), 2, SelectionCfg(abstain_below=0.5))
        assert selection.low_confidence.all()
        assert np.isfinite(selection.force).all(), "a flagged selection still names a force"

    def test_a_confident_selection_is_not_flagged(self) -> None:
        selection = select_forces(peaked_at([2.0]), 1, SelectionCfg(abstain_below=0.5))
        assert not selection.low_confidence.any()

    def test_a_wrongly_shaped_score_is_refused(self) -> None:
        with pytest.raises(ValueError, match="scores"):
            select_forces(lambda forces: np.zeros(len(forces) + 1), 2, SelectionCfg(step=0.5))

    def test_the_score_function_is_called_once_per_grid_point(self) -> None:
        cfg = SelectionCfg(step=0.5)
        calls = []

        def score(forces: np.ndarray) -> np.ndarray:
            calls.append(forces.copy())
            return np.zeros(len(forces))

        select_forces(score, 2, cfg)
        assert len(calls) == len(cfg.grid())
        assert all(len(np.unique(batch)) == 1 for batch in calls), "one force per call, shared by all envs"


class TestSnapping:
    def test_a_regressed_force_lands_on_the_grid(self) -> None:
        """A force regressor and a landscape model must be compared on the same executable
        forces, not one of them allowed arbitrary precision."""
        cfg = SelectionCfg(step=0.05)
        selection = select_nearest([2.03, 3.99, 0.16], cfg)
        assert selection.force.tolist() == pytest.approx([2.05, 4.00, 0.15])
        assert set(selection.force.tolist()) <= set(cfg.grid().tolist())

    def test_a_prediction_beyond_the_range_is_clamped_to_the_grid(self) -> None:
        cfg = SelectionCfg(force_range=(0.15, 4.5), step=0.05)
        assert select_nearest([-3.0, 99.0], cfg).force.tolist() == pytest.approx([0.15, 4.5])


class TestItCannotSeePrivilegedState:
    def test_the_search_takes_a_score_function_and_nothing_else(self) -> None:
        parameters = list(inspect.signature(select_forces).parameters)
        assert parameters == ["score_fn", "num_envs", "cfg"]

    def test_the_module_never_mentions_xi_or_a_label(self) -> None:
        """Selection happens at deployment, where none of these exist."""
        source = inspect.getsource(force_selection)
        for forbidden in ("\\bxi\\b", "success_label", "oracle"):
            assert forbidden.strip("\\b") not in source.replace("_", " ").split()

    def test_selection_does_not_import_the_simulator(self) -> None:
        source = inspect.getsource(force_selection)
        assert "isaaclab" not in source
