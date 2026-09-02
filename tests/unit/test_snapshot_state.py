"""The parts of a snapshot that need no simulator: the filter histories it must carry.

A snapshot's simulator half can only be tested with Isaac Sim running (see
``tests/integration/test_branching.py``). Its *sensor* half is pure Python, and it is the
half that is easy to get wrong: a derived channel is a function of the recent past, so
restoring physics without restoring the filters gives a branch a wrong first velocity.
"""

from __future__ import annotations

import pytest
import torch

from probe_drawer.sensors import CausalDerivative


def feed(derivative: CausalDerivative, samples: list[float]) -> None:
    for sample in samples:
        derivative.update(torch.tensor([sample, sample * 2.0]))


class TestCausalDerivativeState:
    def test_a_fresh_filter_reports_no_samples(self) -> None:
        state = CausalDerivative(dt=1 / 60, window=2).state_dict()
        assert state["previous"] is None
        assert state["value"] is None
        assert state["differences"] == []

    def test_restoring_reproduces_the_filtered_value(self) -> None:
        original = CausalDerivative(dt=1 / 60, window=2)
        feed(original, [0.0, 0.001, 0.003, 0.006])

        restored = CausalDerivative(dt=1 / 60, window=2)
        restored.load_state_dict(original.state_dict())

        assert torch.allclose(restored.filtered, original.filtered)
        assert torch.allclose(restored.raw, original.raw)
        assert torch.allclose(restored.value, original.value)

    def test_restoring_reproduces_the_next_update_too(self) -> None:
        """The point of carrying history: the *following* sample must also agree."""
        original = CausalDerivative(dt=1 / 60, window=3)
        feed(original, [0.0, 0.001, 0.003, 0.006])

        restored = CausalDerivative(dt=1 / 60, window=3)
        restored.load_state_dict(original.state_dict())

        for sample in (0.010, 0.015):
            original.update(torch.tensor([sample, sample * 2.0]))
            restored.update(torch.tensor([sample, sample * 2.0]))
            assert torch.allclose(restored.filtered, original.filtered)

    def test_a_filter_without_history_disagrees(self) -> None:
        """Evidence the history is load-bearing, not decoration."""
        original = CausalDerivative(dt=1 / 60, window=3)
        feed(original, [0.0, 0.001, 0.003, 0.006])

        naive = CausalDerivative(dt=1 / 60, window=3)
        naive.update(original.value.clone())
        naive.update(torch.tensor([0.010, 0.020]))
        original.update(torch.tensor([0.010, 0.020]))

        assert not torch.allclose(naive.filtered, original.filtered)

    def test_the_state_is_a_copy_not_a_view(self) -> None:
        """A snapshot that aliased live buffers would track the present, not the past."""
        derivative = CausalDerivative(dt=1 / 60, window=2)
        feed(derivative, [0.0, 0.001, 0.003])
        state = derivative.state_dict()
        frozen = state["value"].clone()

        feed(derivative, [0.020, 0.050])

        assert torch.allclose(state["value"], frozen)
        assert not torch.allclose(state["value"], derivative.value)

    def test_restoring_after_a_reset_recovers_the_history(self) -> None:
        derivative = CausalDerivative(dt=1 / 60, window=2)
        feed(derivative, [0.0, 0.001, 0.003])
        state = derivative.state_dict()
        expected = derivative.filtered.clone()

        derivative.reset()
        assert derivative.value is None

        derivative.load_state_dict(state)
        assert torch.allclose(derivative.filtered, expected)

    def test_an_incomplete_payload_is_refused(self) -> None:
        derivative = CausalDerivative(dt=1 / 60, window=2)
        with pytest.raises(KeyError, match="missing"):
            derivative.load_state_dict({"previous": None})

    def test_the_window_bounds_the_stored_differences(self) -> None:
        derivative = CausalDerivative(dt=1 / 60, window=2)
        feed(derivative, [0.0, 0.001, 0.003, 0.006, 0.010])
        assert len(derivative.state_dict()["differences"]) == 2
