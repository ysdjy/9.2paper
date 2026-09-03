"""The two hidden-state samplers, and the box mapping they share.

Two samplers exist because a *dataset* wants a plain low-discrepancy fill while a small
*sweep* wants guaranteed corner coverage -- 16 states that span the box rather than sample it.
They share one implementation of the part that is easy to get silently wrong: the affine
scaling onto each axis and ``mu_dynamic = ratio * mu_static``, which is what keeps
``mu_d <= mu_s`` true by construction (D016).
"""

from __future__ import annotations

import numpy as np
import pytest

from probe_drawer.dataset.sampling import (
    SAMPLED_AXES,
    XiSamplerCfg,
    representative_hidden_states,
    sample_hidden_states,
    scale_unit_box,
)
from probe_drawer.experiment_plan import TRAINING_XI_RANGES

BOUNDS = {
    "mass": TRAINING_XI_RANGES.mass,
    "static_friction": TRAINING_XI_RANGES.static_friction,
    "dynamic_friction_ratio": TRAINING_XI_RANGES.dynamic_friction_ratio,
    "damping": TRAINING_XI_RANGES.damping,
}


class TestSharedBoxMapping:
    def test_the_corners_of_the_unit_box_map_to_the_bounds(self) -> None:
        states = scale_unit_box([[0.0] * 4, [1.0] * 4], BOUNDS)
        assert states[0]["mass"] == pytest.approx(TRAINING_XI_RANGES.mass[0])
        assert states[1]["mass"] == pytest.approx(TRAINING_XI_RANGES.mass[1])
        assert states[1]["damping"] == pytest.approx(TRAINING_XI_RANGES.damping[1])

    def test_dynamic_friction_is_the_ratio_times_static(self) -> None:
        """The mapping that makes ``mu_d <= mu_s`` structural rather than checked."""
        state = scale_unit_box([[0.5, 1.0, 1.0, 0.5]], BOUNDS)[0]
        assert state["dynamic_friction"] == pytest.approx(
            TRAINING_XI_RANGES.dynamic_friction_ratio[1] * state["static_friction"]
        )

    def test_a_wrongly_shaped_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match=f"must be \\(n, {len(SAMPLED_AXES)}\\)"):
            scale_unit_box([[0.5, 0.5]], BOUNDS)


class TestBothSamplers:
    @pytest.mark.parametrize(
        "states",
        [sample_hidden_states(XiSamplerCfg(num_states=64)), representative_hidden_states(64)],
        ids=["sobol", "representative"],
    )
    def test_dynamic_friction_never_exceeds_static(self, states) -> None:
        for state in states:
            assert state["dynamic_friction"] <= state["static_friction"] + 1e-12

    @pytest.mark.parametrize(
        "states",
        [sample_hidden_states(XiSamplerCfg(num_states=64)), representative_hidden_states(64)],
        ids=["sobol", "representative"],
    )
    def test_every_value_is_inside_the_training_box(self, states) -> None:
        for state in states:
            assert TRAINING_XI_RANGES.mass[0] <= state["mass"] <= TRAINING_XI_RANGES.mass[1]
            assert (
                TRAINING_XI_RANGES.static_friction[0]
                <= state["static_friction"]
                <= TRAINING_XI_RANGES.static_friction[1]
            )
            assert TRAINING_XI_RANGES.damping[0] <= state["damping"] <= TRAINING_XI_RANGES.damping[1]

    def test_the_two_samplers_are_not_the_same_sequence(self) -> None:
        """They serve different purposes; if they agreed one of them would be redundant."""
        assert sample_hidden_states(XiSamplerCfg(num_states=32)) != representative_hidden_states(32)


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
