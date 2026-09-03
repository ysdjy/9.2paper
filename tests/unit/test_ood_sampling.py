"""Unit tests for out-of-distribution hidden-state sampling. No Isaac Sim required.

The point of this sampler is that an "OOD" state is genuinely OOD. The OOD box contains the
training box, so a naive draw silently mixes in-distribution states into an OOD result -- and
they are the easy ones, so the mixture flatters whatever is measured on it. These tests are
mostly about that not happening.
"""

from __future__ import annotations

import pytest

from probe_drawer.dataset.sampling import (
    SAMPLED_AXES,
    axes_outside,
    sample_ood_hidden_states,
    sampled_axis_values,
)
from probe_drawer.experiment_plan import OOD_XI_RANGES, TRAINING_XI_RANGES

TRAINING = TRAINING_XI_RANGES.as_dict()
OOD = OOD_XI_RANGES.as_dict()


def state(mass=8.0, static=1.5, ratio=0.6, damping=6.0) -> dict:
    """A hidden state in stored form: absolute ``dynamic_friction``, not a ratio."""
    return {
        "mass": mass,
        "static_friction": static,
        "dynamic_friction": ratio * static,
        "damping": damping,
    }


class TestTheRatioIsRecoveredNotGuessed:
    def test_the_stored_absolute_friction_becomes_a_ratio(self) -> None:
        """The ranges are defined on the ratio; comparing an absolute against them is wrong."""
        values = sampled_axis_values(state(static=2.0, ratio=0.4))
        assert values["dynamic_friction_ratio"] == pytest.approx(0.4)
        assert values["static_friction"] == pytest.approx(2.0)
        assert set(values) == set(SAMPLED_AXES)

    def test_a_zero_static_friction_is_refused_rather_than_dividing(self) -> None:
        with pytest.raises(ValueError, match="static_friction must be > 0"):
            sampled_axis_values({"mass": 8.0, "static_friction": 0.0, "dynamic_friction": 0.0, "damping": 6.0})


class TestLocatingAState:
    def test_a_state_inside_every_range_reports_nothing(self) -> None:
        assert axes_outside(state(), TRAINING) == ()

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"mass": 15.0}, ("mass_high",)),
            ({"mass": 3.0}, ("mass_low",)),
            ({"static": 4.0}, ("static_friction_high",)),
            ({"static": 0.3}, ("static_friction_low",)),
            ({"ratio": 0.2}, ("dynamic_friction_ratio_low",)),
            ({"damping": 14.0}, ("damping_high",)),
            ({"damping": 1.2}, ("damping_low",)),
        ],
    )
    def test_it_names_the_axis_and_the_direction(self, kwargs: dict, expected: tuple) -> None:
        """A report has to say *where* a state is unusual, not only that it is."""
        assert axes_outside(state(**kwargs), TRAINING) == expected

    def test_several_novel_axes_are_all_reported(self) -> None:
        outside = axes_outside(state(mass=16.0, static=4.2, damping=0.5), TRAINING)
        assert set(outside) == {"mass_high", "static_friction_high", "damping_low"}

    def test_the_ratio_axis_can_only_be_novel_from_below(self) -> None:
        """Training and OOD share the 1.0 ceiling, which the report should not imply otherwise."""
        assert TRAINING["dynamic_friction_ratio"][1] == OOD["dynamic_friction_ratio"][1]
        assert axes_outside(state(ratio=1.0), TRAINING) == ()


class TestEveryDrawnStateIsGenuinelyOutside:
    def test_all_of_them_have_at_least_one_novel_axis(self) -> None:
        states = sample_ood_hidden_states(64, TRAINING, OOD, seed=7)
        assert len(states) == 64
        assert all(axes_outside(s, TRAINING) for s in states)

    def test_none_of_them_leaves_the_ood_box(self) -> None:
        states = sample_ood_hidden_states(48, TRAINING, OOD, seed=11)
        assert all(not axes_outside(s, OOD) for s in states)

    def test_mu_d_never_exceeds_mu_s(self) -> None:
        """A PhysX requirement, kept true by construction rather than by rejection (D016)."""
        states = sample_ood_hidden_states(48, TRAINING, OOD, seed=13)
        assert all(s["dynamic_friction"] <= s["static_friction"] + 1e-12 for s in states)

    def test_the_draw_is_reproducible_and_seed_dependent(self) -> None:
        assert sample_ood_hidden_states(16, TRAINING, OOD, seed=3) == sample_ood_hidden_states(
            16, TRAINING, OOD, seed=3
        )
        assert sample_ood_hidden_states(16, TRAINING, OOD, seed=3) != sample_ood_hidden_states(
            16, TRAINING, OOD, seed=4
        )

    def test_a_shorter_draw_is_a_prefix_of_a_longer_one(self) -> None:
        """Sobol index stability: extending a pilot must not renumber what it already had."""
        long = sample_ood_hidden_states(32, TRAINING, OOD, seed=5)
        assert sample_ood_hidden_states(8, TRAINING, OOD, seed=5) == long[:8]

    def test_the_novel_axes_are_not_all_the_same_one(self) -> None:
        """Rejection rather than construction, so which axis is novel stays varied."""
        states = sample_ood_hidden_states(64, TRAINING, OOD, seed=17)
        seen = {axis for s in states for axis in axes_outside(s, TRAINING)}
        assert len(seen) >= 5, f"only {seen} appeared; the draw is not exploring the box"


class TestArgumentValidation:
    def test_a_non_positive_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            sample_ood_hidden_states(0, TRAINING, OOD)

    def test_an_ood_range_that_does_not_contain_training_is_refused(self) -> None:
        """Then "outside training" would not mean what the function assumes, so it must not run."""
        narrow = {**OOD, "mass": (6.0, 10.0)}
        with pytest.raises(ValueError, match="does not contain the training range"):
            sample_ood_hidden_states(8, TRAINING, narrow)

    def test_too_little_oversampling_fails_loudly_rather_than_returning_a_short_list(self) -> None:
        with pytest.raises(ValueError, match="raise oversample"):
            sample_ood_hidden_states(8, TRAINING, OOD, oversample=1)
