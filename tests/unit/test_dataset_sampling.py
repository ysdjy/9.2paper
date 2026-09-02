"""What the samplers must do, and what they must not be able to see.

The forbidden-knowledge tests matter more than the distributional ones. A candidate sampler
that peeked at a label would produce a training distribution that depends on the answers, and
nothing downstream would fail -- the numbers would just quietly stop meaning what the paper
says they mean.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from probe_drawer.dataset import (
    ForceSamplerCfg,
    XiSamplerCfg,
    branch_order,
    build_plan,
    candidate_forces,
    sample_hidden_states,
    xi_id,
)
from probe_drawer.dataset import sampling
from probe_drawer.experiment_plan import MAIN_TASK, TRAINING_XI_RANGES


class TestHiddenStateSampling:
    def test_it_draws_the_requested_number(self) -> None:
        assert len(sample_hidden_states(XiSamplerCfg(num_states=64))) == 64

    def test_it_is_reproducible(self) -> None:
        cfg = XiSamplerCfg(num_states=32, seed=11)
        assert sample_hidden_states(cfg) == sample_hidden_states(cfg)

    def test_a_different_seed_gives_different_states(self) -> None:
        first = sample_hidden_states(XiSamplerCfg(num_states=32, seed=1))
        second = sample_hidden_states(XiSamplerCfg(num_states=32, seed=2))
        assert first != second

    def test_index_stability_when_the_dataset_grows(self) -> None:
        """Extending the dataset must not renumber the hidden states it already has."""
        small = sample_hidden_states(XiSamplerCfg(num_states=64))
        large = sample_hidden_states(XiSamplerCfg(num_states=512))
        assert large[:64] == small

    def test_dynamic_friction_never_exceeds_static(self) -> None:
        """PhysX silently discards such a write, so a violating draw would be invisible."""
        for state in sample_hidden_states(XiSamplerCfg(num_states=512)):
            assert state["dynamic_friction"] <= state["static_friction"] + 1e-12

    def test_every_value_lands_inside_its_range(self) -> None:
        cfg = XiSamplerCfg(num_states=256)
        for state in sample_hidden_states(cfg):
            assert cfg.mass[0] <= state["mass"] <= cfg.mass[1]
            assert cfg.static_friction[0] <= state["static_friction"] <= cfg.static_friction[1]
            assert cfg.damping[0] <= state["damping"] <= cfg.damping[1]
            ratio = state["dynamic_friction"] / state["static_friction"]
            assert cfg.dynamic_friction_ratio[0] - 1e-12 <= ratio <= cfg.dynamic_friction_ratio[1] + 1e-12

    def test_the_defaults_match_the_experiment_plan(self) -> None:
        """The sampler duplicates the ranges to stay import-free; they must not drift."""
        cfg = XiSamplerCfg()
        assert cfg.mass == TRAINING_XI_RANGES.mass
        assert cfg.static_friction == TRAINING_XI_RANGES.static_friction
        assert cfg.dynamic_friction_ratio == TRAINING_XI_RANGES.dynamic_friction_ratio
        assert cfg.damping == TRAINING_XI_RANGES.damping

    def test_coverage_is_more_even_than_plain_random(self) -> None:
        """The point of a low-discrepancy sequence, checked rather than assumed.

        Each of the four axes is split into eight bins; a Sobol draw should fill them far
        more evenly than uniform random sampling of the same size.
        """
        states = sample_hidden_states(XiSamplerCfg(num_states=256))
        values = np.array([[s["mass"], s["static_friction"], s["dynamic_friction"], s["damping"]] for s in states])

        sobol_imbalance = []
        for column, (low, high) in enumerate(
            (XiSamplerCfg().mass, XiSamplerCfg().static_friction, (0.0, 3.0), XiSamplerCfg().damping)
        ):
            counts, _ = np.histogram(values[:, column], bins=8, range=(low, high))
            sobol_imbalance.append(counts.std())

        rng = np.random.default_rng(0)
        random_imbalance = []
        for _ in range(4):
            counts, _ = np.histogram(rng.random(256), bins=8, range=(0.0, 1.0))
            random_imbalance.append(counts.std())

        # Compare the two most uniform axes (mass and damping are sampled directly; dynamic
        # friction is a product and so is not uniform by construction).
        assert min(sobol_imbalance[0], sobol_imbalance[3]) < np.mean(random_imbalance)

    def test_a_ratio_above_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mu_d > mu_s"):
            XiSamplerCfg(dynamic_friction_ratio=(0.3, 1.4))

    def test_a_decreasing_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            XiSamplerCfg(mass=(12.0, 4.0))


class TestCandidateForces:
    def test_it_draws_the_requested_count(self) -> None:
        assert len(candidate_forces("abc", ForceSamplerCfg(count=24))) == 24

    def test_every_force_lands_inside_the_task_range(self) -> None:
        cfg = ForceSamplerCfg(force_range=MAIN_TASK.peak_force_range)
        for state in sample_hidden_states(XiSamplerCfg(num_states=64)):
            forces = candidate_forces(xi_id(state), cfg)
            assert min(forces) >= cfg.force_range[0]
            assert max(forces) <= cfg.force_range[1]

    def test_the_forces_are_ascending_and_distinct(self) -> None:
        forces = candidate_forces("abc", ForceSamplerCfg(count=24))
        assert list(forces) == sorted(forces)
        assert len(set(forces)) == 24

    def test_each_stratum_contributes_exactly_one(self) -> None:
        cfg = ForceSamplerCfg(count=24, force_range=(0.15, 4.5))
        low, high = cfg.force_range
        width = (high - low) / cfg.count
        for index, force in enumerate(candidate_forces("abc", cfg)):
            assert low + index * width - 1e-9 <= force <= low + (index + 1) * width + 1e-9

    def test_it_is_reproducible(self) -> None:
        assert candidate_forces("abc") == candidate_forces("abc")

    def test_different_hidden_states_get_different_jitter(self) -> None:
        """Otherwise every drawer would share one force grid and the gaps between grid
        points would never be sampled."""
        assert candidate_forces("abc") != candidate_forces("def")

    def test_it_does_not_depend_on_call_order(self) -> None:
        first = candidate_forces("abc")
        candidate_forces("zzz")
        assert candidate_forces("abc") == first

    def test_the_sampler_cannot_see_a_label(self) -> None:
        """Its whole signature is an identifier and a config; there is nothing to peek at."""
        parameters = list(inspect.signature(candidate_forces).parameters)
        assert parameters == ["state_id", "cfg"]
        forbidden = ("success", "label", "oracle", "best_force", "band", "outcome")
        source = inspect.getsource(sampling)
        offenders = [
            name for name in forbidden if f"{name}=" in source.replace(" ", "") and "reads_labels" not in name
        ]
        assert not offenders, offenders

    def test_the_config_records_that_it_is_label_independent(self) -> None:
        assert ForceSamplerCfg().as_dict()["reads_labels"] is False

    def test_zero_jitter_puts_every_sample_at_its_centre(self) -> None:
        cfg = ForceSamplerCfg(count=4, force_range=(0.0 + 0.1, 4.1), jitter=0.0)
        width = (4.1 - 0.1) / 4
        expected = [0.1 + (index + 0.5) * width for index in range(4)]
        assert candidate_forces("abc", cfg) == pytest.approx(expected)

    def test_an_out_of_bounds_jitter_is_refused(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            ForceSamplerCfg(jitter=0.7)

    def test_a_non_positive_force_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="force_range"):
            ForceSamplerCfg(force_range=(0.0, 4.5))


class TestBranchOrder:
    def test_it_is_a_permutation(self) -> None:
        assert sorted(branch_order("probe-1", 24)) == list(range(24))

    def test_it_is_reproducible(self) -> None:
        assert branch_order("probe-1", 24) == branch_order("probe-1", 24)

    def test_different_probes_get_different_orders(self) -> None:
        """The three repeats of one hidden state must not share an order, or drift could
        still align with force across repeats."""
        assert branch_order("probe-1", 24) != branch_order("probe-2", 24)

    def test_it_is_not_the_identity(self) -> None:
        assert branch_order("probe-1", 24) != tuple(range(24))

    def test_position_and_force_rank_are_weakly_correlated(self) -> None:
        """The whole reason the order is shuffled: branch position must not track force.

        Averaged over many probes, the correlation between a candidate's force rank and the
        position it ran at should be near zero.
        """
        correlations = []
        for index in range(200):
            order = np.array(branch_order(f"probe-{index}", 24))
            position_of_force = np.empty(24, dtype=float)
            position_of_force[order] = np.arange(24, dtype=float)
            correlations.append(np.corrcoef(np.arange(24), position_of_force)[0, 1])
        assert abs(float(np.mean(correlations))) < 0.05

    def test_a_zero_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            branch_order("probe-1", 0)


class TestSamplingPlan:
    def test_the_plan_matches_dataset_v0(self) -> None:
        plan = build_plan(repeats=3)
        assert len(plan.states) == 512
        assert plan.num_probes == 1536
        assert plan.num_candidates == 36864

    def test_every_repeat_of_a_hidden_state_shares_its_force_set(self) -> None:
        """Three probes behind the same (xi, F) question is what makes an empirical success
        probability measurable (D036)."""
        plan = build_plan(repeats=3, xi_cfg=XiSamplerCfg(num_states=8))
        for state in plan.states:
            state_id = xi_id(state)
            assert plan.forces[state_id] == candidate_forces(state_id, plan.force_cfg)

    def test_every_hidden_state_has_a_distinct_identifier(self) -> None:
        plan = build_plan(xi_cfg=XiSamplerCfg(num_states=512))
        assert len(plan.forces) == 512

    def test_the_plan_describes_itself_for_the_manifest(self) -> None:
        described = build_plan(repeats=3, xi_cfg=XiSamplerCfg(num_states=8)).as_dict()
        assert described["num_hidden_states"] == 8
        assert described["probe_repeats"] == 3
        assert described["xi_sampler"]["method"] == "scrambled Sobol"
        assert described["force_sampler"]["reads_labels"] is False

    def test_zero_repeats_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repeats must be >= 1"):
            build_plan(repeats=0)
