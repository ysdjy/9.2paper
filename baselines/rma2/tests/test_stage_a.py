"""Unit tests for Stage A: privileged direct adaptation. No Isaac Sim, no dataset.

Two properties carry most of the weight. The output must stay inside the allowed force range
for *any* input, because a squash that saturates into a clip would silently hand the executor
an illegal force. And Stage A must be unable to read the probe: a Stage A that could see it
would no longer be a privileged ceiling, and the difference from a future Stage B would stop
being the thing measured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rma2.config import StageACfg
from rma2.model import XI_DIMENSIONS, ParameterHead, StageAModel, build_stage_a, xi_moments
from rma2.trainer import force_mae

RANGES = {
    "mass": (4.0, 12.0),
    "static_friction": (0.5, 3.0),
    "dynamic_friction_ratio": (0.3, 1.0),
    "damping": (2.0, 10.0),
}
FORCE_RANGE = (0.5, 6.5)


def batch(count: int = 4, xi_value: float | None = None, history_channels: int = 7):
    xi = torch.randn(count, 4) if xi_value is None else torch.full((count, 4), xi_value)
    return SimpleNamespace(
        xi=xi,
        post_probe=torch.randn(count, 2),
        task_condition=torch.tensor([[0.10, 1.5]] * count),
        history=torch.randn(count, 18, history_channels),
        lengths=torch.full((count,), 18, dtype=torch.long),
    )


def model(cfg: StageACfg | None = None) -> StageAModel:
    return build_stage_a(cfg or StageACfg(), FORCE_RANGE, RANGES).eval()


class TestShapes:
    def test_it_maps_a_batch_to_one_force_each(self) -> None:
        assert model()(batch(6)).shape == (6,)

    def test_the_latent_is_the_configured_width(self) -> None:
        net = model(StageACfg(latent_dim=16))
        assert net.context(batch(5)).shape == (5, 16)

    def test_a_different_latent_width_is_honoured(self) -> None:
        assert model(StageACfg(latent_dim=8)).context(batch(3)).shape == (3, 8)


class TestTheForceIsAlwaysLegal:
    @pytest.mark.parametrize("xi_value", [0.0, 1e3, -1e3, 1e6])
    def test_extreme_inputs_stay_inside_the_range(self, xi_value: float) -> None:
        """A clip would give the same values here and a zero gradient; a squash does not."""
        with torch.no_grad():
            forces = model()(batch(4, xi_value=xi_value))
        assert torch.all(forces >= FORCE_RANGE[0])
        assert torch.all(forces <= FORCE_RANGE[1])

    def test_the_range_comes_from_the_caller_not_the_module(self) -> None:
        net = build_stage_a(StageACfg(), (1.0, 2.0), RANGES).eval()
        with torch.no_grad():
            forces = net(batch(8, xi_value=500.0))
        assert torch.all((forces >= 1.0) & (forces <= 2.0))

    def test_a_decreasing_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be increasing"):
            ParameterHead(4, 4, (8,), (6.5, 0.5))

    def test_the_squash_has_a_live_gradient_at_the_extremes(self) -> None:
        """The reason for a sigmoid rather than a clip: training must still be able to move."""
        net = build_stage_a(StageACfg(), FORCE_RANGE, RANGES)
        data = batch(4, xi_value=50.0)
        net(data).sum().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and any(float(g.abs().sum()) > 0 for g in grads)


class TestStageACannotSeeTheProbe:
    def test_corrupting_the_history_changes_nothing(self) -> None:
        """Structural: a privileged ceiling that peeked at the probe would measure nothing."""
        net = model()
        data = batch(5)
        with torch.no_grad():
            before = net(data)
            data.history = torch.randn_like(data.history) * 1000.0
            data.lengths = torch.ones_like(data.lengths)
            after = net(data)
        assert torch.equal(before, after)

    def test_it_runs_with_no_history_field_at_all(self) -> None:
        """The strongest form of the same check: the attribute need not exist."""
        net = model()
        bare = SimpleNamespace(
            xi=torch.randn(3, 4),
            post_probe=torch.randn(3, 2),
            task_condition=torch.tensor([[0.10, 1.5]] * 3),
        )
        with torch.no_grad():
            assert net(bare).shape == (3,)

    def test_xi_does_change_the_output(self) -> None:
        """The counterpart, so the test above cannot pass by ignoring every input."""
        net = model()
        data = batch(5)
        with torch.no_grad():
            before = net(data)
            data.xi = data.xi + 3.0
            after = net(data)
        assert not torch.allclose(before, after)

    def test_the_conditions_change_the_output(self) -> None:
        net = model()
        data = batch(5)
        with torch.no_grad():
            before = net(data)
            data.post_probe = data.post_probe + 2.0
            after = net(data)
        assert not torch.allclose(before, after)


class TestXiNormalisation:
    def test_the_moments_come_from_the_ranges(self) -> None:
        mean, std = xi_moments(RANGES)
        assert mean == pytest.approx((8.0, 1.75, 1.575, 6.0))
        assert std[0] == pytest.approx(8.0 / 12.0**0.5)

    def test_the_ratio_axis_is_converted_to_an_absolute(self) -> None:
        """The stored third entry is an absolute mu_d; the range declares a ratio.

        Comparing one against the other is a confident wrong answer, so the conversion is
        checked rather than assumed: mu_d spans ratio_low*mu_s_low to ratio_high*mu_s_high.
        """
        mean, _ = xi_moments(RANGES)
        low, high = 0.3 * 0.5, 1.0 * 3.0
        assert mean[2] == pytest.approx((low + high) / 2)

    def test_the_midpoint_normalises_to_zero(self) -> None:
        net = model()
        mean, _ = xi_moments(RANGES)
        assert net.normalise(torch.tensor([list(mean)])).abs().max() < 1e-6

    def test_the_statistics_travel_in_the_state_dict(self) -> None:
        """So a checkpoint cannot be deployed against the wrong normalisation -- which would
        be silent, and would corrupt an out-of-distribution evaluation in particular."""
        state = model().state_dict()
        assert "_xi_mean" in state and "_xi_std" in state

    def test_a_reloaded_checkpoint_reproduces_its_predictions(self) -> None:
        source = model()
        target = model()
        data = batch(6)
        with torch.no_grad():
            assert not torch.allclose(source(data), target(data))
        target.load_state_dict(source.state_dict())
        with torch.no_grad():
            assert torch.equal(source(data), target(data))

    def test_mismatched_statistics_are_refused(self) -> None:
        with pytest.raises(ValueError, match="must each have 4 entries"):
            StageAModel(4, 8, (16,), (16,), FORCE_RANGE, (0.0, 0.0), (1.0, 1.0, 1.0, 1.0))

    def test_the_dimension_order_is_the_stored_one(self) -> None:
        assert XI_DIMENSIONS == ("mass", "static_friction", "dynamic_friction", "damping")


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"latent_dim": 0}, "latent_dim must be >= 1"),
            ({"encoder_units": ()}, "at least one layer"),
            ({"head_units": ()}, "at least one layer"),
            ({"dropout": 1.0}, r"dropout must lie in \[0, 1\)"),
            ({"epochs": 0}, "epochs must be >= 1"),
            ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ],
    )
    def test_it_refuses_nonsense(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            StageACfg(**kwargs)

    def test_the_defaults_match_the_confirmed_design(self) -> None:
        cfg = StageACfg()
        assert cfg.latent_dim == 16, "the audit fixed the latent at 16, matching PspCfg.z_dim"
        assert cfg.learning_rate == 3e-3, "copied from the main project's force-regressor recipe"

    def test_it_serialises_to_plain_types(self) -> None:
        payload = StageACfg().as_dict()
        assert isinstance(payload["encoder_units"], list)


class TestForceMae:
    def test_it_averages_absolute_per_probe_errors(self) -> None:
        assert force_mae({"a": 1.0, "b": 3.0}, {"a": 1.5, "b": 3.5}) == pytest.approx(0.5)

    def test_probes_without_a_target_are_skipped(self) -> None:
        assert force_mae({"a": 1.0, "b": 9.0}, {"a": 1.5}) == pytest.approx(0.5)

    def test_no_overlap_is_nan_rather_than_zero(self) -> None:
        """Zero would read as a perfect fit."""
        import math  # noqa: PLC0415

        assert math.isnan(force_mae({"a": 1.0}, {"b": 1.0}))
