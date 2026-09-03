"""Unit tests for Stage B: latent distillation. No Isaac Sim, no dataset.

Three properties carry the weight, and each corresponds to a way Stage B could quietly stop
being Stage B: the head must actually be frozen (otherwise it is Stage C), the privileged
latent must be detached (otherwise the target drifts toward whatever the adapter finds easy),
and the deployed path must not read ``xi`` (otherwise it is not a probe-based method at all).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from probe_drawer.models import PspCfg
from rma2.config import StageACfg, StageBCfg
from rma2.model import build_stage_a, build_stage_b

RANGES = {
    "mass": (4.0, 12.0),
    "static_friction": (0.5, 3.0),
    "dynamic_friction_ratio": (0.3, 1.0),
    "damping": (2.0, 10.0),
}
FORCE_RANGE = (0.5, 6.5)
CHANNELS = 7


def batch(count: int = 4, steps: int = 18):
    return SimpleNamespace(
        xi=torch.randn(count, 4),
        post_probe=torch.randn(count, 2),
        task_condition=torch.tensor([[0.10, 1.5]] * count),
        history=torch.randn(count, steps, CHANNELS),
        lengths=torch.full((count,), steps, dtype=torch.long),
    )


def pair(latent: int = 16):
    stage_a = build_stage_a(StageACfg(latent_dim=latent), FORCE_RANGE, RANGES)
    return stage_a, build_stage_b(stage_a, CHANNELS, PspCfg(z_dim=latent))


class TestAssembly:
    def test_the_adapter_emits_the_latent_width_the_head_expects(self) -> None:
        _, model = pair(16)
        assert model.context(batch(5)).shape == (5, 16)

    def test_mismatched_latent_widths_are_refused_up_front(self) -> None:
        """Caught here rather than as a matmul error several minutes into an epoch."""
        stage_a = build_stage_a(StageACfg(latent_dim=16), FORCE_RANGE, RANGES)
        with pytest.raises(ValueError, match="latent widths must match"):
            build_stage_b(stage_a, CHANNELS, PspCfg(z_dim=8))

    def test_it_reuses_stage_a_modules_by_reference_not_by_copy(self) -> None:
        """A re-initialised head would distil into something Stage A never trained."""
        stage_a, model = pair()
        assert model.head is stage_a.head
        assert model.privileged is stage_a.encoder

    def test_the_normalisation_is_carried_over_from_stage_a(self) -> None:
        """Normalising xi differently from its teacher would chase a target never produced."""
        stage_a, model = pair()
        assert torch.equal(model._xi_mean, stage_a._xi_mean)
        assert torch.equal(model._xi_std, stage_a._xi_std)

    def test_the_adapter_is_the_main_methods_encoder_class(self) -> None:
        """Held constant across all three probe-based methods, so the comparison is about
        mechanism rather than about which network was chosen."""
        from probe_drawer.models.psp import AdaptationContextEncoder  # noqa: PLC0415

        _, model = pair()
        assert isinstance(model.adapter, AdaptationContextEncoder)


class TestTheFreezeIsRealAndAuditable:
    def test_only_the_adapter_is_trainable(self) -> None:
        _, model = pair()
        model.freeze_all_but_adapter()
        trainable = {name.split(".")[0] for name, p in model.named_parameters() if p.requires_grad}
        assert trainable == {"adapter"}

    def test_it_reports_what_it_froze(self) -> None:
        """Returned rather than done silently, so a test can assert it."""
        _, model = pair()
        frozen = model.freeze_all_but_adapter()
        assert frozen, "nothing was reported as frozen"
        assert all(not name.startswith("adapter.") for name in frozen)
        assert {name.split(".")[0] for name in frozen} == {"privileged", "head"}

    def test_a_distillation_step_leaves_the_head_untouched(self) -> None:
        """The end-to-end check: Stage C is exactly this test failing."""
        _, model = pair()
        model.freeze_all_but_adapter()
        before = [p.detach().clone() for p in model.head.parameters()]
        optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        z_probe, z_priv = model.latents(batch(8))
        loss = ((z_probe - z_priv) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        assert all(torch.equal(a, b) for a, b in zip(before, model.head.parameters(), strict=True))

    def test_the_same_step_does_move_the_adapter(self) -> None:
        """The counterpart, so the test above cannot pass by training nothing at all."""
        _, model = pair()
        model.freeze_all_but_adapter()
        before = [p.detach().clone() for p in model.adapter.parameters()]
        optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        z_probe, z_priv = model.latents(batch(8))
        optimiser.zero_grad(set_to_none=True)
        ((z_probe - z_priv) ** 2).mean().backward()
        optimiser.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.adapter.parameters(), strict=True))


class TestTheTargetIsFixed:
    def test_the_privileged_latent_is_detached(self) -> None:
        """RMA²'s stopgrad. Without it the teacher drifts toward the student."""
        _, model = pair()
        _, z_priv = model.latents(batch(5))
        assert not z_priv.requires_grad

    def test_the_probe_latent_does_carry_gradient(self) -> None:
        _, model = pair()
        z_probe, _ = model.latents(batch(5))
        assert z_probe.requires_grad

    def test_the_target_depends_on_xi_and_not_on_the_probe(self) -> None:
        _, model = pair()
        data = batch(5)
        _, first = model.latents(data)
        data.history = torch.randn_like(data.history)
        _, second = model.latents(data)
        assert torch.equal(first, second)
        data.xi = data.xi + 5.0
        _, third = model.latents(data)
        assert not torch.allclose(first, third)


class TestTheDeployedPathIsProbeOnly:
    def test_corrupting_xi_does_not_change_the_force(self) -> None:
        """``xi`` reaches this class only to build the training target; ``forward`` has no
        path to it, and a Stage B that peeked would not be a probe-based method."""
        _, model = pair()
        model.eval()
        data = batch(5)
        with torch.no_grad():
            before = model(data)
            data.xi = torch.randn_like(data.xi) * 1000.0
            after = model(data)
        assert torch.equal(before, after)

    def test_the_probe_does_change_the_force(self) -> None:
        _, model = pair()
        model.eval()
        data = batch(5)
        with torch.no_grad():
            before = model(data)
            data.history = data.history + 2.0
            after = model(data)
        assert not torch.allclose(before, after)

    def test_the_force_stays_inside_the_allowed_range(self) -> None:
        """The head is Stage A's, so its squash still applies -- verified, not assumed."""
        _, model = pair()
        model.eval()
        data = batch(6)
        data.history = data.history * 1e4
        with torch.no_grad():
            forces = model(data)
        assert torch.all((forces >= FORCE_RANGE[0]) & (forces <= FORCE_RANGE[1]))


class TestItCanActuallyDistil:
    def test_the_latent_error_falls_on_a_batch_it_can_memorise(self) -> None:
        """A tiny overfit. If this cannot fall, nothing about a real run is interpretable."""
        _, model = pair()
        model.freeze_all_but_adapter()
        data = batch(16)
        optimiser = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
        z_probe, z_priv = model.latents(data)
        first = float(((z_probe - z_priv) ** 2).mean())
        for _ in range(200):
            z_probe, z_priv = model.latents(data)
            loss = ((z_probe - z_priv) ** 2).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        assert float(loss) < 0.2 * first, f"latent MSE only fell from {first:.4f} to {float(loss):.4f}"


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"epochs": 0}, "epochs must be >= 1"),
            ({"learning_rate": 0.0}, "learning_rate must be > 0"),
            ({"weight_decay": -1.0}, "weight_decay must be >= 0"),
        ],
    )
    def test_it_refuses_nonsense(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            StageBCfg(**kwargs)

    def test_the_learning_rate_is_rma2s_own(self) -> None:
        """1e-4 from algo/adaptation.py:53, not Stage A's 3e-3."""
        assert StageBCfg().learning_rate == 1e-4
        assert StageBCfg().weight_decay == 0.0
