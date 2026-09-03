r"""Stage A configuration. One dataclass, snapshotted into ``configs/`` per the project's D011.

Every width and rate here is either RMA²'s own or the main project's, and the docstrings say
which. Nothing is tuned against the test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["StageACfg", "StageBCfg"]


@dataclass(frozen=True)
class StageACfg:
    """Sizes and optimisation for privileged direct adaptation.

    Args:
        latent_dim: Width of ``z_priv``. **16**, matching the main method's
            ``PspCfg.z_dim``, not RMA²'s ``d_in - 4`` rule -- which gives 0 for a
            four-dimensional ``xi`` -- and not the design note's provisional 8, which would
            hand this baseline half the bottleneck of the method it is compared against
            (``docs/SETTING_V1_DESIGN_AUDIT.md`` §1).
        encoder_units: Hidden widths of the privileged MLP. RMA²'s ``[128, 128, d_z]``
            (``algo/models.py:87``).
        head_units: Hidden widths of the parameter head. RMA²-style capacity, deliberately
            comparable to the main method's PSP rather than minimal: a strawman baseline
            measures nothing.
        dropout: Applied inside the head only. ``0`` keeps Stage A a clean ceiling estimate --
            the question it answers is what a point regressor *can* do with full ``xi``, so
            regularisation that trades fit for generalisation is off by default.
        epochs: Passes over the training split.
        batch_size: Rows per step. From the main project's ``train_force_regressor``.
        learning_rate: Adam step size. From the same, **not** RMA²'s 1e-4, which belongs to
            its Stage B adapter.
        weight_decay: Adam weight decay, from the same.
        seed: Torch and numpy seed.
        device: ``"cpu"`` or ``"cuda"``.
    """

    latent_dim: int = 16
    encoder_units: tuple[int, ...] = (128, 128)
    head_units: tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.0
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {self.latent_dim}.")
        if not self.encoder_units or not self.head_units:
            raise ValueError("encoder_units and head_units must each have at least one layer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {self.dropout}.")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["encoder_units"] = list(self.encoder_units)
        payload["head_units"] = list(self.head_units)
        return payload


@dataclass(frozen=True)
class StageBCfg:
    """Latent distillation. RMA²'s second stage, and its optimiser.

    Args:
        epochs: Passes over the training split.
        batch_size: Rows per step, from the main project's pipeline.
        learning_rate: Adam step size. **1e-4, RMA²'s own** (``algo/adaptation.py:53``), not
            Stage A's 3e-3 -- the adapter is fitted to a fixed target and RMA² deliberately
            moves it slowly.
        weight_decay: ``0``, as in the official adapter, which uses a plain Adam.
        seed: Torch and numpy seed. Stage B seed ``k`` distils from Stage A seed ``k``, so the
            three runs are three independent teacher-student pairs.
        device: ``"cpu"`` or ``"cuda"``.
    """

    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}.")

    def as_dict(self) -> dict:
        return asdict(self)
