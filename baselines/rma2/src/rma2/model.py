r"""Stage A: privileged direct adaptation. ``xi -> z_priv -> F_peak*``.

The RMA²-style baseline's ceiling. It is given the hidden state exactly, compresses it
through the same bottleneck width the main method's latent uses, and emits **one** force. It
is the privileged *point* regressor, which makes it the missing cell in the paper's
comparison:

===================  ==========================  ==============================
                     point output                success-landscape output
===================  ==========================  ==============================
privileged ``xi``    **Stage A (this)**          privileged teacher
probe only           Direct GRU                  ACE + PSP
===================  ==========================  ==============================

So ``teacher − StageA`` isolates landscape-versus-point with the input held fixed, and
``StageA − DirectGRU`` isolates privileged-versus-probe with the output form held fixed. The
+9.9 pp the main results report between ACE + PSP and Direct GRU currently confounds both.

What is carried over from RMA² verbatim, and why it matters
-----------------------------------------------------------
:class:`PrivilegedEncoder` uses RMA²'s ``MLP`` block -- ``Linear → LayerNorm → ELU`` per
layer, **including the output layer** (``algo/models.py:150-163``). That is not cosmetic: the
output LayerNorm is why ``z_priv`` is bounded and roughly zero-mean per sample, which is what
would make a plain MSE distillation well conditioned in a later Stage B. Reimplementing it
without that would change the distillation problem while claiming to reproduce it, so it is
kept here even though Stage A alone does not need it.

What is deliberately not RMA²
-----------------------------
There is no policy, no reward and no low-level action; the head emits a skill parameter. And
the width rule ``d_z = d_in - 4`` (``models.py:50``) is dropped because it yields 0 for a
four-dimensional ``xi`` -- see :class:`~rma2.config.StageACfg`.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["XI_DIMENSIONS", "ParameterHead", "PrivilegedEncoder", "StageAModel", "build_stage_a", "xi_moments"]

#: Order of the hidden state's four entries, matching ``ProbeBatch.xi``.
#:
#: Note that the *stored* third entry is the absolute ``dynamic_friction``, while
#: ``TRAINING_XI_RANGES`` declares a ``dynamic_friction_ratio``. Normalising an absolute against
#: a ratio bound is a confident wrong answer, so :func:`xi_moments` converts.
XI_DIMENSIONS = ("mass", "static_friction", "dynamic_friction", "damping")


def xi_moments(ranges: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    r"""Per-dimension mean and standard deviation implied by the **training** ranges.

    Taken from the ranges rather than from the data, for two reasons. It is deterministic --
    two runs on two splits normalise identically -- and it is the only defensible choice for
    an out-of-distribution evaluation, where the statistics must be the *training* ones or the
    OOD number means nothing (design note §7). For a uniform draw on :math:`[a, b]` the moments
    are :math:`(a+b)/2` and :math:`(b-a)/\sqrt{12}`.

    Args:
        ranges: ``XiRanges.as_dict()`` -- keyed on ``dynamic_friction_ratio``, not the absolute.

    Returns:
        ``(mean, std)``, each a 4-tuple in :data:`XI_DIMENSIONS` order.
    """
    static = ranges["static_friction"]
    ratio = ranges["dynamic_friction_ratio"]
    # Bounding box of mu_d = ratio * mu_s over the two independent ranges. Both are positive,
    # so the corners are the products of the corners.
    absolute = (ratio[0] * static[0], ratio[1] * static[1])
    bounds = (ranges["mass"], static, absolute, ranges["damping"])
    mean = tuple((low + high) / 2.0 for low, high in bounds)
    std = tuple(max((high - low) / 12.0**0.5, 1e-9) for low, high in bounds)
    return mean, std


class _MlpBlock(nn.Sequential):
    """``Linear → LayerNorm → ELU``, RMA²'s block, applied to the output layer as well."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ELU(),
        )


class PrivilegedEncoder(nn.Module):
    """``xi -> z_priv``. RMA²'s privileged encoder, re-widthed for a four-dimensional ``xi``.

    Args:
        xi_dim: Width of the hidden state. Four in this project (D015).
        latent_dim: Width of ``z_priv``.
        units: Hidden widths.
    """

    def __init__(self, xi_dim: int, latent_dim: int, units: tuple[int, ...]) -> None:
        super().__init__()
        widths = (xi_dim, *units, latent_dim)
        self.net = nn.Sequential(*(_MlpBlock(a, b) for a, b in zip(widths[:-1], widths[1:], strict=True)))

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Args: ``xi`` of shape ``(batch, xi_dim)``, already normalised. Returns ``(batch, d_z)``."""
        return self.net(xi)


class ParameterHead(nn.Module):
    r"""``(z, post_probe_state, task_condition) -> F_peak``, squashed into the allowed range.

    The squash is a sigmoid affine map onto ``[low, high]`` rather than a clip, so the
    gradient never vanishes at a bound and no force limit is hard-coded in the module -- the
    range is passed in from ``probe_drawer.experiment_plan.SETTING_V1_TASK``.

    ``task_condition`` is :math:`(d_\text{goal}, T_\text{goal})`. Both are constant across
    Dataset v1 and so teach the network nothing; they are here because a deployed robot *is*
    told the task, which is the same contract the main method's PSP takes (D044). Feeding them
    is what would let a second task distance be evaluated without a code change.

    Args:
        latent_dim: Width of ``z``.
        condition_dim: Width of the concatenated conditions -- 2 post-probe + 2 task.
        units: Hidden widths.
        force_range: ``(low, high)`` in newtons.
        dropout: Applied between hidden layers.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        units: tuple[int, ...],
        force_range: tuple[float, float],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        low, high = force_range
        if not high > low:
            raise ValueError(f"force_range must be increasing, got {force_range}.")
        self.register_buffer("_low", torch.tensor(float(low)))
        self.register_buffer("_span", torch.tensor(float(high) - float(low)))

        widths = (latent_dim + condition_dim, *units)
        layers: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            layers.append(_MlpBlock(a, b))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(widths[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self, context: torch.Tensor, post_probe: torch.Tensor, task_condition: torch.Tensor
    ) -> torch.Tensor:
        """Args: ``(batch, d_z)``, ``(batch, 2)``, ``(batch, 2)``. Returns ``(batch,)`` newtons."""
        features = torch.cat([context, post_probe, task_condition], dim=-1)
        raw = self.net(features).squeeze(-1)
        return self._low + self._span * torch.sigmoid(raw)


class StageAModel(nn.Module):
    """``PrivilegedEncoder + ParameterHead``. Reads ``xi``; never deployed on a robot.

    ``forward`` takes a whole :class:`~probe_drawer.training.dataloader.ProbeBatch` for
    symmetry with the main project's models, and touches only ``xi``, ``post_probe`` and
    ``task_condition``. It deliberately does **not** read ``history``: a Stage A that could
    see the probe would no longer be a privileged ceiling, and the difference from Stage B
    would stop being the thing measured.
    """

    def __init__(
        self,
        xi_dim: int,
        latent_dim: int,
        encoder_units: tuple[int, ...],
        head_units: tuple[int, ...],
        force_range: tuple[float, float],
        xi_mean: tuple[float, ...],
        xi_std: tuple[float, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(xi_mean) != xi_dim or len(xi_std) != xi_dim:
            raise ValueError(
                f"xi_mean and xi_std must each have {xi_dim} entries, got "
                f"{len(xi_mean)} and {len(xi_std)}."
            )
        # Buffers, not arguments: the normalisation then travels inside ``state_dict`` and a
        # checkpoint cannot be deployed against the wrong statistics. Getting this wrong is
        # silent, and it is exactly what would corrupt an OOD evaluation.
        self.register_buffer("_xi_mean", torch.tensor(xi_mean, dtype=torch.float32))
        self.register_buffer("_xi_std", torch.tensor(xi_std, dtype=torch.float32))
        self.encoder = PrivilegedEncoder(xi_dim, latent_dim, encoder_units)
        self.head = ParameterHead(latent_dim, 4, head_units, force_range, dropout)

    def normalise(self, xi: torch.Tensor) -> torch.Tensor:
        """``xi`` in physical units to standardised units, using the training moments."""
        return (xi - self._xi_mean) / self._xi_std

    def context(self, batch) -> torch.Tensor:
        return self.encoder(self.normalise(batch.xi))

    def forward(self, batch) -> torch.Tensor:
        """``(batch,)`` predicted ``F_peak`` in newtons."""
        return self.head(self.context(batch), batch.post_probe, batch.task_condition)


def build_stage_a(cfg, force_range: tuple[float, float], xi_ranges: dict) -> StageAModel:
    """Assemble Stage A from a :class:`~rma2.config.StageACfg` and the **training** xi ranges."""
    mean, std = xi_moments(xi_ranges)
    return StageAModel(
        xi_dim=len(XI_DIMENSIONS),
        latent_dim=cfg.latent_dim,
        encoder_units=tuple(cfg.encoder_units),
        head_units=tuple(cfg.head_units),
        force_range=force_range,
        xi_mean=mean,
        xi_std=std,
        dropout=cfg.dropout,
    )
