r"""Stage A training: fit ``F_peak`` to the shared per-probe reference force.

**The recipe is copied from the main project on purpose**, not chosen. Direct GRU is trained by
``scripts/train_models.py::train_force_regressor`` -- MSE on the per-probe reference force,
Adam at 3e-3 with 1e-4 weight decay, batches of 256 candidate rows, and the epoch selected on
the mean absolute per-probe error over the validation split. Stage A uses the same data
pipeline, the same target dictionary, the same optimiser and the same selection rule, so that
the **only** difference between the two is what the network reads and how it is wired
(``docs/SETTING_V1_DESIGN_AUDIT.md`` §2). A trainer of this baseline's own devising would make
the comparison partly about optimisation.

Two consequences of reusing that pipeline, both deliberate:

* Rows are **per candidate**, so each probe appears about 32 times per epoch with the same
  target. For a point regressor that is redundant rather than wrong -- it is an epoch
  multiplier -- and matching Direct GRU's exposure is worth more than saving the time.
* Invalid rows are dropped by ``SampleDataset``, and the target is computed on what survives,
  exactly as the main project does it.

The target itself is ``probe_drawer.training.metrics.reference_force_per_probe``: the candidate
whose displacement landed closest to the goal. It is shared with Direct GRU, the ridge and the
linear fit, and is defined for **every** probe including the ones no candidate solved, so the
hardest states are not silently dropped. The design note's max-margin alternative was rejected
because over 32 jittered candidates spanning 6 N the margin it estimates is mostly grid noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from probe_drawer.training import SampleDataset, make_loader

__all__ = ["StageAResult", "StageBResult", "train_stage_a", "train_stage_b"]


@dataclass
class StageAResult:
    """A trained Stage A, with everything needed to audit and redeploy it."""

    model: torch.nn.Module
    cfg: object
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_force_mae: float = float("inf")
    best_state: dict = field(default_factory=dict)

    def restore_best(self) -> torch.nn.Module:
        """Load the selected epoch's weights. Called before any reported number."""
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.model


def _per_probe_predictions(model: torch.nn.Module, dataset: SampleDataset, device: str) -> dict[str, float]:
    """One force per probe, averaging the identical per-candidate rows of that probe.

    Stage A's prediction does not depend on the candidate, so the rows of a probe should agree
    to floating point; the mean is taken anyway rather than the first, because relying on that
    agreement would hide a bug that made them differ.
    """
    model.eval()
    totals: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in make_loader(dataset, batch_size=512, shuffle=False):
            predicted = model(batch.to(device)).cpu().numpy()
            for probe, value in zip(batch.probe_ids, predicted, strict=True):
                totals.setdefault(probe, []).append(float(value))
    return {probe: float(np.mean(values)) for probe, values in totals.items()}


def force_mae(predictions: dict[str, float], targets: dict[str, float]) -> float:
    """Mean absolute per-probe force error, in newtons. ``nan`` when nothing overlaps."""
    errors = [abs(predictions[probe] - targets[probe]) for probe in predictions if probe in targets]
    return float(np.mean(errors)) if errors else float("nan")


def train_stage_a(
    model: torch.nn.Module,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    targets: dict[str, dict[str, float]],
    cfg,
) -> StageAResult:
    """Fit Stage A and select the epoch by validation force MAE.

    Args:
        model: A :class:`~rma2.model.StageAModel`.
        train_dataset: Training rows, already scaled by the shared ``FeatureScaler``.
        val_dataset: Validation rows.
        targets: ``{"train": {probe_id: force}, "val": ...}`` from
            ``reference_force_per_probe``.
        cfg: A :class:`~rma2.config.StageACfg`.

    Returns:
        A :class:`StageAResult` whose ``best_state`` is the selected epoch.

    Raises:
        KeyError: If a training row's probe has no target, which would mean the target
            dictionary was built from a different split than the dataset.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(cfg.device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = make_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    result = StageAResult(model=model, cfg=cfg)
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in loader:
            # Recovered through the probe id rather than by position, because the loader
            # shuffles.
            target = torch.tensor(
                [targets["train"][probe] for probe in batch.probe_ids], dtype=torch.float32
            ).to(cfg.device)
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch.to(cfg.device)), target)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))

        validation = force_mae(_per_probe_predictions(model, val_dataset, cfg.device), targets["val"])
        result.history.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(losses)) if losses else float("nan"),
                "val_force_mae": validation,
            }
        )
        if np.isfinite(validation) and validation < result.best_val_force_mae:
            result.best_val_force_mae = validation
            result.best_epoch = epoch
            result.best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    return result


@dataclass
class StageBResult:
    """A trained adapter, with the distillation curve and its downstream diagnostic."""

    model: torch.nn.Module
    cfg: object
    frozen_parameters: list[str] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_latent_mse: float = float("inf")
    best_state: dict = field(default_factory=dict)

    def restore_best(self) -> torch.nn.Module:
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.model


def _latent_mse(model: torch.nn.Module, dataset: SampleDataset, device: str) -> float:
    """Mean squared latent error over a split, in the same units the loss uses."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in make_loader(dataset, batch_size=512, shuffle=False):
            z_probe, z_priv = model.latents(batch.to(device))
            total += float(((z_probe - z_priv) ** 2).mean()) * len(batch)
            count += len(batch)
    return total / count if count else float("nan")


def train_stage_b(
    model: torch.nn.Module,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    targets: dict[str, dict[str, float]],
    cfg,
) -> StageBResult:
    r"""Distil the privileged latent into the probe adapter. RMA²'s Stage 2, verbatim in form.

    .. math:: L_B = \operatorname{mean}ig( (z_	ext{probe} - \operatorname{sg}(z_	ext{priv}))^2 ig)

    Only the adapter is trained; the privileged encoder and the parameter head are frozen by
    the same named-parameter sweep the official code uses (``algo/adaptation.py:47-53``), and
    the frozen names are recorded on the result so the freeze is auditable rather than assumed.

    **No force loss.** That is the point of the stage: RMA² never lets a parameter-level
    objective reach the adapter, and adding one would make this Stage C under another name.
    The consequence is that the epoch is selected on **validation latent MSE**, the objective
    actually being optimised, and the downstream force MAE is recorded alongside as a
    *diagnostic only*. Selecting on the force error would leak exactly the signal the stage is
    defined to withhold.

    One official mechanism is deliberately absent and is not a shortcut: RMA² gathers its
    history while the policy acts on the estimated latent, so its adapter is trained on its
    own induced distribution. Here the probe is a fixed open-loop excitation that does not
    depend on the latent at all, so there is no distribution to drift and the distillation is
    offline over a fixed dataset.

    Args:
        model: A :class:`~rma2.model.StageBModel` built around a trained Stage A.
        train_dataset: Training rows, scaled by the shared ``FeatureScaler``.
        val_dataset: Validation rows.
        targets: Reference forces, used only for the diagnostic.
        cfg: A :class:`~rma2.config.StageBCfg`.

    Returns:
        A :class:`StageBResult`.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(cfg.device)
    frozen = model.freeze_all_but_adapter()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("nothing is trainable after the freeze; the adapter was frozen too.")

    optimiser = torch.optim.Adam(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loader = make_loader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    result = StageBResult(model=model, cfg=cfg, frozen_parameters=frozen)
    result.history.append(
        {
            "epoch": -1,
            "train_latent_mse": float("nan"),
            "val_latent_mse": _latent_mse(model, val_dataset, cfg.device),
            "val_force_mae": force_mae(
                _per_probe_predictions(model, val_dataset, cfg.device), targets["val"]
            ),
        }
    )

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in loader:
            moved = batch.to(cfg.device)
            z_probe, z_priv = model.latents(moved)
            loss = ((z_probe - z_priv) ** 2).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))

        latent = _latent_mse(model, val_dataset, cfg.device)
        result.history.append(
            {
                "epoch": epoch,
                "train_latent_mse": float(np.mean(losses)) if losses else float("nan"),
                "val_latent_mse": latent,
                "val_force_mae": force_mae(
                    _per_probe_predictions(model, val_dataset, cfg.device), targets["val"]
                ),
            }
        )
        if np.isfinite(latent) and latent < result.best_val_latent_mse:
            result.best_val_latent_mse = latent
            result.best_epoch = epoch
            result.best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    return result
