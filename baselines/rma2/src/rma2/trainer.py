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

__all__ = ["StageAResult", "train_stage_a"]


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
