"""The training loop: teacher first, then student, with the losses the physics allows.

Order matters and is not arbitrary. The privileged teacher is trained first because it
answers a prerequisite question: is the success landscape learnable *at all* from the four
hidden values? If it is not, the data or the task formulation is wrong and training a student
would only obscure that. Only once the teacher works does the student get trained, and it is
measured against the teacher as an upper bound.

**Why latent matching is off by default.** The obvious student loss is
``||z_ace - z_priv||^2``. It is wrong here, and measurably so: Phase 10 showed the probe
barely responds to damping -- ``b`` from 2 to 11 N*s/m leaves the probe duration and
breakaway force essentially unchanged -- so a teacher free to encode ``b`` in ``z_priv``
would set the student a target it cannot observe. Worse, the same phase showed required force
is driven almost entirely by dynamic friction, so encoding ``b`` would not even help. The
student's objective is therefore the task itself, with optional *logit* distillation so it
matches the teacher's landscape rather than the teacher's internal coordinates
(``docs/DECISIONS.md`` D039). ``latent_weight`` exists, defaults to 0, and any run that
raises it says so in its config.

**Class imbalance is handled by reweighting, not resampling.** About 6 % of candidates
succeed. ``pos_weight`` in the loss changes the gradient without changing which rows exist,
so the evaluation set stays the real distribution -- resampling the training set and then
evaluating on a resampled set would report a success rate that no drawer has.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from probe_drawer.models.psp import PspCfg, StudentModel, TeacherModel, build_student, build_teacher
from probe_drawer.training.dataloader import ProbeBatch, SampleDataset, make_loader
from probe_drawer.training.metrics import (
    classification_metrics,
    reference_force_per_probe,
    selection_metrics,
)

__all__ = ["TrainCfg", "TrainedModel", "train_student", "train_teacher"]


@dataclass(frozen=True)
class TrainCfg:
    """Optimisation settings.

    Args:
        epochs: Maximum passes over the training split.
        batch_size: Rows per step.
        learning_rate: Adam step size.
        weight_decay: Adam weight decay.
        patience: Epochs without validation improvement before stopping. ``0`` disables.
        pos_weight: Multiplier on the positive class in the loss. ``None`` computes it as
            ``negatives / positives`` on the training split, which equalises the two classes'
            total gradient contribution.
        distillation_weight: Weight on matching the teacher's *logits*. ``0`` trains the
            student on the task alone.
        distillation_temperature: Softening applied to both logits before matching. Above 1
            emphasises the landscape's shape over its confident extremes.
        latent_weight: Weight on ``||z_ace - z_priv||^2``. Zero by default -- see the module
            docstring.
        seed: Torch and numpy seed.
        device: ``"cpu"`` or ``"cuda"``.
        monitor: Validation key to select the best epoch on.
    """

    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 12
    pos_weight: float | None = None
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    latent_weight: float = 0.0
    seed: int = 0
    device: str = "cpu"
    monitor: str = "selected_success_rate_feasible_only"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainedModel:
    """A trained model with everything needed to reproduce and audit its numbers."""

    model: nn.Module
    cfg: TrainCfg
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_score: float = float("-inf")
    best_state: dict = field(default_factory=dict)
    label_distribution: dict = field(default_factory=dict)

    def restore_best(self) -> nn.Module:
        """Load the best epoch's weights. Called before any reported evaluation."""
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.model


def _resolve_pos_weight(dataset: SampleDataset, cfg: TrainCfg) -> tuple[torch.Tensor, dict]:
    """The positive-class multiplier, and the label distribution it came from."""
    labels = np.array([float(sample.success) for sample in dataset.samples])
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    weight = cfg.pos_weight if cfg.pos_weight is not None else (negatives / max(positives, 1.0))
    return (
        torch.tensor(weight, dtype=torch.float32),
        {
            "rows": len(labels),
            "positives": int(positives),
            "negatives": int(negatives),
            "positive_fraction": float(labels.mean()) if len(labels) else 0.0,
            "pos_weight": float(weight),
            "resampled": False,
        },
    )


def _predict(model: nn.Module, loader, device: str) -> dict:
    """Logits and the fields the metrics need, over a whole loader. No gradients, eval mode."""
    was_training = model.training
    model.eval()
    logits, labels, forces, probes, states = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            logits.append(model(moved).cpu().numpy())
            labels.append(batch.success.numpy())
            forces.append(batch.candidate_force.numpy())
            probes.extend(batch.probe_ids)
            states.extend(batch.xi_ids)
    if was_training:
        model.train()
    return {
        "logits": np.concatenate(logits) if logits else np.array([]),
        "labels": np.concatenate(labels) if labels else np.array([]),
        "forces": np.concatenate(forces) if forces else np.array([]),
        "probe_ids": probes,
        "xi_ids": states,
    }


def evaluate(model: nn.Module, dataset: SampleDataset, cfg: TrainCfg, batch_size: int = 512) -> dict:
    """Classification and selection metrics on one split.

    The *unscaled* candidate forces come from the samples rather than the batch, because the
    batch carries standardised values and a force error in standard deviations would not be
    comparable to anything.
    """
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    predictions = _predict(model, loader, cfg.device)
    if not len(predictions["labels"]):
        return {"rows": 0}

    raw_forces = np.array([sample.candidate_peak_force for sample in dataset.samples], dtype=float)
    reference = reference_force_per_probe(dataset.samples)
    metrics = classification_metrics(predictions["labels"], predictions["logits"])
    metrics.update(
        selection_metrics(
            predictions["probe_ids"], raw_forces, predictions["labels"], predictions["logits"], reference
        )
    )
    metrics.pop("selected_force", None)
    metrics.pop("curve", None)
    return metrics


def _run_epochs(
    model: nn.Module,
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    cfg: TrainCfg,
    step_fn,
    label_distribution: dict,
) -> TrainedModel:
    """Shared loop: train, evaluate, keep the best epoch, stop when it stops improving."""
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = make_loader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    trained = TrainedModel(model=model, cfg=cfg, label_distribution=label_distribution)
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        started = time.perf_counter()
        losses = []
        for batch in loader:
            optimiser.zero_grad(set_to_none=True)
            loss = step_fn(batch.to(cfg.device))
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))

        validation = evaluate(model, val_dataset, cfg)
        score = validation.get(cfg.monitor, float("nan"))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "seconds": time.perf_counter() - started,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        trained.history.append(record)

        improved = np.isfinite(score) and score > trained.best_score
        if improved:
            trained.best_score = float(score)
            trained.best_epoch = epoch
            trained.best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if cfg.patience and stale >= cfg.patience:
                break

    trained.restore_best()
    return trained


def train_teacher(
    train_dataset: SampleDataset, val_dataset: SampleDataset, cfg: TrainCfg, psp: PspCfg | None = None
) -> TrainedModel:
    """Train ``E_priv + PSP`` on binary success. The upper bound, and the Phase 11 gate."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_teacher(psp).to(cfg.device)
    weight, distribution = _resolve_pos_weight(train_dataset, cfg)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight.to(cfg.device))

    def step(batch: ProbeBatch) -> torch.Tensor:
        return criterion(model(batch), batch.success)

    return _run_epochs(model, train_dataset, val_dataset, cfg, step, distribution)


def train_student(
    train_dataset: SampleDataset,
    val_dataset: SampleDataset,
    cfg: TrainCfg,
    num_channels: int,
    teacher: TeacherModel | None = None,
    psp: PspCfg | None = None,
) -> TrainedModel:
    """Train ``ACE + PSP`` on the task, optionally distilling the teacher's landscape.

    Args:
        teacher: A trained teacher. Required if ``distillation_weight`` or ``latent_weight``
            is non-zero, and frozen throughout -- the student must chase a fixed target.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_student(num_channels, psp).to(cfg.device)
    weight, distribution = _resolve_pos_weight(train_dataset, cfg)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight.to(cfg.device))

    needs_teacher = cfg.distillation_weight > 0.0 or cfg.latent_weight > 0.0
    if needs_teacher and teacher is None:
        raise ValueError(
            "distillation_weight or latent_weight is non-zero but no teacher was given."
        )
    if teacher is not None:
        teacher = teacher.to(cfg.device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    temperature = max(cfg.distillation_temperature, 1e-6)

    def step(batch: ProbeBatch) -> torch.Tensor:
        student_logit = model(batch)
        loss = criterion(student_logit, batch.success)
        if teacher is None:
            return loss
        with torch.no_grad():
            teacher_logit = teacher(batch)
        if cfg.distillation_weight > 0.0:
            # Matching soft targets, not latents: the student copies the teacher's landscape
            # without being told to reproduce coordinates it cannot observe.
            soft = torch.sigmoid(teacher_logit / temperature)
            loss = loss + cfg.distillation_weight * nn.functional.binary_cross_entropy_with_logits(
                student_logit / temperature, soft
            )
        if cfg.latent_weight > 0.0:
            loss = loss + cfg.latent_weight * nn.functional.mse_loss(
                model.context(batch), teacher.context(batch)
            )
        return loss

    return _run_epochs(model, train_dataset, val_dataset, cfg, step, distribution)


def save_run(
    directory: Path,
    trained: TrainedModel,
    metrics: dict,
    extra: dict | None = None,
    predictions: dict | None = None,
) -> Path:
    """Write one run's artefacts: config, metrics, per-epoch history, weights, predictions."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"train": trained.cfg.as_dict(), **(extra or {})}, indent=2, default=float)
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": trained.best_epoch,
                "best_score": trained.best_score,
                "monitor": trained.cfg.monitor,
                "label_distribution": trained.label_distribution,
                **metrics,
            },
            indent=2,
            default=float,
        )
    )
    if trained.history:
        keys = sorted({key for record in trained.history for key in record})
        lines = [",".join(keys)]
        lines += [",".join(str(record.get(key, "")) for key in keys) for record in trained.history]
        (directory / "history.csv").write_text("\n".join(lines) + "\n")
    torch.save({"state_dict": trained.model.state_dict(), "cfg": trained.cfg.as_dict()}, directory / "best.pt")
    if predictions:
        np.savez_compressed(directory / "predictions.npz", **predictions)
    return directory
