"""Training: batching, metrics, and the loop.

Nothing here starts a simulator. The dataset is read from disk and the models are pure
torch, so training runs on a machine with no Isaac Sim installed.
"""

from probe_drawer.training.dataloader import (
    FeatureScaler,
    ProbeBatch,
    SampleDataset,
    collate_samples,
    make_loader,
)
from probe_drawer.training.metrics import (
    average_precision,
    calibration_error,
    classification_metrics,
    empirical_success_probability,
    reference_force_per_probe,
    roc_auc,
    selection_metrics,
    success_forces_per_probe,
)
from probe_drawer.training.trainer import (
    TrainCfg,
    TrainedModel,
    evaluate,
    save_run,
    train_student,
    train_teacher,
)

__all__ = [
    "FeatureScaler",
    "ProbeBatch",
    "SampleDataset",
    "TrainCfg",
    "TrainedModel",
    "average_precision",
    "calibration_error",
    "classification_metrics",
    "collate_samples",
    "empirical_success_probability",
    "evaluate",
    "make_loader",
    "reference_force_per_probe",
    "roc_auc",
    "save_run",
    "selection_metrics",
    "success_forces_per_probe",
    "train_student",
    "train_teacher",
]
