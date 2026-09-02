"""The formal training-dataset schema and its leak-free splitting.

This package is the boundary between the simulator and a model. It owns two things and no
physics:

* the shape of one training sample, and the identifiers that record which rows are not
  independent (:mod:`~probe_drawer.dataset.schema`);
* grouped splitting, and the assertion that a split does not leak
  (:mod:`~probe_drawer.dataset.splits`).

Nothing here imports Isaac Lab, so it is usable wherever the data is, not only on a machine
with a simulator. See ``docs/DATASET_SCHEMA.md``.
"""

from probe_drawer.dataset.schema import (
    SPLIT_LEVELS,
    XI_DIMENSIONS,
    TrainingSample,
    candidate_id,
    model_input_fields,
    probe_id,
    validate_probe_history,
    xi_id,
)
from probe_drawer.dataset.splits import DataSplit, SplitCfg, assert_no_leakage, split_samples

__all__ = [
    "SPLIT_LEVELS",
    "XI_DIMENSIONS",
    "DataSplit",
    "SplitCfg",
    "TrainingSample",
    "assert_no_leakage",
    "candidate_id",
    "model_input_fields",
    "probe_id",
    "split_samples",
    "validate_probe_history",
    "xi_id",
]
