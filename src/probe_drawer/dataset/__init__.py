"""The formal training-dataset schema and its leak-free splitting.

This package is the boundary between the simulator and a model. It owns two things and no
physics:

* the shape of one training sample, and the identifiers that record which rows are not
  independent (:mod:`~probe_drawer.dataset.schema`);
* which drawers and which forces to record, and what the samplers are forbidden to see
  (:mod:`~probe_drawer.dataset.sampling`);
* the normalised on-disk layout, so a probe recording is stored once rather than per
  candidate (:mod:`~probe_drawer.dataset.storage`);
* grouped splitting, and the assertion that a split does not leak
  (:mod:`~probe_drawer.dataset.splits`).

Nothing here imports Isaac Lab, so it is usable wherever the data is, not only on a machine
with a simulator. See ``docs/DATASET_SCHEMA.md``.
"""

from probe_drawer.dataset.sampling import (
    ForceSamplerCfg,
    SamplingPlan,
    XiSamplerCfg,
    branch_order,
    build_plan,
    candidate_forces,
    sample_hidden_states,
)
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
from probe_drawer.dataset.storage import DatasetStore, DatasetWriter, ProbeRecord, group_samples_by_probe

__all__ = [
    "SPLIT_LEVELS",
    "XI_DIMENSIONS",
    "DataSplit",
    "DatasetStore",
    "DatasetWriter",
    "ForceSamplerCfg",
    "ProbeRecord",
    "SamplingPlan",
    "SplitCfg",
    "TrainingSample",
    "XiSamplerCfg",
    "assert_no_leakage",
    "branch_order",
    "build_plan",
    "candidate_forces",
    "candidate_id",
    "group_samples_by_probe",
    "model_input_fields",
    "probe_id",
    "sample_hidden_states",
    "split_samples",
    "validate_probe_history",
    "xi_id",
]
