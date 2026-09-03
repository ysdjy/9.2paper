"""Batching variable-length probe histories, and normalising without leaking.

Two problems live here, and both have a wrong answer that is easy to reach.

**Variable length.** A probe stops when the drawer has moved 3 mm, so its recording is as
long as that drawer needs -- 16 to 46 steps in the pilot. Padding the histories *on disk* to
a fixed length would bake a modelling decision irreversibly into the data, so the dataset
keeps them ragged (D037) and padding happens here, per batch, to that batch's own longest
sequence. Every batch carries the true lengths and a mask, and the encoder is expected to use
one of them; a model that silently averaged over padding would train on zeros it will never
see at deployment.

**Normalisation.** Channel statistics computed over the whole dataset leak test information
into training -- quietly, and in a way no split check would catch, because no *row* crosses
the boundary. :class:`FeatureScaler` is therefore fitted on the training subset alone and
then applied unchanged to validation and test.

Nothing here imports Isaac Lab. ``probe_drawer.observations`` is the channel registry and is
simulator-free by design, so the deployability rule still has one definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from probe_drawer.dataset.schema import XI_DIMENSIONS
from probe_drawer.observations import DEFAULT_ACE_INPUT, validate_model_input

__all__ = [
    "LABEL_FIELDS",
    "POST_PROBE_FIELDS",
    "TASK_CONDITION_FIELDS",
    "FeatureScaler",
    "ProbeBatch",
    "SampleDataset",
    "collate_samples",
    "make_loader",
]

#: Order of the post-probe state's two entries, wherever it appears as a vector.
POST_PROBE_FIELDS = ("displacement", "velocity")

#: Order of the task condition's two entries. ``(d_goal, T_goal)`` -- what the task asks,
#: which Setting V1 conditions on rather than adapts (``docs/DECISIONS.md`` D044).
TASK_CONDITION_FIELDS = ("goal_displacement", "duration")

#: Label names a batch can be trained against.
#:
#: ``reach_success`` is the primary metric (D046) and is what Setting V1 trains on;
#: ``success`` is the strict label Dataset v0 carries. They are separate entries rather than
#: one configurable field because a v0 dataset has no ``reach_success`` to give, and reading
#: an absent label must fail loudly rather than fall back.
LABEL_FIELDS = ("reach_success", "success")


@dataclass
class ProbeBatch:
    """One batch, padded to its own longest history.

    Attributes:
        history: ``(batch, time, channel)``, padded with zeros.
        lengths: ``(batch,)`` true lengths, on the CPU as ``pack_padded_sequence`` requires.
        mask: ``(batch, time)`` boolean, ``True`` on real steps.
        candidate_force: ``(batch,)`` the force each row asks about (N).
        post_probe: ``(batch, 2)`` displacement and velocity where the execution starts.
        task_condition: ``(batch, 2)`` ``d_goal`` (m) and ``T_goal`` (s). Deployable: the task
            tells the robot both.
        success: ``(batch,)`` float strict labels in ``{0, 1}``.
        reach_success: ``(batch,)`` float primary labels in ``{0, 1}``, or ``nan`` on a
            Dataset v0 row, which predates the split. ``nan`` rather than a substituted value
            so that training on a dataset that cannot supply this label fails loudly.
        final_displacement: ``(batch,)`` :math:`d_\text{total}(T)` (m) -- the auxiliary target.
        final_velocity: ``(batch,)`` :math:`v(T)` (m/s) -- the other auxiliary target.
        valid: ``(batch,)`` whether the episode stayed inside the operating region.
        xi: ``(batch, 4)`` the hidden state. **Privileged** -- for the teacher and for
            analysis only. It travels in the batch because the teacher needs it; keeping it
            out of the student is the student's responsibility, enforced in
            :mod:`probe_drawer.models.ace`.
        channels: Which channels ``history``'s last axis holds, in order.
        probe_ids: For grouping metrics back to probes.
        xi_ids: For grouping metrics back to hidden states.
    """

    history: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    candidate_force: torch.Tensor
    post_probe: torch.Tensor
    task_condition: torch.Tensor
    success: torch.Tensor
    reach_success: torch.Tensor
    final_displacement: torch.Tensor
    final_velocity: torch.Tensor
    valid: torch.Tensor
    xi: torch.Tensor
    channels: tuple[str, ...]
    probe_ids: tuple[str, ...] = ()
    xi_ids: tuple[str, ...] = ()

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def label(self, name: str) -> torch.Tensor:
        """The named training label, refusing to hand back one the dataset never recorded.

        A Dataset v0 row has no ``reach_success``: a v0 negative failed on position or on
        terminal velocity and the row does not say which. Substituting ``success`` there would
        train on a strictly harder label while reporting the easier one's name, so the ``nan``
        the loader stores is turned into an error here instead.

        Raises:
            ValueError: If ``name`` is not a label field, or the label is not recorded.
        """
        if name not in LABEL_FIELDS:
            raise ValueError(f"unknown label {name!r}; expected one of {LABEL_FIELDS}.")
        values = getattr(self, name)
        if torch.isnan(values).any():
            raise ValueError(
                f"this dataset does not record {name!r} (Dataset v0 predates it). Train on "
                f"'success', or regenerate with scripts/generate_dataset.py --setting v1."
            )
        return values

    def to(self, device: str | torch.device) -> ProbeBatch:
        """Move the tensors, except ``lengths``, which must stay on the CPU."""
        return ProbeBatch(
            history=self.history.to(device),
            lengths=self.lengths,
            mask=self.mask.to(device),
            candidate_force=self.candidate_force.to(device),
            post_probe=self.post_probe.to(device),
            task_condition=self.task_condition.to(device),
            success=self.success.to(device),
            reach_success=self.reach_success.to(device),
            final_displacement=self.final_displacement.to(device),
            final_velocity=self.final_velocity.to(device),
            valid=self.valid.to(device),
            xi=self.xi.to(device),
            channels=self.channels,
            probe_ids=self.probe_ids,
            xi_ids=self.xi_ids,
        )


@dataclass
class FeatureScaler:
    """Per-channel standardisation, fitted on the training subset only.

    Attributes:
        channels: Channel order the statistics correspond to.
        mean: ``(channel,)`` means over all real (unpadded) steps of the fitting samples.
        std: ``(channel,)`` standard deviations, floored away from zero.
        force_mean, force_std: Statistics for the candidate force.
        post_probe_mean, post_probe_std: Statistics for the two post-probe entries.
        fitted_on: How many samples the statistics came from, recorded so a run's config
            shows what the scaler saw.
    """

    channels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    force_mean: float
    force_std: float
    post_probe_mean: np.ndarray
    post_probe_std: np.ndarray
    fitted_on: int = 0

    #: Smallest standard deviation used as a divisor.
    #:
    #: Some channels are nearly constant across a probe -- a drawer that never breaks away
    #: has a flat velocity trace -- and dividing by their true standard deviation would
    #: amplify quantisation noise into the dominant signal.
    FLOOR = 1e-6

    @classmethod
    def fit(cls, samples: Sequence, channels: tuple[str, ...] = DEFAULT_ACE_INPUT) -> FeatureScaler:
        """Fit on ``samples`` -- which must be the training subset and nothing else."""
        if not samples:
            raise ValueError("cannot fit a scaler on no samples.")
        validate_model_input(channels)

        stacked = [
            np.stack([np.asarray(sample.probe_history[name], dtype=np.float64) for name in channels], axis=1)
            for sample in samples
        ]
        pooled = np.concatenate(stacked, axis=0)
        forces = np.array([sample.candidate_peak_force for sample in samples], dtype=np.float64)
        post = np.array(
            [[sample.post_probe_state[name] for name in POST_PROBE_FIELDS] for sample in samples],
            dtype=np.float64,
        )
        return cls(
            channels=tuple(channels),
            mean=pooled.mean(axis=0),
            std=np.maximum(pooled.std(axis=0), cls.FLOOR),
            force_mean=float(forces.mean()),
            force_std=float(max(forces.std(), cls.FLOOR)),
            post_probe_mean=post.mean(axis=0),
            post_probe_std=np.maximum(post.std(axis=0), cls.FLOOR),
            fitted_on=len(samples),
        )

    def transform_history(self, values: np.ndarray) -> np.ndarray:
        """Standardise a ``(time, channel)`` array."""
        return (values - self.mean) / self.std

    def transform_force(self, value: float) -> float:
        return (value - self.force_mean) / self.force_std

    def transform_post_probe(self, values: np.ndarray) -> np.ndarray:
        return (values - self.post_probe_mean) / self.post_probe_std

    def as_dict(self) -> dict:
        return {
            "channels": list(self.channels),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "force_mean": self.force_mean,
            "force_std": self.force_std,
            "post_probe_mean": self.post_probe_mean.tolist(),
            "post_probe_std": self.post_probe_std.tolist(),
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> FeatureScaler:
        return cls(
            channels=tuple(payload["channels"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            force_mean=payload["force_mean"],
            force_std=payload["force_std"],
            post_probe_mean=np.asarray(payload["post_probe_mean"], dtype=np.float64),
            post_probe_std=np.asarray(payload["post_probe_std"], dtype=np.float64),
            fitted_on=payload.get("fitted_on", 0),
        )


@dataclass
class SampleDataset(Dataset):
    """A list of samples, with the scaler applied on access.

    Args:
        samples: The rows.
        channels: Which history channels the model sees. Validated against the observation
            registry, so a privileged channel cannot reach a model even by a typo here.
        scaler: Fitted on the training subset. ``None`` leaves the values raw, which is only
            for tests and for inspecting the data.
        drop_invalid: Whether to drop rows that fell outside the operating region. The
            default drops them and records how many, because an invalid row is evidence about
            the rig rather than about the drawer -- but it is a *visible* decision, not one
            buried in the loader.
    """

    samples: list
    channels: tuple[str, ...] = DEFAULT_ACE_INPUT
    scaler: FeatureScaler | None = None
    drop_invalid: bool = True
    dropped: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        validate_model_input(self.channels)
        if self.drop_invalid:
            kept = [sample for sample in self.samples if sample.valid]
            self.dropped = len(self.samples) - len(kept)
            self.samples = kept
        if self.scaler is not None and tuple(self.scaler.channels) != tuple(self.channels):
            raise ValueError(
                f"scaler was fitted on {self.scaler.channels}, this dataset uses {self.channels}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        history = np.stack(
            [np.asarray(sample.probe_history[name], dtype=np.float32) for name in self.channels], axis=1
        )
        post = np.array([sample.post_probe_state[name] for name in POST_PROBE_FIELDS], dtype=np.float32)
        force = float(sample.candidate_peak_force)
        if self.scaler is not None:
            history = self.scaler.transform_history(history).astype(np.float32)
            post = self.scaler.transform_post_probe(post).astype(np.float32)
            force = self.scaler.transform_force(force)
        # The task condition is deliberately not scaled. Both entries are O(0.1-2) in SI
        # units, so they are already well conditioned, and standardising a quantity that is
        # constant within a dataset divides by a floor rather than by a spread.
        condition = np.array(
            [getattr(sample, name) for name in TASK_CONDITION_FIELDS], dtype=np.float32
        )
        return {
            "history": torch.from_numpy(np.ascontiguousarray(history)),
            "post_probe": torch.from_numpy(post),
            "task_condition": torch.from_numpy(condition),
            "candidate_force": float(force),
            "success": float(sample.success),
            "reach_success": float("nan") if sample.reach_success is None else float(sample.reach_success),
            "final_displacement": float(sample.final_total_displacement),
            "final_velocity": float(sample.final_velocity),
            "valid": float(sample.valid),
            "xi": torch.tensor([sample.xi[name] for name in XI_DIMENSIONS], dtype=torch.float32),
            "probe_id": sample.probe_id,
            "xi_id": sample.xi_id,
        }


def collate_samples(items: Sequence[dict], channels: tuple[str, ...] = DEFAULT_ACE_INPUT) -> ProbeBatch:
    """Pad a batch to its own longest history and record the true lengths.

    Padded with zeros, and the zeros are never silently meaningful: ``lengths`` feeds
    ``pack_padded_sequence`` and ``mask`` is available for anything that pools over time.
    """
    lengths = torch.tensor([item["history"].shape[0] for item in items], dtype=torch.long)
    longest = int(lengths.max())
    channel_count = items[0]["history"].shape[1]

    history = torch.zeros(len(items), longest, channel_count, dtype=torch.float32)
    mask = torch.zeros(len(items), longest, dtype=torch.bool)
    for index, item in enumerate(items):
        steps = item["history"].shape[0]
        history[index, :steps] = item["history"]
        mask[index, :steps] = True

    return ProbeBatch(
        history=history,
        lengths=lengths,
        mask=mask,
        candidate_force=torch.tensor([item["candidate_force"] for item in items], dtype=torch.float32),
        post_probe=torch.stack([item["post_probe"] for item in items]),
        task_condition=torch.stack([item["task_condition"] for item in items]),
        success=torch.tensor([item["success"] for item in items], dtype=torch.float32),
        reach_success=torch.tensor([item["reach_success"] for item in items], dtype=torch.float32),
        final_displacement=torch.tensor(
            [item["final_displacement"] for item in items], dtype=torch.float32
        ),
        final_velocity=torch.tensor([item["final_velocity"] for item in items], dtype=torch.float32),
        valid=torch.tensor([item["valid"] for item in items], dtype=torch.float32),
        xi=torch.stack([item["xi"] for item in items]),
        channels=tuple(channels),
        probe_ids=tuple(item["probe_id"] for item in items),
        xi_ids=tuple(item["xi_id"] for item in items),
    )


def make_loader(
    dataset: SampleDataset,
    batch_size: int = 256,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """A loader that pads per batch.

    ``num_workers`` defaults to 0: the dataset is small enough to sit in memory, and worker
    processes would each copy it.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=lambda items: collate_samples(items, dataset.channels),
    )
