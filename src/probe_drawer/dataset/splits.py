"""Grouped train/validation/test splits, and the check that they are leak-free.

The only splitting offered is grouped splitting. A random split over rows would be a bug
this module exists to make impossible: many rows share one probe episode and one hidden
state, so a random row split trains and tests on the same drawer and the same recording, and
reports memorisation as generalisation (``docs/DECISIONS.md`` D031).

Groups are assigned by hashing the group key, not by shuffling with a seed. That makes a
split *stable*: adding hidden states to the dataset later does not move the existing ones
between splits, so a model trained on an earlier version can still be evaluated honestly on
the later version's test set.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from probe_drawer.dataset.schema import SPLIT_LEVELS, TrainingSample

__all__ = ["NESTING", "DataSplit", "SplitCfg", "assert_no_leakage", "split_samples"]

#: The three group levels, finest first. Each contains the ones before it: every candidate
#: belongs to one probe, and every probe to one hidden state.
NESTING = ("candidate_id", "probe_id", "xi_id")


@dataclass(frozen=True)
class SplitCfg:
    """How to divide the dataset.

    Args:
        level: The group key. Must be one of :data:`~probe_drawer.dataset.schema.SPLIT_LEVELS`
            -- ``"xi_id"`` to test on unseen drawers, ``"probe_id"`` for the looser
            probe-level split.
        train_fraction: Share of *groups*, not rows, used for training.
        val_fraction: Share of groups used for validation. The remainder is the test set.
        salt: Changes the assignment. A different salt gives a different, still stable,
            partition -- for a repeated-splits study, not for retrying until the numbers look
            better.
    """

    level: str = "xi_id"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    salt: str = "phase10"

    def __post_init__(self) -> None:
        if self.level not in SPLIT_LEVELS:
            raise ValueError(
                f"level must be one of {SPLIT_LEVELS}, not {self.level!r}. Splitting on "
                "candidate_id (i.e. per row) leaks the probe and the hidden state."
            )
        if not 0.0 < self.train_fraction < 1.0 or not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1) and val_fraction in [0, 1)")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                f"train ({self.train_fraction}) + val ({self.val_fraction}) leaves no test set"
            )


@dataclass(frozen=True)
class DataSplit:
    """The three subsets, plus the group bookkeeping that lets a reviewer audit them."""

    train: list[TrainingSample]
    val: list[TrainingSample]
    test: list[TrainingSample]
    level: str
    groups: dict[str, str]

    def counts(self) -> dict:
        """Row and group counts per subset."""
        return {
            "level": self.level,
            "rows": {"train": len(self.train), "val": len(self.val), "test": len(self.test)},
            "groups": dict(Counter(self.groups.values())),
        }


def _position(group: str, salt: str) -> float:
    """Map a group key to a stable point in ``[0, 1)``."""
    digest = hashlib.sha1(f"{salt}|{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_samples(samples: Iterable[TrainingSample], cfg: SplitCfg | None = None) -> DataSplit:
    """Partition samples by group.

    Invalid rows are *not* dropped here. Dropping them is a training decision that should be
    visible in the training script, not buried in the splitter.
    """
    cfg = cfg or SplitCfg()
    rows = list(samples)
    subsets: dict[str, list[TrainingSample]] = {"train": [], "val": [], "test": []}
    assignment: dict[str, str] = {}

    for sample in rows:
        group = getattr(sample, cfg.level)
        if group not in assignment:
            position = _position(group, cfg.salt)
            if position < cfg.train_fraction:
                assignment[group] = "train"
            elif position < cfg.train_fraction + cfg.val_fraction:
                assignment[group] = "val"
            else:
                assignment[group] = "test"
        subsets[assignment[group]].append(sample)

    return DataSplit(
        train=subsets["train"], val=subsets["val"], test=subsets["test"], level=cfg.level, groups=assignment
    )


def assert_no_leakage(split: DataSplit) -> None:
    """Raise if a group at or below the split level appears in more than one subset.

    A ``probe_id`` split is only leak-free with respect to probes -- two probes of the same
    drawer can land on opposite sides -- so this checks exactly what the chosen level claims
    and no more, rather than certifying the whole split.
    """
    subsets = {"train": split.train, "val": split.val, "test": split.test}
    problems: list[str] = []

    # Separating a coarse group separates every finer one inside it, but not the other way
    # round: a probe-level split may still put two probes of the same drawer on both sides.
    # So only the levels at or below the one split on are verified.
    for level in NESTING[: NESTING.index(split.level) + 1]:
        seen: dict[str, str] = {}
        shared: dict[str, tuple[str, str]] = {}
        for name, rows in subsets.items():
            for value in {getattr(sample, level) for sample in rows}:
                if value in seen and seen[value] != name:
                    shared[value] = (seen[value], name)
                else:
                    seen[value] = name
        if shared:
            example = next(iter(shared.items()))
            problems.append(
                f"{len(shared)} {level} value(s) appear in two subsets, e.g. {example[0]} in "
                f"{example[1][0]} and {example[1][1]}"
            )

    if problems:
        raise AssertionError("the split leaks: " + "; ".join(problems))
