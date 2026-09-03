"""On-disk layout: normalised, so a probe history is stored once and not 24 times.

One probe answers 24 candidate forces. Writing the probe's recording into each of those 24
rows would multiply the largest part of the dataset by 24 for no information, so storage is
normalised along the same three levels the identifiers already nest in::

    dataset_v0/
    |-- manifest.json          what this dataset is, and what produced it
    |-- hidden_states.jsonl    one line per xi: identifier, index, the four values
    |-- probes.jsonl           one line per probe: summary, post-probe state, history file
    |-- probes/<probe_id>.npz  the variable-length recording, one array per channel
    |-- candidates.jsonl       one line per candidate: its force, its labels
    |-- splits.json            group assignments
    `-- audit.json             the integrity report

Histories are kept at the **raw control rate and their true length**. Padding or resampling
them to a fixed length would be a modelling decision baked irreversibly into the data; it
belongs in the DataLoader, where it can be changed (D037).

The reader materialises :class:`TrainingSample` objects, so callers never need to know the
layout is normalised. The 24 samples of one probe *share* one history dict rather than
copying it, so the in-memory representation is normalised too.

JSONL rather than one big JSON: generation streams, and a run that dies partway leaves a
readable file rather than a truncated one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from probe_drawer.dataset.schema import TrainingSample, validate_probe_history

__all__ = ["DatasetStore", "DatasetWriter", "ProbeRecord", "group_samples_by_probe"]

#: Files that make up a dataset. Named once so the writer, the reader and the audit agree.
MANIFEST = "manifest.json"
HIDDEN_STATES = "hidden_states.jsonl"
PROBES = "probes.jsonl"
CANDIDATES = "candidates.jsonl"
SPLITS = "splits.json"
AUDIT = "audit.json"
HISTORY_DIR = "probes"


@dataclass
class ProbeRecord:
    """One probe episode: its identity, what it measured, and where it left the drawer.

    Attributes:
        probe_id: Identifier, from :func:`~probe_drawer.dataset.schema.probe_id`.
        xi_id: The hidden state this probe ran against.
        repeat_index: Which of the hidden state's independent repeats this is.
        summary: Scalar probe features.
        post_probe_state: ``{displacement, velocity}`` at the moment the execution starts.
        history: ``{channel: 1-D array}``, all channels the same length. Only ``DEPLOYABLE``
            channels may appear; the writer enforces it.
        diagnostics: Extra channels recorded but excluded from the model's input. Stored in
            the same file, under a ``diagnostic/`` prefix, so a later ablation can reach them
            without regenerating anything.
    """

    probe_id: str
    xi_id: str
    repeat_index: int
    summary: dict
    post_probe_state: dict
    history: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        if not self.history:
            return 0
        return len(next(iter(self.history.values())))

    def metadata(self) -> dict:
        """Everything except the arrays, for ``probes.jsonl``."""
        return {
            "probe_id": self.probe_id,
            "xi_id": self.xi_id,
            "repeat_index": self.repeat_index,
            "summary": self.summary,
            "post_probe_state": self.post_probe_state,
            "num_steps": self.num_steps,
            "channels": sorted(self.history),
            "diagnostic_channels": sorted(self.diagnostics),
        }


class DatasetWriter:
    """Streams a dataset to disk.

    Used as a context manager. Writes each record as it arrives so that a generation run
    which dies partway leaves everything up to that point readable, and closes by writing the
    manifest -- whose presence is therefore the signal that a dataset is complete.
    """

    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = Path(root)
        self.manifest = dict(manifest)
        self._history_dir = self.root / HISTORY_DIR
        self._counts = {"hidden_states": 0, "probes": 0, "candidates": 0}
        self._probe_ids: set[str] = set()
        self._candidate_ids: set[str] = set()
        self._handles: dict[str, object] = {}

    def __enter__(self) -> DatasetWriter:
        self._history_dir.mkdir(parents=True, exist_ok=True)
        for name in (HIDDEN_STATES, PROBES, CANDIDATES):
            self._handles[name] = (self.root / name).open("w")
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        # Only stamp the manifest on a clean exit: its absence is how a reader tells a
        # half-written dataset from a finished one.
        if exc_info[0] is None:
            self.manifest["counts"] = dict(self._counts)
            (self.root / MANIFEST).write_text(json.dumps(self.manifest, indent=2, default=float))

    def _write(self, name: str, payload: dict) -> None:
        handle = self._handles.get(name)
        if handle is None:
            raise RuntimeError("DatasetWriter must be used as a context manager.")
        handle.write(json.dumps(payload, default=float) + "\n")
        handle.flush()

    def add_hidden_state(self, state_id: str, index: int, xi: dict, oracle_feasible: bool | None = None) -> None:
        """Record one hidden state.

        Args:
            oracle_feasible: Whether a dense Oracle has established that *some* force
                succeeds. ``None`` means unknown, which is the honest value for a Sobol draw
                with no dense sweep behind it -- an all-negative row set is not proof of
                infeasibility, only of the candidates that were tried (D038).
        """
        self._write(
            HIDDEN_STATES,
            {"xi_id": state_id, "index": index, "xi": xi, "oracle_feasible": oracle_feasible},
        )
        self._counts["hidden_states"] += 1

    def add_probe(self, record: ProbeRecord) -> None:
        """Record one probe: its metadata to JSONL, its arrays to a compressed NPZ."""
        if record.probe_id in self._probe_ids:
            raise ValueError(f"probe {record.probe_id} written twice.")
        validate_probe_history(record.history)
        lengths = {name: len(values) for name, values in record.history.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"probe {record.probe_id} has ragged channels: {lengths}")

        payload = {name: np.asarray(values, dtype=np.float32) for name, values in record.history.items()}
        payload.update(
            {f"diagnostic/{name}": np.asarray(values, dtype=np.float32) for name, values in record.diagnostics.items()}
        )
        np.savez_compressed(self._history_dir / f"{record.probe_id}.npz", **payload)

        self._write(PROBES, record.metadata())
        self._probe_ids.add(record.probe_id)
        self._counts["probes"] += 1

    def add_candidate(self, row: dict) -> None:
        """Record one candidate row.

        Raises:
            ValueError: If the row is a duplicate, or points at a probe that has not been
                written. A dangling reference would surface much later as a loader crash.
        """
        candidate = row["candidate_id"]
        if candidate in self._candidate_ids:
            raise ValueError(f"candidate {candidate} written twice.")
        if row["probe_id"] not in self._probe_ids:
            raise ValueError(
                f"candidate {candidate} references probe {row['probe_id']}, which has not been "
                "written. Write the probe first so the reference cannot dangle."
            )
        self._write(CANDIDATES, row)
        self._candidate_ids.add(candidate)
        self._counts["candidates"] += 1


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


class DatasetStore:
    """Reads a normalised dataset and materialises :class:`TrainingSample` objects."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / MANIFEST).exists():
            raise FileNotFoundError(
                f"{self.root} has no {MANIFEST}. Either the path is wrong or a generation run "
                "did not finish -- the manifest is written last, on purpose."
            )
        self.manifest = json.loads((self.root / MANIFEST).read_text())
        self._histories: dict[str, dict] = {}

    @property
    def hidden_states(self) -> list[dict]:
        return list(_read_jsonl(self.root / HIDDEN_STATES))

    @property
    def probes(self) -> list[dict]:
        return list(_read_jsonl(self.root / PROBES))

    @property
    def candidates(self) -> list[dict]:
        return list(_read_jsonl(self.root / CANDIDATES))

    def probe_history(self, probe: str, include_diagnostics: bool = False) -> dict:
        """The recorded channels for one probe, cached.

        Cached because 24 candidates share one probe: reading the file per candidate would
        do 24x the I/O and hold 24 copies.
        """
        key = f"{probe}:{include_diagnostics}"
        if key not in self._histories:
            path = self.root / HISTORY_DIR / f"{probe}.npz"
            if not path.exists():
                raise FileNotFoundError(f"probe {probe} is referenced but {path} is missing.")
            with np.load(path) as payload:
                self._histories[key] = {
                    name: payload[name].copy()
                    for name in payload.files
                    if include_diagnostics or not name.startswith("diagnostic/")
                }
        return self._histories[key]

    def load_samples(self, include_diagnostics: bool = False) -> list[TrainingSample]:
        """Join the three levels into flat samples.

        The samples of one probe share its history dict rather than copying it, so the
        in-memory footprint stays normalised as well.
        """
        by_id = {probe["probe_id"]: probe for probe in self.probes}
        xi_by_id = {state["xi_id"]: state["xi"] for state in self.hidden_states}

        samples = []
        for row in self.candidates:
            probe = by_id.get(row["probe_id"])
            if probe is None:
                raise KeyError(f"candidate {row['candidate_id']} references unknown probe {row['probe_id']}.")
            samples.append(
                TrainingSample(
                    candidate_id=row["candidate_id"],
                    probe_id=row["probe_id"],
                    xi_id=row["xi_id"],
                    xi=xi_by_id[row["xi_id"]],
                    probe_history=self.probe_history(row["probe_id"], include_diagnostics),
                    probe_summary=probe["summary"],
                    post_probe_state=probe["post_probe_state"],
                    candidate_peak_force=row["candidate_peak_force"],
                    branch_index=row["branch_index"],
                    duration=row["duration"],
                    goal_displacement=row["goal_displacement"],
                    final_total_displacement=row["final_total_displacement"],
                    final_velocity=row["final_velocity"],
                    success=row["success"],
                    valid=row["valid"],
                    # ``get`` rather than ``[]``: Dataset v0 predates these three, and reading
                    # an old dataset must keep working rather than start raising (D046).
                    reach_success=row.get("reach_success"),
                    stable_success=row.get("stable_success"),
                    termination_reason=row.get("termination_reason"),
                    invalid_reasons=row.get("invalid_reasons", []),
                )
            )
        return samples

    def write_splits(self, payload: dict) -> Path:
        path = self.root / SPLITS
        path.write_text(json.dumps(payload, indent=2, default=float))
        return path

    def write_audit(self, payload: dict) -> Path:
        path = self.root / AUDIT
        path.write_text(json.dumps(payload, indent=2, default=float))
        return path

    def read_splits(self) -> dict:
        return json.loads((self.root / SPLITS).read_text())

    def sequence_lengths(self) -> list[int]:
        """Probe history lengths, from the metadata -- no arrays loaded."""
        return [probe["num_steps"] for probe in self.probes]

    def describe(self) -> dict:
        counts = self.manifest.get("counts", {})
        return {
            "root": str(self.root),
            "dataset_version": self.manifest.get("dataset_version"),
            "counts": counts,
            "history_files": len(list((self.root / HISTORY_DIR).glob("*.npz"))),
        }


def group_samples_by_probe(samples: Iterable[TrainingSample]) -> dict[str, list[TrainingSample]]:
    """``probe_id -> its candidate samples``, in file order."""
    grouped: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.probe_id, []).append(sample)
    return grouped
