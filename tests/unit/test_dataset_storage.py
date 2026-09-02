"""The on-disk layout: it must round-trip, and it must refuse a dangling reference.

The normalisation is the point. A probe history stored once and referenced 24 times is 24x
smaller and, more importantly, cannot disagree with itself -- but it introduces references
that can dangle, so the writer refuses to create one and the reader refuses to ignore one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from probe_drawer.dataset import DatasetStore, DatasetWriter, ProbeRecord, probe_id, xi_id

from dataset_fixtures import CHANNELS, PROBE_TASK, make_history, make_xi, write_dataset

class TestRoundTrip:
    def test_a_written_dataset_reads_back(self, tmp_path) -> None:
        written = write_dataset(tmp_path / "ds")
        store = DatasetStore(tmp_path / "ds")

        assert store.manifest["counts"] == {"hidden_states": 2, "probes": 4, "candidates": 12}
        assert len(store.load_samples()) == len(written["candidates"])

    def test_the_samples_carry_their_joined_fields(self, tmp_path) -> None:
        write_dataset(tmp_path / "ds")
        sample = DatasetStore(tmp_path / "ds").load_samples()[0]

        assert set(sample.xi) == {"mass", "static_friction", "dynamic_friction", "damping"}
        assert set(sample.probe_history) == set(CHANNELS)
        assert sample.post_probe_state == {"displacement": 0.0035, "velocity": 0.0002}
        assert sample.probe_summary["duration"] == pytest.approx(0.5, abs=0.02)

    def test_one_probe_history_is_stored_once(self, tmp_path) -> None:
        """The reason storage is normalised: 3 candidates, 1 file."""
        write_dataset(tmp_path / "ds", states=1, repeats=1)
        files = list((tmp_path / "ds" / "probes").glob("*.npz"))
        assert len(files) == 1
        assert len(DatasetStore(tmp_path / "ds").load_samples()) == 3

    def test_the_samples_of_one_probe_share_its_history_object(self, tmp_path) -> None:
        """Normalised in memory too, not just on disk."""
        write_dataset(tmp_path / "ds", states=1, repeats=1)
        samples = DatasetStore(tmp_path / "ds").load_samples()
        assert samples[0].probe_history is samples[1].probe_history

    def test_histories_keep_their_true_lengths(self, tmp_path) -> None:
        """Padding on disk would bake a modelling decision into the data (D037)."""
        write_dataset(tmp_path / "ds", states=1, repeats=2, lengths=(23, 31))
        assert sorted(DatasetStore(tmp_path / "ds").sequence_lengths()) == [23, 31]

    def test_diagnostics_are_stored_but_not_loaded_by_default(self, tmp_path) -> None:
        """Rich logging, selective model input."""
        write_dataset(tmp_path / "ds", states=1, repeats=1)
        store = DatasetStore(tmp_path / "ds")
        assert "diagnostic/measured_force" not in store.probe_history(store.probes[0]["probe_id"])
        with_diagnostics = store.probe_history(store.probes[0]["probe_id"], include_diagnostics=True)
        assert "diagnostic/measured_force" in with_diagnostics

    def test_the_channel_values_survive_exactly(self, tmp_path) -> None:
        write_dataset(tmp_path / "ds", states=1, repeats=1, lengths=(23,))
        history = DatasetStore(tmp_path / "ds").load_samples()[0].probe_history
        assert np.array_equal(history["drawer_position"], np.linspace(0.0, 1.0, 23, dtype=np.float32))

    def test_splits_and_audit_round_trip(self, tmp_path) -> None:
        write_dataset(tmp_path / "ds")
        store = DatasetStore(tmp_path / "ds")
        store.write_splits({"level": "xi_id", "groups": {"a": "train"}})
        store.write_audit({"passes": True})
        assert store.read_splits()["level"] == "xi_id"
        assert json.loads((tmp_path / "ds" / "audit.json").read_text())["passes"] is True


class TestRefusals:
    def test_a_candidate_before_its_probe_is_refused(self, tmp_path) -> None:
        """A dangling reference would surface much later, as a loader crash."""
        with pytest.raises(ValueError, match="has not been written"):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_candidate({"candidate_id": "c1", "probe_id": "missing"})

    def test_a_duplicate_probe_is_refused(self, tmp_path) -> None:
        xi = make_xi()
        record = ProbeRecord(probe_id(xi, 0, PROBE_TASK), xi_id(xi), 0, {}, {}, make_history(5))
        with pytest.raises(ValueError, match="written twice"):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_probe(record)
                writer.add_probe(record)

    def test_a_duplicate_candidate_is_refused(self, tmp_path) -> None:
        xi = make_xi()
        probe = probe_id(xi, 0, PROBE_TASK)
        with pytest.raises(ValueError, match="written twice"):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_probe(ProbeRecord(probe, xi_id(xi), 0, {}, {}, make_history(5)))
                for _ in range(2):
                    writer.add_candidate({"candidate_id": "c1", "probe_id": probe})

    def test_a_ragged_history_is_refused(self, tmp_path) -> None:
        history = {"commanded_force": np.zeros(5, np.float32), "drawer_position": np.zeros(4, np.float32)}
        with pytest.raises(ValueError, match="ragged"):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_probe(ProbeRecord("p1", "x1", 0, {}, {}, history))

    def test_a_privileged_channel_in_a_history_is_refused(self, tmp_path) -> None:
        """The deployability rule applies at the write boundary, not only at read time."""
        history = {"drawer_position": np.zeros(5, np.float32), "drawer_resistance_force": np.zeros(5, np.float32)}
        with pytest.raises(ValueError, match="cannot be inputs"):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_probe(ProbeRecord("p1", "x1", 0, {}, {}, history))

    def test_writing_outside_a_context_manager_is_refused(self, tmp_path) -> None:
        writer = DatasetWriter(tmp_path / "ds", {})
        with pytest.raises(RuntimeError, match="context manager"):
            writer.add_hidden_state("x1", 0, make_xi())

    def test_a_missing_manifest_is_refused(self, tmp_path) -> None:
        """The manifest is written last, so its absence means the run did not finish."""
        (tmp_path / "ds").mkdir()
        with pytest.raises(FileNotFoundError, match="did not finish"):
            DatasetStore(tmp_path / "ds")

    def test_a_failed_run_leaves_no_manifest(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            with DatasetWriter(tmp_path / "ds", {}) as writer:
                writer.add_hidden_state("x1", 0, make_xi())
                writer.add_candidate({"candidate_id": "c1", "probe_id": "missing"})
        assert not (tmp_path / "ds" / "manifest.json").exists()
        assert (tmp_path / "ds" / "hidden_states.jsonl").exists(), "partial output stays readable"

    def test_a_missing_history_file_is_reported_clearly(self, tmp_path) -> None:
        write_dataset(tmp_path / "ds", states=1, repeats=1)
        store = DatasetStore(tmp_path / "ds")
        next((tmp_path / "ds" / "probes").glob("*.npz")).unlink()
        with pytest.raises(FileNotFoundError, match="is referenced but"):
            store.load_samples()
