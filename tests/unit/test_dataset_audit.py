"""The audit must fail on a broken dataset, not only pass on a good one.

Every gate gets a test that breaks exactly what it guards. An audit that only ever passes is
indistinguishable from an audit that does nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from probe_drawer.dataset import DatasetStore, SplitCfg, split_samples
from probe_drawer.dataset.audit import audit_dataset

from dataset_fixtures import write_dataset


@pytest.fixture
def dataset(tmp_path):
    """A structurally sound small dataset."""
    root = tmp_path / "ds"
    write_dataset(root, states=8, repeats=3, forces=(0.5, 1.5, 2.5, 3.5))
    return root


def run_audit(root, level: str = "xi_id") -> dict:
    store = DatasetStore(root)
    split = split_samples(store.load_samples(), SplitCfg(level=level))
    return audit_dataset(store, split)


def rewrite(root, name: str, rows: list[dict]) -> None:
    (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))


class TestAPassingDataset:
    def test_a_sound_dataset_passes_every_gate(self, dataset) -> None:
        report = run_audit(dataset)
        assert report["all_gates_passed"], report["failed_gates"]

    def test_it_reports_the_distributions(self, dataset) -> None:
        distributions = run_audit(dataset)["distributions"]
        assert distributions["rows"] == 96
        assert distributions["probes"] == 24
        assert distributions["hidden_states"] == 8
        assert distributions["sequence_length"]["distinct"] >= 2

    def test_it_reports_positives_per_probe(self, dataset) -> None:
        """The statistic that decides whether the candidate budget is enough."""
        distributions = run_audit(dataset)["distributions"]
        assert distributions["probes_with_at_least_one"] + distributions["probes_with_no_positive"] == 24

    def test_it_notes_an_empty_subset_without_failing(self, dataset) -> None:
        """With few groups a hashed split can legitimately leave one empty."""
        report = run_audit(dataset)
        split = report["checks"]["split_has_no_leakage"]
        assert "empty_subsets" in split
        assert split["passes"]


class TestGatesCatchBreakage:
    def test_a_duplicate_candidate_id_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        rows = store.candidates
        rows[1]["candidate_id"] = rows[0]["candidate_id"]
        rewrite(dataset, "candidates.jsonl", rows)
        report = run_audit(dataset)
        assert "identifiers_unique" in report["failed_gates"]

    def test_a_dangling_probe_reference_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        rows = store.candidates
        rows[0]["probe_id"] = "does-not-exist"
        rewrite(dataset, "candidates.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "probe_references_resolve" in report["failed_gates"]

    def test_a_missing_history_file_fails(self, dataset) -> None:
        next((dataset / "probes").glob("*.npz")).unlink()
        report = audit_dataset(DatasetStore(dataset), None)
        assert "probe_references_resolve" in report["failed_gates"]

    def test_a_candidate_disagreeing_with_its_probe_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        rows = store.candidates
        rows[0]["xi_id"] = "some-other-state"
        rewrite(dataset, "candidates.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "probes_belong_to_one_hidden_state" in report["failed_gates"]

    def test_repeats_with_different_force_grids_fail(self, dataset) -> None:
        """Three repeats must answer the same questions, or (xi, F) is not repeated."""
        store = DatasetStore(dataset)
        rows = store.candidates
        rows[0]["candidate_peak_force"] = 99.0
        rewrite(dataset, "candidates.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "repeats_share_a_candidate_grid" in report["failed_gates"]

    def test_a_nan_label_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        rows = store.candidates
        rows[0]["final_total_displacement"] = float("nan")
        rewrite(dataset, "candidates.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "finite_values" in report["failed_gates"]

    def test_a_nan_in_a_history_fails(self, dataset) -> None:
        path = next((dataset / "probes").glob("*.npz"))
        with np.load(path) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        arrays["drawer_position"][2] = np.nan
        np.savez_compressed(path, **arrays)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "finite_values" in report["failed_gates"]

    def test_an_incomplete_hidden_state_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        rows = store.hidden_states
        del rows[0]["xi"]["damping"]
        rewrite(dataset, "hidden_states.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "no_privileged_fields_in_model_input" in report["failed_gates"]

    def test_a_count_mismatch_fails(self, dataset) -> None:
        manifest = json.loads((dataset / "manifest.json").read_text())
        manifest["counts"]["candidates"] += 1
        (dataset / "manifest.json").write_text(json.dumps(manifest))
        report = audit_dataset(DatasetStore(dataset), None)
        assert "counts_match_the_manifest" in report["failed_gates"]

    def test_force_ordered_branches_fail(self, dataset) -> None:
        """The gate that guards the shuffle: branch position must not track force."""
        store = DatasetStore(dataset)
        rows = store.candidates
        by_probe: dict[str, list[dict]] = {}
        for row in rows:
            by_probe.setdefault(row["probe_id"], []).append(row)
        for group in by_probe.values():
            for position, row in enumerate(sorted(group, key=lambda item: item["candidate_peak_force"])):
                row["branch_index"] = position
        rewrite(dataset, "candidates.jsonl", rows)
        report = audit_dataset(DatasetStore(dataset), None)
        assert "branch_index_decorrelated_from_force" in report["failed_gates"]
        assert report["checks"]["branch_index_decorrelated_from_force"]["mean_correlation"] > 0.9

    def test_a_leaking_split_fails(self, dataset) -> None:
        store = DatasetStore(dataset)
        samples = store.load_samples()
        split = split_samples(samples, SplitCfg(level="xi_id"))
        leaked = type(split)(
            train=split.train + split.test[:1],
            val=split.val,
            test=split.test,
            level=split.level,
            groups=split.groups,
        )
        report = audit_dataset(store, leaked)
        assert "split_has_no_leakage" in report["failed_gates"]


class TestCorrelationGateScales:
    def test_the_gate_loosens_on_a_small_dataset(self, tmp_path) -> None:
        """A shuffle cannot be shown to be biased from 12 probes of 4 candidates."""
        root = tmp_path / "small"
        write_dataset(root, states=4, repeats=3, forces=(0.5, 1.5, 2.5, 3.5))
        check = audit_dataset(DatasetStore(root), None)["checks"][
            "branch_index_decorrelated_from_force"
        ]
        assert check["tolerance"] > 0.2

    def test_the_gate_tightens_as_the_dataset_grows(self, tmp_path) -> None:
        small = tmp_path / "small"
        large = tmp_path / "large"
        write_dataset(small, states=4, repeats=3, forces=(0.5, 1.5, 2.5, 3.5))
        write_dataset(large, states=32, repeats=3, forces=(0.5, 1.5, 2.5, 3.5))
        tolerances = [
            audit_dataset(DatasetStore(path), None)["checks"][
                "branch_index_decorrelated_from_force"
            ]["tolerance"]
            for path in (small, large)
        ]
        assert tolerances[1] < tolerances[0]
