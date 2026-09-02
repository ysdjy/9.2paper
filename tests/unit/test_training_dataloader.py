"""Batching and normalisation: the two places a silent leak or a silent bug would live.

The padding-equivalence test is the important one. A GRU fed a zero-padded batch without
using the lengths still produces *plausible* numbers -- it has simply also consumed steps
that will never exist at deployment. Nothing fails; the model is just quietly worse. So the
test asserts the batched encoder output equals the one-sequence-at-a-time output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from probe_drawer.dataset import TrainingSample, candidate_id, probe_id, xi_id
from probe_drawer.models import PspCfg, build_student, build_teacher
from probe_drawer.training import FeatureScaler, SampleDataset, collate_samples, make_loader

CHANNELS = ("commanded_force", "drawer_position", "drawer_velocity")
PROBE_TASK = {"initial_force": 1.0, "max_force": 6.0, "target_displacement": 0.003, "max_velocity": 0.08}


def make_xi(mass: float = 8.0) -> dict:
    return {"mass": mass, "static_friction": 1.25, "dynamic_friction": 0.8, "damping": 6.0}


def make_sample(steps: int, force: float = 2.0, *, mass: float = 8.0, success: bool = True, valid: bool = True):
    xi = make_xi(mass)
    probe = probe_id(xi, 0, PROBE_TASK)
    rng = np.random.default_rng(steps)
    return TrainingSample(
        candidate_id=candidate_id(probe, force, 1.5, 0.04),
        probe_id=probe,
        xi_id=xi_id(xi),
        xi=xi,
        probe_history={name: rng.normal(size=steps).astype(np.float32) for name in CHANNELS},
        probe_summary={"duration": steps / 60},
        post_probe_state={"displacement": 0.0035, "velocity": 0.0002},
        candidate_peak_force=force,
        branch_index=0,
        duration=1.5,
        goal_displacement=0.04,
        final_total_displacement=0.041,
        final_velocity=0.01,
        success=success,
        valid=valid,
    )


def make_samples(lengths=(23, 31, 42), **kwargs) -> list:
    return [make_sample(steps, force=1.0 + index, mass=4.0 + index, **kwargs)
            for index, steps in enumerate(lengths)]


class TestDynamicPadding:
    def test_a_batch_is_padded_to_its_own_longest(self) -> None:
        dataset = SampleDataset(make_samples((23, 31, 42)), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], CHANNELS)

        assert batch.history.shape == (3, 42, 3)
        assert batch.lengths.tolist() == [23, 31, 42]

    def test_different_batches_pad_to_different_lengths(self) -> None:
        """The point of padding per batch rather than per dataset."""
        dataset = SampleDataset(make_samples((16, 20, 40, 46)), channels=CHANNELS)
        short = collate_samples([dataset[0], dataset[1]], CHANNELS)
        long = collate_samples([dataset[2], dataset[3]], CHANNELS)
        assert short.history.shape[1] == 20
        assert long.history.shape[1] == 46

    def test_the_mask_marks_exactly_the_real_steps(self) -> None:
        dataset = SampleDataset(make_samples((23, 42)), channels=CHANNELS)
        batch = collate_samples([dataset[0], dataset[1]], CHANNELS)
        assert batch.mask[0, :23].all() and not batch.mask[0, 23:].any()
        assert batch.mask[1].all()
        assert batch.mask.sum(dim=1).tolist() == batch.lengths.tolist()

    def test_the_padding_is_zero_and_the_real_values_survive(self) -> None:
        samples = make_samples((23, 42))
        dataset = SampleDataset(samples, channels=CHANNELS)
        batch = collate_samples([dataset[0], dataset[1]], CHANNELS)

        assert torch.all(batch.history[0, 23:] == 0.0)
        assert torch.allclose(
            batch.history[0, :23, 1],
            torch.from_numpy(samples[0].probe_history["drawer_position"]),
        )

    def test_lengths_stay_on_the_cpu(self) -> None:
        """``pack_padded_sequence`` requires it, and a CUDA move would raise."""
        dataset = SampleDataset(make_samples(), channels=CHANNELS)
        batch = collate_samples([dataset[0]], CHANNELS).to("cpu")
        assert batch.lengths.device.type == "cpu"

    def test_the_loader_pads_each_batch(self) -> None:
        dataset = SampleDataset(make_samples((16, 20, 40, 46)), channels=CHANNELS)
        shapes = [batch.history.shape[1] for batch in make_loader(dataset, batch_size=2)]
        assert shapes == [20, 46]


class TestPaddingDoesNotChangeTheEncoder:
    def test_a_padded_batch_matches_one_sequence_at_a_time(self) -> None:
        """The equivalence the whole variable-length design rests on."""
        torch.manual_seed(0)
        student = build_student(len(CHANNELS), PspCfg(z_dim=8, gru_hidden=16, hidden=32)).eval()
        dataset = SampleDataset(make_samples((23, 31, 42)), channels=CHANNELS)
        items = [dataset[index] for index in range(3)]

        with torch.no_grad():
            batched = student.encoder(*_as_batch(items))
            alone = torch.cat([student.encoder(*_as_batch([item])) for item in items])

        assert torch.allclose(batched, alone, atol=1e-6), (batched - alone).abs().max()

    def test_the_batch_order_does_not_matter(self) -> None:
        torch.manual_seed(0)
        student = build_student(len(CHANNELS), PspCfg(z_dim=8, gru_hidden=16, hidden=32)).eval()
        dataset = SampleDataset(make_samples((23, 31, 42)), channels=CHANNELS)
        items = [dataset[index] for index in range(3)]

        with torch.no_grad():
            forward = student.encoder(*_as_batch(items))
            reverse = student.encoder(*_as_batch(items[::-1]))

        assert torch.allclose(forward, reverse.flip(0), atol=1e-6)

    def test_extra_padding_changes_nothing(self) -> None:
        """Adding a longer sequence to the batch must not perturb the shorter ones."""
        torch.manual_seed(0)
        student = build_student(len(CHANNELS), PspCfg(z_dim=8, gru_hidden=16, hidden=32)).eval()
        dataset = SampleDataset(make_samples((23, 31, 42)), channels=CHANNELS)

        with torch.no_grad():
            pair = student.encoder(*_as_batch([dataset[0], dataset[1]]))
            triple = student.encoder(*_as_batch([dataset[0], dataset[1], dataset[2]]))

        assert torch.allclose(pair, triple[:2], atol=1e-6)

    def test_a_wrong_channel_count_is_refused(self) -> None:
        student = build_student(7)
        with pytest.raises(ValueError, match="built for 7 channels"):
            student.encoder(torch.zeros(2, 5, 3), torch.tensor([5, 5]))


def _as_batch(items):
    batch = collate_samples(items, CHANNELS)
    return batch.history, batch.lengths


class TestScalerDoesNotLeak:
    def test_it_is_fitted_only_on_what_it_is_given(self) -> None:
        """Statistics from the whole dataset would leak test information invisibly: no row
        crosses the split, so no split check would catch it."""
        train = make_samples((23, 31))
        everything = train + make_samples((42, 46))
        assert FeatureScaler.fit(train, CHANNELS).mean.tolist() != FeatureScaler.fit(everything, CHANNELS).mean.tolist()
        assert FeatureScaler.fit(train, CHANNELS).fitted_on == 2

    def test_it_standardises_the_fitting_data(self) -> None:
        samples = make_samples((40, 46, 42))
        scaler = FeatureScaler.fit(samples, CHANNELS)
        pooled = np.concatenate(
            [np.stack([s.probe_history[name] for name in CHANNELS], axis=1) for s in samples]
        )
        transformed = scaler.transform_history(pooled)
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)
        assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-6)

    def test_a_constant_channel_does_not_explode(self) -> None:
        sample = make_sample(20)
        sample.probe_history["drawer_velocity"] = np.zeros(20, dtype=np.float32)
        scaler = FeatureScaler.fit([sample], CHANNELS)
        assert np.all(np.isfinite(scaler.transform_history(np.zeros((20, 3)))))

    def test_it_round_trips(self) -> None:
        scaler = FeatureScaler.fit(make_samples(), CHANNELS)
        restored = FeatureScaler.from_dict(scaler.as_dict())
        assert restored.channels == scaler.channels
        assert np.allclose(restored.mean, scaler.mean)
        assert restored.force_mean == scaler.force_mean

    def test_a_mismatched_scaler_is_refused(self) -> None:
        scaler = FeatureScaler.fit(make_samples(), CHANNELS)
        with pytest.raises(ValueError, match="fitted on"):
            SampleDataset(make_samples(), channels=("commanded_force", "drawer_position"), scaler=scaler)

    def test_fitting_on_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            FeatureScaler.fit([], CHANNELS)

    def test_a_privileged_channel_cannot_be_requested(self) -> None:
        with pytest.raises(ValueError, match="cannot be inputs"):
            SampleDataset(make_samples(), channels=("drawer_position", "drawer_resistance_force"))


class TestInvalidRows:
    def test_they_are_dropped_and_counted(self) -> None:
        samples = make_samples((23, 31, 42)) + [make_sample(25, valid=False)]
        dataset = SampleDataset(samples, channels=CHANNELS)
        assert len(dataset) == 3
        assert dataset.dropped == 1

    def test_keeping_them_is_possible_and_explicit(self) -> None:
        samples = make_samples((23, 31)) + [make_sample(25, valid=False)]
        dataset = SampleDataset(samples, channels=CHANNELS, drop_invalid=False)
        assert len(dataset) == 3
        assert dataset.dropped == 0


class TestPrivilegedIsolation:
    def test_the_student_output_does_not_depend_on_xi(self) -> None:
        """Structural, not conventional: the student has no path to ``batch.xi``."""
        torch.manual_seed(0)
        student = build_student(len(CHANNELS), PspCfg(z_dim=8, gru_hidden=16, hidden=32)).eval()
        dataset = SampleDataset(make_samples(), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], CHANNELS)

        with torch.no_grad():
            before = student(batch)
            batch.xi = torch.randn_like(batch.xi) * 100.0
            after = student(batch)

        assert torch.equal(before, after)

    def test_the_teacher_output_does_depend_on_xi(self) -> None:
        """The complementary check: otherwise the isolation test would pass on a broken
        teacher too."""
        torch.manual_seed(0)
        teacher = build_teacher(PspCfg(z_dim=8, hidden=32)).eval()
        dataset = SampleDataset(make_samples(), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], CHANNELS)

        with torch.no_grad():
            before = teacher(batch)
            batch.xi = torch.randn_like(batch.xi) * 100.0
            after = teacher(batch)

        assert not torch.allclose(before, after)

    def test_the_student_has_no_xi_shaped_parameter(self) -> None:
        student = build_student(len(CHANNELS))
        assert not any("privileged" in name.lower() for name, _ in student.named_parameters())
