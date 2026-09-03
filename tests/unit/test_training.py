"""Training: it must learn, it must reload, and the student must not see xi.

The overfitting test is the one that catches a broken pipeline. A loss that decreases proves
the optimiser runs; a model that drives a tiny batch to near-zero loss proves the gradients
actually reach the parameters that matter and the labels line up with the inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from probe_drawer.dataset import TrainingSample, candidate_id, probe_id, xi_id
from probe_drawer.models import PspCfg
from probe_drawer.models.baselines import FeatureRegression, FixedForceBaseline
from probe_drawer.models.psp import build_student
from probe_drawer.training import (
    SampleDataset,
    TrainCfg,
    classification_metrics,
    empirical_success_probability,
    evaluate,
    reference_force_per_probe,
    roc_auc,
    selection_metrics,
    train_student,
    train_teacher,
)
from probe_drawer.training.dataloader import collate_samples

CHANNELS = ("commanded_force", "drawer_position", "drawer_velocity")
PROBE_TASK = {"initial_force": 1.0, "max_force": 6.0, "target_displacement": 0.003, "max_velocity": 0.08}
TINY = PspCfg(z_dim=8, hidden=32, gru_hidden=16, dropout=0.0)


def make_sample(mass: float, force: float, repeat: int, steps: int, success: bool):
    """A sample whose history encodes ``mass``, so a model can in principle learn from it."""
    xi = {"mass": mass, "static_friction": 0.5 + mass / 10, "dynamic_friction": 0.4, "damping": 6.0}
    probe = probe_id(xi, repeat, PROBE_TASK)
    ramp = np.linspace(0.0, mass / 12.0, steps, dtype=np.float32)
    return TrainingSample(
        candidate_id=candidate_id(probe, force, 1.5, 0.04),
        probe_id=probe,
        xi_id=xi_id(xi),
        xi=xi,
        probe_history={
            "commanded_force": ramp * 2.0,
            "drawer_position": ramp * 0.003,
            "drawer_velocity": ramp * 0.01,
        },
        probe_summary={"displacement_per_newton": 0.008 - mass * 0.0004, "duration": steps / 60},
        post_probe_state={"displacement": 0.0035, "velocity": 0.0002},
        candidate_peak_force=force,
        branch_index=repeat,
        duration=1.5,
        goal_displacement=0.04,
        final_total_displacement=0.04 if success else 0.02,
        final_velocity=0.01,
        success=success,
        # A Setting V1 sample. Reach and stable coincide here because the synthetic rule has
        # only one failure mode; the tests that need them to differ say so explicitly.
        reach_success=success,
        stable_success=success,
        termination_reason="duration_completed",
        valid=True,
    )


def as_dataset_v0(samples: list) -> list:
    """Strip the labels Dataset v0 never recorded, to exercise the refusal path (D046)."""
    return [replace(sample, reach_success=None, stable_success=None) for sample in samples]


def make_learnable_set(num_states: int = 24, forces=(0.5, 1.5, 2.5, 3.5)) -> list:
    """A synthetic task with a real rule: the required force rises with mass.

    A heavier drawer needs more force, so ``success`` is true for the candidate nearest
    ``0.4 + mass / 4``. Learnable, and not trivially so.
    """
    samples = []
    for index in range(num_states):
        mass = 4.0 + 8.0 * index / max(num_states - 1, 1)
        needed = 0.4 + mass / 4.0
        nearest = min(forces, key=lambda force: abs(force - needed))
        for repeat in range(2):
            for force in forces:
                samples.append(
                    make_sample(mass, force, repeat, 20 + index % 7, success=force == nearest)
                )
    return samples


@pytest.fixture
def datasets():
    samples = make_learnable_set()
    split = int(len(samples) * 0.75)
    return (
        SampleDataset(samples[:split], channels=CHANNELS),
        SampleDataset(samples[split:], channels=CHANNELS),
    )


class TestTheTeacherLearns:
    def test_it_overfits_a_tiny_set(self) -> None:
        """The pipeline check: gradients reach the parameters and labels match the inputs."""
        samples = make_learnable_set(num_states=4)
        dataset = SampleDataset(samples, channels=CHANNELS)
        cfg = TrainCfg(epochs=200, batch_size=len(samples), learning_rate=0.02, patience=0, seed=0)
        trained = train_teacher(dataset, dataset, cfg, TINY)

        assert trained.history[-1]["train_loss"] < 0.25 * trained.history[0]["train_loss"]
        assert evaluate(trained.restore_best(), dataset, cfg)["auroc"] > 0.95

    def test_the_loss_decreases(self, datasets) -> None:
        train, val = datasets
        cfg = TrainCfg(epochs=25, batch_size=64, patience=0, seed=0)
        history = train_teacher(train, val, cfg, TINY).history
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_it_records_the_label_distribution_and_does_not_resample(self, datasets) -> None:
        """Reweighting, not resampling: the evaluation set must stay the real distribution."""
        train, val = datasets
        trained = train_teacher(train, val, TrainCfg(epochs=2, patience=0), TINY)
        distribution = trained.label_distribution
        assert distribution["resampled"] is False
        assert distribution["rows"] == len(train)
        assert distribution["pos_weight"] > 1.0

    def test_the_best_epoch_is_restored_not_the_last(self, datasets) -> None:
        train, val = datasets
        trained = train_teacher(train, val, TrainCfg(epochs=12, patience=0, seed=0), TINY)
        assert 0 <= trained.best_epoch < len(trained.history)
        assert trained.best_score >= max(
            record.get("val_" + trained.cfg.monitor, float("-inf")) for record in trained.history
        ) - 1e-9


class TestTheStudentLearns:
    def test_it_overfits_a_tiny_set(self) -> None:
        samples = make_learnable_set(num_states=4)
        dataset = SampleDataset(samples, channels=CHANNELS)
        cfg = TrainCfg(epochs=200, batch_size=len(samples), learning_rate=0.02, patience=0, seed=0)
        trained = train_student(dataset, dataset, cfg, len(CHANNELS), psp=TINY)
        assert trained.history[-1]["train_loss"] < 0.35 * trained.history[0]["train_loss"]

    def test_distillation_requires_a_teacher(self, datasets) -> None:
        train, val = datasets
        cfg = TrainCfg(epochs=1, distillation_weight=1.0)
        with pytest.raises(ValueError, match="no teacher"):
            train_student(train, val, cfg, len(CHANNELS), psp=TINY)

    def test_latent_matching_requires_a_teacher(self, datasets) -> None:
        train, val = datasets
        with pytest.raises(ValueError, match="no teacher"):
            train_student(train, val, TrainCfg(epochs=1, latent_weight=1.0), len(CHANNELS), psp=TINY)

    def test_latent_matching_is_off_by_default(self) -> None:
        """The probe cannot observe damping, so a latent target would be unreachable (D039)."""
        assert TrainCfg().latent_weight == 0.0

    def test_distillation_runs_with_a_frozen_teacher(self, datasets) -> None:
        train, val = datasets
        teacher = train_teacher(train, val, TrainCfg(epochs=4, patience=0, seed=0), TINY)
        before = [parameter.detach().clone() for parameter in teacher.model.parameters()]

        cfg = TrainCfg(epochs=4, patience=0, seed=0, distillation_weight=0.5)
        train_student(train, val, cfg, len(CHANNELS), teacher=teacher.model, psp=TINY)

        for parameter, original in zip(teacher.model.parameters(), before, strict=True):
            assert torch.equal(parameter, original), "the teacher must not be updated"


class TestCheckpoints:
    def test_reloading_reproduces_the_predictions(self, datasets, tmp_path) -> None:
        train, val = datasets
        cfg = TrainCfg(epochs=6, patience=0, seed=0)
        trained = train_teacher(train, val, cfg, TINY)
        model = trained.restore_best().eval()

        from probe_drawer.models import build_teacher  # noqa: PLC0415

        path = tmp_path / "best.pt"
        torch.save({"state_dict": model.state_dict()}, path)
        reloaded = build_teacher(TINY)
        reloaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        reloaded.eval()

        first = evaluate(model, val, cfg)
        second = evaluate(reloaded, val, cfg)
        assert first["auroc"] == pytest.approx(second["auroc"])
        assert first["bce"] == pytest.approx(second["bce"])

    def test_a_seed_makes_a_run_reproducible(self, datasets) -> None:
        train, val = datasets
        cfg = TrainCfg(epochs=5, patience=0, seed=7)
        first = train_teacher(train, val, cfg, TINY)
        second = train_teacher(train, val, cfg, TINY)
        assert first.history[-1]["train_loss"] == pytest.approx(second.history[-1]["train_loss"])

    def test_different_seeds_give_different_runs(self, datasets) -> None:
        train, val = datasets
        first = train_teacher(train, val, TrainCfg(epochs=5, patience=0, seed=1), TINY)
        second = train_teacher(train, val, TrainCfg(epochs=5, patience=0, seed=2), TINY)
        assert first.history[-1]["train_loss"] != second.history[-1]["train_loss"]


class TestMetrics:
    def test_auroc_is_one_for_a_perfect_ranking(self) -> None:
        assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_auroc_is_half_for_a_constant_score(self) -> None:
        assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.5] * 4)) == pytest.approx(0.5)

    def test_auroc_is_undefined_with_one_class(self) -> None:
        """Not 1.0 and not 0.5 -- undefined, so it cannot be quoted as a result."""
        assert np.isnan(roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))

    def test_classification_metrics_are_finite(self) -> None:
        labels = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        metrics = classification_metrics(labels, np.array([-2.0, 1.0, -1.0, 2.0, -3.0]))
        assert all(np.isfinite(metrics[key]) for key in ("auroc", "auprc", "bce", "brier", "ece"))

    def test_selection_uses_the_top_scoring_candidate(self) -> None:
        result = selection_metrics(
            probe_ids=["p", "p", "p"],
            forces=[1.0, 2.0, 3.0],
            labels=[0.0, 1.0, 0.0],
            scores=[0.1, 0.9, 0.2],
        )
        assert result["selected_success_rate"] == pytest.approx(1.0)
        assert result["probes"] == 1

    def test_infeasible_probes_are_separated_out(self) -> None:
        """A probe with no succeeding candidate cannot be chosen correctly; counting it
        measures the dataset's coverage, not the model."""
        result = selection_metrics(
            probe_ids=["a", "a", "b", "b"],
            forces=[1.0, 2.0, 1.0, 2.0],
            labels=[0.0, 1.0, 0.0, 0.0],
            scores=[0.1, 0.9, 0.9, 0.1],
        )
        assert result["selected_success_rate"] == pytest.approx(0.5)
        assert result["selected_success_rate_feasible_only"] == pytest.approx(1.0)
        assert result["feasible_probes"] == 1

    def test_the_reference_force_is_defined_for_every_probe(self) -> None:
        samples = make_learnable_set(num_states=3)
        reference = reference_force_per_probe(samples)
        assert len(reference) == len({sample.probe_id for sample in samples})

    def test_the_empirical_success_probability_is_computed_not_stored(self) -> None:
        """The dataset keeps binary labels; probabilities are an analysis product (D036)."""
        samples = make_learnable_set(num_states=2)
        probabilities = empirical_success_probability(samples)
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())
        assert all(sample.success in (True, False) for sample in samples)


class TestBaselines:
    def test_a_single_feature_regression_recovers_a_linear_rule(self) -> None:
        samples = make_learnable_set(num_states=24)
        targets = [0.4 + sample.xi["mass"] / 4.0 for sample in samples]
        model = FeatureRegression(features=("displacement_per_newton",)).fit(samples, targets)
        predictions = model.predict(samples)
        assert float(np.mean(np.abs(predictions - targets))) < 0.05

    def test_ridge_shrinks_the_coefficients(self) -> None:
        samples = make_learnable_set(num_states=24)
        targets = [0.4 + sample.xi["mass"] / 4.0 for sample in samples]
        features = ("displacement_per_newton", "duration")
        plain = FeatureRegression(features=features, alpha=0.0).fit(samples, targets)
        ridged = FeatureRegression(features=features, alpha=50.0).fit(samples, targets)
        assert np.abs(ridged.coefficients).sum() < np.abs(plain.coefficients).sum()

    def test_predicting_before_fitting_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\) before predict"):
            FeatureRegression(features=("duration",)).predict(make_learnable_set(num_states=2))

    def test_the_fixed_force_baseline_picks_the_best_single_force(self) -> None:
        samples = make_learnable_set(num_states=24)
        forces = [sample.candidate_peak_force for sample in samples]
        labels = [float(sample.success) for sample in samples]
        baseline = FixedForceBaseline().fit(forces, labels)
        assert min(forces) <= baseline.force <= max(forces)
        assert baseline.predict(3).tolist() == [baseline.force] * 3


class TestTheTaskConditionedHead:
    """Setting V1's PSP: conditioned on the task, and predicting what the execution will do."""

    def test_the_condition_vector_is_the_documented_five(self) -> None:
        from probe_drawer.models.psp import CONDITION_DIM, CONDITION_FIELDS  # noqa: PLC0415

        assert CONDITION_FIELDS == (
            "candidate_peak_force",
            "post_probe_displacement",
            "post_probe_velocity",
            "goal_displacement",
            "duration",
        )
        assert CONDITION_DIM == 5

    def test_the_task_condition_reaches_the_batch_as_d_goal_then_t_goal(self) -> None:
        batch = collate_samples(
            [SampleDataset(make_learnable_set(4), channels=CHANNELS)[index] for index in range(3)],
            channels=CHANNELS,
        )
        assert batch.task_condition.shape == (3, 2)
        assert batch.task_condition[0].tolist() == pytest.approx([0.04, 1.5])

    def test_the_head_returns_a_logit_and_two_auxiliary_predictions(self) -> None:
        dataset = SampleDataset(make_learnable_set(6), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(5)], channels=CHANNELS)
        prediction = build_student(len(CHANNELS), TINY).predict(batch)
        assert prediction.logit.shape == (5,)
        assert prediction.displacement.shape == (5,)
        assert prediction.velocity.shape == (5,)

    def test_forward_is_exactly_the_logit_of_predict(self) -> None:
        """``model(batch)`` keeps meaning "the task output", so losses and distillation are
        unchanged by the auxiliary head's existence."""
        dataset = SampleDataset(make_learnable_set(6), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(5)], channels=CHANNELS)
        model = build_student(len(CHANNELS), TINY).eval()
        with torch.no_grad():
            assert torch.equal(model(batch), model.predict(batch).logit)

    def test_the_task_condition_changes_the_prediction(self) -> None:
        """Otherwise the extra inputs are decoration and the contract is not real."""
        dataset = SampleDataset(make_learnable_set(6), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(5)], channels=CHANNELS)
        model = build_student(len(CHANNELS), TINY).eval()
        with torch.no_grad():
            before = model(batch)
            batch.task_condition = batch.task_condition * torch.tensor([2.5, 1.3])
            after = model(batch)
        assert not torch.allclose(before, after)

    def test_the_student_still_cannot_see_xi(self) -> None:
        """The auxiliary head must not have opened a path to the privileged channel."""
        dataset = SampleDataset(make_learnable_set(6), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(5)], channels=CHANNELS)
        model = build_student(len(CHANNELS), TINY).eval()
        with torch.no_grad():
            before = model.predict(batch)
            batch.xi = torch.randn_like(batch.xi) * 100.0
            after = model.predict(batch)
        assert torch.equal(before.logit, after.logit)
        assert torch.equal(before.displacement, after.displacement)


class TestLabelSelection:
    """A dataset that cannot supply the requested label must say so, not substitute."""

    def test_training_on_reach_success_fails_loudly_on_a_dataset_v0_row(self) -> None:
        dataset = SampleDataset(as_dataset_v0(make_learnable_set(6)), channels=CHANNELS)
        with pytest.raises(ValueError, match="does not record 'reach_success'"):
            train_teacher(dataset, dataset, TrainCfg(epochs=1, device="cpu"), TINY)

    def test_the_same_dataset_trains_fine_on_the_strict_label(self) -> None:
        dataset = SampleDataset(as_dataset_v0(make_learnable_set(6)), channels=CHANNELS)
        trained = train_teacher(dataset, dataset, TrainCfg(epochs=1, device="cpu", label="success"), TINY)
        assert trained.label_distribution["label"] == "success"

    def test_the_batch_refuses_an_unrecorded_label(self) -> None:
        dataset = SampleDataset(as_dataset_v0(make_learnable_set(4)), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], channels=CHANNELS)
        with pytest.raises(ValueError, match="does not record 'reach_success'"):
            batch.label("reach_success")
        assert batch.label("success").shape == (3,)

    def test_an_unknown_label_name_is_rejected(self) -> None:
        dataset = SampleDataset(make_learnable_set(4), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], channels=CHANNELS)
        with pytest.raises(ValueError, match="unknown label"):
            batch.label("stable_success")

    def test_reach_and_stable_can_disagree_within_one_batch(self) -> None:
        """The case the split exists for: arrived, but still moving."""
        samples = [
            replace(sample, reach_success=True, stable_success=False, success=False)
            for sample in make_learnable_set(4)
        ]
        dataset = SampleDataset(samples, channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], channels=CHANNELS)
        assert batch.label("reach_success").tolist() == [1.0, 1.0, 1.0]
        assert batch.label("success").tolist() == [0.0, 0.0, 0.0]


class TestTheAuxiliaryLoss:
    def test_it_is_zero_when_the_predictions_are_exact(self) -> None:
        from probe_drawer.training.trainer import _auxiliary_loss  # noqa: PLC0415
        from probe_drawer.models.psp import PspPrediction  # noqa: PLC0415

        dataset = SampleDataset(make_learnable_set(4), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], channels=CHANNELS)
        exact = PspPrediction(
            logit=torch.zeros(3),
            displacement=batch.final_displacement.clone(),
            velocity=batch.final_velocity.clone(),
        )
        assert float(_auxiliary_loss(exact, batch)) == pytest.approx(0.0, abs=1e-12)

    def test_a_displacement_error_of_one_goal_costs_one_half(self) -> None:
        """Normalised by ``d_goal``, so the loss means the same thing at another distance."""
        from probe_drawer.training.trainer import _auxiliary_loss  # noqa: PLC0415
        from probe_drawer.models.psp import PspPrediction  # noqa: PLC0415

        dataset = SampleDataset(make_learnable_set(4), channels=CHANNELS)
        batch = collate_samples([dataset[index] for index in range(3)], channels=CHANNELS)
        offset = PspPrediction(
            logit=torch.zeros(3),
            displacement=batch.final_displacement + batch.task_condition[:, 0],
            velocity=batch.final_velocity.clone(),
        )
        assert float(_auxiliary_loss(offset, batch)) == pytest.approx(0.5, abs=1e-6)

    def test_disabling_it_leaves_the_classifier_trainable(self) -> None:
        dataset = SampleDataset(make_learnable_set(8), channels=CHANNELS)
        trained = train_teacher(
            dataset, dataset, TrainCfg(epochs=2, device="cpu", auxiliary_weight=0.0), TINY
        )
        assert trained.history and np.isfinite(trained.history[-1]["train_loss"])
