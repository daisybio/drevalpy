"""Tests for the single model + single fold execution unit (:func:`drevalpy.single`)."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from drevalpy import single
from drevalpy.data import split
from drevalpy.errors import InvalidInputError
from drevalpy.evaluation import Metric
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.registry.splitter import SplitValidationError
from drevalpy.types import Fold, PairMask
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult
from tests._fake_wandb import FakeWandb, install_fake_wandb

FIT_MODULE = importlib.import_module("drevalpy.experiment._fit")


class _StubModel(DRPModel):
    """A ``DRPModel`` that records what ``single`` hands it and predicts ``0..n-1`` (or a constant)."""

    early_stopping: bool = False
    instances: list[_StubModel] = []  # plain class-level config
    prediction_value: float | None = None

    def __init__(self, spec, hyperparameters=None, *, check_values=True, response_transformation=None) -> None:
        super().__init__(
            spec, hyperparameters, check_values=check_values, response_transformation=response_transformation
        )
        self.received_hyperparameters = hyperparameters
        self.train_calls = []
        self.predict_calls = []
        type(self).instances.append(self)

    def train(self, dataset, scope, *, early_stopping_scope=None, model_checkpoint_dir="checkpoints", n_jobs=1):
        super().train(
            dataset,
            scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=model_checkpoint_dir,
            n_jobs=n_jobs,
        )
        self.train_calls.append(
            {
                "scope": scope,
                "early_stopping_scope": early_stopping_scope,
                "model_checkpoint_dir": model_checkpoint_dir,
                "n_jobs": n_jobs,
                "response_transformation": self._fitted_response_transformation,
            }
        )

        def fake_prediction(batch):
            if self.prediction_value is None:
                return np.arange(batch.n_pairs, dtype=float)
            return np.full(batch.n_pairs, self.prediction_value, dtype=float)

        self._predictor._predict = fake_prediction

    def predict(self, dataset, scope):
        self.predict_calls.append(scope)
        return super().predict(dataset, scope)


@pytest.fixture
def stub_model(monkeypatch):
    cls = type("_FreshStubModel", (_StubModel,), {"instances": []})
    cls.recipe = ModelSpec("StubModel", ModelSpec.from_name("ElasticNet").config)
    monkeypatch.setattr(FIT_MODULE, "DRPModel", cls)
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda spec: cls.early_stopping))
    return cls


@pytest.fixture
def folds(synthetic_dataset) -> list[Fold]:
    return split(synthetic_dataset, "LSO", n_splits=2)


@pytest.fixture
def tuner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``hpam_tune``; the kwargs of its last call are collected in the returned dict."""
    seen: dict[str, Any] = {}

    def fake_tune(execution, dataset, **kwargs: Any):
        seen.update(kwargs, execution=execution, dataset=dataset)
        return {"predictor.elasticNet.alpha": 0.2}, []

    monkeypatch.setattr(FIT_MODULE, "hpam_tune", fake_tune)
    return seen


def _fold_with_val_pairs(shape: tuple[int, int], n_val: int) -> Fold:
    train = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    val = np.zeros(shape, dtype=bool)
    train[:2, :] = True
    test[2, :] = True
    val.reshape(-1)[shape[1] * 3 : shape[1] * 3 + n_val] = True
    return Fold(
        train=PairMask(train),
        test=PairMask(test),
        val=PairMask(val),
        metadata={"split_mode": "LSO", "fold_index": 0, "fold_id": "abc123"},
    )


def _train_responses(dataset, fold: Fold) -> np.ndarray:
    pairs = fold.train_val.pairs
    responses = dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    return responses[~np.isnan(responses)]


def test_end_to_end_with_a_real_model(synthetic_dataset) -> None:
    fold = split(synthetic_dataset, "LSO", n_splits=2)[0]
    result = single(ModelSpec.from_name("ElasticNet"), synthetic_dataset, fold, hyperparameter_tuning=False)
    pairs = fold.test.pairs

    assert isinstance(result, RunResult)
    assert (result.model_name, result.dataset_name) == ("ElasticNet", synthetic_dataset.name)
    assert len(result.predictions) == len(result.ground_truth) == len(pairs)
    assert np.isfinite(result.predictions).all()
    assert set(result.metrics) == set(Metric)
    assert result.best_hyperparameters == dict(ModelSpec.from_name("ElasticNet").hyperparameters)
    assert result.trials is None
    np.testing.assert_array_equal(result.sample_ids, synthetic_dataset.sample_ids[pairs[:, 0]])
    np.testing.assert_array_equal(result.drug_ids, synthetic_dataset.drug_ids[pairs[:, 1]])
    expected = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    np.testing.assert_allclose(result.ground_truth, expected, equal_nan=True)


class TestFoldMetadata:
    def test_fold_metadata_is_copied_not_aliased(self, synthetic_dataset, folds, stub_model) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[1], hyperparameter_tuning=False)

        assert result.split_mode == folds[1].metadata["split_mode"]
        assert result.fold_index == folds[1].metadata["fold_index"]
        assert result.fold_id == folds[1].metadata["fold_id"]
        assert {key: result.fold_metadata[key] for key in folds[1].metadata} == folds[1].metadata
        assert "execution" in result.fold_metadata
        result.fold_metadata["injected"] = "value"
        assert "injected" not in folds[1].metadata

    def test_absent_metadata_falls_back_to_defaults(self, synthetic_dataset, stub_model) -> None:
        shape = synthetic_dataset.response_matrix.shape
        fold = Fold(
            train=PairMask(np.eye(*shape, dtype=bool)),
            test=PairMask(np.flipud(np.eye(*shape, dtype=bool))),
            val=PairMask(np.zeros(shape, dtype=bool)),
        )

        result = single(stub_model.recipe, synthetic_dataset, fold, hyperparameter_tuning=False)

        assert (result.split_mode, result.fold_index, result.fold_id) == ("", 0, "")

    def test_the_dataset_randomization_tag_is_propagated(self, synthetic_dataset, folds, stub_model) -> None:
        randomized = synthetic_dataset.with_randomized_views(
            ["gene_expression"], random_state=0, randomization=("SVRC", "gene_expression")
        )

        result = single(stub_model.recipe, randomized, folds[0], hyperparameter_tuning=False)

        assert result.randomization == ("SVRC", "gene_expression")


class TestTraining:
    def test_trains_on_train_val_and_predicts_on_test(self, synthetic_dataset, folds, stub_model) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        model = stub_model.instances[0]
        np.testing.assert_array_equal(model.train_calls[0]["scope"].mask, folds[0].train_val.mask)
        np.testing.assert_array_equal(model.predict_calls[0].mask, folds[0].test.mask)
        assert model.received_hyperparameters == dict(stub_model.recipe.hyperparameters)
        assert model.hyperparameters == dict(stub_model.recipe.hyperparameters)
        assert model.train_calls[0]["model_checkpoint_dir"]

    def test_metrics_are_empty_when_no_prediction_is_valid(self, synthetic_dataset, folds, stub_model) -> None:
        stub_model.prediction_value = float("nan")

        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert result.metrics == {}

    @pytest.mark.parametrize(
        ("early_stopping", "n_val", "expects_scope"),
        [
            pytest.param(False, 4, False, id="unsupported"),
            pytest.param(True, 4, True, id="carved-out"),
            pytest.param(True, 1, False, id="single-val-pair"),
            pytest.param(True, 0, False, id="no-val-pairs"),
        ],
    )
    def test_the_early_stopping_scope(
        self, synthetic_dataset, stub_model, early_stopping, n_val, expects_scope
    ) -> None:
        stub_model.early_stopping = early_stopping
        fold = _fold_with_val_pairs(synthetic_dataset.response_matrix.shape, n_val)

        single(stub_model.recipe, synthetic_dataset, fold, hyperparameter_tuning=False)

        scope = stub_model.instances[0].train_calls[0]["early_stopping_scope"]
        assert (scope is not None) is expects_scope
        if expects_scope:
            expected, _ = fold.early_stopping_mask()
            np.testing.assert_array_equal(scope.mask, expected.mask)
            assert scope.mask.any()
            assert (scope.mask & ~fold.val.mask).sum() == 0


class TestResponseTransformation:
    def test_the_model_gets_a_transformer_fitted_on_the_training_scope(
        self, synthetic_dataset, folds, stub_model
    ) -> None:
        """Regression: predictions were inverse-transformed but training stayed raw."""
        scaler = StandardScaler()

        single(
            stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, response_transformation=scaler
        )

        handed = stub_model.instances[0].train_calls[0]["response_transformation"]
        assert float(handed.mean_[0]) == pytest.approx(float(_train_responses(synthetic_dataset, folds[0]).mean()))
        assert not hasattr(scaler, "mean_")

    def test_predictions_are_mapped_back_to_the_response_scale(self, synthetic_dataset, folds, stub_model) -> None:
        result = single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=False,
            response_transformation=StandardScaler(),
        )

        raw = np.arange(len(folds[0].test.pairs), dtype=float)
        reference = StandardScaler().fit(_train_responses(synthetic_dataset, folds[0]).reshape(-1, 1))
        np.testing.assert_allclose(result.predictions, reference.inverse_transform(raw.reshape(-1, 1)).ravel())

    def test_no_transformation_hands_the_model_nothing(self, synthetic_dataset, folds, stub_model) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert stub_model.instances[0].train_calls[0]["response_transformation"] is None
        np.testing.assert_array_equal(result.predictions, np.arange(len(folds[0].test.pairs), dtype=float))


class TestHyperparameterTuning:
    def test_trials_and_the_tuner_inputs(self, synthetic_dataset, folds, stub_model, monkeypatch) -> None:
        seen: dict[str, Any] = {}
        trials = [
            TrialResult({"predictor.elasticNet.alpha": 0.1}, {"RMSE": 1.0}, "RMSE", np.array([1.0, 2.0])),
            TrialResult({"predictor.elasticNet.alpha": 0.9}, {"RMSE": 2.0}, "RMSE", np.array([3.0, 4.0])),
        ]

        def fake_tune(execution, dataset, **kwargs: Any):
            seen.update(kwargs, execution=execution, dataset=dataset)
            return {"predictor.elasticNet.alpha": 0.1}, trials

        monkeypatch.setattr(FIT_MODULE, "hpam_tune", fake_tune)

        result = single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=True,
            hpo_metric="RMSE",
            hpo_num_trials=3,
            hpo_random_state=11,
        )

        assert result.trials == trials
        assert result.best_hyperparameters == {"predictor.elasticNet.alpha": 0.1}
        execution = seen["execution"]
        assert execution.model is stub_model.recipe
        assert seen["dataset"] is synthetic_dataset
        assert execution.scopes.stopping is None
        np.testing.assert_array_equal(execution.scopes.training.mask, folds[0].train.mask)
        np.testing.assert_array_equal(execution.scopes.validation.mask, folds[0].val.mask)
        assert (execution.hyperparameter_tuning, execution.hpo_num_trials, execution.hpo_metric) == (True, 3, "RMSE")
        assert execution.hpo_random_state == 11

    def test_early_stopping_narrows_the_validation_scope_handed_to_the_tuner(
        self, synthetic_dataset, folds, stub_model, tuner
    ) -> None:
        stub_model.early_stopping = True

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=True)

        es_expected, val_expected = folds[0].early_stopping_mask()
        np.testing.assert_array_equal(tuner["execution"].scopes.stopping.mask, es_expected.mask)
        np.testing.assert_array_equal(tuner["execution"].scopes.validation.mask, val_expected.mask)


@pytest.mark.parametrize("tuning", [False, True])
def test_final_fit_excludes_stopping_holdout_and_transforms_training_only(
    stub_model, synthetic_dataset, monkeypatch, tuning
):
    stub_model.early_stopping = True
    fold = _fold_with_val_pairs(synthetic_dataset.response_matrix.shape, 8)
    es, val = fold.early_stopping_mask()
    seen = []

    def tune(execution, dataset, **kwargs):
        seen.append({**kwargs, "execution": execution})
        return dict(stub_model.recipe.hyperparameters), []

    monkeypatch.setattr(FIT_MODULE, "hpam_tune", tune)
    single(
        stub_model.recipe,
        synthetic_dataset,
        fold,
        hyperparameter_tuning=tuning,
        response_transformation=StandardScaler(),
    )
    call = stub_model.instances[-1].train_calls[0]
    assert not (call["scope"].mask & es.mask).any()
    assert call["scope"] == fold.train | val
    assert call["early_stopping_scope"] == es
    pairs = call["scope"].pairs
    expected = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    np.testing.assert_allclose(call["response_transformation"].mean_, [np.nanmean(expected)])
    if tuning:
        assert seen[0]["execution"].scopes.stopping == es
        assert seen[0]["execution"].scopes.validation == val
        assert seen[0]["execution"].scopes.training == fold.train


@pytest.mark.parametrize("defect", ["shape", "pair_overlap", "entity_overlap"])
def test_direct_folds_are_validated_before_training(stub_model, synthetic_dataset, defect):
    fold = _fold_with_val_pairs(synthetic_dataset.response_matrix.shape, 4)
    if defect == "shape":
        fold = replace(fold, test=PairMask(np.zeros((1, 1), dtype=bool)))
    elif defect == "pair_overlap":
        fold = replace(fold, test=PairMask(fold.train.mask.copy()))
    else:
        train, test = fold.train.mask.copy(), fold.test.mask.copy()
        train[0, 0], test[0, 0] = False, True
        fold = replace(fold, train=PairMask(train), test=PairMask(test))

    with pytest.raises(SplitValidationError):
        single(stub_model.recipe, synthetic_dataset, fold, hyperparameter_tuning=False)
    assert stub_model.instances == []


class TestNJobs:
    def test_it_reaches_the_final_fit_and_the_tuner_but_not_the_result(
        self, synthetic_dataset, folds, stub_model, tuner
    ) -> None:
        one = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)
        two = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, n_jobs=3)
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=True, n_jobs=2)

        assert [call.train_calls[0]["n_jobs"] for call in stub_model.instances] == [1, 3, 2]
        assert tuner["n_jobs"] == 2
        assert one.fold_metadata == two.fold_metadata

    @pytest.mark.parametrize("n_jobs", [0, -2])
    def test_an_unusable_count_is_rejected_before_any_fit(self, synthetic_dataset, folds, stub_model, n_jobs) -> None:
        with pytest.raises(InvalidInputError, match="n_jobs"):
            single(stub_model.recipe, synthetic_dataset, folds[0], n_jobs=n_jobs)

        assert stub_model.instances == []


class TestSeeding:
    @pytest.fixture
    def seeded(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        calls: list[int] = []
        monkeypatch.setattr(FIT_MODULE, "seed_everything", calls.append)
        return calls

    def test_the_final_fit_is_seeded_from_the_identity_and_recorded(
        self, synthetic_dataset, folds, stub_model, seeded
    ) -> None:
        first = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=3)
        single(stub_model.recipe, synthetic_dataset, folds[1], hyperparameter_tuning=False, seed=3)
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=3)

        # Each run seeds before tuning and again before the final fit, with the same seed.
        assert len(seeded) == 6
        assert seeded[0::2] == seeded[1::2]
        assert seeded[0] == seeded[4]
        assert seeded[0] != seeded[2]
        assert first.fold_metadata["execution"]["seed"] == 3

    def test_hpo_and_the_final_fit_are_both_seeded_and_the_sampler_defaults_to_the_seed(
        self, synthetic_dataset, folds, stub_model, seeded, monkeypatch
    ) -> None:
        sampler_states: list[int] = []

        def fake_tune(execution, dataset, **kwargs: Any):
            sampler_states.append(execution.hpo_random_state)
            return {"predictor.elasticNet.alpha": 0.2}, []

        monkeypatch.setattr(FIT_MODULE, "hpam_tune", fake_tune)

        single(stub_model.recipe, synthetic_dataset, folds[0], seed=8)
        single(stub_model.recipe, synthetic_dataset, folds[0], seed=8, hpo_random_state=2)

        assert sampler_states == [8, 2]
        assert len(seeded) == 4
        assert seeded[0] == seeded[1]


@pytest.mark.slow
def test_neural_predictions_follow_the_seed(synthetic_dataset, folds) -> None:
    spec = ModelSpec.from_name("SimpleNeuralNetwork").with_hyperparameters(
        {"units_per_layer": [4, 4], "max_epochs": 2, "batch_size": 8}, check_values=False
    )

    def predict(seed: int) -> np.ndarray:
        torch.manual_seed(np.random.randint(0, 2**31))  # dirty the global RNGs so only ``seed`` can explain equality
        return single(spec, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=seed).predictions

    first = predict(1)

    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, predict(1))
    assert not np.array_equal(first, predict(2))


@pytest.mark.slow
class TestWandbTracking:
    @pytest.fixture
    def fake_wandb(self, monkeypatch: pytest.MonkeyPatch) -> FakeWandb:
        return install_fake_wandb(monkeypatch)

    def test_trials_and_the_final_fit_each_get_a_run(self, synthetic_dataset, fake_wandb: FakeWandb) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[1]

        result = single(
            ModelSpec.from_name("ElasticNet"), synthetic_dataset, fold, hpo_num_trials=2, wandb_project="proj"
        )

        assert fake_wandb.run_names() == [
            "ElasticNet_split1_trial0",
            "ElasticNet_split1_trial1",
            "ElasticNet_split1_final",
        ]
        final = fake_wandb.runs[-1]
        assert final.project == "proj"
        assert final.tags == ["ElasticNet", "LSO", synthetic_dataset.name, "final"]
        assert final.config["model_name"] == "ElasticNet"
        assert final.config["dataset"] == synthetic_dataset.name
        assert final.config["split_mode"] == "LSO"
        assert final.config["fold_index"] == 1
        assert final.config["hyperparameter_tuning"] is True
        assert final.config.items() >= result.best_hyperparameters.items()
        assert final.summary == {f"test_{key}": value for key, value in result.metrics.items()}
        assert all(run.finished and run.exit_code is None for run in fake_wandb.runs)
        assert fake_wandb.runs[0].config["fold_index"] == 1

    @pytest.mark.parametrize("project", ["p", None])
    def test_without_tuning_only_the_final_run_exists_and_without_a_project_nothing_is_tracked(
        self, synthetic_dataset, fake_wandb: FakeWandb, project
    ) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]

        single(
            ModelSpec.from_name("ElasticNet"),
            synthetic_dataset,
            fold,
            hyperparameter_tuning=False,
            wandb_project=project,
        )

        if project is None:
            assert fake_wandb.runs == []
        else:
            assert fake_wandb.run_names() == ["ElasticNet_split0_final"]
            assert fake_wandb.runs[0].config["hyperparameter_tuning"] is False

    def test_a_failing_final_fit_marks_its_run_failed(
        self, synthetic_dataset, fake_wandb: FakeWandb, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]

        def explode(self, *args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(DRPModel, "train", explode)

        with pytest.raises(RuntimeError, match="boom"):
            single(
                ModelSpec.from_name("ElasticNet"),
                synthetic_dataset,
                fold,
                hyperparameter_tuning=False,
                wandb_project="p",
            )

        assert fake_wandb.runs[0].exit_code == 1
