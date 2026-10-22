"""Tests for the single model + single fold execution unit.

:func:`drevalpy.single` is the seam every entry point funnels through, but
it was effectively untested on CI: ``tests/test_integration.py`` only exercises it
when a real ``CTRPv1`` ``.h5mu`` happens to be cached, so a clean checkout skipped
it entirely. This module replaces that path with the in-memory
``synthetic_dataset`` fixture.

Two levels are used deliberately:

* :class:`TestEndToEnd` trains a real ``ElasticNet`` on real split masks, so the
  happy path is honest rather than mocked.
* The remaining classes drive a stub model. ``single``'s own contract with a
  model is only ``get_model_name`` / ``supports_early_stopping`` /
  ``get_default_hyperparameters`` / ``train`` / ``predict``, and a stub is the
  only way to pin down exactly which ``early_stopping_scope`` it was handed or to
  assert the response-transformation arithmetic against known predictions.
"""

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
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.registry.splitter import SplitValidationError
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results.run import RunResult

#: ``single`` lives in the private ``drevalpy._single`` module so that the public
#: ``drevalpy.single`` name can be the re-exported function.
SINGLE_MODULE = importlib.import_module("drevalpy._single")


class _StubModel(DRPModel):
    """Minimal stand-in for a ``DRPModel`` subclass.

    Class attributes configure the branches ``single`` selects on; instance
    attributes record what ``train`` was handed. Instances are collected on the
    class so a test can reach the one ``single`` created internally.
    """

    early_stopping: bool = False
    default_hyperparameters: dict[str, Any] = {"alpha": 0.5}  # plain class-level config
    instances: list[_StubModel] = []  # plain class-level config
    prediction_value: float | None = None

    def __init__(self, spec, hyperparameters, *, response_transformation=None) -> None:
        super().__init__(spec, hyperparameters, response_transformation=response_transformation)
        self.received_hyperparameters = hyperparameters
        self.train_calls = []
        self.predict_calls = []
        type(self).instances.append(self)

    @classmethod
    def get_model_name(cls) -> str:
        return "StubModel"

    @classmethod
    def supports_early_stopping(cls) -> bool:
        return cls.early_stopping

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        return dict(cls.default_hyperparameters)

    def train(self, dataset, scope, *, early_stopping_scope=None, model_checkpoint_dir="checkpoints"):
        super().train(
            dataset, scope, early_stopping_scope=early_stopping_scope, model_checkpoint_dir=model_checkpoint_dir
        )
        self.train_calls.append(
            {
                "dataset": dataset,
                "scope": scope,
                "early_stopping_scope": early_stopping_scope,
                "model_checkpoint_dir": model_checkpoint_dir,
                "response_transformation": self._pipeline._response_transform,
            }
        )

        def fake_prediction(batch):
            if self.prediction_value is None:
                return np.arange(batch.n_pairs, dtype=float)
            return np.full(batch.n_pairs, self.prediction_value, dtype=float)

        self._pipeline._predictor._predict = fake_prediction

    def predict(self, dataset, scope):
        self.predict_calls.append(scope)
        return super().predict(dataset, scope)


@pytest.fixture
def stub_model(monkeypatch):
    """Record orchestration while exercising the actual fitted model pipeline."""
    cls = type("_FreshStubModel", (_StubModel,), {"instances": []})
    cls.recipe = ModelSpec("StubModel", ModelSpec.from_name("ElasticNet").config)
    monkeypatch.setattr(SINGLE_MODULE, "DRPModel", cls)
    monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda spec: cls.early_stopping))
    monkeypatch.setattr(ModelSpec, "default_hyperparameters", property(lambda spec: dict(cls.default_hyperparameters)))
    return cls


@pytest.fixture
def folds(synthetic_dataset) -> list[SplitMasks]:
    """Two real LSO folds over the synthetic dataset."""
    return split(synthetic_dataset, "LSO", n_splits=2)


def _masks_with_val_pairs(shape: tuple[int, int], n_val: int) -> SplitMasks:
    """Build masks whose ``val`` holds exactly ``n_val`` pairs."""
    train = np.zeros(shape, dtype=bool)
    test = np.zeros(shape, dtype=bool)
    val = np.zeros(shape, dtype=bool)
    train[:2, :] = True
    test[2, :] = True
    flat_val = val.reshape(-1)
    flat_val[shape[1] * 3 : shape[1] * 3 + n_val] = True
    return SplitMasks(
        train=SplitMask(train),
        test=SplitMask(test),
        val=SplitMask(val),
        metadata={"split_mode": "LSO", "fold_index": 0, "fold_id": "abc123"},
    )


class TestEndToEnd:
    """A real model on real folds, so the happy path is not mocked."""

    @pytest.fixture(scope="class")
    def result(self, synthetic_dataset) -> RunResult:
        elastic_net = ModelSpec.from_name("ElasticNet")
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]
        return single(elastic_net, synthetic_dataset, fold, hyperparameter_tuning=False)

    def test_returns_a_run_result(self, result: RunResult) -> None:
        assert isinstance(result, RunResult)

    def test_identifies_the_model_and_dataset(self, result: RunResult, synthetic_dataset) -> None:
        assert result.model_name == "ElasticNet"
        assert result.dataset_name == synthetic_dataset.name

    def test_predicts_one_value_per_test_pair(self, result: RunResult, synthetic_dataset) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]

        assert len(result.predictions) == len(fold.test.pairs)
        assert len(result.ground_truth) == len(result.predictions)

    def test_predictions_are_finite(self, result: RunResult) -> None:
        assert np.isfinite(result.predictions).all()

    def test_reports_every_available_metric(self, result: RunResult) -> None:
        assert set(result.metrics) == set(AVAILABLE_METRICS)

    def test_records_the_default_hyperparameters(self, result: RunResult) -> None:
        assert result.best_hyperparameters == ModelSpec.from_name("ElasticNet").default_hyperparameters

    def test_skipping_hpo_records_no_trials(self, result: RunResult) -> None:
        assert result.trials is None

    def test_ids_line_up_with_the_test_pairs(self, result: RunResult, synthetic_dataset) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]
        pairs = fold.test.pairs

        np.testing.assert_array_equal(result.sample_ids, synthetic_dataset.sample_ids[pairs[:, 0]])
        np.testing.assert_array_equal(result.drug_ids, synthetic_dataset.drug_ids[pairs[:, 1]])

    def test_ground_truth_is_read_from_the_response_matrix(self, result: RunResult, synthetic_dataset) -> None:
        fold = split(synthetic_dataset, "LSO", n_splits=2)[0]
        pairs = fold.test.pairs
        expected = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]

        np.testing.assert_allclose(result.ground_truth, expected, equal_nan=True)


class TestFoldMetadata:
    def test_split_metadata_is_copied_onto_the_result(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[1], hyperparameter_tuning=False)

        assert result.split_mode == folds[1].metadata["split_mode"]
        assert result.fold_index == folds[1].metadata["fold_index"]
        assert result.fold_id == folds[1].metadata["fold_id"]
        assert {key: result.fold_metadata[key] for key in folds[1].metadata} == folds[1].metadata
        assert "execution" in result.fold_metadata

    def test_fold_metadata_does_not_alias_the_split_masks(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[1], hyperparameter_tuning=False)

        result.fold_metadata["injected"] = "value"

        assert "injected" not in folds[1].metadata

    def test_absent_metadata_falls_back_to_defaults(self, synthetic_dataset, stub_model: type[_StubModel]) -> None:
        shape = synthetic_dataset.response_matrix.shape
        masks = SplitMasks(
            train=SplitMask(np.eye(*shape, dtype=bool)),
            test=SplitMask(np.flipud(np.eye(*shape, dtype=bool))),
            val=SplitMask(np.zeros(shape, dtype=bool)),
        )

        result = single(stub_model.recipe, synthetic_dataset, masks, hyperparameter_tuning=False)

        assert result.split_mode == ""
        assert result.fold_index == 0
        assert result.fold_id == ""

    def test_the_dataset_randomization_tag_is_propagated(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        randomized = synthetic_dataset.with_randomized_views(
            ["gene_expression"], random_state=0, randomization=("SVRC", "gene_expression")
        )

        result = single(stub_model.recipe, randomized, folds[0], hyperparameter_tuning=False)

        assert result.randomization == ("SVRC", "gene_expression")


class TestTraining:
    def test_trains_on_the_merged_train_and_val_mask(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        scope = stub_model.instances[0].train_calls[0]["scope"]
        np.testing.assert_array_equal(scope.mask, folds[0].train_val.mask)

    def test_predicts_on_the_test_mask(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        np.testing.assert_array_equal(stub_model.instances[0].predict_calls[0].mask, folds[0].test.mask)

    def test_the_model_is_built_from_the_default_hyperparameters(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert stub_model.instances[0].received_hyperparameters == {"alpha": 0.5}

    def test_the_checkpoint_directory_exists_during_training(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert stub_model.instances[0].train_calls[0]["model_checkpoint_dir"]

    def test_metrics_are_empty_when_no_prediction_is_valid(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        stub_model.prediction_value = float("nan")

        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert result.metrics == {}


class TestEarlyStopping:
    def test_a_model_without_support_gets_no_early_stopping_scope(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        stub_model.early_stopping = False

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert stub_model.instances[0].train_calls[0]["early_stopping_scope"] is None

    def test_a_supporting_model_gets_a_carved_out_scope(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        stub_model.early_stopping = True

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        scope = stub_model.instances[0].train_calls[0]["early_stopping_scope"]
        expected, _ = folds[0].early_stopping_mask()
        assert scope is not None
        np.testing.assert_array_equal(scope.mask, expected.mask)

    def test_the_early_stopping_scope_is_a_subset_of_val(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        stub_model.early_stopping = True

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        scope = stub_model.instances[0].train_calls[0]["early_stopping_scope"]
        assert scope is not None
        assert scope.any()
        assert (scope.mask & ~folds[0].val.mask).sum() == 0

    @pytest.mark.parametrize(
        ("n_val", "expects_scope"),
        [
            pytest.param(1, False, id="a-single-val-pair-is-not-enough"),
            pytest.param(0, False, id="no-val-pairs-at-all"),
            pytest.param(4, True, id="several-val-pairs-carve-a-scope"),
        ],
    )
    def test_the_val_mask_must_hold_more_than_one_pair(
        self,
        synthetic_dataset,
        stub_model: type[_StubModel],
        n_val: int,
        expects_scope: bool,
    ) -> None:
        stub_model.early_stopping = True
        masks = _masks_with_val_pairs(synthetic_dataset.response_matrix.shape, n_val)

        single(stub_model.recipe, synthetic_dataset, masks, hyperparameter_tuning=False)

        scope = stub_model.instances[0].train_calls[0]["early_stopping_scope"]
        assert (scope is not None) is expects_scope


class TestResponseTransformation:
    def _train_responses(self, synthetic_dataset, masks: SplitMasks) -> np.ndarray:
        pairs = masks.train_val.pairs
        responses = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
        return responses[~np.isnan(responses)]

    def test_the_caller_s_transformer_is_left_unfitted(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        scaler = StandardScaler()

        single(
            stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, response_transformation=scaler
        )

        assert not hasattr(scaler, "mean_")

    def test_the_model_is_trained_on_the_transformed_target_space(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        """The original bug: predictions were inverse-transformed but training stayed raw.

        ``single`` therefore has to hand the model a *fitted* transformer, whose
        statistics come from the train_val scope it also inverts with.
        """
        single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=False,
            response_transformation=StandardScaler(),
        )

        handed = stub_model.instances[0].train_calls[0]["response_transformation"]
        expected_mean = float(self._train_responses(synthetic_dataset, folds[0]).mean())
        assert float(handed.mean_[0]) == pytest.approx(expected_mean)

    def test_no_transformation_hands_the_model_nothing(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        assert stub_model.instances[0].train_calls[0]["response_transformation"] is None

    def test_predictions_are_mapped_back_to_the_response_scale(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        result = single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=False,
            response_transformation=StandardScaler(),
        )

        raw = np.arange(len(folds[0].test.pairs), dtype=float)
        reference = StandardScaler().fit(self._train_responses(synthetic_dataset, folds[0]).reshape(-1, 1))
        expected = reference.inverse_transform(raw.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(result.predictions, expected)

    def test_the_transform_is_fitted_on_non_nan_training_responses_only(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        stub_model.prediction_value = 0.0

        result = single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=False,
            response_transformation=StandardScaler(),
        )

        # A zero prediction inverse-transforms to the fitted mean.
        expected_mean = float(self._train_responses(synthetic_dataset, folds[0]).mean())
        np.testing.assert_allclose(result.predictions, expected_mean, rtol=1e-6)

    def test_no_transformation_leaves_predictions_untouched(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False)

        np.testing.assert_array_equal(result.predictions, np.arange(len(folds[0].test.pairs), dtype=float))


class TestHyperparameterTuning:
    def test_tuning_records_one_trial_per_drawn_configuration(
        self,
        synthetic_dataset,
        folds: list[SplitMasks],
        stub_model: type[_StubModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        raw_trials = [
            ({"alpha": 0.1}, {"RMSE": 1.0}, np.array([1.0, 2.0])),
            ({"alpha": 0.9}, {"RMSE": 2.0}, np.array([3.0, 4.0])),
        ]
        monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", lambda **kwargs: ({"alpha": 0.1}, raw_trials))

        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=True, hpo_metric="RMSE")

        assert result.trials is not None
        assert [t.hyperparameters for t in result.trials] == [{"alpha": 0.1}, {"alpha": 0.9}]
        assert [t.optimization_metric for t in result.trials] == ["RMSE", "RMSE"]
        assert result.best_hyperparameters == {"alpha": 0.1}

    def test_the_tuner_receives_the_folds_scopes_and_hpo_settings(
        self,
        synthetic_dataset,
        folds: list[SplitMasks],
        stub_model: type[_StubModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_tune(**kwargs: Any):
            seen.update(kwargs)
            return {"alpha": 0.2}, []

        monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", fake_tune)

        single(
            stub_model.recipe,
            synthetic_dataset,
            folds[0],
            hyperparameter_tuning=True,
            hpo_metric="MSE",
            hpo_num_trials=3,
            hpo_random_state=11,
        )

        assert seen["model_spec"] is stub_model.recipe
        assert seen["mudataset"] is synthetic_dataset
        assert seen["metric"] == "MSE"
        assert seen["early_stopping_scope"] is None
        np.testing.assert_array_equal(seen["train_scope"].mask, folds[0].train.mask)
        np.testing.assert_array_equal(seen["val_scope"].mask, folds[0].val.mask)
        assert seen["hpo_config"].n_trials == 3
        assert seen["hpo_config"].random_state == 11

    def test_early_stopping_narrows_the_validation_scope_handed_to_the_tuner(
        self,
        synthetic_dataset,
        folds: list[SplitMasks],
        stub_model: type[_StubModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_model.early_stopping = True
        seen: dict[str, Any] = {}

        def fake_tune(**kwargs: Any):
            seen.update(kwargs)
            return {"alpha": 0.2}, []

        monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", fake_tune)

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=True)

        es_expected, val_expected = folds[0].early_stopping_mask()
        np.testing.assert_array_equal(seen["early_stopping_scope"].mask, es_expected.mask)
        np.testing.assert_array_equal(seen["val_scope"].mask, val_expected.mask)


@pytest.mark.parametrize("tuning", [False, True])
def test_final_fit_excludes_stopping_holdout_and_transforms_training_only(
    stub_model,
    synthetic_dataset,
    monkeypatch,
    tuning,
):
    stub_model.early_stopping = True
    masks = _masks_with_val_pairs(synthetic_dataset.response_matrix.shape, 8)
    es, val = masks.early_stopping_mask()
    seen = []

    def tune(**kwargs):
        seen.append(kwargs)
        return stub_model.get_default_hyperparameters(), []

    monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", tune)
    single(
        stub_model.recipe,
        synthetic_dataset,
        masks,
        hyperparameter_tuning=tuning,
        response_transformation=StandardScaler(),
    )
    call = stub_model.instances[-1].train_calls[0]
    assert not (call["scope"] & es).any()
    assert call["scope"] == masks.train | val
    assert call["early_stopping_scope"] == es
    pairs = call["scope"].pairs
    expected = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    np.testing.assert_allclose(call["response_transformation"].mean_, [np.nanmean(expected)])
    if tuning:
        assert seen[0]["early_stopping_scope"] == es
        assert seen[0]["val_scope"] == val
        assert seen[0]["train_scope"] == masks.train


@pytest.mark.parametrize("defect", ["shape", "pair_overlap", "entity_overlap"])
def test_direct_masks_are_validated_before_training(stub_model, synthetic_dataset, defect):
    shape = synthetic_dataset.response_matrix.shape
    masks = _masks_with_val_pairs(shape, 4)
    if defect == "shape":
        masks = replace(masks, test=SplitMask(np.zeros((1, 1), dtype=bool)))
    elif defect == "pair_overlap":
        masks = replace(masks, test=SplitMask(masks.train.mask.copy()))
    else:
        # Distinct responses still leak the same sample in LSO mode.
        masks.train.mask[0, 0] = False
        masks.test.mask[0, 0] = True

    with pytest.raises(SplitValidationError):
        single(stub_model.recipe, synthetic_dataset, masks, hyperparameter_tuning=False)
    assert stub_model.instances == []


class TestSeeding:
    """``seed`` drives the global RNGs right before the HPO search and the final fit."""

    @pytest.fixture
    def seeded(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        calls: list[int] = []
        monkeypatch.setattr(SINGLE_MODULE, "seed_everything", calls.append)
        return calls

    def test_the_final_fit_is_seeded_from_the_identity(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel], seeded: list[int]
    ) -> None:
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=3)
        single(stub_model.recipe, synthetic_dataset, folds[1], hyperparameter_tuning=False, seed=3)
        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=3)

        assert seeded[0] == seeded[2]
        assert seeded[0] != seeded[1]

    def test_hpo_and_the_final_fit_are_both_seeded(
        self,
        synthetic_dataset,
        folds: list[SplitMasks],
        stub_model: type[_StubModel],
        seeded: list[int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", lambda **kwargs: ({"alpha": 0.2}, []))

        single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=True, seed=3)

        assert len(seeded) == 2
        assert seeded[0] == seeded[1]

    def test_the_hpo_sampler_defaults_to_the_seed(
        self,
        synthetic_dataset,
        folds: list[SplitMasks],
        stub_model: type[_StubModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: list[int] = []

        def fake_tune(**kwargs: Any):
            seen.append(kwargs["hpo_config"].random_state)
            return {"alpha": 0.2}, []

        monkeypatch.setattr(SINGLE_MODULE, "hpam_tune", fake_tune)

        single(stub_model.recipe, synthetic_dataset, folds[0], seed=8)
        single(stub_model.recipe, synthetic_dataset, folds[0], seed=8, hpo_random_state=2)

        assert seen == [8, 2]

    def test_the_seed_is_recorded_in_the_result_metadata(
        self, synthetic_dataset, folds: list[SplitMasks], stub_model: type[_StubModel]
    ) -> None:
        result = single(stub_model.recipe, synthetic_dataset, folds[0], hyperparameter_tuning=False, seed=3)

        assert result.fold_metadata["execution"]["seed"] == 3


@pytest.mark.slow
class TestNeuralReproducibility:
    """The same seed reproduces a torch model's predictions; another seed does not."""

    @pytest.fixture
    def tiny_network(self, monkeypatch: pytest.MonkeyPatch) -> ModelSpec:
        hyperparameters = {"units_per_layer": [4, 4], "max_epochs": 2, "batch_size": 8}
        monkeypatch.setattr(ModelSpec, "default_hyperparameters", property(lambda spec: dict(hyperparameters)))
        return ModelSpec.from_name("SimpleNeuralNetwork")

    def _predict(self, spec: ModelSpec, dataset, fold: SplitMasks, seed: int) -> np.ndarray:
        # Dirty the global RNGs so only ``seed`` can explain equal predictions.
        torch.manual_seed(np.random.randint(0, 2**31))
        return single(spec, dataset, fold, hyperparameter_tuning=False, seed=seed).predictions

    def test_same_seed_gives_identical_predictions(
        self, tiny_network: ModelSpec, synthetic_dataset, folds: list[SplitMasks]
    ) -> None:
        first = self._predict(tiny_network, synthetic_dataset, folds[0], seed=1)
        second = self._predict(tiny_network, synthetic_dataset, folds[0], seed=1)

        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)

    def test_a_different_seed_gives_different_predictions(
        self, tiny_network: ModelSpec, synthetic_dataset, folds: list[SplitMasks]
    ) -> None:
        first = self._predict(tiny_network, synthetic_dataset, folds[0], seed=1)
        other = self._predict(tiny_network, synthetic_dataset, folds[0], seed=2)

        assert not np.array_equal(first, other)


def test_wandb_project_is_accepted_and_defaults_to_none() -> None:
    import inspect

    assert inspect.signature(single).parameters["wandb_project"].default is None
