"""Tests for :mod:`drevalpy._train`, the final-model fit behind :func:`drevalpy.train`.

The happy path trains a real ``ElasticNet`` and a real single-drug model on the synthetic
dataset. The scope arithmetic is pinned with a recording ``DRPModel``, because only the
scopes handed to ``train`` show which pairs were fitted on and which were held out.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import drevalpy
from drevalpy.data import CurveQuality, synthetic
from drevalpy.errors import InvalidInputError, UnknownNameError
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.types import SplitMask

TRAIN_MODULE = importlib.import_module("drevalpy._train")


class _RecordingModel(DRPModel):
    """A ``DRPModel`` that records the scopes of its fit and fits nothing."""

    fits: list[dict[str, Any]] = []  # plain class-level log, reset by the fixture

    def train(self, dataset, scope, *, early_stopping_scope=None, model_checkpoint_dir=None) -> None:
        type(self).fits.append({"scope": scope, "early_stopping_scope": early_stopping_scope})


@pytest.fixture
def recorded_fits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the ``DRPModel`` of ``train`` by one that records the scopes it is fitted on."""
    cls = type("_FreshRecordingModel", (_RecordingModel,), {"fits": []})
    monkeypatch.setattr(TRAIN_MODULE, "DRPModel", cls)
    return cls.fits


def _observed(dataset) -> np.ndarray:
    return np.isfinite(dataset.response_matrix)


class TestFinalFit:
    """A real model is trained on the dataset and predicts it."""

    @pytest.fixture(scope="class")
    def fitted(self, synthetic_dataset) -> DRPModel:
        return drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False)

    def test_is_exported_from_the_package(self) -> None:
        assert drevalpy.train is TRAIN_MODULE.train

    def test_returns_a_fitted_model_of_the_requested_recipe(self, fitted: DRPModel) -> None:
        assert isinstance(fitted, DRPModel)
        assert fitted.name == "ElasticNet"

    def test_predicts_every_pair_of_the_dataset(self, fitted: DRPModel, synthetic_dataset) -> None:
        scope = SplitMask(_observed(synthetic_dataset))

        predictions = fitted.predict(synthetic_dataset, scope)

        assert predictions.shape == (len(scope),)
        assert np.isfinite(predictions).all()

    def test_uses_the_default_hyperparameters_without_tuning(self, fitted: DRPModel) -> None:
        assert fitted.hyperparameters == ModelSpec.from_name("ElasticNet").default_hyperparameters

    def test_accepts_a_model_spec(self, synthetic_dataset) -> None:
        fitted = drevalpy.train(ModelSpec.from_name("ElasticNet"), synthetic_dataset, hyperparameter_tuning=False)

        assert fitted.name == "ElasticNet"

    def test_the_same_seed_gives_the_same_model(self, synthetic_dataset, fitted: DRPModel) -> None:
        again = drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False, seed=42)
        scope = SplitMask(_observed(synthetic_dataset))

        np.testing.assert_array_equal(again.predict(synthetic_dataset, scope), fitted.predict(synthetic_dataset, scope))


class TestSingleDrugModel:
    """A single-drug model fits one estimator per drug through its own routing."""

    def test_fits_an_estimator_for_every_drug(self, synthetic_dataset) -> None:
        fitted = drevalpy.train("SingleDrugElasticNet", synthetic_dataset, hyperparameter_tuning=False)

        estimators = fitted._pipeline._predictor._estimators
        assert set(estimators) == {str(drug) for drug in synthetic_dataset.drug_ids}

    def test_predicts_every_drug(self, synthetic_dataset) -> None:
        fitted = drevalpy.train("SingleDrugElasticNet", synthetic_dataset, hyperparameter_tuning=False)
        scope = SplitMask(_observed(synthetic_dataset))

        assert np.isfinite(fitted.predict(synthetic_dataset, scope)).all()


class TestResponseTransformation:
    """The transformation is fitted on the training pairs and inverted when predicting."""

    def test_predictions_are_in_response_units(self, synthetic_dataset) -> None:
        fitted = drevalpy.train(
            "NaivePredictor",
            synthetic_dataset,
            hyperparameter_tuning=False,
            response_transformation=StandardScaler(),
        )
        scope = SplitMask(_observed(synthetic_dataset))

        predictions = fitted.predict(synthetic_dataset, scope)

        np.testing.assert_allclose(predictions, np.nanmean(synthetic_dataset.response_matrix), atol=0.05)
        assert hasattr(fitted._pipeline._response_transform, "mean_")

    def test_does_not_fit_the_callers_prototype(self, synthetic_dataset) -> None:
        prototype = StandardScaler()

        drevalpy.train(
            "NaivePredictor", synthetic_dataset, hyperparameter_tuning=False, response_transformation=prototype
        )

        assert not hasattr(prototype, "mean_")


class TestScopes:
    """Every usable pair is fitted on, except the slice a run would set aside."""

    def test_a_model_without_early_stopping_trains_on_every_observed_pair(
        self, synthetic_dataset, recorded_fits: list[dict[str, Any]]
    ) -> None:
        drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False)

        (fit,) = recorded_fits
        assert fit["early_stopping_scope"] is None
        assert fit["scope"] == SplitMask(_observed(synthetic_dataset))

    def test_an_early_stopping_model_sets_aside_a_quarter_of_the_validation_slice(
        self, synthetic_dataset, recorded_fits: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda spec: True))
        n_pairs = int(_observed(synthetic_dataset).sum())
        n_validation = max(1, int(n_pairs * TRAIN_MODULE.VALIDATION_RATIO))

        drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False)

        (fit,) = recorded_fits
        stopping, training = fit["early_stopping_scope"], fit["scope"]
        assert len(stopping) == max(1, int(n_validation * 0.25))
        assert not (stopping & training).any()
        assert (stopping | training) == SplitMask(_observed(synthetic_dataset))

    def test_the_seed_picks_the_held_out_pairs(
        self, synthetic_dataset, recorded_fits: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda spec: True))

        for seed in (1, 1, 2):
            drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False, seed=seed)

        first, repeated, other = (fit["early_stopping_scope"] for fit in recorded_fits)
        assert first == repeated
        assert first != other

    def test_pairs_failing_the_quality_rule_are_not_used(self, recorded_fits: list[dict[str, Any]]) -> None:
        dataset = synthetic(low_quality_fraction=0.2)

        drevalpy.train("ElasticNet", dataset, hyperparameter_tuning=False)
        drevalpy.train("ElasticNet", dataset, hyperparameter_tuning=False, quality=CurveQuality.off())

        default_fit, unfiltered_fit = recorded_fits
        assert len(default_fit["scope"]) < len(unfiltered_fit["scope"]) == int(_observed(dataset).sum())


class TestTuning:
    """Tuning scores on the validation slice and the final fit uses the winner."""

    def test_the_tuned_hyperparameters_are_used_and_the_scopes_are_disjoint(
        self, synthetic_dataset, recorded_fits: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_tune(**kwargs: Any) -> tuple[dict[str, Any], list]:
            calls.append(kwargs)
            return {"alpha": 0.5}, []

        monkeypatch.setattr(TRAIN_MODULE, "hpam_tune", fake_tune)

        fitted = drevalpy.train(
            "ElasticNet", synthetic_dataset, hpo_metric="Pearson", hpo_num_trials=3, hpo_random_state=9
        )

        (call,) = calls
        assert fitted.hyperparameters["alpha"] == 0.5
        assert call["metric"] == "Pearson"
        assert call["hpo_config"].n_trials == 3
        assert call["hpo_config"].random_state == 9
        assert not (call["train_scope"] & call["val_scope"]).any()
        assert call["val_scope"].any()
        assert recorded_fits[0]["scope"] == call["train_scope"] | call["val_scope"]


@pytest.mark.slow
class TestRealTuning:
    """Tuning runs for real once."""

    def test_tuning_a_real_model_returns_a_fitted_model(self, synthetic_dataset) -> None:
        fitted = drevalpy.train("ElasticNet", synthetic_dataset, hpo_num_trials=2)

        scope = SplitMask(_observed(synthetic_dataset))
        assert np.isfinite(fitted.predict(synthetic_dataset, scope)).all()


class TestInputs:
    """Bad inputs raise the library's input errors."""

    def test_an_unknown_model_name_is_rejected(self, synthetic_dataset) -> None:
        with pytest.raises(UnknownNameError):
            drevalpy.train("NoSuchModel", synthetic_dataset)

    def test_a_dataset_name_is_loaded(self, synthetic_dataset, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TRAIN_MODULE, "load", lambda name: synthetic_dataset)

        fitted = drevalpy.train("NaivePredictor", "anything", hyperparameter_tuning=False)

        assert fitted.name == "NaivePredictor"

    def test_fewer_than_two_usable_pairs_are_rejected(self) -> None:
        dataset = synthetic(n_samples=2, n_drugs=1, missing_fraction=0.0)
        dataset.mdata["response"].X[:] = np.nan
        dataset.mdata["response"].X[0, 0] = 1.0

        with pytest.raises(InvalidInputError, match="at least 2"):
            drevalpy.train("NaivePredictor", dataset, hyperparameter_tuning=False, quality=CurveQuality.off())
