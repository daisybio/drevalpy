"""Tests for :mod:`drevalpy._train`, the final-model fit behind :func:`drevalpy.train`."""

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

    def train(self, dataset, scope, *, early_stopping_scope=None, model_checkpoint_dir=None, n_jobs=1) -> None:
        type(self).fits.append({"scope": scope, "early_stopping_scope": early_stopping_scope, "n_jobs": n_jobs})


@pytest.fixture
def recorded_fits(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    cls = type("_FreshRecordingModel", (_RecordingModel,), {"fits": []})
    monkeypatch.setattr(TRAIN_MODULE, "DRPModel", cls)
    return cls.fits


@pytest.fixture
def tuner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_tune(**kwargs: Any) -> tuple[dict[str, Any], list]:
        seen.update(kwargs)
        return {"alpha": 0.5}, []

    monkeypatch.setattr(TRAIN_MODULE, "hpam_tune", fake_tune)
    return seen


def _observed(dataset) -> np.ndarray:
    return np.isfinite(dataset.response_matrix)


def test_a_real_model_is_fitted_and_predicts_every_pair(synthetic_dataset) -> None:
    fitted = drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False)
    scope = SplitMask(_observed(synthetic_dataset))

    predictions = fitted.predict(synthetic_dataset, scope)

    assert drevalpy.train is TRAIN_MODULE.train
    assert fitted.name == "ElasticNet"
    assert fitted.hyperparameters == ModelSpec.from_name("ElasticNet").default_hyperparameters
    assert predictions.shape == (len(scope),)
    assert np.isfinite(predictions).all()
    again = drevalpy.train(ModelSpec.from_name("ElasticNet"), synthetic_dataset, hyperparameter_tuning=False, seed=42)
    np.testing.assert_array_equal(again.predict(synthetic_dataset, scope), predictions)


def test_a_single_drug_model_fits_an_estimator_for_every_drug(synthetic_dataset) -> None:
    fitted = drevalpy.train("SingleDrugElasticNet", synthetic_dataset, hyperparameter_tuning=False)
    scope = SplitMask(_observed(synthetic_dataset))

    assert set(fitted._pipeline._predictor._estimators) == {str(drug) for drug in synthetic_dataset.drug_ids}
    assert np.isfinite(fitted.predict(synthetic_dataset, scope)).all()


def test_the_response_transformation_is_fitted_on_a_copy_and_inverted_when_predicting(synthetic_dataset) -> None:
    prototype = StandardScaler()
    fitted = drevalpy.train(
        "NaivePredictor", synthetic_dataset, hyperparameter_tuning=False, response_transformation=prototype
    )

    predictions = fitted.predict(synthetic_dataset, SplitMask(_observed(synthetic_dataset)))

    np.testing.assert_allclose(predictions, np.nanmean(synthetic_dataset.response_matrix), atol=0.05)
    assert hasattr(fitted._pipeline._response_transform, "mean_")
    assert not hasattr(prototype, "mean_")


class TestScopes:
    def test_a_model_without_early_stopping_trains_on_every_observed_pair(
        self, synthetic_dataset, recorded_fits
    ) -> None:
        drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False)

        (fit,) = recorded_fits
        assert fit["early_stopping_scope"] is None
        assert fit["scope"] == SplitMask(_observed(synthetic_dataset))

    def test_an_early_stopping_model_sets_aside_a_seeded_quarter_of_the_validation_slice(
        self, synthetic_dataset, recorded_fits, monkeypatch
    ) -> None:
        monkeypatch.setattr(ModelSpec, "supports_early_stopping", property(lambda spec: True))
        n_validation = max(1, int(int(_observed(synthetic_dataset).sum()) * TRAIN_MODULE.VALIDATION_RATIO))

        for seed in (1, 1, 2):
            drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False, seed=seed)

        first, repeated, other = recorded_fits
        stopping, training = first["early_stopping_scope"], first["scope"]
        assert len(stopping) == max(1, int(n_validation * 0.25))
        assert not (stopping & training).any()
        assert (stopping | training) == SplitMask(_observed(synthetic_dataset))
        assert repeated["early_stopping_scope"] == stopping
        assert other["early_stopping_scope"] != stopping

    def test_pairs_failing_the_quality_rule_are_not_used(self, recorded_fits) -> None:
        dataset = synthetic(low_quality_fraction=0.2)

        drevalpy.train("ElasticNet", dataset, hyperparameter_tuning=False)
        drevalpy.train("ElasticNet", dataset, hyperparameter_tuning=False, quality=CurveQuality.off())

        default_fit, unfiltered_fit = recorded_fits
        assert len(default_fit["scope"]) < len(unfiltered_fit["scope"]) == int(_observed(dataset).sum())


class TestTuning:
    def test_the_tuned_hyperparameters_are_used_and_the_scopes_are_disjoint(
        self, synthetic_dataset, recorded_fits, tuner
    ) -> None:
        fitted = drevalpy.train(
            "ElasticNet", synthetic_dataset, hpo_metric="Pearson", hpo_num_trials=3, hpo_random_state=9, n_jobs=3
        )

        assert fitted.hyperparameters["alpha"] == 0.5
        assert tuner["metric"] == "Pearson"
        assert (tuner["hpo_config"].n_trials, tuner["hpo_config"].random_state, tuner["n_jobs"]) == (3, 9, 3)
        assert not (tuner["train_scope"] & tuner["val_scope"]).any()
        assert tuner["val_scope"].any()
        assert recorded_fits[0]["scope"] == tuner["train_scope"] | tuner["val_scope"]

    @pytest.mark.slow
    def test_tuning_a_real_model_returns_a_fitted_model(self, synthetic_dataset) -> None:
        fitted = drevalpy.train("ElasticNet", synthetic_dataset, hpo_num_trials=2)

        assert np.isfinite(fitted.predict(synthetic_dataset, SplitMask(_observed(synthetic_dataset)))).all()


class TestInputs:
    def test_an_unknown_model_name_is_rejected(self, synthetic_dataset) -> None:
        with pytest.raises(UnknownNameError):
            drevalpy.train("NoSuchModel", synthetic_dataset)

    def test_a_dataset_name_is_loaded(self, synthetic_dataset, monkeypatch) -> None:
        monkeypatch.setattr(TRAIN_MODULE, "load", lambda name: synthetic_dataset)

        assert drevalpy.train("NaivePredictor", "anything", hyperparameter_tuning=False).name == "NaivePredictor"

    def test_fewer_than_two_usable_pairs_are_rejected(self) -> None:
        dataset = synthetic(n_samples=2, n_drugs=1, missing_fraction=0.0)
        dataset.mdata["response"].X[:] = np.nan
        dataset.mdata["response"].X[0, 0] = 1.0

        with pytest.raises(InvalidInputError, match="at least 2"):
            drevalpy.train("NaivePredictor", dataset, hyperparameter_tuning=False, quality=CurveQuality.off())

    def test_n_jobs_reaches_the_final_fit_and_an_unusable_count_is_rejected(
        self, recorded_fits, synthetic_dataset
    ) -> None:
        drevalpy.train("ElasticNet", synthetic_dataset, hyperparameter_tuning=False, n_jobs=2)
        assert recorded_fits[0]["n_jobs"] == 2

        with pytest.raises(InvalidInputError, match="n_jobs"):
            drevalpy.train("ElasticNet", synthetic_dataset, n_jobs=0)
