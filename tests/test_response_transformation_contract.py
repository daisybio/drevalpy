"""End-to-end contract for the ``response_transformation`` pipeline setting.

Not a module mirror: the behaviour only exists once ``utils/response_transform.py``, ``models/_pipeline.py``,
``models/tuning/hpo_runtime.py`` and ``_single.py`` agree. Fit the scaler on the training scope only, train on
transformed targets, inverse-transform the predictions, score against untouched ground truth.

``KNNRegressor`` is affine-equivariant, so a symmetric transform/inverse round trip must be a no-op for it; the
group-mean transforms must break that identity for a multi-drug model but not for ``SingleDrugElasticNet``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy import single
from drevalpy.data import split
from drevalpy.models import DRPModel, ModelSpec
from drevalpy.models.tuning.config import build_experiment_hpo_config
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results.run import RunResult
from drevalpy.utils.response_transform import GroupMeanCenterer

HPO_TRIALS = 2


def _fitted_transform(prototype, dataset, scope):
    model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=prototype)
    model.train(dataset, scope)
    return model._pipeline._response_transform


class _LogCapture(logging.Handler):
    """Collect one logger's formatted records (``caplog`` is function scoped, the tuning fixture is not)."""

    def __init__(self, logger_name: str) -> None:
        super().__init__()
        self._logger = logging.getLogger(logger_name)
        self._chunks: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self._chunks.append(self.format(record))

    def __enter__(self) -> _LogCapture:
        self._logger.addHandler(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._logger.removeHandler(self)

    @property
    def text(self) -> str:
        return "\n".join(self._chunks)


@pytest.fixture(scope="module")
def fold(synthetic_dataset) -> SplitMasks:
    return split(synthetic_dataset, "LSO", n_splits=2)[0]


def _responses(mudataset, scope) -> np.ndarray:
    pairs = scope.pairs
    responses = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    return responses[~np.isnan(responses)].astype(np.float64)


def _drug_means(mudataset, scope) -> dict[str, float]:
    rows, cols = scope.pairs[:, 0], scope.pairs[:, 1]
    values = mudataset.response_matrix[rows, cols]
    drugs = np.asarray(mudataset.drug_ids)[cols]
    keep = ~np.isnan(values)
    return {str(drug): float(values[keep & (drugs == drug)].mean()) for drug in np.unique(drugs[keep])}


def _trial_metrics(dataset, fold: SplitMasks, transformation) -> dict[str, float]:
    trial_model = DRPModel(
        ModelSpec.from_name("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5}, response_transformation=transformation
    )
    metrics, predictions = _mu_evaluate_trial_all_metrics(
        trial_model,
        mudataset=dataset,
        train_scope=fold.train,
        val_scope=fold.val,
        early_stopping_scope=None,
        model_checkpoint_dir=None,
    )
    assert len(predictions) > 0
    return metrics


def test_standardizing_the_target_is_a_no_op_for_an_affine_equivariant_model(synthetic_dataset, fold) -> None:
    spec = ModelSpec.from_name("KNNRegressor")
    raw = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
    raw_again = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
    standard = single(
        spec, synthetic_dataset, fold, hyperparameter_tuning=False, response_transformation=StandardScaler()
    )
    pairs = fold.test.pairs

    np.testing.assert_array_equal(raw_again.predictions, raw.predictions)  # determinism control
    np.testing.assert_allclose(standard.predictions, raw.predictions, rtol=1e-5, atol=1e-6)
    assert standard.metrics["RMSE"] == pytest.approx(raw.metrics["RMSE"], rel=1e-5)
    expected_truth = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    np.testing.assert_allclose(standard.ground_truth, expected_truth, equal_nan=True)
    # raw training plus an inverse transform anyway (the original bug) is a shift by the training mean
    reference = StandardScaler().fit(_responses(synthetic_dataset, fold.train_val).reshape(-1, 1))
    double_scaled = reference.inverse_transform(raw.predictions.reshape(-1, 1)).ravel()
    assert not np.allclose(standard.predictions, double_scaled, rtol=1e-3)


class TestFittedOnTheTrainingScopeOnly:
    def test_no_prototype_means_no_transformation(self, synthetic_dataset, fold) -> None:
        assert _fitted_transform(None, synthetic_dataset, fold.train) is None

    def test_the_statistics_are_those_of_the_training_scope(self, synthetic_dataset, fold) -> None:
        prototype = StandardScaler()
        fitted = _fitted_transform(prototype, synthetic_dataset, fold.train)
        on_test = _fitted_transform(StandardScaler(), synthetic_dataset, fold.test)
        train = _responses(synthetic_dataset, fold.train)
        full_mean = float(np.nanmean(synthetic_dataset.response_matrix))

        assert float(fitted.mean_[0]) == pytest.approx(train.mean())
        assert float(fitted.scale_[0]) == pytest.approx(train.std())
        assert abs(train.mean() - full_mean) > 1e-2  # the premise: a leak would be visible
        assert float(fitted.mean_[0]) != pytest.approx(full_mean, abs=1e-3)
        assert float(fitted.mean_[0]) != pytest.approx(float(on_test.mean_[0]), abs=1e-3)
        assert fitted is not prototype
        assert not hasattr(prototype, "mean_")

    def test_unmeasured_pairs_do_not_poison_the_fit(self, synthetic_dataset) -> None:
        everything = SplitMask(np.ones(synthetic_dataset.response_matrix.shape, dtype=bool))

        fitted = _fitted_transform(StandardScaler(), synthetic_dataset, everything)

        assert float(fitted.mean_[0]) == pytest.approx(float(np.nanmean(synthetic_dataset.response_matrix)))


class TestHpoRoundTrip:
    """Tuning with a transform must complete instead of raising ``NotFittedError``."""

    def test_a_single_trial_evaluation_completes(self, synthetic_dataset, fold) -> None:
        assert np.isfinite(_trial_metrics(synthetic_dataset, fold, StandardScaler())["RMSE"])

    def test_a_tuning_run_records_every_trial_without_failures(self, synthetic_dataset, fold) -> None:
        """The Optuna objective swallows trial exceptions: an empty trial list was the only visible symptom."""
        prototype = StandardScaler()
        records = _LogCapture("drevalpy.models.tuning.hpo")
        with records, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            best_params, trials = hpam_tune(
                model_spec=ModelSpec.from_name("ElasticNet"),
                mudataset=synthetic_dataset,
                train_scope=fold.train,
                val_scope=fold.val,
                early_stopping_scope=None,
                response_transformation=prototype,
                metric="RMSE",
                hpo_config=build_experiment_hpo_config("RMSE", n_trials=HPO_TRIALS, random_state=0),
            )

        assert "NotFittedError" not in records.text
        assert "Optuna trial" not in records.text
        assert len(trials) == HPO_TRIALS
        assert all(np.isfinite(metrics["RMSE"]) for _, metrics, _ in trials)
        assert not [str(w.message) for w in caught if "did not find a valid configuration" in str(w.message)]
        assert best_params
        assert not hasattr(prototype, "mean_")


class TestGroupMeanTransformations:
    @pytest.fixture(scope="class")
    def runs(self, synthetic_dataset, fold: SplitMasks) -> dict[str, RunResult]:
        out = {}
        for model in ("NaivePredictor", "KNNRegressor"):
            spec = ModelSpec.from_name(model)
            out[model] = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
            for name, levels in (("drug_mean", ("drug",)), ("drug_tissue_mean", ("drug", "tissue"))):
                out[f"{model}/{name}"] = single(
                    spec,
                    synthetic_dataset,
                    fold,
                    hyperparameter_tuning=False,
                    response_transformation=GroupMeanCenterer(levels),
                )
        return out

    def test_the_naive_mean_of_the_residuals_is_the_training_drug_mean(self, runs, synthetic_dataset, fold) -> None:
        means = _drug_means(synthetic_dataset, fold.train_val)
        result = runs["NaivePredictor/drug_mean"]

        np.testing.assert_allclose(result.predictions, [means[str(drug)] for drug in result.drug_ids], rtol=1e-6)

    def test_the_group_means_are_those_of_the_training_scope(self, synthetic_dataset, fold) -> None:
        fitted = _fitted_transform(GroupMeanCenterer(), synthetic_dataset, fold.train)

        means = _drug_means(synthetic_dataset, fold.train)
        np.testing.assert_allclose(fitted.level_means_[0], [means[key] for key in fitted.level_keys_[0]], rtol=1e-6)

    @pytest.mark.parametrize("name", ["drug_mean", "drug_tissue_mean"])
    def test_a_multi_drug_model_changes_but_stays_on_the_original_scale(self, runs, name) -> None:
        raw, transformed = runs["KNNRegressor"], runs[f"KNNRegressor/{name}"]

        assert not np.allclose(transformed.predictions, raw.predictions, rtol=1e-3)
        np.testing.assert_array_equal(transformed.ground_truth, raw.ground_truth)
        assert abs(transformed.predictions.mean() - raw.predictions.mean()) < raw.predictions.std()

    def test_a_single_drug_model_is_indifferent_to_the_drug_offset(self, synthetic_dataset, fold) -> None:
        spec = ModelSpec.from_name("SingleDrugElasticNet")
        raw = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
        centred = single(
            spec, synthetic_dataset, fold, hyperparameter_tuning=False, response_transformation=GroupMeanCenterer()
        )

        np.testing.assert_allclose(centred.predictions, raw.predictions, rtol=1e-4, atol=1e-4)

    def test_a_hyperparameter_trial_completes(self, synthetic_dataset, fold) -> None:
        metrics = _trial_metrics(synthetic_dataset, fold, GroupMeanCenterer(("drug", "tissue")))

        assert np.isfinite(metrics["RMSE"])
