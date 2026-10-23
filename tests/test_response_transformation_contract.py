"""End-to-end contract for the ``response_transformation`` pipeline setting.

This file is deliberately not a module mirror: the behaviour it pins is a
pipeline-wide contract that only exists once ``drevalpy/utils/response_transform.py``,
``models/_pipeline.py``, ``models/drp_model.py``,
``models/tuning/hpo_runtime.py`` and ``_single.py`` agree with each other. Fitting
it into any one of those mirrors would hide the fact that it is the *seam* being
tested, not a module.

The contract, in one line: fit the scaler on the training scope only, train on
transformed targets, inverse-transform the predictions, and score against
untouched ground truth.

Two properties make that contract falsifiable end to end:

* **Affine equivalence.** A k-nearest-neighbour regressor is affine-equivariant -
  the neighbours are chosen from the features alone and the prediction is a mean
  of their targets - so a symmetric transform/inverse round trip is a provable
  no-op for ``KNNRegressor``. Any asymmetry breaks the identity: training on raw
  targets and inverse-transforming anyway shifts every prediction by the training
  mean, and transforming without inverting shrinks them onto unit variance.
  A tree is not used: the synthetic omics views share hidden factors, so many
  features offer equally good splits and ``HistGradientBoostingRegressor`` breaks
  such near-ties by float32 rounding, which depends on the target's scale.
* **HPO round trip.** Nothing in ``hpam_tune`` used to fit the transformer, so a
  non-``None`` value raised ``NotFittedError`` on the first trial - and the
  Optuna objective swallows trial exceptions, so the only visible symptom was an
  empty trial list. Both are asserted.

A third property covers the non-affine case. ``drug_mean`` and ``drug_tissue_mean`` are not
equivariant for a model that sees several drugs, so the affine identity above must *break*
for them (the contrast keeps that identity honest), while a model that sees one drug at a time,
here ``SingleDrugElasticNet``, still returns the same predictions: for it the group offset is a
constant shift of the drug's own targets, which its unpenalized intercept absorbs.

Between the two sits the fit itself: the scaler's statistics must come from the
training scope alone, which is the one place the design can leak held-out
responses into training.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

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


def _fitted_transform(prototype, dataset, scope):
    model = DRPModel(ModelSpec.from_name("NaivePredictor"), response_transformation=prototype)
    model.train(dataset, scope)
    return model._pipeline._response_transform


#: Two trials is the smallest budget that still proves the objective ran more
#: than once; the suite pays for every extra fit.
HPO_TRIALS = 2


class _LogCapture(logging.Handler):
    """Collect one logger's records, formatted traceback included.

    ``caplog`` cannot be used here: the tuning run is shared by several
    assertions and therefore lives in a class-scoped fixture, while ``caplog`` is
    function scoped. The formatted text is what matters - ``logger.exception``
    puts the exception type in the traceback rather than the message.
    """

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
    """The first of two LSO folds over the session-wide synthetic dataset."""
    return split(synthetic_dataset, "LSO", n_splits=2)[0]


def _train_responses(mudataset, scope) -> np.ndarray:
    """Return the non-NaN raw responses inside *scope*."""
    pairs = scope.pairs
    responses = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    return responses[~np.isnan(responses)].astype(np.float64)


class TestAffineEquivalence:
    """A nearest-neighbour mean must be indifferent to a symmetric affine transform."""

    @pytest.fixture(scope="class")
    def runs(self, synthetic_dataset, fold: SplitMasks) -> dict[str, RunResult]:
        """Three ``single`` runs: raw, raw again, and standardized.

        The second raw run is the determinism control. Without it a broken
        equivalence assertion is indistinguishable from a predictor that simply
        does not reproduce its own predictions.
        """
        model_class = ModelSpec.from_name("KNNRegressor")
        return {
            "raw": single(model_class, synthetic_dataset, fold, hyperparameter_tuning=False),
            "raw_again": single(model_class, synthetic_dataset, fold, hyperparameter_tuning=False),
            "standard": single(
                model_class,
                synthetic_dataset,
                fold,
                hyperparameter_tuning=False,
                response_transformation=StandardScaler(),
            ),
        }

    def test_the_predictor_is_deterministic(self, runs: dict[str, RunResult]) -> None:
        np.testing.assert_array_equal(runs["raw_again"].predictions, runs["raw"].predictions)

    def test_standardizing_the_target_leaves_predictions_unchanged(self, runs: dict[str, RunResult]) -> None:
        np.testing.assert_allclose(runs["standard"].predictions, runs["raw"].predictions, rtol=1e-5, atol=1e-6)

    def test_standardizing_the_target_leaves_rmse_unchanged(self, runs: dict[str, RunResult]) -> None:
        assert runs["standard"].metrics["RMSE"] == pytest.approx(runs["raw"].metrics["RMSE"], rel=1e-5)

    def test_ground_truth_is_never_transformed(
        self, runs: dict[str, RunResult], synthetic_dataset, fold: SplitMasks
    ) -> None:
        pairs = fold.test.pairs
        expected = synthetic_dataset.response_matrix[pairs[:, 0], pairs[:, 1]]

        np.testing.assert_allclose(runs["standard"].ground_truth, expected, equal_nan=True)

    def test_predictions_are_not_inverse_transformed_twice(
        self, runs: dict[str, RunResult], synthetic_dataset, fold: SplitMasks
    ) -> None:
        """Pin the original bug: raw training plus an inverse transform anyway.

        That combination reproduces ``raw`` predictions pushed through
        ``inverse_transform``, which on this fixture is a shift of roughly the
        training mean, so it is comfortably distinguishable from equality.
        """
        reference = StandardScaler().fit(_train_responses(synthetic_dataset, fold.train_val).reshape(-1, 1))
        double_scaled = reference.inverse_transform(runs["raw"].predictions.reshape(-1, 1)).ravel()

        assert not np.allclose(runs["standard"].predictions, double_scaled, rtol=1e-3)


class TestFittedOnTheTrainingScopeOnly:
    """``_fitted_transform`` must not see held-out responses.

    The scaler's statistics are the leakage channel: fitted on the whole matrix
    they carry the test fold's mean and variance into training, which is a real
    if mild leak and makes folds incomparable.
    """

    def test_no_prototype_means_no_transformation(self, synthetic_dataset, fold: SplitMasks) -> None:
        assert _fitted_transform(None, synthetic_dataset, fold.train) is None

    def test_the_mean_is_the_training_scope_mean(self, synthetic_dataset, fold: SplitMasks) -> None:
        fitted = _fitted_transform(StandardScaler(), synthetic_dataset, fold.train)

        expected = _train_responses(synthetic_dataset, fold.train).mean()
        assert float(fitted.mean_[0]) == pytest.approx(expected)

    def test_the_scale_is_the_training_scope_deviation(self, synthetic_dataset, fold: SplitMasks) -> None:
        fitted = _fitted_transform(StandardScaler(), synthetic_dataset, fold.train)

        expected = _train_responses(synthetic_dataset, fold.train).std()
        assert float(fitted.scale_[0]) == pytest.approx(expected)

    def test_the_mean_is_not_the_full_matrix_mean(self, synthetic_dataset, fold: SplitMasks) -> None:
        train_mean = _train_responses(synthetic_dataset, fold.train).mean()
        full_mean = float(np.nanmean(synthetic_dataset.response_matrix))
        # Guard the test's own premise: on this fixture the two differ by ~0.11,
        # so an equality below is evidence of leakage and not of a tied fixture.
        assert abs(train_mean - full_mean) > 1e-2

        fitted = _fitted_transform(StandardScaler(), synthetic_dataset, fold.train)

        assert float(fitted.mean_[0]) != pytest.approx(full_mean, abs=1e-3)

    def test_a_different_scope_yields_different_statistics(self, synthetic_dataset, fold: SplitMasks) -> None:
        on_train = _fitted_transform(StandardScaler(), synthetic_dataset, fold.train)
        on_test = _fitted_transform(StandardScaler(), synthetic_dataset, fold.test)

        assert float(on_train.mean_[0]) != pytest.approx(float(on_test.mean_[0]), abs=1e-3)

    def test_unmeasured_pairs_do_not_poison_the_fit(self, synthetic_dataset) -> None:
        everything = SplitMask(np.ones(synthetic_dataset.response_matrix.shape, dtype=bool))

        fitted = _fitted_transform(StandardScaler(), synthetic_dataset, everything)

        assert np.isfinite(fitted.mean_).all()
        assert float(fitted.mean_[0]) == pytest.approx(float(np.nanmean(synthetic_dataset.response_matrix)))

    def test_the_prototype_is_cloned_rather_than_fitted(self, synthetic_dataset, fold: SplitMasks) -> None:
        prototype = StandardScaler()

        fitted = _fitted_transform(prototype, synthetic_dataset, fold.train)

        assert fitted is not prototype
        assert not hasattr(prototype, "mean_")


class TestHpoRoundTrip:
    """Tuning with a transform must complete instead of raising ``NotFittedError``."""

    def test_a_single_trial_evaluation_completes(self, synthetic_dataset, fold: SplitMasks) -> None:
        """The trial helpers own the fit, so they take an *unfitted* prototype.

        This is the narrowest reproduction of the crash: unlike the Optuna
        objective, ``_mu_evaluate_trial_all_metrics`` does not swallow
        exceptions, so a missing fit surfaces as ``NotFittedError`` here.
        """
        trial_model = DRPModel(
            ModelSpec.from_name("ElasticNet"), {"alpha": 0.1, "l1_ratio": 0.5}, response_transformation=StandardScaler()
        )

        metrics, predictions = _mu_evaluate_trial_all_metrics(
            trial_model,
            mudataset=synthetic_dataset,
            train_scope=fold.train,
            val_scope=fold.val,
            early_stopping_scope=None,
            model_checkpoint_dir=None,
        )

        assert np.isfinite(metrics["RMSE"])
        assert len(predictions) > 0

    @pytest.fixture(scope="class")
    def tuning(self, synthetic_dataset, fold: SplitMasks) -> dict[str, Any]:
        """One ``hpam_tune`` call with a standard transform on a regularized linear model.

        Trial exceptions are logged rather than raised, and a study with no valid
        trial falls back to the defaults with a warning, so both channels are
        captured here and asserted separately below.
        """
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
        return {
            "prototype": prototype,
            "best_params": best_params,
            "trials": trials,
            "log_text": records.text,
            "warnings": [str(entry.message) for entry in caught],
        }

    def test_no_trial_raises_not_fitted_error(self, tuning: dict[str, Any]) -> None:
        assert "NotFittedError" not in tuning["log_text"]

    def test_no_trial_fails_at_all(self, tuning: dict[str, Any]) -> None:
        assert "Optuna trial" not in tuning["log_text"]

    def test_every_trial_is_recorded(self, tuning: dict[str, Any]) -> None:
        """An empty list is how the swallowed ``NotFittedError`` used to present."""
        assert len(tuning["trials"]) == HPO_TRIALS

    def test_every_trial_scores_a_finite_metric(self, tuning: dict[str, Any]) -> None:
        assert all(np.isfinite(metrics["RMSE"]) for _, metrics, _ in tuning["trials"])

    def test_tuning_does_not_fall_back_to_the_defaults(self, tuning: dict[str, Any]) -> None:
        assert not [message for message in tuning["warnings"] if "did not find a valid configuration" in message]
        assert tuning["best_params"]

    def test_the_callers_prototype_is_left_unfitted(self, tuning: dict[str, Any]) -> None:
        """Each pipeline fits its own clone."""
        assert not hasattr(tuning["prototype"], "mean_")


def _drug_means(mudataset, scope) -> dict[str, float]:
    """Return the mean raw response per drug id inside *scope*."""
    rows, cols = scope.pairs[:, 0], scope.pairs[:, 1]
    values = mudataset.response_matrix[rows, cols]
    drugs = np.asarray(mudataset.drug_ids)[cols]
    keep = ~np.isnan(values)
    return {str(drug): float(values[keep & (drugs == drug)].mean()) for drug in np.unique(drugs[keep])}


class TestGroupMeanTransformations:
    """``drug_mean`` and ``drug_tissue_mean`` fit on the training scope and are undone on the way out."""

    @pytest.fixture(scope="class")
    def runs(self, synthetic_dataset, fold: SplitMasks) -> dict[str, RunResult]:
        """Naive-mean and KNN runs raw, with ``drug_mean`` and with ``drug_tissue_mean``."""
        out = {}
        for model in ("NaivePredictor", "KNNRegressor"):
            spec = ModelSpec.from_name(model)
            out[model] = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
            for name in ("drug_mean", "drug_tissue_mean"):
                out[f"{model}/{name}"] = single(
                    spec,
                    synthetic_dataset,
                    fold,
                    hyperparameter_tuning=False,
                    response_transformation=GroupMeanCenterer(("drug",) if name == "drug_mean" else ("drug", "tissue")),
                )
        return out

    def test_the_naive_mean_of_the_residuals_is_the_training_drug_mean(
        self, runs: dict[str, RunResult], synthetic_dataset, fold: SplitMasks
    ) -> None:
        """Residuals have mean zero, so the inverse transform alone puts back each drug's training mean."""
        means = _drug_means(synthetic_dataset, fold.train_val)
        result = runs["NaivePredictor/drug_mean"]

        expected = [means[str(drug)] for drug in result.drug_ids]
        np.testing.assert_allclose(result.predictions, expected, rtol=1e-6)

    def test_the_group_means_are_those_of_the_training_scope(self, synthetic_dataset, fold: SplitMasks) -> None:
        fitted = _fitted_transform(GroupMeanCenterer(), synthetic_dataset, fold.train)

        means = _drug_means(synthetic_dataset, fold.train)
        np.testing.assert_allclose(fitted.level_means_[0], [means[key] for key in fitted.level_keys_[0]], rtol=1e-6)

    def test_the_transformation_changes_the_predictions_of_a_multi_drug_model(self, runs: dict[str, RunResult]) -> None:
        raw = runs["KNNRegressor"].predictions

        assert not np.allclose(runs["KNNRegressor/drug_mean"].predictions, raw, rtol=1e-3)
        assert not np.allclose(runs["KNNRegressor/drug_tissue_mean"].predictions, raw, rtol=1e-3)

    def test_the_ground_truth_is_never_transformed(self, runs: dict[str, RunResult]) -> None:
        for name in ("drug_mean", "drug_tissue_mean"):
            np.testing.assert_array_equal(runs[f"KNNRegressor/{name}"].ground_truth, runs["KNNRegressor"].ground_truth)

    def test_predictions_come_back_on_the_original_scale(self, runs: dict[str, RunResult]) -> None:
        """A missing inverse would leave residuals around zero instead of values around the response mean."""
        raw = runs["KNNRegressor"].predictions

        for name in ("drug_mean", "drug_tissue_mean"):
            assert abs(runs[f"KNNRegressor/{name}"].predictions.mean() - raw.mean()) < raw.std()

    def test_a_single_drug_model_is_indifferent_to_the_drug_offset(self, synthetic_dataset, fold: SplitMasks) -> None:
        spec = ModelSpec.from_name("SingleDrugElasticNet")
        raw = single(spec, synthetic_dataset, fold, hyperparameter_tuning=False)
        centred = single(
            spec,
            synthetic_dataset,
            fold,
            hyperparameter_tuning=False,
            response_transformation=GroupMeanCenterer(),
        )

        np.testing.assert_allclose(centred.predictions, raw.predictions, rtol=1e-4, atol=1e-4)

    def test_a_hyperparameter_trial_completes(self, synthetic_dataset, fold: SplitMasks) -> None:
        trial_model = DRPModel(
            ModelSpec.from_name("ElasticNet"),
            {"alpha": 0.1, "l1_ratio": 0.5},
            response_transformation=GroupMeanCenterer(("drug", "tissue")),
        )

        metrics, _ = _mu_evaluate_trial_all_metrics(
            trial_model,
            mudataset=synthetic_dataset,
            train_scope=fold.train,
            val_scope=fold.val,
            early_stopping_scope=None,
            model_checkpoint_dir=None,
        )

        assert np.isfinite(metrics["RMSE"])
