"""End-to-end contract for the ``response_transformation`` pipeline setting.

This file is deliberately not a module mirror: the behaviour it pins is a
pipeline-wide contract that only exists once ``drevalpy/utils/response_transform.py``,
``models/component_stack.py``, ``models/drp_model.py``,
``models/tuning/hpo_runtime.py`` and ``_single.py`` agree with each other. Fitting
it into any one of those mirrors would hide the fact that it is the *seam* being
tested, not a module.

The contract, in one line: fit the scaler on the training scope only, train on
transformed targets, inverse-transform the predictions, and score against
untouched ground truth.

Two properties make that contract falsifiable end to end:

* **Affine equivalence.** Squared-error trees are affine-equivariant - splits are
  chosen by a gain that scales uniformly with the target and leaves are means -
  so a symmetric transform/inverse round trip is a provable no-op for
  ``GradientBoosting``. Any asymmetry breaks the identity: training on raw
  targets and inverse-transforming anyway shifts every prediction by the training
  mean, and transforming without inverting shrinks them onto unit variance.
  ``GradientBoosting`` (``HistGradientBoostingRegressor``) is used rather than
  ``randomForest`` because its default ``random_state`` is ``None``, so two
  bootstrap-sampled fits of the forest would disagree for reasons unrelated to
  the transform.
* **HPO round trip.** Nothing in ``hpam_tune`` used to fit the transformer, so a
  non-``None`` value raised ``NotFittedError`` on the first trial - and the
  Optuna objective swallows trial exceptions, so the only visible symptom was an
  empty trial list. Both are asserted.

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
from drevalpy.models import construct_model
from drevalpy.models.tuning.config import build_experiment_hpo_config
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.results.run import RunResult
from drevalpy.utils import fit_response_transformation

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
    """The first of two LCO folds over the session-wide synthetic dataset."""
    return split(synthetic_dataset, "LCO", n_splits=2)[0]


def _train_responses(mudataset, scope) -> np.ndarray:
    """Return the non-NaN raw responses inside *scope*."""
    pairs = scope.pairs
    responses = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]
    return responses[~np.isnan(responses)].astype(np.float64)


class TestAffineEquivalence:
    """A squared-error tree must be indifferent to a symmetric affine transform."""

    @pytest.fixture(scope="class")
    def runs(self, synthetic_dataset, fold: SplitMasks) -> dict[str, RunResult]:
        """Three ``single`` runs: raw, raw again, and standardized.

        The second raw run is the determinism control. Without it a broken
        equivalence assertion is indistinguishable from a predictor that simply
        does not reproduce its own predictions.
        """
        model_class = construct_model("GradientBoosting")
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
    """``fit_response_transformation`` must not see held-out responses.

    The scaler's statistics are the leakage channel: fitted on the whole matrix
    they carry the test fold's mean and variance into training, which is a real
    if mild leak and makes folds incomparable.
    """

    def test_no_prototype_means_no_transformation(self, synthetic_dataset, fold: SplitMasks) -> None:
        assert fit_response_transformation(None, synthetic_dataset, fold.train) is None

    def test_the_mean_is_the_training_scope_mean(self, synthetic_dataset, fold: SplitMasks) -> None:
        fitted = fit_response_transformation(StandardScaler(), synthetic_dataset, fold.train)

        expected = _train_responses(synthetic_dataset, fold.train).mean()
        assert float(fitted.mean_[0]) == pytest.approx(expected)

    def test_the_scale_is_the_training_scope_deviation(self, synthetic_dataset, fold: SplitMasks) -> None:
        fitted = fit_response_transformation(StandardScaler(), synthetic_dataset, fold.train)

        expected = _train_responses(synthetic_dataset, fold.train).std()
        assert float(fitted.scale_[0]) == pytest.approx(expected)

    def test_the_mean_is_not_the_full_matrix_mean(self, synthetic_dataset, fold: SplitMasks) -> None:
        train_mean = _train_responses(synthetic_dataset, fold.train).mean()
        full_mean = float(np.nanmean(synthetic_dataset.response_matrix))
        # Guard the test's own premise: on this fixture the two differ by ~0.11,
        # so an equality below is evidence of leakage and not of a tied fixture.
        assert abs(train_mean - full_mean) > 1e-2

        fitted = fit_response_transformation(StandardScaler(), synthetic_dataset, fold.train)

        assert float(fitted.mean_[0]) != pytest.approx(full_mean, abs=1e-3)

    def test_a_different_scope_yields_different_statistics(self, synthetic_dataset, fold: SplitMasks) -> None:
        on_train = fit_response_transformation(StandardScaler(), synthetic_dataset, fold.train)
        on_test = fit_response_transformation(StandardScaler(), synthetic_dataset, fold.test)

        assert float(on_train.mean_[0]) != pytest.approx(float(on_test.mean_[0]), abs=1e-3)

    def test_unmeasured_pairs_do_not_poison_the_fit(self, synthetic_dataset) -> None:
        everything = SplitMask(np.ones(synthetic_dataset.response_matrix.shape, dtype=bool))

        fitted = fit_response_transformation(StandardScaler(), synthetic_dataset, everything)

        assert np.isfinite(fitted.mean_).all()
        assert float(fitted.mean_[0]) == pytest.approx(float(np.nanmean(synthetic_dataset.response_matrix)))

    def test_the_prototype_is_cloned_rather_than_fitted(self, synthetic_dataset, fold: SplitMasks) -> None:
        prototype = StandardScaler()

        fitted = fit_response_transformation(prototype, synthetic_dataset, fold.train)

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
        trial_model = construct_model("ElasticNet")({"alpha": 0.1, "l1_ratio": 0.5})

        metrics, predictions = _mu_evaluate_trial_all_metrics(
            trial_model,
            mudataset=synthetic_dataset,
            train_scope=fold.train,
            val_scope=fold.val,
            early_stopping_scope=None,
            response_transformation=StandardScaler(),
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
                model_class=construct_model("ElasticNet"),
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
        """Only ``fit_response_transformation`` fits, and it fits a clone."""
        assert not hasattr(tuning["prototype"], "mean_")
