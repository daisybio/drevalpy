"""Tests for HPO trial construction with immutable ModelSpec inputs."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.tuning.hpo_runtime import _construct_trial_model, _mu_evaluate_trial_all_metrics
from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity


def _evaluate(predictions, **kwargs):
    model = MagicMock()
    model.predict.return_value = predictions
    scopes = lso_split_masks()
    call = {
        "mudataset": synthetic_mudataset_identity(),
        "train_scope": scopes.train,
        "val_scope": scopes.test,
        "early_stopping_scope": None,
        "model_checkpoint_dir": None,
    }
    call.update(kwargs)
    return model, _mu_evaluate_trial_all_metrics(model, **call)


def test_construct_trial_model_keeps_spec_identity() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    trial_model = _construct_trial_model(spec, {"predictor.elasticNet.alpha": 0.5})

    assert trial_model.spec.name == spec.name
    assert trial_model.hyperparameters["alpha"] == 0.5


@pytest.mark.parametrize("supplied", [False, True])
def test_trial_uses_one_training_call_and_keeps_nan_scope_alignment(tmp_path, supplied) -> None:
    dataset = synthetic_mudataset_identity()
    model, (metrics, predictions) = _evaluate(
        np.array([np.nan, 5.0]),
        mudataset=dataset,
        model_checkpoint_dir=tmp_path if supplied else None,
        trial_number=3,
    )

    model.train.assert_called_once()
    assert model.train.call_args.args == (dataset, lso_split_masks().train)
    assert model.train.call_args.kwargs["early_stopping_scope"] is None
    if supplied:
        assert model.train.call_args.kwargs["model_checkpoint_dir"] == tmp_path / "trial_3"
    assert metrics["MSE"] == 1.0
    assert np.isnan(metrics["R^2"])
    np.testing.assert_array_equal(predictions, [np.nan, 5.0])


def test_trial_rejects_prediction_truncation() -> None:
    with pytest.raises(ValueError, match="shape"):
        _evaluate(np.array([1.0]))


def test_trial_with_no_joint_valid_pairs_keeps_empty_metric_dictionary() -> None:
    _, (metrics, predictions) = _evaluate(np.full(2, np.nan))

    assert metrics == {}
    assert predictions.shape == (2,)


def test_trial_hands_n_jobs_to_the_training_call() -> None:
    model, _ = _evaluate(np.array([1.0, 2.0]), n_jobs=3)
    assert model.train.call_args.kwargs["n_jobs"] == 3
    model, _ = _evaluate(np.array([1.0, 2.0]))
    assert model.train.call_args.kwargs["n_jobs"] == 1
