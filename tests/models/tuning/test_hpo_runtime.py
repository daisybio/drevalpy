"""Tests for HPO trial construction with immutable ModelSpec inputs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from drevalpy.models import ModelSpec
from drevalpy.models.tuning.hpo_runtime import (
    _construct_trial_model,
)


def test_construct_trial_model_keeps_spec_identity() -> None:
    spec = ModelSpec.from_name("ElasticNet")
    trial_model = _construct_trial_model(spec, {"predictor.elasticNet.alpha": 0.5})

    assert trial_model.spec.name == spec.name
    assert trial_model.hyperparameters["alpha"] == 0.5


@pytest.mark.parametrize("supplied", [False, True])
def test_trial_uses_one_training_call_and_keeps_nan_scope_alignment(tmp_path, supplied) -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity

    dataset = synthetic_mudataset_identity()
    scopes = lso_split_masks()
    model = MagicMock()
    model.predict.return_value = np.array([np.nan, 5.0])
    metrics, predictions = _mu_evaluate_trial_all_metrics(
        model,
        mudataset=dataset,
        train_scope=scopes.train,
        val_scope=scopes.test,
        early_stopping_scope=None,
        model_checkpoint_dir=tmp_path if supplied else None,
        trial_number=3,
    )
    model.train.assert_called_once()
    assert model.train.call_args.args == (dataset, scopes.train)
    assert model.train.call_args.kwargs["early_stopping_scope"] is None
    if supplied:
        assert model.train.call_args.kwargs["model_checkpoint_dir"] == tmp_path / "trial_3"
    assert metrics["MSE"] == 1.0
    assert np.isnan(metrics["R^2"])
    np.testing.assert_array_equal(predictions, [np.nan, 5.0])


def test_trial_rejects_prediction_truncation() -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity

    model = MagicMock()
    model.predict.return_value = np.array([1.0])
    scopes = lso_split_masks()
    with pytest.raises(ValueError, match="shape"):
        _mu_evaluate_trial_all_metrics(
            model,
            mudataset=synthetic_mudataset_identity(),
            train_scope=scopes.train,
            val_scope=scopes.test,
            early_stopping_scope=None,
            model_checkpoint_dir=None,
        )


def test_trial_with_no_joint_valid_pairs_keeps_empty_metric_dictionary() -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity

    model = MagicMock()
    model.predict.return_value = np.full(2, np.nan)
    scopes = lso_split_masks()
    metrics, predictions = _mu_evaluate_trial_all_metrics(
        model,
        mudataset=synthetic_mudataset_identity(),
        train_scope=scopes.train,
        val_scope=scopes.test,
        early_stopping_scope=None,
        model_checkpoint_dir=None,
    )
    assert metrics == {}
    assert predictions.shape == (2,)


def test_trial_hands_n_jobs_to_the_training_call() -> None:
    import numpy as np

    from drevalpy.models.tuning.hpo_runtime import _mu_evaluate_trial_all_metrics
    from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity

    model = MagicMock()
    model.predict.return_value = np.array([1.0, 2.0])
    scopes = lso_split_masks()
    kwargs = {
        "mudataset": synthetic_mudataset_identity(),
        "train_scope": scopes.train,
        "val_scope": scopes.test,
        "early_stopping_scope": None,
        "model_checkpoint_dir": None,
    }

    _mu_evaluate_trial_all_metrics(model, n_jobs=3, **kwargs)
    assert model.train.call_args.kwargs["n_jobs"] == 3
    _mu_evaluate_trial_all_metrics(model, **kwargs)
    assert model.train.call_args.kwargs["n_jobs"] == 1
