"""Focused HPO lifecycle tests for ModelSpec based tuning."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.errors import DrEvalError
from drevalpy.models import ModelSpec
from drevalpy.models.config import from_spec
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.hpo import HPOTrialsFailedError, hpam_tune
from drevalpy.types import SplitMask
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def _tune(spec: ModelSpec, n_trials: int, **kwargs):
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    scope = SplitMask.from_pairs(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), shape=dataset.response_matrix.shape)
    return hpam_tune(
        model_spec=spec,
        mudataset=dataset,
        train_scope=scope,
        val_scope=scope,
        early_stopping_scope=None,
        metric="RMSE",
        hpo_config=HPOConfig.from_metric("RMSE", n_trials=n_trials),
        **kwargs,
    )


def test_zero_trials_returns_spec_defaults() -> None:
    spec = ModelSpec.from_name("ElasticNet")

    best, trials = _tune(spec, 0)

    assert best == spec.default_hyperparameters
    assert trials == []


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics", return_value=({"RMSE": 0.1}, np.zeros(4)))
def test_tuning_passes_model_spec_and_n_jobs_to_each_trial(mock_evaluate) -> None:
    best, _ = _tune(ModelSpec.from_name("ElasticNet"), 2, n_jobs=4)

    assert "alpha" in best
    assert mock_evaluate.call_count == 2
    assert [call.kwargs["n_jobs"] for call in mock_evaluate.call_args_list] == [4, 4]


@patch("drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics", side_effect=RuntimeError("boom"))
def test_all_failed_trials_raise_without_default_fallback(mock_evaluate) -> None:
    with pytest.raises(HPOTrialsFailedError, match="All 2 hyperparameter trials failed"):
        _tune(ModelSpec.from_name("ElasticNet"), 2)
    assert issubclass(HPOTrialsFailedError, DrEvalError)
    assert issubclass(HPOTrialsFailedError, RuntimeError)


@patch(
    "drevalpy.models.tuning.hpo._mu_evaluate_trial_all_metrics",
    side_effect=[({"RMSE": 1.0 - 0.1 * i}, np.zeros(4)) for i in range(4)],
)
def test_best_hyperparameters_carry_the_decoded_pow2_value_the_trials_trained_with(mock_evaluate) -> None:
    spec = ModelSpec("FingerprintElasticNet", from_spec("pca[gene_expression]:fingerprints:elasticNet"))

    best, trials = _tune(spec, 4)

    best_trial_params, _, _ = trials[-1]
    assert best["n_bits"] == best_trial_params["drug_featurizer.fingerprints.n_bits"]
    assert best["n_bits"] in {512, 1024, 2048}
    assert spec.resolve(best).values["drug_featurizer.fingerprints.n_bits"] == best["n_bits"]
