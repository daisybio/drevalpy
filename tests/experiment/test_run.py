"""Orchestration tests for experiment_run."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.experiment.run import _normalize_baselines, drug_response_experiment_impl
from drevalpy.models._model_lookup import get_model_class


def test_normalize_baselines_injects_naive_mean_effects() -> None:
    ridge = get_model_class("Ridge")
    normalized = _normalize_baselines([ridge])
    names = {m.get_model_name() for m in normalized}
    assert "NaiveMeanEffectsPredictor" in names
    assert "Ridge" in names


def test_normalize_baselines_none_defaults_to_naive_mean_effects() -> None:
    normalized = _normalize_baselines(None)
    assert len(normalized) == 1
    assert normalized[0].get_model_name() == "NaiveMeanEffectsPredictor"


@patch("drevalpy.experiment.run.consolidate_single_drug_model_predictions_impl")
@patch("drevalpy.experiment.run._run_one_model")
@patch("drevalpy.experiment.run.make_model_list", return_value={"NaivePredictor": None})
@patch("drevalpy.experiment.run.prepare_response_splits_impl", return_value=1)
def test_drug_response_experiment_runs_consolidation(
    _prepare_splits: MagicMock,
    _make_model_list: MagicMock,
    _run_one_model: MagicMock,
    consolidate: MagicMock,
    tmp_path,
) -> None:
    response = DrugResponseDataset(
        response=np.array([1.0, 2.0]),
        cell_line_ids=np.array(["A", "B"]),
        drug_ids=np.array(["d1", "d1"]),
        dataset_name="Toy_Data",
    )
    model = get_model_class("NaivePredictor")
    drug_response_experiment_impl(
        models=[model],
        response_data=response,
        baselines=[],
        n_cv_splits=1,
        path_out=str(tmp_path / "out"),
        path_data=str(tmp_path / "data"),
        hyperparameter_tuning=False,
    )
    _run_one_model.assert_called_once()
    consolidate.assert_called_once()


@patch("drevalpy.experiment.run._run_one_model")
@patch("drevalpy.experiment.run.make_model_list", return_value={"NaivePredictor": None})
@patch("drevalpy.experiment.run.prepare_response_splits_impl", return_value=1)
def test_drug_response_experiment_skips_baseline_stress_when_only_baselines(
    _prepare_splits: MagicMock,
    _make_model_list: MagicMock,
    run_one_model: MagicMock,
    tmp_path,
) -> None:
    """Baselines still run through _run_one_model; stress tests are gated inside the fold loop."""
    response = DrugResponseDataset(
        response=np.array([1.0]),
        cell_line_ids=np.array(["A"]),
        drug_ids=np.array(["d1"]),
        dataset_name="Toy_Data",
    )
    baseline = get_model_class("NaiveMeanEffectsPredictor")
    with patch("drevalpy.experiment.run.consolidate_single_drug_model_predictions_impl"):
        drug_response_experiment_impl(
            models=[],
            baselines=[baseline],
            response_data=response,
            n_cv_splits=1,
            path_out=str(tmp_path / "out"),
            hyperparameter_tuning=False,
        )
    assert run_one_model.call_count == 1
