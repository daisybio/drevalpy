"""Tests for the per-drug three-omic predictor shell shared by MOLIR and SuperFELTR."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature._omic_helpers import OMIC_BLOCK_NAMES
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors.literature.test_omic_helpers import _batch


@pytest.mark.parametrize("predictor_cls", [MOLIRPredictor, SuperFELTRPredictor])
def test_the_shell_is_shared_and_a_failed_restore_leaves_the_predictor_untouched(
    predictor_cls: type[SingleDrugOmicsPredictor],
) -> None:
    assert issubclass(predictor_cls, SingleDrugOmicsPredictor)
    assert predictor_cls.required_sample_blocks == OMIC_BLOCK_NAMES
    assert predictor_cls().is_fitted() is False
    assert predictor_cls().get_state() == {}

    predictor = predictor_cls({"epochs": 7})
    state = {"algorithms": {"d1": save_trusted_mapping({})}, "predictor_hyperparameters": {"epochs": 1}}
    with pytest.raises(PredictorStateError, match="payload for 'd1' is missing"):
        predictor.set_state(state)

    assert predictor._hyperparameters["epochs"] == 7
    assert predictor.is_fitted() is False


class _RecordingPredictor(SingleDrugOmicsPredictor):
    def _fit_single_drug(self, drug_id: str, batch: Any) -> str | None:
        return None if drug_id == "d2" else f"model-{drug_id}"

    def _predict_single_drug(self, drug_id: str, model: Any, batch: Any) -> Any:
        raise NotImplementedError

    def _serialize_model(self, drug_id: str, model: Any) -> bytes:
        raise NotImplementedError

    def _restore_model(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError


def test_models_are_collected_per_drug_and_every_drug_records_feature_names() -> None:
    predictor = _RecordingPredictor()

    predictor._fit(_batch())

    assert predictor._models == {"d1": "model-d1"}
    assert set(predictor._feature_names) == {"d1", "d2"}


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_drugs_are_fitted_through_the_shared_fan_out(monkeypatch: pytest.MonkeyPatch, n_jobs: int) -> None:
    from drevalpy.components.predictors.literature import _single_drug_omics

    seen: list[int] = []
    real = _single_drug_omics.run_per_drug
    monkeypatch.setattr(
        _single_drug_omics, "run_per_drug", lambda fit, tasks, jobs: seen.append(jobs) or real(fit, tasks, jobs)
    )

    _RecordingPredictor()._fit(replace(_batch(), training_context=TrainingContext(n_jobs=n_jobs)))

    assert seen == [n_jobs]
