"""Tests for the per-drug three-omic predictor shell shared by MOLIR and SuperFELTR.

``literature/_single_drug_omics.py`` holds the class variables, per-drug routing and
``algorithms`` state layout that both single-drug predictors inherit.
"""

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
class TestSingleDrugOmicsShell:
    def test_both_predictors_share_the_shell(self, predictor_cls: type[SingleDrugOmicsPredictor]) -> None:
        assert issubclass(predictor_cls, SingleDrugOmicsPredictor)
        assert predictor_cls.required_sample_blocks == OMIC_BLOCK_NAMES

    def test_an_unfitted_predictor_has_no_state(self, predictor_cls: type[SingleDrugOmicsPredictor]) -> None:
        predictor = predictor_cls()

        assert predictor.is_fitted() is False
        assert predictor.get_state() == {}

    def test_a_payload_missing_a_required_entry_is_a_predictor_state_error(
        self, predictor_cls: type[SingleDrugOmicsPredictor]
    ) -> None:
        state = {"algorithms": {"d1": save_trusted_mapping({})}, "predictor_hyperparameters": {}}

        with pytest.raises(PredictorStateError, match="payload for 'd1' is missing"):
            predictor_cls().set_state(state)

    def test_a_failed_restore_leaves_the_predictor_untouched(
        self, predictor_cls: type[SingleDrugOmicsPredictor]
    ) -> None:
        predictor = predictor_cls({"epochs": 7})
        state = {"algorithms": {"d1": save_trusted_mapping({})}, "predictor_hyperparameters": {"epochs": 1}}

        with pytest.raises(PredictorStateError):
            predictor.set_state(state)

        assert predictor._hyperparameters["epochs"] == 7
        assert predictor.is_fitted() is False


class _RecordingPredictor(SingleDrugOmicsPredictor):
    """Fits ``"model-<drug>"`` per drug, except that ``d2`` gets no model."""

    def _fit_single_drug(self, drug_id: str, batch: Any) -> str | None:
        return None if drug_id == "d2" else f"model-{drug_id}"

    def _predict_single_drug(self, drug_id: str, model: Any, batch: Any) -> Any:
        raise NotImplementedError

    def _serialize_model(self, drug_id: str, model: Any) -> bytes:
        raise NotImplementedError

    def _restore_model(self, payload: dict[str, Any]) -> Any:
        raise NotImplementedError


class TestPerDrugFit:
    def test_the_models_are_collected_from_what_each_drug_returns(self) -> None:
        predictor = _RecordingPredictor()

        predictor._fit(_batch())

        assert predictor._models == {"d1": "model-d1"}

    def test_every_drug_records_its_feature_names_even_without_a_model(self) -> None:
        predictor = _RecordingPredictor()

        predictor._fit(_batch())

        assert set(predictor._feature_names) == {"d1", "d2"}

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_the_drugs_are_fitted_through_the_shared_fan_out(
        self, monkeypatch: pytest.MonkeyPatch, n_jobs: int
    ) -> None:
        from drevalpy.components.predictors.literature import _single_drug_omics

        seen: list[int] = []
        real = _single_drug_omics.run_per_drug
        monkeypatch.setattr(
            _single_drug_omics, "run_per_drug", lambda fit, tasks, jobs: seen.append(jobs) or real(fit, tasks, jobs)
        )
        batch = replace(_batch(), training_context=TrainingContext(n_jobs=n_jobs))

        _RecordingPredictor()._fit(batch)

        assert seen == [n_jobs]
