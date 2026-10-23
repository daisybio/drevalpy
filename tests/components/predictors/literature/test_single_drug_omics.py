"""Tests for the per-drug three-omic predictor shell shared by MOLIR and SuperFELTR.

``literature/_single_drug_omics.py`` holds the class variables, per-drug routing and
``algorithms`` state layout that both single-drug predictors inherit.
"""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.literature._omic_helpers import OMIC_BLOCK_NAMES
from drevalpy.components.predictors.literature._single_drug_omics import SingleDrugOmicsPredictor
from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.utils.torch_io import save_trusted_mapping


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
