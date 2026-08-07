"""Tests for FeatureDatasetBlockPredictor lifecycle hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_dataset_block import FeatureDatasetBlockPredictor
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import FeatureDataset


class _FakeAlgorithm(LiteratureTrainingMixin):
    def __init__(self) -> None:
        self.trained = False
        self.marker = 0.0

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return {"marker": 1.0}

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters
        self.marker = float(hyperparameters.get("marker", 1.0))

    def train(
        self,
        output,
        cell_line_input,
        drug_input=None,
        output_earlystopping=None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.trained = True
        _ = output, cell_line_input, drug_input, output_earlystopping, model_checkpoint_dir

    def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
        return np.full(len(cell_line_ids), self.marker, dtype=np.float64)


class _FakePredictor(FeatureDatasetBlockPredictor):
    required_cell_line_blocks = ("gene_expression",)
    required_drug_blocks = ("fingerprints",)

    @property
    def _algorithm_cls(self) -> type[LiteratureTrainingMixin]:
        return _FakeAlgorithm

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        assert isinstance(algorithm, _FakeAlgorithm)
        return {"trained": algorithm.trained, "marker": algorithm.marker}

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> LiteratureTrainingMixin:
        algorithm = _FakeAlgorithm()
        algorithm.trained = bool(payload.get("trained", False))
        algorithm.marker = float(payload.get("marker", 0.0))
        return algorithm

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        return dict(_FakeAlgorithm.get_default_hyperparameters())


_FakePredictor.cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
_FakePredictor.drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


class _StrictFittedPredictor(_FakePredictor):
    def _is_algorithm_fitted(self, algorithm: LiteratureTrainingMixin | None) -> bool:
        return algorithm is not None and isinstance(algorithm, _FakeAlgorithm) and algorithm.trained


class _ValidatedPayloadPredictor(_FakePredictor):
    def _validate_restored_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("trained") is not True:
            raise PredictorStateError("payload missing trained flag")


def _batch() -> ModelInputBatch:
    return ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2"]),
        drug_ids=np.array(["d1", "d2"]),
        response=np.array([1.0, 2.0]),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1]),
        drug_pair_idx=np.array([0, 1]),
        cell_line_blocks={
            "gene_expression": FeatureBlock(
                values=np.array([[0.1], [0.2]], dtype=np.float32),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
        },
        drug_blocks={
            "fingerprints": FeatureBlock(
                values=np.array([[1.0], [2.0]], dtype=np.float32),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
        },
        training_context=TrainingContext(checkpoint_dir=Path("checkpoints")),
    )


def test_fit_predict_and_state_round_trip() -> None:
    predictor = _FakePredictor({"marker": 3.5})
    assert not predictor.is_fitted()
    predictor.fit(_batch())
    assert predictor.is_fitted()
    predictions = predictor.predict(_batch())
    assert predictions.tolist() == [3.5, 3.5]

    restored = _FakePredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted()
    assert restored.predict(_batch()).tolist() == [3.5, 3.5]


def test_set_state_requires_payload_blob() -> None:
    predictor = _FakePredictor()
    with pytest.raises(PredictorStateError, match="payload byte blob"):
        predictor.set_state({})


def test_is_algorithm_fitted_hook() -> None:
    predictor = _StrictFittedPredictor()
    assert not predictor.is_fitted()
    predictor._algorithm = _FakeAlgorithm()
    assert not predictor.is_fitted()
    predictor._algorithm.trained = True
    assert predictor.is_fitted()


def test_validate_restored_payload_hook() -> None:
    predictor = _FakePredictor()
    predictor.fit(_batch())
    state = predictor.get_state()
    bad = _ValidatedPayloadPredictor()
    # Corrupt by exporting an untrained algorithm payload shape via set_state after empty fit path
    from drevalpy.components.predictors.literature._torch_state import save_object_mapping

    with pytest.raises(PredictorStateError, match="trained flag"):
        bad.set_state(
            {
                "payload": save_object_mapping(
                    {
                        "trained": False,
                        "marker": 1.0,
                        "predictor_hyperparameters": {},
                    }
                )
            }
        )
    _ = state


def test_materialize_inputs_override() -> None:
    class _CustomMaterialize(_FakePredictor):
        def _materialize_inputs(self, batch: ModelInputBatch):
            cell_lines, drugs = super()._materialize_inputs(batch)
            assert isinstance(cell_lines, FeatureDataset)
            assert drugs is not None
            return cell_lines, drugs

    predictor = _CustomMaterialize()
    predictor.fit(_batch())
    assert predictor.is_fitted()


def test_set_engine_preload_state() -> None:
    predictor = _FakePredictor()
    predictor.set_engine_preload_state({"preloaded": True})
    assert predictor._engine_preload_state == {"preloaded": True}
