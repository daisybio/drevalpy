"""Tests for per-drug literature block predictor routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat
from drevalpy.components.feature_block import FeatureBlock
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._torch_state import save_object_mapping
from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.components.predictors.single_drug_block import SingleDrugBlockPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset


class _FakeAlgorithm(LiteratureTrainingMixin):
    instances: ClassVar[list[_FakeAlgorithm]] = []

    def __init__(self) -> None:
        self.trained_drug_ids: list[str] = []
        _FakeAlgorithm.instances.append(self)

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        self.hyperparameters = hyperparameters

    def train(
        self,
        output,
        cell_line_input,
        drug_input=None,
        output_earlystopping=None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.trained_drug_ids = list(np.unique(output.drug_ids))

    def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
        return np.full(len(cell_line_ids), float(self.trained_drug_ids[0].replace("d", "")), dtype=np.float64)


class _FakePredictor(SingleDrugBlockPredictor):
    required_cell_line_blocks = ("gene_expression",)
    required_drug_blocks = ("identity",)

    @property
    def _algorithm_cls(self) -> type[_FakeAlgorithm]:
        return _FakeAlgorithm

    def _export_algorithm_state(self, algorithm: LiteratureTrainingMixin) -> dict[str, Any]:
        assert isinstance(algorithm, _FakeAlgorithm)
        return {"trained_drug_ids": algorithm.trained_drug_ids}

    def _apply_algorithm_state(self, payload: dict[str, Any]) -> _FakeAlgorithm:
        algorithm = _FakeAlgorithm()
        algorithm.trained_drug_ids = list(payload.get("trained_drug_ids", []))
        return algorithm


_FakePredictor.cell_line_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
_FakePredictor.drug_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


def _omics_batch() -> ModelInputBatch:
    return ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl2", "cl1", "cl2"]),
        drug_ids=np.array(["d1", "d1", "d2", "d2"]),
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_entity_ids=np.array(["cl1", "cl2"]),
        drug_entity_ids=np.array(["d1", "d2"]),
        cell_line_features=np.empty((0, 0), dtype=np.float32),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1, 0, 1]),
        drug_pair_idx=np.array([0, 0, 1, 1]),
        cell_line_blocks={
            "gene_expression": FeatureBlock(
                values=np.array([[0.1], [0.2]], dtype=np.float32),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
        },
        drug_blocks={
            "identity": FeatureBlock(
                values=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
            "identity_categories": FeatureBlock(
                values=np.array(["d1", "d2"]),
                format=FeatureFormat.NUMERIC_MATRIX,
            ),
        },
        training_context=TrainingContext(checkpoint_dir="checkpoints"),
    )


@pytest.fixture(autouse=True)
def _reset_fake_algorithms() -> None:
    _FakeAlgorithm.instances = []


def test_fit_trains_one_algorithm_per_drug() -> None:
    predictor = _FakePredictor()
    predictor.fit(_omics_batch())
    assert set(predictor._algorithms) == {"d1", "d2"}
    assert len(_FakeAlgorithm.instances) == 2


def test_predict_routes_rows_and_returns_nan_for_unknown_drug() -> None:
    predictor = _FakePredictor()
    batch = _omics_batch()
    predictor.fit(batch)
    predictions = predictor.predict(
        ModelInputBatch(
            cell_line_ids=np.array(["cl1", "cl2", "cl1"]),
            drug_ids=np.array(["d1", "d2", "d3"]),
            response=None,
            cell_line_entity_ids=batch.cell_line_entity_ids,
            drug_entity_ids=np.array(["d1", "d2", "d3"]),
            cell_line_features=batch.cell_line_features,
            drug_features=None,
            cell_line_pair_idx=np.array([0, 1, 0]),
            drug_pair_idx=np.array([0, 1, 2]),
            cell_line_blocks=batch.cell_line_blocks,
            drug_blocks={
                "identity": FeatureBlock(
                    values=np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    format=FeatureFormat.NUMERIC_MATRIX,
                ),
                "identity_categories": FeatureBlock(
                    values=np.array(["d1", "d2", "d3"]),
                    format=FeatureFormat.NUMERIC_MATRIX,
                ),
            },
        )
    )
    assert predictions[0] == 1.0
    assert predictions[1] == 2.0
    assert np.isnan(predictions[2])


def test_legacy_state_predicts_single_drug_batch_only() -> None:
    predictor = _FakePredictor()
    predictor.set_state(
        {
            "payload": save_object_mapping(
                {
                    "trained_drug_ids": ["d1"],
                    "predictor_hyperparameters": {},
                }
            )
        }
    )
    batch = _omics_batch()
    single = batch.subset_pairs(np.array([True, False, False, False]))
    predictions = predictor.predict(single)
    assert predictions.tolist() == [1.0]

    mixed = ModelInputBatch(
        cell_line_ids=np.array(["cl1", "cl1"]),
        drug_ids=np.array(["d1", "d2"]),
        response=None,
        cell_line_entity_ids=batch.cell_line_entity_ids,
        drug_entity_ids=batch.drug_entity_ids,
        cell_line_features=batch.cell_line_features,
        drug_features=None,
        cell_line_pair_idx=np.array([0, 0]),
        drug_pair_idx=np.array([0, 1]),
        cell_line_blocks=batch.cell_line_blocks,
        drug_blocks=batch.drug_blocks,
    )
    with pytest.raises(PredictorStateError, match="legacy state supports only single-drug batches"):
        predictor.predict(mixed)


def test_state_round_trip_preserves_per_drug_algorithms() -> None:
    predictor = _FakePredictor()
    predictor.fit(_omics_batch())
    restored = _FakePredictor()
    restored.set_state(predictor.get_state())
    assert set(restored._algorithms) == {"d1", "d2"}


def test_per_drug_checkpoint_dirs_are_isolated() -> None:
    predictor = _FakePredictor()
    batch = _omics_batch()
    seen_dirs: list[str] = []

    def _capture_train(algorithm_cls, hyperparameters, preload_state, sub, cell_lines, drugs):
        seen_dirs.append(sub.training_context.checkpoint_dir)
        algorithm = algorithm_cls()
        algorithm.configure(hyperparameters)
        algorithm.train(
            DrugResponseDataset(
                response=sub.response,
                cell_line_ids=sub.cell_line_ids,
                drug_ids=sub.drug_ids,
            ),
            cell_lines,
            drugs,
        )
        return algorithm

    with patch(
        "drevalpy.components.predictors.single_drug_block.train_fitted_algorithm",
        side_effect=_capture_train,
    ):
        predictor.fit(batch)
    assert len(seen_dirs) == 2
    assert len(set(seen_dirs)) == 2
    assert all(Path(path).as_posix().startswith("checkpoints/drug_") for path in seen_dirs)
