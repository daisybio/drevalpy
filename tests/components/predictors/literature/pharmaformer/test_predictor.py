"""Smoke test mirror for pharmaformer predictor package."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.pharmaformer.predictor import (
    PharmaFormerPredictor,
    _build_combined_model,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    seed_literature,
    trainable_parameter_vector,
)


def test_pharmaformer_predictor_registry_name() -> None:
    assert get_predictor("pharmaFormer") is PharmaFormerPredictor


def test_set_state_requires_the_model_state() -> None:
    payload = save_trusted_mapping(
        {"predictor_hyperparameters": PharmaFormerPredictor.get_default_hyperparameters(), "gene_input_size": 6}
    )

    with pytest.raises(PredictorStateError, match="missing gene_input_size or model_state"):
        PharmaFormerPredictor().set_state({"payload": payload})


def _trainable_batch(tmp_path) -> ModelInputBatch:
    rng = np.random.default_rng(987)
    cell_ids = np.array(["train-1", "train-2", "val-1", "val-2"])
    drug_ids = np.array(["d1", "d2"])
    training = ResponseBatch(
        response=np.array([-1.0, 0.5, 1.5, 3.0], dtype=np.float32),
        sample_ids=np.array(["train-1", "train-1", "train-2", "train-2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )
    validation = ResponseBatch(
        response=np.array([0.25, 2.25], dtype=np.float32),
        sample_ids=np.array(["val-1", "val-2"]),
        drug_ids=np.array(["d1", "d2"]),
    )
    return ModelInputBatch.from_response(
        training,
        sample=FeaturizerResult.from_blocks(
            {"gene_expression": numeric_feature_block(rng.normal(size=(4, 6)).astype(np.float32))}, cell_ids
        ),
        drug=FeaturizerResult.from_blocks(
            {"bpe_smiles": numeric_feature_block(rng.normal(size=(2, 128)).astype(np.float32))}, drug_ids
        ),
        sample_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=validation,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )


class TestTinyPharmaFormerFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int, tmp_path) -> None:
        hyperparameters = {
            "epochs": 1,
            "patience": 1,
            "batch_size": 4,
            "lr": 0.001,
            "dropout": 0.0,
            "gene_hidden_size": 16,
            "drug_hidden_size": 16,
            "feature_dim": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
        }
        batch = _trainable_batch(tmp_path)
        seed_literature(seed)
        device = PharmaFormerPredictor(hyperparameters)._device
        initial = trainable_parameter_vector(_build_combined_model(6, hyperparameters, device))

        seed_literature(seed)
        predictor = PharmaFormerPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)
        expected = np.asarray(predictor.predict(batch))
        assert np.isfinite(expected).all()
        assert np.isfinite(np.mean((expected - batch.response) ** 2))
        assert_training_predictions("pharmaformer", seed, expected)
        assert_frozen_reference("pharmaformer", seed)

        restored = PharmaFormerPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(expected, restored.predict(batch), rtol=1e-5, atol=1e-5)
