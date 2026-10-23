"""Smoke test mirror for molir predictor package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.predictors.literature.molir.predictor import MOLIRPredictor
from drevalpy.components.predictors.literature.molir.utils import MOLIModel
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors.literature._helpers import (
    assert_corrected_training_result,
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    seed_literature,
    single_drug_omics_batch,
    trainable_parameter_vector,
)


def _loss_model(gamma: float) -> MOLIModel:
    hyperparameters = dict(MOLIRPredictor.get_default_hyperparameters())
    hyperparameters.update(
        {
            "gamma": gamma,
            "mini_batch": 4,
            "h_dim1": 2,
            "h_dim2": 2,
            "h_dim3": 2,
            "dropout_rate": 0.0,
            "epochs": 1,
        }
    )
    model = MOLIModel(hyperparameters, 2, 2, 2)
    model.positive_range = 0.1
    model.negative_range = 0.5
    return model


def test_gamma_weights_triplet_loss_and_its_embedding_gradient() -> None:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.25, 0.0], [0.1, 0.8], [0.9, 0.2]],
        dtype=torch.float32,
    )
    predictions = torch.tensor([[0.2], [0.4], [0.6], [0.8]], dtype=torch.float32)
    response = torch.tensor([0.0, 0.05, 1.0, 1.05], dtype=torch.float32)

    unweighted_embeddings = embeddings.clone().requires_grad_()
    unweighted_loss = _loss_model(0.0)._compute_loss(unweighted_embeddings, predictions, response)
    unweighted_loss.backward()
    weighted_embeddings = embeddings.clone().requires_grad_()
    weighted_loss = _loss_model(1.0)._compute_loss(weighted_embeddings, predictions, response)
    weighted_loss.backward()

    assert weighted_loss > unweighted_loss
    torch.testing.assert_close(unweighted_embeddings.grad, torch.zeros_like(unweighted_embeddings))
    assert torch.count_nonzero(weighted_embeddings.grad).item() > 0


def test_molir_predictor_registry_name() -> None:
    assert get_predictor("molir") is MOLIRPredictor


class TestTinyMOLIRFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state_without_feature_names(self, seed: int, tmp_path) -> None:
        hyperparameters = {
            "epochs": 1,
            # One full batch: BatchNorm over two rows amplifies platform rounding to ~1e-2.
            "mini_batch": 6,
            "h_dim1": 4,
            "h_dim2": 4,
            "h_dim3": 4,
            "dropout_rate": 0.0,
        }
        batch = single_drug_omics_batch(tmp_path)
        seed_literature(seed)
        pristine = MOLIRPredictor(hyperparameters)._build_model((6, 5, 4))
        initial = trainable_parameter_vector(pristine)

        seed_literature(seed)
        predictor = MOLIRPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        fitted = predictor._models["d1"]
        assert_trainable_parameters_changed(initial, fitted)
        assert fitted.checkpoint_callback is not None
        expected = np.asarray(predictor.predict(batch))
        assert np.isfinite(expected).all()
        assert np.isfinite(np.mean((expected - batch.response) ** 2))
        # The gamma correction supersedes the v2 baseline in training.json for MOLIR.
        assert_corrected_training_result(
            "molir_gamma_weighting",
            seed,
            expected,
            fitted.checkpoint_callback.best_model_score,
        )
        assert_frozen_reference("molir", seed)

        restored = MOLIRPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(expected, restored.predict(batch), rtol=1e-5, atol=1e-5)
