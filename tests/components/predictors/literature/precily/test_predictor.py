"""Tests for the Precily predictor and its feed-forward regressor."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.predictors.literature.precily.model_utils import PrecilyNetwork
from drevalpy.components.predictors.literature.precily.predictor import PrecilyPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.experiment._tracking import tracking_run
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from tests._fake_wandb import install_fake_wandb
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_state_round_trip,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    finite_predictions,
    seed_literature,
    trainable_parameter_vector,
    two_by_two_batch,
)


def test_precily_predictor_registry_name() -> None:
    assert get_predictor("precily") is PrecilyPredictor


class TestTinyPrecilyFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_updates_parameters_and_round_trips_state(self, seed: int) -> None:
        batch = two_by_two_batch(
            sample_blocks={
                "pathways": numeric_feature_block(
                    np.array([[0.1, 0.4, 0.8, 1.2], [1.3, 0.7, -0.2, 0.5]], dtype=np.float32)
                )
            },
            drug_blocks={
                "smilesvec": numeric_feature_block(
                    np.array([[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.4, 0.6]], dtype=np.float32)
                )
            },
        )
        hyperparameters = {"epochs": 1, "batch_size": 4, "dropout": 0.0, "learning_rate": 0.01}
        seed_literature(seed)
        pristine = PrecilyPredictor(hyperparameters)
        initial = trainable_parameter_vector(pristine._build_model(8, 0.0))

        seed_literature(seed)
        predictor = PrecilyPredictor(hyperparameters=hyperparameters)
        predictor.fit(batch)
        assert predictor._model is not None
        assert_trainable_parameters_changed(initial, predictor._model)
        expected = finite_predictions(predictor, batch)
        assert_training_predictions("precily", seed, expected)
        assert_frozen_reference("precily", seed)

        assert_state_round_trip(predictor, batch, expected)


def test_set_state_requires_the_model_state() -> None:
    payload = save_trusted_mapping(
        {"predictor_hyperparameters": PrecilyPredictor.get_default_hyperparameters(), "input_dim": 8}
    )

    with pytest.raises(PredictorStateError, match="missing input_dim or model_state"):
        PrecilyPredictor().set_state({"payload": payload})


def test_epoch_losses_are_logged_not_printed_and_sent_to_the_tracking_run(caplog, capsys, monkeypatch) -> None:
    fake = install_fake_wandb(monkeypatch)
    batch = two_by_two_batch(
        sample_blocks={"pathways": numeric_feature_block(np.ones((2, 4), dtype=np.float32))},
        drug_blocks={"smilesvec": numeric_feature_block(np.ones((2, 4), dtype=np.float32))},
    )

    with caplog.at_level("INFO", logger="drevalpy"), tracking_run("proj", name="r"):
        PrecilyPredictor({"epochs": 2, "batch_size": 4, "dropout": 0.0, "learning_rate": 0.01}).fit(batch)

    assert [record.getMessage()[:25] for record in caplog.records if "Training Loss" in record.getMessage()] == [
        "Precily: Epoch [1/2] Trai",
        "Precily: Epoch [2/2] Trai",
    ]
    assert capsys.readouterr().out == ""
    assert [step["epoch"] for step in fake.runs[0].logged] == [1, 2]
    assert all(np.isfinite(step["train_loss"]) for step in fake.runs[0].logged)


def test_network_predicts_one_deterministic_differentiable_scalar_per_row() -> None:
    network = PrecilyNetwork(input_dim=8)
    features = torch.randn(3, 8)

    network.eval()
    with torch.no_grad():
        first = network(features)
        second = network(features)
        single = network(features[:1])
    assert first.shape == (3,)
    assert torch.allclose(first, second)
    assert single.shape == (1,)

    network.train()
    network(features).sum().backward()
    assert network.net[0].weight.grad is not None


def test_network_reproduces_the_reference_layer_widths_and_dropout() -> None:
    network = PrecilyNetwork(input_dim=7)
    linear_shapes = [(layer.in_features, layer.out_features) for layer in network.net if isinstance(layer, nn.Linear)]

    assert linear_shapes == [(7, 1429), (1429, 512), (512, 140), (140, 200), (200, 1)]
    assert {layer.p for layer in network.net if isinstance(layer, nn.Dropout)} == {0.1}
    custom = PrecilyNetwork(input_dim=4, dropout=0.42)
    assert {layer.p for layer in custom.net if isinstance(layer, nn.Dropout)} == {0.42}
