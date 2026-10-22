"""Smoke test mirror for precily predictor package."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.precily.predictor import PrecilyPredictor
from drevalpy.experiment._tracking import tracking_run
from drevalpy.registry._builtins import ensure_predictor_registered
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from tests._fake_wandb import install_fake_wandb
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_trainable_parameters_changed,
    assert_training_predictions,
    seed_literature,
    trainable_parameter_vector,
    two_by_two_batch,
)


def test_precily_predictor_registry_name() -> None:
    ensure_predictor_registered("precily")
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
        expected = np.asarray(predictor.predict(batch))
        assert np.isfinite(expected).all()
        assert np.isfinite(np.mean((expected - batch.response) ** 2))
        assert_training_predictions("precily", seed, expected)
        assert_frozen_reference("precily", seed)

        restored = PrecilyPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(expected, restored.predict(batch), rtol=1e-5, atol=1e-5)


def test_epochs_are_logged_to_the_active_tracking_run(monkeypatch) -> None:
    fake = install_fake_wandb(monkeypatch)
    batch = two_by_two_batch(
        sample_blocks={"pathways": numeric_feature_block(np.ones((2, 4), dtype=np.float32))},
        drug_blocks={"smilesvec": numeric_feature_block(np.ones((2, 4), dtype=np.float32))},
    )

    with tracking_run("proj", name="r"):
        PrecilyPredictor({"epochs": 2, "batch_size": 4, "dropout": 0.0, "learning_rate": 0.01}).fit(batch)

    assert [step["epoch"] for step in fake.runs[0].logged] == [1, 2]
    assert all(np.isfinite(step["train_loss"]) for step in fake.runs[0].logged)
