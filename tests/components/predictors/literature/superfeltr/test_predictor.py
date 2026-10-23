"""Smoke test mirror for superfeltr predictor package."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature.superfeltr.predictor import SuperFELTRPredictor
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_training_checkpoint_score,
    assert_training_predictions,
    seed_literature,
    single_drug_omics_batch,
)


def test_superfeltr_predictor_registry_name() -> None:
    assert get_predictor("superfeltr") is SuperFELTRPredictor


class TestTinySuperFELTRFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fits_triplet_batches_and_round_trips_state_without_feature_names(self, seed: int, tmp_path) -> None:
        seed_literature(seed)
        batch = single_drug_omics_batch(tmp_path)
        predictor = SuperFELTRPredictor(
            hyperparameters={
                "epochs": 1,
                "mini_batch": 2,
                "out_dim_expr_encoder": 3,
                "out_dim_mutation_encoder": 3,
                "out_dim_cnv_encoder": 2,
                "dropout_rate": 0.0,
            }
        )
        predictor.fit(batch)
        fitted = predictor._drug_models["d1"]
        assert fitted.best_checkpoint is not None
        assert np.isfinite(float(fitted.best_checkpoint.best_model_score))
        assert_training_checkpoint_score("superfeltr", seed, fitted.best_checkpoint.best_model_score)
        expected = np.asarray(predictor.predict(batch))
        assert np.isfinite(expected).all()
        assert np.isfinite(np.mean((expected - batch.response) ** 2))
        assert_training_predictions("superfeltr", seed, expected)
        assert_frozen_reference("superfeltr", seed)

        restored = SuperFELTRPredictor()
        restored.set_state(predictor.get_state())
        np.testing.assert_allclose(expected, restored.predict(batch), rtol=1e-5, atol=1e-5)
