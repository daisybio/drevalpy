"""Tests for the SRMF predictor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drevalpy.components.predictors.literature.srmf.predictor import SRMFPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.utils.torch_io import save_trusted_mapping
from tests.components.predictors.literature._helpers import (
    assert_frozen_reference,
    assert_state_round_trip,
    assert_training_predictions,
    finite_predictions,
    seed_literature,
    two_by_two_batch,
)


def test_srmf_predictor_registry_name() -> None:
    assert get_predictor("srmf") is SRMFPredictor


def test_set_state_rejects_an_invalid_blob_and_missing_latent_factors() -> None:
    with pytest.raises(PredictorStateError):
        SRMFPredictor().set_state({"payload": b"invalid"})
    payload = save_trusted_mapping({"predictor_hyperparameters": SRMFPredictor.get_default_hyperparameters()})
    with pytest.raises(PredictorStateError, match="missing best_u"):
        SRMFPredictor().set_state({"payload": payload})


def test_unseen_entity_pairs_fall_back_to_the_training_mean() -> None:
    predictor = SRMFPredictor({"K": 2})
    predictor._best_u = pd.DataFrame([[2.0, 3.0]], index=["d1"])
    predictor._best_v = pd.DataFrame([[5.0, 7.0]], index=["c1"])
    predictor._training_mean = 4.0
    batch = SimpleNamespace(
        drug_ids=np.array(["d1", "missing-drug", "d1", "missing-drug"]),
        sample_ids=np.array(["c1", "c1", "missing-cell", "missing-cell"]),
    )

    predictions = predictor._predict(batch)

    np.testing.assert_allclose(predictions, [31.0, 4.0, 4.0, 4.0])


class TestTinySRMFFit:
    pytestmark = pytest.mark.slow

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_learns_finite_factors_and_round_trips_state(self, seed: int) -> None:
        seed_literature(seed)
        batch = two_by_two_batch(
            sample_blocks={
                "gene_expression": numeric_feature_block(
                    np.array([[0.1, 0.4, 0.9], [1.1, -0.2, 0.5]], dtype=np.float32)
                )
            },
            drug_blocks={
                "fingerprints": numeric_feature_block(
                    np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
                )
            },
        )
        predictor = SRMFPredictor(hyperparameters={"K": 2, "max_iter": 3, "n_features": 4, "seed": seed})
        predictor.fit(batch)
        factors = np.concatenate([predictor._best_u.to_numpy().ravel(), predictor._best_v.to_numpy().ravel()])
        assert np.isfinite(factors).all()
        expected = finite_predictions(predictor, batch)
        assert_training_predictions("srmf", seed, expected)
        assert_frozen_reference("srmf", seed)

        assert_state_round_trip(predictor, batch, expected)
