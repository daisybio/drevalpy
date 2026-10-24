"""Tests for the naive mean-effects predictor."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.naive.effects import NaiveMeanEffectsPredictor
from tests.components.predictors.naive._helpers import naive_batch, one_hot

CELL = one_hot(["cl1", "cl2"], ["cl1", "cl2"])
DRUGS = one_hot(["d1", "d2"], ["d1", "d2"])


def _pair_batch(sample_blocks, response=None, cell_idx=(0, 1), drug_idx=(0, 1), cell=CELL):
    return naive_batch(
        response=response,
        sample_features=cell,
        drug_features=DRUGS,
        sample_pair_idx=np.array(cell_idx, dtype=np.int64),
        drug_pair_idx=np.array(drug_idx, dtype=np.int64),
        sample_blocks=sample_blocks,
    )


@pytest.mark.parametrize("tissue", [None, np.empty((2, 0))], ids=["no-tissue-block", "empty-tissue-block"])
def test_without_tissue_decomposes_cell_and_drug_and_round_trips(tissue) -> None:
    blocks = {"identity": CELL} | ({} if tissue is None else {"tissue": tissue})
    response = np.array([1.0, 2.0, 5.0, 8.0])
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(_pair_batch(blocks, response, (0, 0, 1, 1), (0, 1, 0, 1)))
    batch = _pair_batch(blocks)
    mean = response.mean()
    expected = [
        mean + (np.mean([1.0, 2.0]) - mean) + (np.mean([1.0, 5.0]) - mean),
        mean + (np.mean([5.0, 8.0]) - mean) + (np.mean([2.0, 8.0]) - mean),
    ]
    restored = NaiveMeanEffectsPredictor()
    restored.set_state(predictor.get_state())

    np.testing.assert_allclose(predictor.predict(batch), expected)
    np.testing.assert_allclose(restored.predict(batch), expected)


def test_with_tissue_blocks_adds_tissue_then_within_tissue_cell_effects() -> None:
    cell = one_hot(["cl1", "cl2", "cl3"], ["cl1", "cl2", "cl3"])
    tissue = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # cl1, cl2 -> lung; cl3 -> blood
    blocks = {"identity": cell, "tissue": tissue}
    response = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(_pair_batch(blocks, response, (0, 0, 1, 1, 2, 2), (0, 1, 0, 1, 0, 1), cell))
    preds = predictor.predict(_pair_batch(blocks, None, (0, 1, 2), (0, 1, 0), cell))

    mean, lung, blood = response.mean(), np.mean([1.0, 2.0, 3.0, 4.0]), np.mean([5.0, 6.0])
    d1, d2 = np.mean([1.0, 3.0, 5.0]) - mean, np.mean([2.0, 4.0, 6.0]) - mean
    expected = [
        mean + (lung - mean) + (np.mean([1.0, 2.0]) - lung) + d1,
        mean + (lung - mean) + (np.mean([3.0, 4.0]) - lung) + d2,
        mean + (blood - mean) + (np.mean([5.0, 6.0]) - blood) + d1,
    ]
    np.testing.assert_allclose(preds, expected)
