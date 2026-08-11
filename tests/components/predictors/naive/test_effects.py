"""Tests for naive mean-effects predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.naive.effects import NaiveMeanEffectsPredictor
from tests.components.predictors.naive._helpers import naive_batch, one_hot


def test_naive_mean_effects_without_tissue_decomposes_cell_and_drug() -> None:
    cell_cats = ["cl1", "cl2"]
    drug_cats = ["d1", "d2"]
    cell = one_hot(cell_cats, cell_cats)
    drugs = one_hot(drug_cats, drug_cats)
    response = np.array([1.0, 2.0, 5.0, 8.0])
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(
        naive_batch(
            response=response,
            cell_line_features=cell,
            drug_features=drugs,
            cell_line_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0, 1], dtype=np.int64),
            cell_line_blocks={"identity": cell},
        )
    )
    preds = predictor.predict(
        naive_batch(
            cell_line_features=cell,
            drug_features=drugs,
            cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1], dtype=np.int64),
            cell_line_blocks={"identity": cell},
            cell_line_ids=np.array(["wrong_cl", "also_wrong"]),
            drug_ids=np.array(["wrong_d", "also_wrong_d"]),
        )
    )
    dataset_mean = float(np.mean(response))
    expected = np.array(
        [
            dataset_mean + (np.mean([1.0, 2.0]) - dataset_mean) + (np.mean([1.0, 5.0]) - dataset_mean),
            dataset_mean + (np.mean([5.0, 8.0]) - dataset_mean) + (np.mean([2.0, 8.0]) - dataset_mean),
        ]
    )
    np.testing.assert_allclose(preds, expected)


def test_naive_mean_effects_with_tissue_blocks() -> None:
    cell_cats = ["cl1", "cl2", "cl3"]
    drug_cats = ["d1", "d2"]
    cell = one_hot(cell_cats, cell_cats)
    # cl1,cl2 -> lung; cl3 -> blood
    tissue = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    drugs = one_hot(drug_cats, drug_cats)
    response = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pair_cl = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pair_d = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(
        naive_batch(
            response=response,
            cell_line_features=np.concatenate([cell, tissue], axis=1),
            drug_features=drugs,
            cell_line_pair_idx=pair_cl,
            drug_pair_idx=pair_d,
            cell_line_blocks={"identity": cell, "tissue": tissue},
        )
    )
    preds = predictor.predict(
        naive_batch(
            cell_line_features=np.concatenate([cell, tissue], axis=1),
            drug_features=drugs,
            cell_line_pair_idx=np.array([0, 1, 2], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
            cell_line_blocks={"identity": cell, "tissue": tissue},
        )
    )
    dataset_mean = float(np.mean(response))
    lung_mean = float(np.mean([1.0, 2.0, 3.0, 4.0]))
    blood_mean = float(np.mean([5.0, 6.0]))
    cl1_mean = float(np.mean([1.0, 2.0]))
    cl2_mean = float(np.mean([3.0, 4.0]))
    cl3_mean = float(np.mean([5.0, 6.0]))
    d1_effect = float(np.mean([1.0, 3.0, 5.0]) - dataset_mean)
    d2_effect = float(np.mean([2.0, 4.0, 6.0]) - dataset_mean)
    expected = np.array(
        [
            dataset_mean + (lung_mean - dataset_mean) + (cl1_mean - lung_mean) + d1_effect,
            dataset_mean + (lung_mean - dataset_mean) + (cl2_mean - lung_mean) + d2_effect,
            dataset_mean + (blood_mean - dataset_mean) + (cl3_mean - blood_mean) + d1_effect,
        ]
    )
    np.testing.assert_allclose(preds, expected)


def test_naive_mean_effects_empty_optional_tissue() -> None:
    cell = one_hot(["cl1", "cl2"], ["cl1", "cl2"])
    drugs = one_hot(["d1", "d2"], ["d1", "d2"])
    response = np.array([1.0, 2.0, 5.0, 8.0])
    empty_tissue = np.empty((2, 0), dtype=np.float64)
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(
        naive_batch(
            response=response,
            cell_line_features=cell,
            drug_features=drugs,
            cell_line_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0, 1], dtype=np.int64),
            cell_line_blocks={"identity": cell, "tissue": empty_tissue},
        )
    )
    preds = predictor.predict(
        naive_batch(
            cell_line_features=cell,
            drug_features=drugs,
            cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1], dtype=np.int64),
            cell_line_blocks={"identity": cell, "tissue": empty_tissue},
        )
    )
    dataset_mean = float(np.mean(response))
    expected = np.array(
        [
            dataset_mean + (np.mean([1.0, 2.0]) - dataset_mean) + (np.mean([1.0, 5.0]) - dataset_mean),
            dataset_mean + (np.mean([5.0, 8.0]) - dataset_mean) + (np.mean([2.0, 8.0]) - dataset_mean),
        ]
    )
    np.testing.assert_allclose(preds, expected)


def test_naive_mean_effects_state_roundtrip() -> None:
    cell = one_hot(["cl1"], ["cl1"])
    drugs = one_hot(["d1"], ["d1"])
    predictor = NaiveMeanEffectsPredictor()
    batch = naive_batch(
        response=np.array([4.0]),
        cell_line_features=cell,
        drug_features=drugs,
        cell_line_pair_idx=np.array([0], dtype=np.int64),
        drug_pair_idx=np.array([0], dtype=np.int64),
        cell_line_blocks={"identity": cell},
    )
    predictor.fit(batch)
    restored = NaiveMeanEffectsPredictor()
    restored.set_state(predictor.get_state())
    np.testing.assert_allclose(restored.predict(batch), predictor.predict(batch))
