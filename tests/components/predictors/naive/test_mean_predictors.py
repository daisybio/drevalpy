"""Tests for the naive mean predictors: global, per-entity, tissue-aware and additive effects."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.components.predictors.naive.effects import NaiveMeanEffectsPredictor
from drevalpy.components.predictors.naive.entity_mean import NaiveDrugMeanPredictor, NaiveSampleMeanPredictor
from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
from drevalpy.components.predictors.naive.tissue import NaiveTissueDrugMeanPredictor, NaiveTissueMeanPredictor
from drevalpy.models import DRPModel, ModelSpec
from tests.components.predictors.naive._helpers import naive_batch, one_hot
from tests.models.synthetic_fixtures import lso_split_masks, synthetic_mudataset_identity

CELL = one_hot(["cl1", "cl2"], ["cl1", "cl2"])
DRUGS = one_hot(["d1", "d2"], ["d1", "d2"])
CATEGORIES = ["e1", "e2"]
FEATURES = one_hot(CATEGORIES, CATEGORIES)


class _SampleSide(SingleEntityNaivePredictor):
    pass


class _DrugSide(SingleEntityNaivePredictor):
    _feature_side = "drug"


def _entity_batch(side: str, idx: list[int], response: list[float] | None = None, features=FEATURES):
    return naive_batch(
        response=None if response is None else np.array(response),
        **{f"{side}_features": features, f"{side}_pair_idx": np.array(idx, dtype=np.int64)},
    )


def test_predicts_the_dataset_mean_for_any_batch_and_round_trips_state() -> None:
    predictor = NaiveMeanPredictor()
    predictor.fit(naive_batch(response=np.array([1.0, 5.0]), sample_ids=np.array(["x", "y"])))
    restored = NaiveMeanPredictor()
    restored.set_state(predictor.get_state())

    assert predictor.predict(naive_batch(n_pairs=3, sample_ids=np.array(["a", "b", "c"]))).tolist() == [3.0] * 3
    assert restored.predict(naive_batch(n_pairs=2)).tolist() == [3.0, 3.0]


@pytest.mark.parametrize(
    "predictor_class",
    [
        NaiveMeanPredictor,
        NaiveTissueMeanPredictor,
        NaiveTissueDrugMeanPredictor,
        NaiveMeanEffectsPredictor,
        _SampleSide,
    ],
)
def test_fitting_without_a_response_and_predicting_before_fit_are_rejected(predictor_class) -> None:
    with pytest.raises(RuntimeError, match="batch.response is required for fit"):
        predictor_class()._fit(naive_batch(n_pairs=2))
    with pytest.raises(RuntimeError, match="Call fit before predict"):
        predictor_class()._predict(naive_batch(n_pairs=2))


def test_single_entity_base_reads_named_blocks_and_defaults_to_the_sample_side() -> None:
    assert issubclass(SingleEntityNaivePredictor, BlockPredictor)
    assert SingleEntityNaivePredictor.input_interface == "block"
    assert SingleEntityNaivePredictor._feature_side == "sample"


@pytest.mark.parametrize(("cls", "side"), [(_SampleSide, "sample"), (_DrugSide, "drug")])
def test_recovers_per_entity_means_and_falls_back_to_the_dataset_mean(cls, side) -> None:
    predictor = cls()
    assert not predictor.is_fitted()
    assert predictor.get_state() == {}

    predictor.fit(_entity_batch(side, [0, 0, 1, 1], [1.0, 3.0, 5.0, 7.0]))
    preds = predictor.predict(_entity_batch(side, [0, 1, 2], features=np.vstack([FEATURES, np.zeros((1, 2))])))

    assert predictor.is_fitted()
    np.testing.assert_allclose(preds, [2.0, 6.0, 4.0])


def test_single_entity_fit_and_predict_guards() -> None:
    predictor = _SampleSide()

    with pytest.raises(ValueError, match="require response values during fit"):
        predictor.fit(naive_batch(n_pairs=2))
    with pytest.raises(RuntimeError, match="Call fit before predict"):
        predictor.predict(naive_batch(n_pairs=2))


def test_single_entity_state_round_trip_reproduces_predictions_and_is_json_friendly() -> None:
    predictor = _SampleSide()
    batch = _entity_batch("sample", [0, 1], [2.0, 8.0])
    predictor.fit(batch)
    state = predictor.get_state()
    restored = _SampleSide()
    restored.set_state(state)

    assert isinstance(state["dataset_mean"], float)
    assert isinstance(state["effects"], list)
    assert restored.is_fitted()
    np.testing.assert_allclose(restored.predict(batch), predictor.predict(batch))
    empty = _SampleSide()
    empty.set_state({})
    assert not empty.is_fitted()


def test_drug_mean_follows_the_matrix_not_decoy_ids_and_round_trips() -> None:
    predictor = NaiveDrugMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0]),
            drug_features=FEATURES,
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
            drug_ids=np.array(["not_d1", "not_d2", "still_not_d1"]),
        )
    )
    batch = naive_batch(
        drug_features=FEATURES,
        drug_pair_idx=np.array([0, 1], dtype=np.int64),
        drug_ids=np.array(["decoy_a", "decoy_b"]),
    )
    restored = NaiveDrugMeanPredictor()
    restored.set_state(predictor.get_state())

    assert predictor.predict(batch).tolist() == [3.0, 3.0]
    assert restored.predict(batch).tolist() == [3.0, 3.0]


def test_sample_mean_falls_back_for_an_unseen_zero_row() -> None:
    predictor = NaiveSampleMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 6.0]),
            sample_features=FEATURES,
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
        )
    )
    preds = predictor.predict(
        naive_batch(
            sample_features=np.vstack([FEATURES, np.zeros((1, 2))]),
            sample_pair_idx=np.array([0, 1, 2], dtype=np.int64),
        )
    )

    assert preds.tolist() == [2.0, 6.0, 4.0]


def test_naive_tissue_mean_requires_tissue_features() -> None:
    batch = naive_batch(response=np.array([1.0]), sample_features=np.empty((1, 0), dtype=np.float64))
    empty = naive_batch(n_pairs=2, sample_features=np.empty((2, 0), dtype=np.float64))
    fitted = NaiveTissueMeanPredictor()
    fitted.fit(naive_batch(response=np.array([1.0, 2.0]), sample_features=CELL))

    with pytest.raises(ValueError, match="requires tissue"):
        NaiveTissueMeanPredictor().fit(batch)
    with pytest.raises(ValueError, match="requires tissue"):
        fitted.predict(empty)


def test_naive_tissue_mean_predicts_from_matrix_not_ids_and_round_trips() -> None:
    tissue_features = one_hot(["lung", "blood"], ["lung", "blood"])
    predictor = NaiveTissueMeanPredictor()
    predictor.fit(
        naive_batch(
            response=np.array([2.0, 4.0]),
            sample_features=tissue_features,
            sample_pair_idx=np.array([0, 1], dtype=np.int64),
            sample_ids=np.array(["ignore_a", "ignore_b"]),
        )
    )
    batch = naive_batch(
        sample_features=tissue_features,
        sample_pair_idx=np.array([0, 1], dtype=np.int64),
        sample_ids=np.array(["other_a", "other_b"]),
    )
    restored = NaiveTissueMeanPredictor()
    restored.set_state(predictor.get_state())

    assert predictor.predict(batch).tolist() == [2.0, 4.0]
    assert restored.predict(batch).tolist() == [2.0, 4.0]


def test_naive_tissue_drug_mean_interaction_table_and_guards() -> None:
    predictor = NaiveTissueDrugMeanPredictor()
    with pytest.raises(ValueError, match="requires tissue"):
        predictor.fit(naive_batch(response=np.array([1.0]), sample_features=np.empty((1, 0), dtype=np.float64)))
    predictor.fit(
        naive_batch(
            response=np.array([1.0, 3.0, 5.0]),
            sample_features=CELL,
            drug_features=DRUGS,
            sample_pair_idx=np.array([0, 0, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0], dtype=np.int64),
        )
    )
    preds = predictor.predict(
        naive_batch(
            sample_features=CELL,
            drug_features=DRUGS,
            sample_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0, 1], dtype=np.int64),
        )
    )
    assert preds.tolist() == [1.0, 3.0, 5.0, 3.0]
    with pytest.raises(ValueError, match="requires tissue"):
        predictor.predict(naive_batch(n_pairs=2, sample_features=np.empty((2, 0), dtype=np.float64)))


def test_naive_tissue_model_trains_predicts_and_reloads(tmp_path) -> None:
    mudataset = synthetic_mudataset_identity()
    mudataset.mdata.obs["tissue"] = ["Lung", "Blood"]
    split = lso_split_masks()
    model = DRPModel(ModelSpec.from_name("NaiveTissueMeanPredictor"))
    model.train(mudataset, split.train)
    preds = model.predict(mudataset, split.test)
    assert np.isfinite(preds).all()

    model.save(str(tmp_path / "model"))
    loaded = type(model).load(str(tmp_path / "model"))

    assert np.allclose(preds, loaded.predict(mudataset, split.test))


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
