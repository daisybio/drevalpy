"""Tests for the naive mean predictors: global, per-entity, tissue-aware and additive effects."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.components.predictors.naive.effects import NaiveMeanEffectsPredictor
from drevalpy.components.predictors.naive.entity_mean import NaiveDrugMeanPredictor, NaiveSampleMeanPredictor
from drevalpy.components.predictors.naive.mean import NaiveMeanPredictor
from drevalpy.components.predictors.naive.tissue import NaiveTissueDrugMeanPredictor, NaiveTissueMeanPredictor
from drevalpy.models import DRPModel, ModelSpec
from tests.components.predictors._helpers import pickle_round_trip
from tests.components.predictors.naive._helpers import naive_batch, one_hot
from tests.models.synthetic_fixtures import lso_fold, synthetic_model_dataset_identity

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


def test_predicts_the_dataset_mean_for_any_batch_and_round_trips_through_pickle() -> None:
    predictor = NaiveMeanPredictor()
    predictor.fit(naive_batch(response=np.array([1.0, 5.0]), sample_ids=np.array(["x", "y"])))
    restored = pickle_round_trip(predictor)

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
    with pytest.raises(ValueError, match="require response values during fit"):
        predictor_class().fit(naive_batch(n_pairs=2))
    with pytest.raises(RuntimeError, match="Call fit before predict"):
        predictor_class()._predict(naive_batch(n_pairs=2))


def test_single_entity_base_reads_named_blocks_and_defaults_to_the_sample_side() -> None:
    assert SingleEntityNaivePredictor.sample_inputs == SingleEntityNaivePredictor.drug_inputs == Inputs.blocks()
    assert SingleEntityNaivePredictor._feature_side == "sample"


@pytest.mark.parametrize(("cls", "side"), [(_SampleSide, "sample"), (_DrugSide, "drug")])
def test_recovers_per_entity_means_and_falls_back_to_the_dataset_mean(cls, side) -> None:
    predictor = cls()
    assert not predictor.is_fitted()

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


def test_single_entity_pickle_round_trip_reproduces_predictions() -> None:
    predictor = _SampleSide()
    batch = _entity_batch("sample", [0, 1], [2.0, 8.0])
    predictor.fit(batch)
    restored = pickle_round_trip(predictor)

    assert restored.is_fitted()
    np.testing.assert_allclose(restored.predict(batch), predictor.predict(batch))
    assert not pickle_round_trip(_SampleSide()).is_fitted()


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
    restored = pickle_round_trip(predictor)

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
    restored = pickle_round_trip(predictor)

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
    dataset = synthetic_model_dataset_identity()
    dataset.mdata.obs["tissue"] = ["Lung", "Blood"]
    fold = lso_fold()
    model = DRPModel(ModelSpec.from_name("NaiveTissueMeanPredictor"))
    model.train(dataset, fold.train)
    preds = model.predict(dataset, fold.test)
    assert np.isfinite(preds).all()

    model.save(str(tmp_path / "model"))
    loaded = type(model).load(str(tmp_path / "model"))

    assert np.allclose(preds, loaded.predict(dataset, fold.test))


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
    blocks = {"oneHot": CELL} | ({} if tissue is None else {"tissue": tissue})
    response = np.array([1.0, 2.0, 5.0, 8.0])
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(_pair_batch(blocks, response, (0, 0, 1, 1), (0, 1, 0, 1)))
    batch = _pair_batch(blocks)
    mean = response.mean()
    expected = [
        mean + (np.mean([1.0, 2.0]) - mean) + (np.mean([1.0, 5.0]) - mean),
        mean + (np.mean([5.0, 8.0]) - mean) + (np.mean([2.0, 8.0]) - mean),
    ]
    restored = pickle_round_trip(predictor)

    np.testing.assert_allclose(predictor.predict(batch), expected)
    np.testing.assert_allclose(restored.predict(batch), expected)


def test_with_tissue_blocks_adds_tissue_then_within_tissue_cell_effects() -> None:
    cell = one_hot(["cl1", "cl2", "cl3"], ["cl1", "cl2", "cl3"])
    tissue = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # cl1, cl2 -> lung; cl3 -> blood
    blocks = {"oneHot": cell, "tissue": tissue}
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


def test_tissue_drug_mean_falls_back_to_the_dataset_mean_for_an_unseen_tissue_drug_or_combination() -> None:
    tissues = np.vstack([CELL, np.zeros((1, 2))])  # third tissue row is unseen in training
    drugs = np.vstack([DRUGS, np.zeros((1, 2))])  # third drug row is unseen in training
    predictor = NaiveTissueDrugMeanPredictor()
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
            sample_features=tissues,
            drug_features=drugs,
            sample_pair_idx=np.array([2, 0, 1, 0, 2], dtype=np.int64),
            drug_pair_idx=np.array([0, 2, 1, 1, 2], dtype=np.int64),
        )
    )

    # unseen tissue, unseen drug, combination (t2, d2) without training pairs, a seen combination, both unseen
    assert preds.tolist() == [3.0, 3.0, 3.0, 3.0, 3.0]


def test_naive_mean_effects_adds_no_effect_for_an_unseen_sample_tissue_or_drug() -> None:
    cell = np.vstack([one_hot(["cl1", "cl2"], ["cl1", "cl2"]), np.zeros((1, 2))])
    tissue = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    drugs = np.vstack([DRUGS, np.zeros((1, 2))])
    response = np.array([1.0, 2.0, 5.0, 8.0])
    blocks = {"oneHot": cell, "tissue": tissue}
    predictor = NaiveMeanEffectsPredictor()
    predictor.fit(
        naive_batch(
            response=response,
            sample_features=cell,
            drug_features=drugs,
            sample_pair_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            drug_pair_idx=np.array([0, 1, 0, 1], dtype=np.int64),
            sample_blocks=blocks,
        )
    )
    mean = response.mean()
    # tissue effect: t1 = mean(1, 2) - mean, t2 = mean(5, 8) - mean; sample effects vanish within a tissue's mean
    t1, t2 = np.mean([1.0, 2.0]) - mean, np.mean([5.0, 8.0]) - mean
    c1, c2 = np.mean([1.0, 2.0]) - mean - t1, np.mean([5.0, 8.0]) - mean - t2
    d1, d2 = np.mean([1.0, 5.0]) - mean, np.mean([2.0, 8.0]) - mean

    preds = predictor.predict(
        naive_batch(
            sample_features=cell,
            drug_features=drugs,
            sample_pair_idx=np.array([0, 2, 1, 2], dtype=np.int64),
            drug_pair_idx=np.array([2, 0, 1, 2], dtype=np.int64),
            sample_blocks=blocks,
        )
    )

    # unseen drug; unseen sample and tissue; fully seen pair; unseen sample, tissue and drug
    np.testing.assert_allclose(preds, [mean + t1 + c1, mean + d1, mean + t2 + c2 + d2, mean])


@pytest.mark.parametrize("side", ["sample", "drug"])
def test_single_entity_rejects_a_matrix_that_is_not_one_hot(side) -> None:
    predictor = _SampleSide() if side == "sample" else _DrugSide()
    multi_hot = np.ones((2, 2))

    with pytest.raises(ValueError, match="need one-hot rows"):
        predictor.fit(_entity_batch(side, [0, 1], [1.0, 2.0], features=multi_hot))


def test_group_means_match_a_dense_one_hot_reference_on_random_data() -> None:
    rng = np.random.default_rng(0)
    n_samples, n_drugs, n_tissues, n_pairs = 30, 12, 4, 400
    sample_idx = rng.integers(0, n_samples, n_pairs)
    drug_idx = rng.integers(0, n_drugs, n_pairs)
    tissue_of = rng.integers(0, n_tissues, n_samples)
    cell = np.eye(n_samples)
    drugs = np.eye(n_drugs)
    tissue = np.eye(n_tissues)[tissue_of]
    response = rng.normal(size=n_pairs)
    blocks = {"oneHot": cell, "tissue": tissue}
    kwargs = {"sample_features": cell, "drug_features": drugs, "sample_blocks": blocks}
    train_s, train_d, train_y = sample_idx[:300], drug_idx[:300], response[:300]
    test_s, test_d = sample_idx[300:], drug_idx[300:]

    def dense_means(design, y):
        counts = design.sum(axis=0)
        return np.divide(design.T @ y, counts, out=np.zeros(design.shape[1]), where=counts > 0), counts

    mean = train_y.mean()
    t_design, d_design = tissue[train_s], drugs[train_d]
    t_means, t_counts = dense_means(t_design, train_y)
    t_eff = np.where(t_counts > 0, t_means - mean, 0.0)
    c_means, c_counts = dense_means(cell[train_s], train_y - mean - t_design @ t_eff)
    c_eff = np.where(c_counts > 0, c_means, 0.0)
    d_means, d_counts = dense_means(d_design, train_y)
    d_eff = np.where(d_counts > 0, d_means - mean, 0.0)
    expected = mean + cell[test_s] @ c_eff + tissue[test_s] @ t_eff + drugs[test_d] @ d_eff
    joint_counts = np.einsum("ni,nj->ij", t_design, d_design)
    joint_sums = np.einsum("ni,nj,n->ij", t_design, d_design, train_y)
    joint = np.where(
        joint_counts > 0,
        np.divide(joint_sums, joint_counts, out=np.zeros_like(joint_sums), where=joint_counts > 0) - mean,
        0.0,
    )
    expected_joint = mean + np.einsum("ni,ij,nj->n", tissue[test_s], joint, drugs[test_d])

    effects = NaiveMeanEffectsPredictor()
    effects.fit(naive_batch(response=train_y, sample_pair_idx=train_s, drug_pair_idx=train_d, **kwargs))
    tissue_drug = NaiveTissueDrugMeanPredictor()
    tissue_drug.fit(
        naive_batch(
            response=train_y,
            sample_features=tissue,
            drug_features=drugs,
            sample_pair_idx=train_s,
            drug_pair_idx=train_d,
        )
    )
    test_batch = naive_batch(sample_pair_idx=test_s, drug_pair_idx=test_d, **kwargs)
    tissue_drug_batch = naive_batch(
        sample_features=tissue, drug_features=drugs, sample_pair_idx=test_s, drug_pair_idx=test_d
    )

    np.testing.assert_allclose(effects.predict(test_batch), expected, rtol=1e-12)
    np.testing.assert_allclose(tissue_drug.predict(tissue_drug_batch), expected_joint, rtol=1e-12)
