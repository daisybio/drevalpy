"""Tests for the EnsembleMF predictor and its batch helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.ensemblemf.data import sorted_entities, split_pairs
from drevalpy.components.predictors.literature.ensemblemf.predictor import EnsembleMFPredictor
from drevalpy.components.predictors.literature.ensemblemf.training import PairTensors
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors.literature._helpers import entity_pair_batch

TINY = {
    "n_ensemble": 2,
    "max_epochs": 3,
    "patience": 2,
    "hidden_dim": 16,
    "emb_dim": 8,
    "mlp_hidden": 8,
    "batch_size": 8,
    "dropout": 0.0,
}

SAMPLE_IDS = np.array(["s3", "s0", "s2", "s1", "v1", "v0"])
DRUG_IDS = np.array(["d2", "d0", "d1"])


def _tables() -> tuple[FeaturizerResult, FeaturizerResult]:
    rng = np.random.default_rng(3)
    sample = FeaturizerResult.from_blocks(
        {"gene_expression": numeric_feature_block(rng.normal(size=(6, 9)).astype(np.float32))}, SAMPLE_IDS
    )
    drug = FeaturizerResult.from_blocks(
        {"fingerprints": numeric_feature_block(rng.integers(0, 2, size=(3, 12)).astype(np.float32))}, DRUG_IDS
    )
    return sample, drug


def _batch(tmp_path, *, early_stopping: bool = True) -> ModelInputBatch:
    sample, drug = _tables()
    return entity_pair_batch(
        sample,
        drug,
        np.repeat(["s0", "s1", "s2", "s3"], 2),
        np.tile(["d0", "d1"], 4),
        response=np.arange(8, dtype=np.float32) / 4.0,
        early_stopping=early_stopping,
        checkpoint_dir=tmp_path,
    )


def _query(sample_ids: list[str], drug_ids: list[str]) -> ModelInputBatch:
    return entity_pair_batch(*_tables(), sample_ids, drug_ids)


def test_registry_name_and_published_defaults() -> None:
    assert get_predictor("ensembleMF") is EnsembleMFPredictor
    defaults = EnsembleMFPredictor.get_default_hyperparameters()
    assert (defaults["n_ensemble"], defaults["hidden_dim"], defaults["emb_dim"]) == (20, 256, 128)
    assert (defaults["batch_size"], defaults["max_epochs"], defaults["patience"]) == (2048, 300, 25)
    assert defaults["use_drug_id_embedding"] is True


def test_fit_predict_one_value_per_pair_and_averages_the_members(tmp_path) -> None:
    predictor = EnsembleMFPredictor(TINY)
    predictor.fit(_batch(tmp_path))
    query = _query(["s0", "v0", "s3"], ["d0", "d1", "d1"])

    prediction = predictor.predict(query)

    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert len(predictor._nets) == 2
    members = []
    sample, drug = _tables()
    x_sample = torch.as_tensor(sample.blocks["gene_expression"].values)
    x_drug = torch.as_tensor(drug.blocks["fingerprints"].values)
    for net in predictor._nets:
        rows = predictor._embedding_rows(net, DRUG_IDS)
        z_sample, z_drug = net.encode(x_sample, x_drug, rows)
        members.append(
            net.score_pairs(
                z_sample[torch.as_tensor(query.sample_pair_idx)], z_drug[torch.as_tensor(query.drug_pair_idx)]
            )
            .detach()
            .numpy()
        )
    np.testing.assert_allclose(prediction, np.mean(members, axis=0), rtol=1e-6)


def test_members_use_consecutive_seeds_and_fit_is_reproducible_without_touching_global_rng(tmp_path) -> None:
    different = EnsembleMFPredictor({**TINY, "seed": 7, "max_epochs": 1})
    different.fit(_batch(tmp_path))
    weights = [net.state_dict()["sample_encoder.input_proj.weight"] for net in different._nets]
    assert not torch.equal(weights[0], weights[1])

    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    first, second = EnsembleMFPredictor(TINY), EnsembleMFPredictor(TINY)
    first.fit(_batch(tmp_path))
    torch.testing.assert_close(torch.rand(3), expected)
    second.fit(_batch(tmp_path))
    query = _query(["s1", "v1"], ["d1", "d0"])
    np.testing.assert_array_equal(first.predict(query), second.predict(query))


def test_without_an_early_stopping_set_a_seeded_tenth_is_held_out(tmp_path) -> None:
    first, second = EnsembleMFPredictor(TINY), EnsembleMFPredictor(TINY)
    first.fit(_batch(tmp_path, early_stopping=False))
    second.fit(_batch(tmp_path, early_stopping=False))
    query = _query(["s2"], ["d0"])

    np.testing.assert_array_equal(first.predict(query), second.predict(query))
    assert len(first._nets) == 2


def test_the_fallback_split_needs_two_pairs(tmp_path) -> None:
    batch = _batch(tmp_path, early_stopping=False).subset_pairs(np.array([True] + [False] * 7))

    with pytest.raises(ValueError, match="at least two training pairs"):
        EnsembleMFPredictor(TINY).fit(batch)


def test_only_drugs_seen_in_training_get_an_id_embedding(tmp_path) -> None:
    predictor = EnsembleMFPredictor({**TINY, "n_ensemble": 1})
    predictor.fit(_batch(tmp_path))
    net = predictor._nets[0]

    rows = predictor._embedding_rows(net, DRUG_IDS)

    np.testing.assert_array_equal(predictor._drug_seen, [True, True, False])
    assert predictor._drug_ids.tolist() == ["d0", "d1", "d2"]
    assert rows[0].abs().sum() == 0
    torch.testing.assert_close(rows[1], net.drug_id_emb.weight[0])
    torch.testing.assert_close(rows[2], net.drug_id_emb.weight[1])
    unseen = predictor._embedding_rows(net, np.array(["unseen", "d1"]))
    assert unseen[0].abs().sum() == 0
    torch.testing.assert_close(unseen[1], net.drug_id_emb.weight[1])


def test_embedding_can_be_switched_off(tmp_path) -> None:
    predictor = EnsembleMFPredictor({**TINY, "use_drug_id_embedding": False, "use_mlp_head": False})
    predictor.fit(_batch(tmp_path))

    assert not hasattr(predictor._nets[0], "drug_id_emb")
    assert np.isfinite(predictor.predict(_query(["s0"], ["d2"]))).all()


def test_state_round_trips_the_predictions(tmp_path) -> None:
    predictor = EnsembleMFPredictor(TINY)
    predictor.fit(_batch(tmp_path))
    query = _query(["s0", "v1", "s2"], ["d2", "d1", "d0"])

    restored = EnsembleMFPredictor()
    restored.set_state(predictor.get_state())

    assert restored.is_fitted()
    assert restored._hyperparameters["n_ensemble"] == 2
    np.testing.assert_allclose(restored.predict(query), predictor.predict(query), rtol=1e-6)


def test_unfitted_predictor_has_no_state_and_refuses_to_predict() -> None:
    predictor = EnsembleMFPredictor()

    assert predictor.get_state() == {}
    assert not predictor.is_fitted()
    with pytest.raises(ValueError, match="not initialized"):
        predictor.predict(_query(["s0"], ["d0"]))
    payload = save_trusted_mapping({"predictor_hyperparameters": TINY, "input_dims": [9, 12], "member_states": []})
    with pytest.raises(PredictorStateError, match="missing input_dims or member_states"):
        predictor.set_state({"payload": payload})


def test_a_batch_without_drug_features_is_rejected(tmp_path) -> None:
    sample, _ = _tables()
    batch = ModelInputBatch.from_response(
        ResponseBatch(
            response=np.zeros(2, dtype=np.float32), sample_ids=np.array(["s0", "s1"]), drug_ids=np.array(["d0", "d0"])
        ),
        sample=sample,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )

    with pytest.raises(RuntimeError, match="requires drug features"):
        EnsembleMFPredictor(TINY).fit(batch)


def _pairs(n: int) -> PairTensors:
    return PairTensors(torch.arange(n), torch.arange(n) % 3, torch.arange(n, dtype=torch.float32))


def test_sorted_entities_orders_rows_by_id_and_maps_original_rows() -> None:
    ids = np.array(["b", "c", "a"])
    values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    result = FeaturizerResult.from_blocks({"x": numeric_feature_block(values)}, ids)

    sorted_ids, matrix, position = sorted_entities(result, "x")

    assert sorted_ids.tolist() == ["a", "b", "c"]
    assert matrix[:, 0].tolist() == [3.0, 1.0, 2.0]
    assert position.tolist() == [1, 2, 0]


def test_split_pairs_keeps_a_supplied_validation_set_and_holds_out_a_seeded_tenth_otherwise() -> None:
    train, validation = _pairs(10), _pairs(3)
    assert split_pairs(train, validation, 0) == (train, validation)

    train_a, validation_a = split_pairs(_pairs(30), None, 7)
    train_b, validation_b = split_pairs(_pairs(30), None, 7)
    assert (len(train_a), len(validation_a)) == (27, 3)
    assert set(train_a.sample_idx.tolist()).isdisjoint(validation_a.sample_idx.tolist())
    assert train_a.sample_idx.tolist() == train_b.sample_idx.tolist()
    assert validation_a.sample_idx.tolist() == validation_b.sample_idx.tolist()
    unseeded_train, unseeded_validation = split_pairs(_pairs(20), None, -1)
    assert len(unseeded_train) + len(unseeded_validation) == 20


def test_split_pairs_needs_two_pairs() -> None:
    with pytest.raises(ValueError, match="at least two training pairs"):
        split_pairs(_pairs(1), None, 0)
