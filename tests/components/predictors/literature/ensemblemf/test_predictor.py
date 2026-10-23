"""Tests for the EnsembleMF predictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.ensemblemf.predictor import EnsembleMFPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.utils.torch_io import save_trusted_mapping

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

#: Deliberately unsorted, so the predictor's own ordering is exercised.
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
    """Train on s0..s3 x d0,d1 (never d2); optionally stop on v0, v1 x d0, d1."""
    sample, drug = _tables()
    pair_samples = np.repeat(["s0", "s1", "s2", "s3"], 2)
    pair_drugs = np.tile(["d0", "d1"], 4)
    response = np.arange(8, dtype=np.float32) / 4.0
    stopping = ResponseBatch(
        response=np.array([0.5, 1.5], dtype=np.float32),
        sample_ids=np.array(["v0", "v1"]),
        drug_ids=np.array(["d0", "d1"]),
    )
    row = {sid: i for i, sid in enumerate(SAMPLE_IDS)}
    drug_row = {did: i for i, did in enumerate(DRUG_IDS)}
    return ModelInputBatch.from_response(
        ResponseBatch(response=response, sample_ids=pair_samples, drug_ids=pair_drugs),
        sample=sample,
        drug=drug,
        sample_pair_idx=np.array([row[s] for s in pair_samples]),
        drug_pair_idx=np.array([drug_row[d] for d in pair_drugs]),
        early_stopping_response=stopping if early_stopping else None,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )


def _query(sample_ids: list[str], drug_ids: list[str]) -> ModelInputBatch:
    sample, drug = _tables()
    row = {sid: i for i, sid in enumerate(SAMPLE_IDS)}
    drug_row = {did: i for i, did in enumerate(DRUG_IDS)}
    return ModelInputBatch.from_response(
        ResponseBatch(
            response=np.zeros(len(sample_ids), dtype=np.float32),
            sample_ids=np.array(sample_ids),
            drug_ids=np.array(drug_ids),
        ),
        sample=sample,
        drug=drug,
        sample_pair_idx=np.array([row[s] for s in sample_ids]),
        drug_pair_idx=np.array([drug_row[d] for d in drug_ids]),
    )


def test_registry_name() -> None:
    assert get_predictor("ensembleMF") is EnsembleMFPredictor


def test_defaults_are_the_1_6_1_values() -> None:
    defaults = EnsembleMFPredictor.get_default_hyperparameters()

    assert defaults["n_ensemble"] == 20
    assert defaults["hidden_dim"] == 256
    assert defaults["emb_dim"] == 128
    assert defaults["learning_rate"] == 3e-4
    assert defaults["batch_size"] == 2048
    assert defaults["max_epochs"] == 300
    assert defaults["patience"] == 25
    assert defaults["seed"] == 0
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


def test_members_use_consecutive_seeds(tmp_path) -> None:
    predictor = EnsembleMFPredictor({**TINY, "seed": 7, "max_epochs": 1})
    predictor.fit(_batch(tmp_path))
    state = [net.state_dict()["sample_encoder.input_proj.weight"] for net in predictor._nets]

    assert not torch.equal(state[0], state[1])


def test_fit_is_reproducible_for_a_fixed_seed(tmp_path) -> None:
    first, second = EnsembleMFPredictor(TINY), EnsembleMFPredictor(TINY)
    first.fit(_batch(tmp_path))
    second.fit(_batch(tmp_path))
    query = _query(["s1", "v1"], ["d1", "d0"])

    np.testing.assert_array_equal(first.predict(query), second.predict(query))


def test_fit_does_not_consume_the_global_rng(tmp_path) -> None:
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)

    EnsembleMFPredictor(TINY).fit(_batch(tmp_path))

    torch.testing.assert_close(torch.rand(3), expected)


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


def test_a_drug_unknown_to_training_is_scored_from_its_fingerprint(tmp_path) -> None:
    predictor = EnsembleMFPredictor({**TINY, "n_ensemble": 1})
    predictor.fit(_batch(tmp_path))
    net = predictor._nets[0]
    rows = predictor._embedding_rows(net, np.array(["unseen", "d1"]))

    assert rows[0].abs().sum() == 0
    torch.testing.assert_close(rows[1], net.drug_id_emb.weight[1])


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


def test_set_state_requires_the_members() -> None:
    payload = save_trusted_mapping({"predictor_hyperparameters": TINY, "input_dims": [9, 12], "member_states": []})

    with pytest.raises(PredictorStateError, match="missing input_dims or member_states"):
        EnsembleMFPredictor().set_state({"payload": payload})


def test_a_batch_without_drug_features_is_rejected(tmp_path) -> None:
    sample, _ = _tables()
    batch = ModelInputBatch.from_response(
        ResponseBatch(
            response=np.zeros(2, dtype=np.float32), sample_ids=np.array(["s0", "s1"]), drug_ids=np.array(["d0", "d0"])
        ),
        sample=sample,
        sample_pair_idx=np.array([1, 3]),
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )

    with pytest.raises(RuntimeError, match="requires drug features"):
        EnsembleMFPredictor(TINY).fit(batch)
