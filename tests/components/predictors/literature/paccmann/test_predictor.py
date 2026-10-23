"""Tests for the PaccMann predictor."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.paccmann.predictor import (
    TOKENS_BLOCK,
    VARIANTS_BLOCK,
    PaccMannPredictor,
)
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.utils.torch_io import save_trusted_mapping

TINY = {
    "epochs": 3,
    "patience": 2,
    "batch_size": 4,
    "smiles_embedding_size": 4,
    "filters": [3, 3, 3],
    "molecule_heads": [1, 1, 1, 1],
    "gene_heads": [1, 1, 1, 1],
    "smiles_attention_size": 5,
    "stacked_dense_hidden_sizes": [8, 4],
    "dropout": 0.0,
}
PADDING, VARIANTS, VOCABULARY, GENES = 16, 3, 10, 7
SAMPLE_IDS = np.array(["s0", "s1", "s2", "s3", "v0", "v1"])
DRUG_IDS = np.array(["d0", "d1", "d2"])


def _tables(*, metadata: bool = True) -> tuple[FeaturizerResult, FeaturizerResult]:
    rng = np.random.default_rng(5)
    sample = FeaturizerResult.from_blocks(
        {"gene_expression": numeric_feature_block(rng.normal(size=(6, GENES)).astype(np.float32))}, SAMPLE_IDS
    )
    info = {"vocabulary_size": VOCABULARY, "padding_length": PADDING, "n_variants": VARIANTS} if metadata else None
    drug = FeaturizerResult.from_blocks(
        {
            TOKENS_BLOCK: numeric_feature_block(
                rng.integers(2, VOCABULARY, size=(3, PADDING)).astype(np.float32), metadata=info
            ),
            VARIANTS_BLOCK: numeric_feature_block(
                rng.integers(2, VOCABULARY, size=(3, VARIANTS * PADDING)).astype(np.float32), metadata=info
            ),
        },
        DRUG_IDS,
    )
    return sample, drug


def _batch(tmp_path, *, early_stopping: bool = True, n_pairs: int = 8, metadata: bool = True) -> ModelInputBatch:
    sample, drug = _tables(metadata=metadata)
    pair_samples = np.repeat(["s0", "s1", "s2", "s3"], 2)[:n_pairs]
    pair_drugs = np.tile(["d0", "d1"], 4)[:n_pairs]
    row = {sid: i for i, sid in enumerate(SAMPLE_IDS)}
    drug_row = {did: i for i, did in enumerate(DRUG_IDS)}
    stopping = ResponseBatch(
        response=np.array([0.5, 1.5], dtype=np.float32),
        sample_ids=np.array(["v0", "v1"]),
        drug_ids=np.array(["d0", "d1"]),
    )
    return ModelInputBatch.from_response(
        ResponseBatch(
            response=np.arange(n_pairs, dtype=np.float32) / 4.0, sample_ids=pair_samples, drug_ids=pair_drugs
        ),
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
    assert get_predictor("paccMann") is PaccMannPredictor


def test_defaults_are_the_1_6_1_values() -> None:
    defaults = PaccMannPredictor.get_default_hyperparameters()

    assert defaults["epochs"] == 10
    assert defaults["batch_size"] == 64
    assert defaults["filters"] == [16, 16, 16]
    assert defaults["molecule_heads"] == [2, 2, 2, 2]
    assert defaults["stacked_dense_hidden_sizes"] == [512, 256]
    assert defaults["batch_norm"] is True
    assert PaccMannPredictor.get_hyperparameter_space() == {}


def test_fit_predict_one_finite_value_per_pair(tmp_path) -> None:
    predictor = PaccMannPredictor(TINY)
    predictor.fit(_batch(tmp_path))

    prediction = predictor.predict(_query(["s0", "v0", "s3"], ["d0", "d1", "d2"]))

    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()
    assert predictor._architecture == {
        "smiles_vocabulary_size": VOCABULARY,
        "smiles_padding_length": PADDING,
        "number_of_genes": GENES,
        "n_variants": VARIANTS,
    }


def test_prediction_reads_the_plain_tokens_not_the_variants(tmp_path) -> None:
    predictor = PaccMannPredictor(TINY)
    predictor.fit(_batch(tmp_path))
    query = _query(["s1", "v1"], ["d2", "d0"])
    expected = predictor.predict(query)
    query.drug.blocks[VARIANTS_BLOCK].values[:] = 0

    np.testing.assert_array_equal(predictor.predict(query), expected)
    query.drug.blocks[TOKENS_BLOCK].values[:] = 3
    assert not np.array_equal(predictor.predict(query), expected)


def test_early_stopping_restores_the_best_epoch(tmp_path) -> None:
    predictor = PaccMannPredictor({**TINY, "epochs": 4, "patience": 1})
    predictor.fit(_batch(tmp_path))

    assert predictor.is_fitted()
    assert not predictor._model.training


def test_without_an_early_stopping_set_the_full_epoch_budget_runs(tmp_path) -> None:
    predictor = PaccMannPredictor({**TINY, "epochs": 2})
    predictor.fit(_batch(tmp_path, early_stopping=False))

    assert np.isfinite(predictor.predict(_query(["s0"], ["d2"]))).all()


def test_a_trailing_batch_of_one_is_dropped_only_when_it_is_not_the_only_batch(tmp_path) -> None:
    predictor = PaccMannPredictor({**TINY, "batch_size": 4})
    # 5 pairs -> batches of 4 and 1; the lone row would break batch norm.
    predictor.fit(_batch(tmp_path, n_pairs=5))
    loader, _ = predictor._loaders(_batch(tmp_path, n_pairs=5))
    assert len(loader) == 1

    # A single pair is the only batch and must be kept.
    loader, _ = predictor._loaders(_batch(tmp_path, n_pairs=1))
    assert len(loader) == 1


def test_each_row_draws_one_of_the_drug_variants(tmp_path) -> None:
    predictor = PaccMannPredictor(TINY)
    seen = []

    class _Probe(torch.nn.Module):
        def forward(self, smiles, genes):  # noqa: ANN001, ANN202
            seen.append(smiles.clone())
            return genes[:, :1]

    batch = _batch(tmp_path)
    variants = batch.drug.blocks[VARIANTS_BLOCK].values
    loader, _ = predictor._loaders(batch)
    gene_inputs, drug_inputs, targets = next(iter(loader))
    predictor._device = torch.device("cpu")

    predictor._batch_loss(_Probe(), VARIANTS)((gene_inputs, drug_inputs, targets))

    assert seen[0].shape == (drug_inputs.shape[0], PADDING)
    for row, tokens in zip(drug_inputs.numpy(), seen[0].numpy(), strict=True):
        assert any(np.array_equal(tokens, row.reshape(VARIANTS, PADDING)[v]) for v in range(VARIANTS))
    assert variants.shape[1] == VARIANTS * PADDING


def test_state_round_trips_the_predictions(tmp_path) -> None:
    predictor = PaccMannPredictor(TINY)
    predictor.fit(_batch(tmp_path))
    query = _query(["s0", "v1", "s2"], ["d2", "d1", "d0"])

    restored = PaccMannPredictor()
    restored.set_state(predictor.get_state())

    assert restored.is_fitted()
    assert restored._hyperparameters["epochs"] == 3
    np.testing.assert_allclose(restored.predict(query), predictor.predict(query), rtol=1e-6)


def test_unfitted_predictor_has_no_state_and_refuses_to_predict() -> None:
    predictor = PaccMannPredictor()

    assert predictor.get_state() == {}
    assert not predictor.is_fitted()
    with pytest.raises(ValueError, match="not initialized"):
        predictor.predict(_query(["s0"], ["d0"]))


def test_set_state_requires_the_architecture() -> None:
    payload = save_trusted_mapping({"predictor_hyperparameters": TINY, "architecture": {}, "model_state": b""})

    with pytest.raises(PredictorStateError, match="missing architecture or model_state"):
        PaccMannPredictor().set_state({"payload": payload})


def test_blocks_without_vocabulary_metadata_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="vocabulary metadata"):
        PaccMannPredictor(TINY).fit(_batch(tmp_path, metadata=False))


def test_a_batch_without_drug_features_is_rejected(tmp_path) -> None:
    sample, _ = _tables()
    batch = ModelInputBatch.from_response(
        ResponseBatch(
            response=np.zeros(2, dtype=np.float32), sample_ids=np.array(["s0", "s1"]), drug_ids=np.array(["d0", "d0"])
        ),
        sample=sample,
        sample_pair_idx=np.array([0, 1]),
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )

    with pytest.raises(RuntimeError, match="requires drug features"):
        PaccMannPredictor(TINY).fit(batch)
