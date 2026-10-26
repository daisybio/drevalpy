"""Tests for the PaccMann predictor and network."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.components.predictors.literature.paccmann.network import (
    ContextAttentionLayer,
    PaccMannConfig,
    PaccMannNetwork,
)
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
from tests.components.predictors.literature._helpers import entity_pair_batch

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
                rng.integers(2, VOCABULARY, size=(3, PADDING)).astype(np.float32), annotations=info
            ),
            VARIANTS_BLOCK: numeric_feature_block(
                rng.integers(2, VOCABULARY, size=(3, VARIANTS * PADDING)).astype(np.float32), annotations=info
            ),
        },
        DRUG_IDS,
    )
    return sample, drug


def _batch(tmp_path, *, early_stopping: bool = True, n_pairs: int = 8, metadata: bool = True) -> ModelInputBatch:
    sample, drug = _tables(metadata=metadata)
    return entity_pair_batch(
        sample,
        drug,
        np.repeat(["s0", "s1", "s2", "s3"], 2)[:n_pairs],
        np.tile(["d0", "d1"], 4)[:n_pairs],
        response=np.arange(n_pairs, dtype=np.float32) / 4.0,
        early_stopping=early_stopping,
        checkpoint_dir=tmp_path,
    )


def _query(sample_ids: list[str], drug_ids: list[str]) -> ModelInputBatch:
    return entity_pair_batch(*_tables(), sample_ids, drug_ids)


def test_registry_name_and_published_defaults() -> None:
    assert get_predictor("paccMann") is PaccMannPredictor
    defaults = PaccMannPredictor.get_default_hyperparameters()
    assert (defaults["epochs"], defaults["batch_size"], defaults["filters"]) == (10, 64, [16, 16, 16])
    assert defaults["molecule_heads"] == [2, 2, 2, 2]
    assert defaults["stacked_dense_hidden_sizes"] == [512, 256]
    assert defaults["batch_norm"] is True
    assert PaccMannPredictor.get_hyperparameter_space() == {}


def test_fit_predict_reads_the_plain_tokens_not_the_variants(tmp_path) -> None:
    predictor = PaccMannPredictor(TINY)
    predictor.fit(_batch(tmp_path))
    query = _query(["s0", "v0", "s3"], ["d0", "d1", "d2"])

    expected = predictor.predict(query)

    assert expected.shape == (3,)
    assert np.isfinite(expected).all()
    assert predictor._architecture == {
        "smiles_vocabulary_size": VOCABULARY,
        "smiles_padding_length": PADDING,
        "number_of_genes": GENES,
        "n_variants": VARIANTS,
    }
    query.drug.blocks[VARIANTS_BLOCK].values[:] = 0
    np.testing.assert_array_equal(predictor.predict(query), expected)
    query.drug.blocks[TOKENS_BLOCK].values[:] = 3
    assert not np.array_equal(predictor.predict(query), expected)


def test_early_stopping_restores_the_best_epoch_and_the_budget_runs_without_a_stopping_set(tmp_path) -> None:
    stopped = PaccMannPredictor({**TINY, "epochs": 4, "patience": 1})
    stopped.fit(_batch(tmp_path))
    assert stopped.is_fitted()
    assert not stopped._model.training

    full = PaccMannPredictor({**TINY, "epochs": 2})
    full.fit(_batch(tmp_path, early_stopping=False))
    assert np.isfinite(full.predict(_query(["s0"], ["d2"]))).all()


def test_a_trailing_batch_of_one_is_dropped_only_when_it_is_not_the_only_batch(tmp_path) -> None:
    predictor = PaccMannPredictor({**TINY, "batch_size": 4})
    predictor.fit(_batch(tmp_path, n_pairs=5))
    loader, _ = predictor._loaders(_batch(tmp_path, n_pairs=5))
    assert len(loader) == 1

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


def test_unfitted_predictor_has_no_state_and_refuses_to_predict_or_restore() -> None:
    predictor = PaccMannPredictor()

    assert predictor.get_state() == {}
    assert not predictor.is_fitted()
    with pytest.raises(ValueError, match="not initialized"):
        predictor.predict(_query(["s0"], ["d0"]))
    payload = save_trusted_mapping({"predictor_hyperparameters": TINY, "architecture": {}, "model_state": b""})
    with pytest.raises(PredictorStateError, match="missing architecture or model_state"):
        predictor.set_state({"payload": payload})


def test_inputs_missing_vocabulary_metadata_or_drug_features_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="vocabulary metadata"):
        PaccMannPredictor(TINY).fit(_batch(tmp_path, metadata=False))

    sample, _ = _tables()
    batch = ModelInputBatch.from_response(
        ResponseBatch(
            response=np.zeros(2, dtype=np.float32), sample_ids=np.array(["s0", "s1"]), drug_ids=np.array(["d0", "d0"])
        ),
        sample=sample,
        training_context=TrainingContext(checkpoint_dir=tmp_path),
    )
    with pytest.raises(RuntimeError, match="requires drug features"):
        PaccMannPredictor(TINY).fit(batch)


SMALL = {
    "smiles_vocabulary_size": 12,
    "smiles_embedding_size": 4,
    "smiles_padding_length": 16,
    "number_of_genes": 7,
    "molecule_heads": [1, 2, 1, 1],
    "gene_heads": [1, 1, 1, 1],
    "filters": [3, 3, 3],
    "smiles_attention_size": 5,
    "stacked_dense_hidden_sizes": [8, 4],
    "dropout": 0.0,
    "batch_norm": True,
}


def _network() -> PaccMannNetwork:
    return PaccMannNetwork(PaccMannConfig.from_hyperparameters(SMALL))


def test_config_resolves_kernel_windows_and_ignores_unknown_keys() -> None:
    config = PaccMannConfig.from_hyperparameters({**SMALL, "epochs": 10, "learning_rate": 0.1})

    assert config.resolved_kernel_sizes == [(3, 4), (5, 4), (11, 4)]
    assert config.number_of_genes == 7


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"gene_heads": [1, 1]}, "same length"),
        ({"kernel_sizes": [(3, 4), (5, 4)]}, "filters and kernel_sizes"),
        ({"filters": [3, 3], "kernel_sizes": [(3, 4), (5, 4)]}, "one more entry than filters"),
    ],
)
def test_config_rejects_inconsistent_lengths(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PaccMannConfig.from_hyperparameters({**SMALL, **override})


def test_attention_layer_weights_sum_to_one_and_can_keep_the_sequence_axis() -> None:
    layer = ContextAttentionLayer(4, 6, 1, 7, attention_size=5, temperature=1.0)
    attended, alphas = layer(torch.randn(3, 6, 4), torch.randn(3, 7, 1))
    assert attended.shape == (3, 4)
    assert alphas.shape == (3, 6)
    torch.testing.assert_close(alphas.sum(dim=1), torch.ones(3))

    sequence_layer = ContextAttentionLayer(1, 7, 4, 6, attention_size=1, temperature=1.0)
    attended, _ = sequence_layer(torch.randn(3, 7, 1), torch.randn(3, 6, 4), average_seq=False)
    assert attended.shape == (3, 7)

    single = ContextAttentionLayer(4, 6, 1, 1, attention_size=5, temperature=1.0)
    assert isinstance(single.context_hidden_projection, torch.nn.Identity)


def test_network_predicts_one_value_per_row_and_follows_the_configuration() -> None:
    network = _network().eval()

    output = network(torch.randint(0, 12, (5, 16)), torch.randn(5, 7))

    assert output.shape == (5, 1)
    assert torch.isfinite(output).all()
    assert network(torch.randint(0, 12, (1, 16)), torch.randn(1, 7)).shape == (1, 1)
    assert len(network.molecule_attentions) == sum(SMALL["molecule_heads"])
    assert len(network.gene_attentions) == sum(SMALL["gene_heads"])
    assert len(network.convolutions) == 3


def test_batch_norm_can_be_switched_off() -> None:
    network = PaccMannNetwork(PaccMannConfig.from_hyperparameters({**SMALL, "batch_norm": False}))

    assert isinstance(network.batch_norm, torch.nn.Identity)
