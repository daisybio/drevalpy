"""Tests for SMILES handling: source lookup, tokenizer, the smilesTokens featurizer and BPE PharmaFormer."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.bpe_pharmaformer import BpePharmaformerDrugFeaturizer
from drevalpy.components.featurizers.drug.drug_graph import as_pyg_graph
from drevalpy.components.featurizers.drug.smiles_tokenizer import (
    BASE_VOCABULARY,
    PAD_IDX,
    UNK_IDX,
    build_vocabulary,
    encode_smiles,
    randomize_smiles,
    tokenize_smiles,
)
from drevalpy.components.featurizers.drug.smiles_tokens import (
    TOKENS_BLOCK,
    VARIANTS_BLOCK,
    SmilesTokensFeaturizer,
)
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import featurizer_registry
from drevalpy.types.data.feature_source import FeatureSource
from tests._import_shims import block_imports
from tests.components.featurizers.drug._helpers import DrugSource
from tests.models.synthetic_fixtures import synthetic_model_dataset_gene_expression_fingerprints


def test_returns_none_without_a_mudata_backing_or_smiles_column() -> None:
    assert DrugSource({"v": {"d1": [1.0]}}).smiles(np.array(["d1"])) is None
    dataset = synthetic_model_dataset_gene_expression_fingerprints()
    source = FeatureSource(dataset, "drug")
    assert source.smiles(dataset.drug_ids) is None


def test_returns_smiles_indexed_by_the_requested_drug_ids(drug_source: FeatureSource, drug_ids: np.ndarray) -> None:
    drug_ids = drug_ids[:3]

    smiles = drug_source.smiles(drug_ids)

    assert smiles is not None
    assert list(smiles.index) == list(drug_ids)
    assert all(isinstance(value, str) and value for value in smiles)
    unknown = drug_source.smiles(np.array(["not-a-drug"]))
    assert unknown is not None and unknown.isna().all()


def test_portable_graph_conversion_preserves_arrays_and_index_dtype() -> None:
    import torch

    payload = {"x": np.array([[1.0, 2.0], [3.0, 4.0]]), "edge_index": np.array([[0, 1], [1, 0]])}
    graph = as_pyg_graph(payload)
    assert graph.edge_index.dtype == torch.long
    np.testing.assert_array_equal(graph.x.numpy(), payload["x"])
    assert isinstance(payload["x"], np.ndarray)
    assert as_pyg_graph(graph) is graph
    assert as_pyg_graph(None) is None


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.mark.parametrize(
    ("smiles", "tokens"),
    [
        ("CCO", ["C", "C", "O"]),
        ("CCl", ["C", "Cl"]),
        ("BrC", ["Br", "C"]),
        ("C[C@@H](N)C(=O)O", ["C", "[C@@H]", "(", "N", ")", "C", "(", "=", "O", ")", "O"]),
        ("[Pt+2]", ["[Pt+2]"]),
        ("C%12CC%12", ["C", "%12", "C", "C", "%12"]),
        ("c1ccncc1", ["c", "1", "c", "c", "n", "c", "c", "1"]),
        ("F/C=C/F", ["F", "/", "C", "=", "C", "/", "F"]),
        ("", []),
    ],
)
def test_tokenize_smiles_splits_at_atoms_not_characters(smiles: str, tokens: list[str]) -> None:
    assert tokenize_smiles(smiles) == tokens


def test_vocabulary_starts_with_pad_and_unk_then_follows_first_appearance() -> None:
    vocabulary = build_vocabulary(["CCl", "OC"])

    assert vocabulary == {"<pad>": 0, "<unk>": 1, "C": 2, "Cl": 3, "O": 4}
    assert (vocabulary["<pad>"], vocabulary["<unk>"]) == (PAD_IDX, UNK_IDX)
    assert dict(BASE_VOCABULARY) == {"<pad>": 0, "<unk>": 1}


def test_encode_pads_truncates_and_maps_unknown_tokens_to_unk() -> None:
    vocabulary = build_vocabulary(["CCO"])

    encoded = encode_smiles(["CCO", "CN", "CCOCCO"], vocabulary, 4)

    np.testing.assert_array_equal(encoded, [[2, 2, 3, 0], [2, 1, 0, 0], [2, 2, 3, 2]])
    assert encoded.dtype == np.int64
    assert encode_smiles([], build_vocabulary([]), 5).shape == (0, 5)


def test_randomize_keeps_the_original_first_and_the_same_molecule() -> None:
    from rdkit import Chem

    variants = randomize_smiles([ASPIRIN], 6)[0]

    assert variants[0] == ASPIRIN
    assert len(variants) == 6
    assert len(set(variants)) > 1
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))
    assert {Chem.MolToSmiles(Chem.MolFromSmiles(variant)) for variant in variants} == {canonical}


def test_randomize_is_reproducible_and_shares_one_generator_across_molecules() -> None:
    assert randomize_smiles([ASPIRIN, "CCO"], 5) == randomize_smiles([ASPIRIN, "CCO"], 5)
    assert randomize_smiles([ASPIRIN, "CCOC(=O)C"], 5)[1] != randomize_smiles(["CCOC(=O)C"], 5)[0]


def test_randomize_repeats_a_smiles_rdkit_cannot_parse() -> None:
    assert randomize_smiles(["not a smiles"], 3) == [["not a smiles"] * 3]


SMILES = {"d1": "CCO", "d2": "CC(=O)Oc1ccccc1C(=O)O", "d3": "CCCl", "d4": "c1ccncc1"}
IDS = np.array(list(SMILES))


def _source(smiles: dict[str, Any] | None = None) -> DrugSource:
    return DrugSource(smiles=SMILES if smiles is None else smiles)


def _blocks(featurizer: SmilesTokensFeaturizer, ids: np.ndarray, source: DrugSource | None = None):
    return featurizer.transform(source or _source(), ids).blocks


def test_registers_and_declares_both_output_blocks() -> None:
    assert featurizer_registry("drug").get("smilesTokens") is SmilesTokensFeaturizer
    assert SmilesTokensFeaturizer.side == "drug"
    assert [spec.name for spec in SmilesTokensFeaturizer.output_blocks({})] == [TOKENS_BLOCK, VARIANTS_BLOCK]
    assert SmilesTokensFeaturizer.get_hyperparameter_space() == {}


@pytest.mark.parametrize("kwargs", [{"padding_length": 0}, {"n_variants": 0}])
def test_rejects_non_positive_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(InvalidInputError, match="padding_length >= 1 and n_variants >= 1"):
        SmilesTokensFeaturizer(**kwargs)


def test_tokens_block_holds_the_padded_token_indices_and_block_metadata() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=3).fit(_source(), IDS)

    result = featurizer.transform(_source(), IDS)

    tokens = result.blocks[TOKENS_BLOCK].values
    assert tokens.shape == (4, 8)
    assert tokens.dtype == np.float32
    np.testing.assert_array_equal(result.matrix, tokens)
    vocabulary_size = int(tokens.max()) + 1
    for block in result.blocks.values():
        annotations = dict(block.annotations or {})
        assert annotations["padding_length"] == 8
        assert annotations["n_variants"] == 3
        assert int(annotations["vocabulary_size"]) >= vocabulary_size


def test_variant_bank_matches_the_sorted_unique_smiles_draw() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=40, n_variants=5).fit(_source(), IDS)

    blocks = _blocks(featurizer, IDS)
    variants = blocks[VARIANTS_BLOCK].values

    unique = sorted(SMILES.values())
    bank = dict(zip(unique, randomize_smiles(unique, 5), strict=True))
    flat = [variant for molecule in bank.values() for variant in molecule]
    vocabulary = build_vocabulary(flat)
    assert variants.shape == (4, 5 * 40)
    assert not np.any(variants == 1)
    assert blocks[VARIANTS_BLOCK].annotations["vocabulary_size"] == len(vocabulary)
    np.testing.assert_array_equal(blocks[TOKENS_BLOCK].values, encode_smiles(SMILES.values(), vocabulary, 40))
    for row, smiles in enumerate(SMILES.values()):
        expected = encode_smiles(bank[smiles], vocabulary, 40).reshape(-1)
        np.testing.assert_array_equal(variants[row], expected)
        np.testing.assert_array_equal(variants[row, :40], blocks[TOKENS_BLOCK].values[row])


def test_fit_is_fold_specific_and_drugs_outside_the_fit_use_their_own_smiles() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=3).fit(_source(), np.array(["d1"]))

    tokens = featurizer.transform(_source(), np.array(["d1", "d3"])).matrix
    assert 1 not in tokens[0]
    np.testing.assert_array_equal(tokens[1][:3], [2, 2, 1])

    blocks = _blocks(featurizer, np.array(["d3"]))
    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, np.tile(blocks[TOKENS_BLOCK].values, (1, 3)))


def test_without_augmentation_the_bank_is_the_smiles_itself() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=20, augment=False).fit(_source(), IDS)

    blocks = _blocks(featurizer, IDS)

    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, blocks[TOKENS_BLOCK].values)
    assert blocks[VARIANTS_BLOCK].annotations["n_variants"] == 1


def test_fitting_twice_on_the_same_drugs_draws_the_same_bank_and_pickling_keeps_it() -> None:
    first = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), IDS)
    second = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), IDS[::-1])
    restored = pickle.loads(pickle.dumps(first))  # noqa: S301 -- pickled on this line

    expected = _blocks(first, IDS)
    for other in (second, restored):
        for name, block in _blocks(other, IDS).items():
            np.testing.assert_array_equal(block.values, expected[name].values)


def test_drugs_without_a_smiles_are_invalid() -> None:
    source = _source({**SMILES, "d5": None})
    ids = np.array(["d1", "d5"])
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=2)

    np.testing.assert_array_equal(featurizer.valid_mask(source, ids), [True, False])
    assert featurizer.valid_mask(DrugSource({"v": {"d1": [1.0]}}), np.array(["d1"])).tolist() == [True]


_IDS_SLICE = slice(0, 2)


def _fitted(
    drug_source: FeatureSource, drug_ids: np.ndarray, **kwargs
) -> tuple[BpePharmaformerDrugFeaturizer, np.ndarray]:
    drug_ids = drug_ids[_IDS_SLICE]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, **kwargs)
    return featurizer.fit(drug_source, drug_ids), drug_ids


def test_bpe_registration_and_hyperparameter_space() -> None:
    assert featurizer_registry("drug").get("bpePharmaformer") is BpePharmaformerDrugFeaturizer
    assert BpePharmaformerDrugFeaturizer.output_blocks({})[0].name == "bpe_smiles"
    assert set(BpePharmaformerDrugFeaturizer.get_hyperparameter_space()) == {"num_symbols", "max_length"}


@pytest.mark.parametrize(
    ("featurizer_cls", "fit_error"),
    [(SmilesTokensFeaturizer, "no SMILES available"), (BpePharmaformerDrugFeaturizer, "Cannot learn BPE codes")],
)
def test_transform_before_fit_and_fit_without_smiles_raise(featurizer_cls, fit_error) -> None:
    source = DrugSource({"v": {"d1": [1.0]}})
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer_cls().transform(source, np.array(["d1"]))
    with pytest.raises(ValueError, match=fit_error):
        featurizer_cls().fit(source, np.array(["d1"]))


def test_transform_without_smiles_raises(drug_source: FeatureSource, drug_ids: np.ndarray) -> None:
    featurizer, _ = _fitted(drug_source, drug_ids, max_length=16)
    with pytest.raises(ValueError, match="Cannot encode BPE"):
        featurizer.transform(DrugSource({"v": {"d1": [1.0]}}), np.array(["d1"]))


def test_fit_then_transform_emits_a_padded_token_matrix_with_one_block(
    drug_source: FeatureSource, drug_ids: np.ndarray
) -> None:
    featurizer, drug_ids = _fitted(drug_source, drug_ids, max_length=32)

    result = featurizer.transform(drug_source, drug_ids)

    assert result.matrix.shape == (2, 32)
    assert result.matrix.dtype == np.float32
    assert set(result.blocks) == {"bpe_smiles"}
    assert result.blocks["bpe_smiles"].feature_names is None
    assert featurizer.transform(drug_source, np.array([], dtype=str)).matrix.shape == (0, 32)


def test_transform_truncates_long_sequences_and_pads_short_ones(
    drug_source: FeatureSource, drug_ids: np.ndarray
) -> None:
    short, drug_ids = _fitted(drug_source, drug_ids, max_length=4)
    assert np.all(short.transform(drug_source, drug_ids).matrix > 0)
    padded, _ = _fitted(drug_source, drug_ids, max_length=512)
    assert np.any(padded.transform(drug_source, drug_ids).matrix[0] == 0.0)


def test_transform_leaves_unknown_drugs_as_zero_rows(drug_source: FeatureSource, drug_ids: np.ndarray) -> None:
    featurizer, _ = _fitted(drug_source, drug_ids, max_length=16)

    matrix = featurizer.transform(drug_source, np.array(["not-a-drug"])).matrix

    np.testing.assert_allclose(matrix, np.zeros((1, 16), dtype=np.float32))


def test_a_fitted_bpe_featurizer_pickles(drug_source: FeatureSource, drug_ids: np.ndarray) -> None:
    featurizer, drug_ids = _fitted(drug_source, drug_ids, max_length=32)

    restored = pickle.loads(pickle.dumps(featurizer))  # noqa: S301 -- pickled on this line

    np.testing.assert_array_equal(
        restored.transform(drug_source, drug_ids).matrix, featurizer.transform(drug_source, drug_ids).matrix
    )


def test_learn_bpe_reports_missing_subword_nmt(
    monkeypatch: pytest.MonkeyPatch, drug_source: FeatureSource, drug_ids: np.ndarray
) -> None:
    block_imports(monkeypatch, "subword_nmt")

    with pytest.raises(ImportError, match="subword-nmt is required"):
        BpePharmaformerDrugFeaturizer().fit(drug_source, drug_ids[_IDS_SLICE])


def test_learning_bpe_silences_progress_and_restores_the_module_binding(
    drug_source: FeatureSource, drug_ids: np.ndarray, capsys: pytest.CaptureFixture[str]
) -> None:
    from subword_nmt import learn_bpe as learn_bpe_module

    original = learn_bpe_module.tqdm

    _fitted(drug_source, drug_ids, max_length=16)

    assert learn_bpe_module.tqdm is original
    assert "it/s" not in capsys.readouterr().err
