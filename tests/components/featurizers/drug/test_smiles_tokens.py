"""Tests for the ``smilesTokens`` drug featurizer; the source stub serves SMILES via ``mdata``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from drevalpy.components.featurizers.drug._smiles_tokenizer import (
    build_vocabulary,
    encode_smiles,
    randomize_smiles,
)
from drevalpy.components.featurizers.drug.smiles_tokens import (
    TOKENS_BLOCK,
    VARIANTS_BLOCK,
    SmilesTokensFeaturizer,
)
from drevalpy.errors import InvalidInputError
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from tests.conftest import MockFeatureSource

SMILES = {"d1": "CCO", "d2": "CC(=O)Oc1ccccc1C(=O)O", "d3": "CCCl", "d4": "c1ccncc1"}
IDS = np.array(list(SMILES))


class SmilesSource(MockFeatureSource):
    def __init__(self, smiles: dict[str, Any]) -> None:
        """Serve *smiles* by drug id."""
        super().__init__({drug: {} for drug in smiles})
        frame = pd.DataFrame({"canonical_smiles": pd.Series(smiles)})
        self._mdata = SimpleNamespace(mod={"response": SimpleNamespace(var=frame)})

    @property
    def mdata(self) -> Any:
        return self._mdata


def _source(smiles: dict[str, Any] | None = None) -> SmilesSource:
    return SmilesSource(SMILES if smiles is None else smiles)


def test_registers_and_declares_both_output_blocks() -> None:
    assert drug_featurizer_registry.get("smilesTokens") is SmilesTokensFeaturizer
    assert [spec.name for spec in SmilesTokensFeaturizer.output_block_specs] == [TOKENS_BLOCK, VARIANTS_BLOCK]
    assert SmilesTokensFeaturizer.get_hyperparameter_space() == {}
    assert SmilesTokensFeaturizer().get_state() == {}


@pytest.mark.parametrize("kwargs", [{"padding_length": 0}, {"n_variants": 0}])
def test_rejects_non_positive_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(InvalidInputError, match="padding_length >= 1 and n_variants >= 1"):
        SmilesTokensFeaturizer(**kwargs)


def test_transform_before_fit_and_fit_without_smiles_raise() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        SmilesTokensFeaturizer()._transform_features(_source(), IDS)
    with pytest.raises(ValueError, match="no SMILES available"):
        SmilesTokensFeaturizer().fit(MockFeatureSource({"d1": {}}), entity_ids=np.array(["d1"]))


def test_tokens_block_holds_the_padded_token_indices_and_block_metadata() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=3).fit(_source(), entity_ids=IDS)

    blocks = featurizer.transform_blocks(_source(), IDS)

    tokens = blocks[TOKENS_BLOCK].values
    assert tokens.shape == (4, 8)
    assert tokens.dtype == np.float32
    np.testing.assert_array_equal(tokens, encode_smiles(SMILES.values(), featurizer._vocabulary, 8))
    assert featurizer.output_dim == 8
    for block in blocks.values():
        assert dict(block.metadata or {}) == {
            "vocabulary_size": len(featurizer._vocabulary),
            "padding_length": 8,
            "n_variants": 3,
        }


def test_variant_bank_matches_the_sorted_unique_smiles_draw() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=40, n_variants=5).fit(_source(), entity_ids=IDS)

    variants = featurizer.transform_blocks(_source(), IDS)[VARIANTS_BLOCK].values

    unique = sorted(SMILES.values())
    bank = dict(zip(unique, randomize_smiles(unique, 5), strict=True))
    flat = [variant for molecule in bank.values() for variant in molecule]
    vocabulary = build_vocabulary(flat)
    assert variants.shape == (4, 5 * 40)
    assert not np.any(variants == 1)
    for row, smiles in enumerate(SMILES.values()):
        expected = encode_smiles(bank[smiles], vocabulary, 40).reshape(-1)
        np.testing.assert_array_equal(variants[row], expected)
        np.testing.assert_array_equal(variants[row, :40], featurizer.transform(_source(), IDS)[row])


def test_fit_is_fold_specific_and_drugs_outside_the_fit_use_their_own_smiles() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=3).fit(_source(), entity_ids=np.array(["d1"]))

    tokens = featurizer.transform(_source(), np.array(["d1", "d3"]))
    assert 1 not in tokens[0]
    np.testing.assert_array_equal(tokens[1][:3], [2, 2, 1])

    blocks = featurizer.transform_blocks(_source(), np.array(["d3"]))
    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, np.tile(blocks[TOKENS_BLOCK].values, (1, 3)))


def test_without_augmentation_the_bank_is_the_smiles_itself() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=20, augment=False).fit(_source(), entity_ids=IDS)

    blocks = featurizer.transform_blocks(_source(), IDS)

    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, blocks[TOKENS_BLOCK].values)
    assert blocks[VARIANTS_BLOCK].metadata["n_variants"] == 1


def test_fitting_twice_on_the_same_drugs_draws_the_same_bank_and_state_round_trips() -> None:
    first = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), entity_ids=IDS)
    second = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), entity_ids=IDS[::-1])
    restored = SmilesTokensFeaturizer(padding_length=999)
    restored.set_state(first.get_state())

    expected = first.transform_blocks(_source(), IDS)
    for other in (second, restored):
        for name, block in other.transform_blocks(_source(), IDS).items():
            np.testing.assert_array_equal(block.values, expected[name].values)


def test_drugs_without_a_smiles_are_invalid_and_come_back_as_nan_rows() -> None:
    source = _source({**SMILES, "d5": None})
    ids = np.array(["d1", "d5"])
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=2).fit(source, entity_ids=ids)

    np.testing.assert_array_equal(featurizer._detect_valid(source, ids), [True, False])
    blocks = featurizer.transform_blocks(source, ids)
    assert not np.isnan(blocks[TOKENS_BLOCK].values[0]).any()
    assert np.isnan(blocks[TOKENS_BLOCK].values[1]).all()
    assert SmilesTokensFeaturizer()._detect_valid(MockFeatureSource({"d1": {}}), np.array(["d1"])).tolist() == [True]
