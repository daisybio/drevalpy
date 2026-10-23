"""Tests for the ``smilesTokens`` drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.smiles_tokens`. The source is a stub whose
``mdata`` serves the SMILES column, so nothing here needs a dataset.
"""

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
    """Feature source whose ``mdata`` serves a ``canonical_smiles`` column."""

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


def test_registers_under_its_name() -> None:
    assert drug_featurizer_registry.get("smilesTokens") is SmilesTokensFeaturizer


def test_declares_both_output_blocks() -> None:
    names = [spec.name for spec in SmilesTokensFeaturizer.output_block_specs]

    assert names == [TOKENS_BLOCK, VARIANTS_BLOCK]


@pytest.mark.parametrize("kwargs", [{"padding_length": 0}, {"n_variants": 0}])
def test_rejects_non_positive_sizes(kwargs: dict[str, int]) -> None:
    with pytest.raises(InvalidInputError, match="padding_length >= 1 and n_variants >= 1"):
        SmilesTokensFeaturizer(**kwargs)


def test_hyperparameter_space_is_empty() -> None:
    assert SmilesTokensFeaturizer.get_hyperparameter_space() == {}


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        SmilesTokensFeaturizer()._transform_features(_source(), IDS)


def test_fit_without_smiles_raises() -> None:
    with pytest.raises(ValueError, match="no SMILES available"):
        SmilesTokensFeaturizer().fit(MockFeatureSource({"d1": {}}), entity_ids=np.array(["d1"]))


def test_tokens_block_holds_the_padded_token_indices_of_each_smiles() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=3).fit(_source(), entity_ids=IDS)

    blocks = featurizer.transform_blocks(_source(), IDS)

    tokens = blocks[TOKENS_BLOCK].values
    assert tokens.shape == (4, 8)
    assert tokens.dtype == np.float32
    np.testing.assert_array_equal(tokens, encode_smiles(SMILES.values(), featurizer._vocabulary, 8))
    assert featurizer.output_dim == 8


def test_variant_bank_matches_the_sorted_unique_smiles_draw() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=40, n_variants=5).fit(_source(), entity_ids=IDS)

    variants = featurizer.transform_blocks(_source(), IDS)[VARIANTS_BLOCK].values

    unique = sorted(SMILES.values())
    bank = dict(zip(unique, randomize_smiles(unique, 5), strict=True))
    flat = [variant for molecule in bank.values() for variant in molecule]
    vocabulary = build_vocabulary(flat)
    assert variants.shape == (4, 5 * 40)
    for row, smiles in enumerate(SMILES.values()):
        expected = encode_smiles(bank[smiles], vocabulary, 40).reshape(-1)
        np.testing.assert_array_equal(variants[row], expected)
        np.testing.assert_array_equal(variants[row, :40], featurizer.transform(_source(), IDS)[row])


def test_vocabulary_covers_the_variants_of_the_training_drugs() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=32, n_variants=6).fit(_source(), entity_ids=IDS)
    blocks = featurizer.transform_blocks(_source(), IDS)

    assert int(blocks[VARIANTS_BLOCK].values.max()) < len(featurizer._vocabulary)
    assert not np.any(blocks[VARIANTS_BLOCK].values == 1)


def test_block_metadata_carries_the_sizes_the_network_needs() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=16, n_variants=4).fit(_source(), entity_ids=IDS)

    blocks = featurizer.transform_blocks(_source(), IDS)

    for block in blocks.values():
        assert dict(block.metadata or {}) == {
            "vocabulary_size": len(featurizer._vocabulary),
            "padding_length": 16,
            "n_variants": 4,
        }


def test_fit_is_fold_specific_and_unseen_tokens_become_unknown() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=8, n_variants=2).fit(_source(), entity_ids=np.array(["d1"]))

    tokens = featurizer.transform(_source(), np.array(["d1", "d3"]))

    assert 1 not in tokens[0]
    np.testing.assert_array_equal(tokens[1][:3], [2, 2, 1])


def test_drugs_outside_the_fit_get_their_own_smiles_in_every_variant_slot() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=3).fit(_source(), entity_ids=np.array(["d1"]))

    blocks = featurizer.transform_blocks(_source(), np.array(["d3"]))

    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, np.tile(blocks[TOKENS_BLOCK].values, (1, 3)))


def test_without_augmentation_the_bank_is_the_smiles_itself() -> None:
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=20, augment=False).fit(_source(), entity_ids=IDS)

    blocks = featurizer.transform_blocks(_source(), IDS)

    np.testing.assert_array_equal(blocks[VARIANTS_BLOCK].values, blocks[TOKENS_BLOCK].values)
    assert blocks[VARIANTS_BLOCK].metadata["n_variants"] == 1


def test_fitting_twice_on_the_same_drugs_draws_the_same_bank() -> None:
    first = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), entity_ids=IDS)
    second = SmilesTokensFeaturizer(padding_length=30, n_variants=4).fit(_source(), entity_ids=IDS[::-1])

    np.testing.assert_array_equal(
        first.transform_blocks(_source(), IDS)[VARIANTS_BLOCK].values,
        second.transform_blocks(_source(), IDS)[VARIANTS_BLOCK].values,
    )


def test_drugs_without_a_smiles_are_invalid_and_come_back_as_nan_rows() -> None:
    smiles = {**SMILES, "d5": None}
    source = _source(smiles)
    featurizer = SmilesTokensFeaturizer(padding_length=6, n_variants=2).fit(source, entity_ids=np.array(["d1", "d5"]))

    ids = np.array(["d1", "d5"])
    np.testing.assert_array_equal(featurizer._detect_valid(source, ids), [True, False])
    blocks = featurizer.transform_blocks(source, ids)
    assert not np.isnan(blocks[TOKENS_BLOCK].values[0]).any()
    assert np.isnan(blocks[TOKENS_BLOCK].values[1]).all()


def test_detect_valid_accepts_everything_when_the_source_has_no_smiles() -> None:
    mask = SmilesTokensFeaturizer()._detect_valid(MockFeatureSource({"d1": {}}), np.array(["d1"]))

    assert mask.tolist() == [True]


def test_state_round_trip_reproduces_the_blocks() -> None:
    fitted = SmilesTokensFeaturizer(padding_length=20, n_variants=3).fit(_source(), entity_ids=IDS)
    restored = SmilesTokensFeaturizer(padding_length=999)
    restored.set_state(fitted.get_state())

    for name, block in fitted.transform_blocks(_source(), IDS).items():
        np.testing.assert_array_equal(restored.transform_blocks(_source(), IDS)[name].values, block.values)


def test_state_is_empty_before_fit() -> None:
    assert SmilesTokensFeaturizer().get_state() == {}
