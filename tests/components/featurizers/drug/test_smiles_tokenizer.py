"""Tests for the SMILES tokenizer, vocabulary and augmentation helpers of ``smilesTokens``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug._smiles_tokenizer import (
    BASE_VOCABULARY,
    PAD_IDX,
    UNK_IDX,
    build_vocabulary,
    encode_smiles,
    randomize_smiles,
    tokenize_smiles,
)

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
