"""SMILES tokenization, vocabulary and augmentation behind the ``smilesTokens`` featurizer.

The tokenizer is the atom-level regular expression of ``pytoda.smiles.processing.SMILES_TOKENIZER``,
which the original PaccMann implementation tokenizes with. Splitting by character instead would break
multi-character atoms ("Cl", "Br") and bracket atoms such as "[C@@H]" or "[Pt+2]".

``rdkit`` is imported inside :func:`randomize_smiles` only: this module is imported eagerly with the
featurizer registry, and ``rdkit`` costs a noticeable import. See ``tests/test_import_cost_policy.py``.

The leading underscore keeps the module out of ``registry/_builtins.py::_discover_modules``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Final

import numpy as np

#: Copied verbatim from ``pytoda.smiles.processing.SMILES_TOKENIZER``.
SMILES_TOKENIZER: Final = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|" r"-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)

PAD_IDX: Final = 0
UNK_IDX: Final = 1

#: Vocabulary every fit starts from; learned tokens follow in order of first appearance.
BASE_VOCABULARY: Final[Mapping[str, int]] = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}

#: Seed of the generator that shuffles the atom order of every molecule, so repeated fits on the same
#: SMILES produce the same variant bank.
SMILES_AUGMENTATION_SEED: Final = 42


def tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into atom-level tokens.

    :param smiles: SMILES string.
    :returns: Tokens in order of appearance.
    """
    return [token for token in SMILES_TOKENIZER.split(smiles) if token]


def build_vocabulary(smiles_list: Iterable[str]) -> dict[str, int]:
    """Assign an index to every token, in order of first appearance after ``<pad>`` and ``<unk>``.

    :param smiles_list: SMILES strings to read tokens from.
    :returns: Token to index mapping.
    """
    vocabulary = dict(BASE_VOCABULARY)
    for smiles in smiles_list:
        for token in tokenize_smiles(smiles):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_smiles(smiles_list: Iterable[str], vocabulary: Mapping[str, int], padding_length: int) -> np.ndarray:
    """Encode SMILES strings as zero-padded token index rows.

    Tokens missing from *vocabulary* become ``<unk>``; tokens beyond *padding_length* are cut off.

    :param smiles_list: SMILES strings to encode.
    :param vocabulary: Token to index mapping.
    :param padding_length: Length of every row.
    :returns: Integer array of shape ``(n_smiles, padding_length)``.
    """
    rows = list(smiles_list)
    encoded = np.full((len(rows), padding_length), PAD_IDX, dtype=np.int64)
    for i, smiles in enumerate(rows):
        token_ids = [vocabulary.get(token, UNK_IDX) for token in tokenize_smiles(smiles)][:padding_length]
        encoded[i, : len(token_ids)] = token_ids
    return encoded


def randomize_smiles(smiles_list: list[str], n_variants: int) -> list[list[str]]:
    """Generate equivalent SMILES strings for the same molecules.

    Every molecule is re-serialized from a shuffled atom order, which gives a different string for the
    same molecule. One generator seeded with :data:`SMILES_AUGMENTATION_SEED` serves all molecules in
    the given order. A molecule RDKit cannot parse keeps its original SMILES.

    :param smiles_list: Distinct SMILES strings.
    :param n_variants: Number of variants per molecule, including the original SMILES.
    :returns: The variants of each molecule, the original SMILES first.
    :raises ImportError: If RDKit is not installed.
    """
    try:
        from rdkit import Chem, RDLogger
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("Please install rdkit to augment SMILES for PaccMann: pip install rdkit") from err

    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]  # RDKit warns loudly about SMILES it can still parse
    rng = np.random.default_rng(SMILES_AUGMENTATION_SEED)

    variants = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            variants.append([smiles] * n_variants)
            continue

        atom_order = list(range(molecule.GetNumAtoms()))
        molecule_variants = [smiles]
        for _ in range(n_variants - 1):
            rng.shuffle(atom_order)
            renumbered = Chem.RenumberAtoms(molecule, atom_order)
            molecule_variants.append(Chem.MolToSmiles(renumbered, canonical=False))
        variants.append(molecule_variants)

    return variants
