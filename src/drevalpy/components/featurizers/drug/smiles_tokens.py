"""SMILES token featurizer for PaccMann: padded token rows plus an augmented variant bank per drug.

``rdkit`` is imported lazily by :func:`~drevalpy.components.featurizers.drug.smiles_tokenizer.randomize_smiles`,
and only when augmentation is on. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers.drug.smiles_tokenizer import (
    BASE_VOCABULARY,
    build_vocabulary,
    encode_smiles,
    randomize_smiles,
)
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.errors import InvalidInputError
from drevalpy.registry.featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource, Side

TOKENS_BLOCK = "smiles_tokens"
VARIANTS_BLOCK = "smiles_token_variants"


@register(
    "smilesTokens",
    description="Atom-level SMILES token rows and an augmented bank of equivalent SMILES per drug (set-dependent).",
)
class SmilesTokensFeaturizer(Featurizer):
    """Tokenize SMILES with a vocabulary learned on the training drugs.

    Emits two blocks: ``smiles_tokens`` holds the padded token indices of each drug's own SMILES, and
    ``smiles_token_variants`` holds the ``n_variants`` equivalent SMILES of the drug, flattened to
    ``n_variants * padding_length`` columns. The variants of the training drugs are drawn once at fit, so a
    fit is fold-specific: the vocabulary covers the training SMILES and their variants, and any other token
    maps to ``<unk>``. Drugs outside the fit get their own SMILES in every variant slot.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})
    input_views: ClassVar[tuple[str, ...]] = (TOKENS_BLOCK,)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)

    def __init__(self, *, padding_length: int = 512, n_variants: int = 20, augment: bool = True) -> None:
        """Configure the token rows and the variant bank.

        :param padding_length: Length of every token row; longer SMILES are cut off.
        :param n_variants: Equivalent SMILES per drug in the bank, including the drug's own SMILES.
        :param augment: Draw ``n_variants`` randomized SMILES per training drug. When ``False``, or when
            RDKit is missing, the bank holds the drug's own SMILES only.
        :raises InvalidInputError: If *padding_length* or *n_variants* is below one.
        """
        if padding_length < 1 or n_variants < 1:
            raise InvalidInputError("smilesTokens needs padding_length >= 1 and n_variants >= 1.")
        self._padding_length = int(padding_length)
        self._n_variants = int(n_variants)
        self._augment = bool(augment)
        self._vocabulary: dict[str, int] = dict(BASE_VOCABULARY)
        self._bank: dict[str, list[str]] = {}
        self._bank_width = 1
        self._is_fitted = False

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs.

        :returns: Empty mapping; the featurizer is not searched.
        """
        return {}

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Return the ``smiles_tokens`` and ``smiles_token_variants`` blocks.

        :param config: Featurizer config; unused.
        :returns: Two numeric block specs.
        """
        return (
            BlockSpec(TOKENS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
            BlockSpec(VARIANTS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
        )

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        """Flag drugs that have a non-empty SMILES string.

        :param source: Drug feature source.
        :param ids: Drug identifiers to check.
        :returns: Boolean mask; all ``True`` when the source carries no SMILES at all.
        """
        smiles = source.smiles(ids)
        if smiles is None:
            return np.ones(len(ids), dtype=bool)
        return np.array([isinstance(value, str) and bool(value) for value in smiles.to_numpy()], dtype=bool)

    def _smiles(self, source: FeatureSource, entity_ids: np.ndarray) -> list[str]:
        """Read the SMILES string of every drug.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers.
        :returns: One SMILES string per identifier.
        :raises ValueError: If the source has no SMILES.
        """
        smiles = source.smiles(entity_ids)
        if smiles is None:
            raise ValueError("Cannot tokenize SMILES: no SMILES available.")
        return [str(value) for value in smiles.to_numpy()]

    def _build_bank(self, unique_smiles: list[str]) -> list[list[str]]:
        """Draw the equivalent SMILES of each training molecule, or keep the molecule alone.

        :param unique_smiles: Distinct training SMILES, sorted.
        :returns: The variants of each molecule, the original SMILES first.
        """
        if not self._augment:
            return [[smiles] for smiles in unique_smiles]
        try:
            return randomize_smiles(unique_smiles, self._n_variants)
        except ImportError as err:  # pragma: no cover - depends on the environment
            warnings.warn(f"{err} Tokenizing SMILES without augmentation.", stacklevel=2)
            return [[smiles] for smiles in unique_smiles]

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Learn the vocabulary and draw the variant bank from the training drugs.

        :param source: Drug feature source.
        :param train_ids: Usable training drug IDs.
        :param train_pair_ids: Unused.
        :param es_pair_ids: Unused.
        :returns: This featurizer, fitted.
        """
        unique_smiles = sorted(set(self._smiles(source, train_ids)))
        variants = self._build_bank(unique_smiles)
        self._vocabulary = build_vocabulary(variant for molecule in variants for variant in molecule)
        self._bank = dict(zip(unique_smiles, variants, strict=True))
        self._bank_width = len(variants[0]) if variants else 1
        self._is_fitted = True
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Encode each drug's own SMILES and its variant bank.

        :param source: Drug feature source.
        :param ids: Drug IDs to encode, in output order.
        :returns: Result with the ``smiles_tokens`` and ``smiles_token_variants`` blocks.
        :raises RuntimeError: If called before the featurizer is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("SmilesTokensFeaturizer must be fit before transform")
        smiles = self._smiles(source, ids)
        length = self._padding_length
        tokens = encode_smiles(smiles, self._vocabulary, length)
        variants = np.zeros((len(smiles), self._bank_width * length), dtype=np.int64)
        for row, own in enumerate(smiles):
            bank = self._bank.get(own, [own] * self._bank_width)
            variants[row] = encode_smiles(bank, self._vocabulary, length).reshape(-1)
        metadata = {
            "vocabulary_size": len(self._vocabulary),
            "padding_length": length,
            "n_variants": self._bank_width,
        }
        blocks = {
            TOKENS_BLOCK: numeric_feature_block(tokens.astype(np.float32), annotations=metadata),
            VARIANTS_BLOCK: numeric_feature_block(variants.astype(np.float32), annotations=metadata),
        }
        return FeaturizerResult(ids, blocks, (TOKENS_BLOCK,))
