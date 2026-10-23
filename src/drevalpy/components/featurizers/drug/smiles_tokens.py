"""SMILES token featurizer for PaccMann: padded token rows plus an augmented variant bank per drug.

``rdkit`` is imported lazily by :func:`~drevalpy.components.featurizers.drug._smiles_tokenizer.randomize_smiles`,
and only when augmentation is on. See ``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, cast

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.featurizers.drug._smiles_tokenizer import (
    BASE_VOCABULARY,
    build_vocabulary,
    encode_smiles,
    randomize_smiles,
)
from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.components.featurizers.drug.base import DrugFeaturizer
from drevalpy.errors import InvalidInputError
from drevalpy.registry.drug_featurizer import register
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource

TOKENS_BLOCK = "smiles_tokens"
VARIANTS_BLOCK = "smiles_token_variants"


@register(
    "smilesTokens",
    description="Atom-level SMILES token rows and an augmented bank of equivalent SMILES per drug (set-dependent).",
    contract=FeatureFormat.NUMERIC_MATRIX,
)
class SmilesTokensFeaturizer(DrugFeaturizer):
    """Tokenize SMILES with a vocabulary learned on the training drugs.

    Emits two blocks: ``smiles_tokens`` holds the padded token indices of each drug's own SMILES, and
    ``smiles_token_variants`` holds the ``n_variants`` equivalent SMILES of the drug, flattened to
    ``n_variants * padding_length`` columns. The variants of the training drugs are drawn once at fit, so a
    fit is fold-specific: the vocabulary covers the training SMILES and their variants, and any other token
    maps to ``<unk>``. Drugs outside the fit get their own SMILES in every variant slot.
    """

    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (
        BlockSpec(TOKENS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
        BlockSpec(VARIANTS_BLOCK, FeatureFormat.NUMERIC_MATRIX),
    )
    input_views: ClassVar[tuple[str, ...]] = (TOKENS_BLOCK,)
    source_views: ClassVar[tuple[str, ...]] = ("canonical_smiles",)
    precompute: ClassVar[bool] = False

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

    def _detect_valid(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Flag drugs that have a non-empty SMILES string.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers to check.
        :returns: Boolean mask; all ``True`` when the source carries no SMILES at all.
        """
        smiles = get_smiles_for_entities(source, entity_ids)
        if smiles is None:
            return np.ones(len(entity_ids), dtype=bool)
        return np.array([isinstance(value, str) and bool(value) for value in smiles.to_numpy()], dtype=bool)

    def _smiles(self, source: FeatureSource, entity_ids: np.ndarray) -> list[str]:
        """Read the SMILES string of every drug.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers.
        :returns: One SMILES string per identifier.
        :raises ValueError: If the source has no SMILES.
        """
        smiles = get_smiles_for_entities(source, entity_ids)
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

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> SmilesTokensFeaturizer:
        """Learn the vocabulary and draw the variant bank from the training drugs.

        :param source: Feature source providing drug views.
        :param entity_ids: Training drug identifiers.
        :param pair_expanded_ids: Unused.
        :param pair_expanded_es_ids: Unused.
        :returns: This featurizer, fitted.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        unique_smiles = sorted(set(self._smiles(source, ids)))
        variants = self._build_bank(unique_smiles)
        self._vocabulary = build_vocabulary(variant for molecule in variants for variant in molecule)
        self._bank = dict(zip(unique_smiles, variants, strict=True))
        self._bank_width = len(variants[0]) if variants else 1
        self._is_fitted = True
        return self

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Encode each drug's own SMILES and its variant bank.

        :param source: Feature source providing drug views.
        :param entity_ids: Drug identifiers to encode.
        :returns: Result with the ``smiles_tokens`` and ``smiles_token_variants`` blocks.
        :raises RuntimeError: If called before the featurizer is fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("SmilesTokensFeaturizer must be fit before transform")
        smiles = self._smiles(source, entity_ids)
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
            TOKENS_BLOCK: numeric_feature_block(tokens.astype(np.float32), metadata=metadata),
            VARIANTS_BLOCK: numeric_feature_block(variants.astype(np.float32), metadata=metadata),
        }
        return FeaturizerResult(entity_ids, blocks, (TOKENS_BLOCK,))

    @property
    def output_dim(self) -> int:
        """Return the width of the ``smiles_tokens`` block (the padding length)."""
        return self._padding_length

    def get_state(self) -> dict[str, object]:
        """Persist the vocabulary and the variant bank.

        :returns: Fitted state, or an empty mapping before fitting.
        """
        if not self._is_fitted:
            return {}
        return {
            "padding_length": self._padding_length,
            "vocabulary": dict(self._vocabulary),
            "bank": {smiles: list(variants) for smiles, variants in self._bank.items()},
            "bank_width": self._bank_width,
            "fitted": True,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the vocabulary and variant bank written by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        vocabulary, bank = state.get("vocabulary"), state.get("bank")
        if isinstance(vocabulary, dict):
            self._vocabulary = {str(token): int(index) for token, index in vocabulary.items()}
        if isinstance(bank, dict):
            self._bank = {
                str(smiles): [str(variant) for variant in cast(list, variants)] for smiles, variants in bank.items()
            }
        width = state.get("bank_width")
        if isinstance(width, int):
            self._bank_width = width
        length = state.get("padding_length")
        if isinstance(length, int):
            self._padding_length = length
        self._is_fitted = bool(state.get("fitted"))
