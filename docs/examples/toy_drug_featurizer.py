"""Drug featurizer example: features derived from the identifier alone.

Shows two declarations the sample example does not need. It leaves ``input_views``
empty, which is how a featurizer states that it reads no view, only entity IDs. And
because its block is not named after a view, it overrides ``output_blocks``, so
config validation knows the block before anything is fitted.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np

from drevalpy.plugin import (
    BlockSpec,
    FeatureFormat,
    FeatureSource,
    Featurizer,
    FeaturizerResult,
    Side,
    dense_result,
    register_featurizer,
)

BLOCK = "toy_drug_hash"
N_COLUMNS = 8


@register_featurizer(
    "toyDrugHash",
    description="Deterministic pseudo-random drug features hashed from the drug identifier.",
)
class ToyDrugHashFeaturizer(Featurizer):
    """Hash each drug ID into a fixed-width vector.

    A stand-in for a real embedding: it reads no view, so it works
    against any dataset, including a response-only one.
    """

    sides: ClassVar[frozenset[Side]] = frozenset({"drug"})

    @classmethod
    def output_blocks(cls, config: Mapping[str, Any]) -> tuple[BlockSpec, ...]:
        """Declare the one numeric block, whatever the config."""
        _ = config
        return (BlockSpec(BLOCK, FeatureFormat.NUMERIC_MATRIX),)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        """Do nothing: hashing needs no statistics from the training drugs."""
        _ = source, train_ids, train_pair_ids, es_pair_ids
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        """Return one row of hashed bytes per drug."""
        _ = source
        rows = [self._hash_row(str(entity_id)) for entity_id in ids]
        matrix = np.vstack(rows) if rows else np.empty((0, N_COLUMNS))
        return dense_result(ids, BLOCK, matrix, None)

    @staticmethod
    def _hash_row(entity_id: str) -> np.ndarray:
        digest = hashlib.sha256(entity_id.encode()).digest()[:N_COLUMNS]
        return np.frombuffer(digest, dtype=np.uint8) / 255.0
