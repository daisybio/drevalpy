"""Drug featurizer example: features derived from the identifier alone.

Shows the two declarations the cell-line example does not: a ``contract`` on the
class body rather than in the decorator, and ``entity_id_only``, which is how a
featurizer states that it reads no raw feature view. Leaving both
``entity_id_only`` and ``input_views`` unset makes registration raise.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import numpy as np

from drevalpy.plugin import (
    DrugFeaturizer,
    FeatureBlock,
    FeatureFormat,
    FeatureSource,
    numeric_feature_block,
    register_drug_featurizer,
)

N_COLUMNS = 8


@register_drug_featurizer(
    "toyDrugHash",
    description="Deterministic pseudo-random drug features hashed from the drug identifier.",
)
class ToyDrugHashFeaturizer(DrugFeaturizer):
    """Hash each drug id into a fixed-width vector.

    A stand-in for a real embedding: it reads no feature view, so it works
    against any dataset, including a response-only one.
    """

    contract = FeatureFormat.NUMERIC_MATRIX
    entity_id_only: ClassVar[bool] = True

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> ToyDrugHashFeaturizer:
        """Do nothing: hashing needs no statistics from the training drugs."""
        _ = source, entity_ids, pair_expanded_ids, pair_expanded_es_ids
        return self

    def _transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return one row of hashed bytes per drug."""
        _ = source
        rows = [self._hash_row(str(entity_id)) for entity_id in entity_ids]
        values = np.vstack(rows) if rows else np.empty((0, N_COLUMNS), dtype=np.float32)
        return {"toy_drug_hash": numeric_feature_block(values)}

    @staticmethod
    def _hash_row(entity_id: str) -> np.ndarray:
        digest = hashlib.sha256(entity_id.encode()).digest()[:N_COLUMNS]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0

    @property
    def output_dim(self) -> int:
        """Fixed width, known before fitting."""
        return N_COLUMNS
