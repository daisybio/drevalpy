"""Tissue metadata featurizer for samples."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.components.featurizers.sample.base import SampleFeaturizer
from drevalpy.data.utils import TISSUE_IDENTIFIER
from drevalpy.registry.sample_featurizer import register
from drevalpy.types.data.batch.feature_block import (
    BlockSpec,
    FeaturizerResult,
    global_feature_block,
    numeric_feature_block,
)
from drevalpy.types.data.feature_source import FeatureSource


def _tissue_label(source: FeatureSource, entity_id: str) -> str | None:
    raw = source.get_entity_view(str(entity_id), TISSUE_IDENTIFIER)
    if raw is None:
        return None
    label = np.asarray(raw).reshape(-1)[0]
    if isinstance(label, float) and np.isnan(label):
        return None
    return str(label)


@register(
    "tissue",
    description="One-hot encoding of tissue or lineage labels for sample entities.",
    output_format=FeatureFormat.NUMERIC_MATRIX,
)
class TissueFeaturizer(SampleFeaturizer):
    """Map each sample to a dense one-hot tissue vector."""

    input_views: ClassVar[tuple[str, ...]] = ()
    output_block_specs: ClassVar[tuple[BlockSpec, ...]] = (BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX),)

    def __init__(self, *, allow_missing: bool = False) -> None:
        """Initialize instance state.

        :param allow_missing: Treat samples without a tissue annotation as an extra ``__missing__``
            category instead of raising.
        """
        self._encoder = OneHotCategoryEncoder()
        self._allow_missing = bool(allow_missing)

    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> TissueFeaturizer:
        """Learn the tissue vocabulary from the training samples.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param pair_expanded_ids: Unused.
        :param pair_expanded_es_ids: Unused.
        :returns: This featurizer, fitted.
        :raises ValueError: If a sample lacks a tissue annotation and *allow_missing* is false, or
            no sample has one.
        """
        _ = pair_expanded_ids, pair_expanded_es_ids
        ids = entity_ids if entity_ids is not None else source.identifiers
        available: list[str] = []
        for entity_id in ids:
            label = _tissue_label(source, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in sample_input"
                    raise ValueError(msg)
                continue
            available.append(label)
        if not available:
            if self._allow_missing:
                self._encoder.fit_categories(np.array([], dtype=str))
                return self
            msg = "TissueFeaturizer requires tissue annotations in sample_input"
            raise ValueError(msg)
        self._encoder.fit_categories(np.asarray(available, dtype=str))
        return self

    def _feature_matrix(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return the one-hot tissue matrix for *entity_ids*.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Sample identifiers to encode.
        :returns: Float32 matrix of shape ``(len(entity_ids), output_dim)``.
        :raises ValueError: If a sample lacks a tissue annotation and *allow_missing* is false.
        """
        if self._encoder.output_dim == 0:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        categories: list[str] = []
        for entity_id in entity_ids:
            label = _tissue_label(source, str(entity_id))
            if label is None:
                if not self._allow_missing:
                    msg = "TissueFeaturizer requires tissue annotations in sample_input"
                    raise ValueError(msg)
                categories.append("__missing__")
            else:
                categories.append(label)
        return self._encoder.transform(np.asarray(categories, dtype=str))

    def _transform_features(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> FeaturizerResult:
        """Emit the one-hot ``tissue`` block and the fitted category labels as metadata.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Sample identifiers to encode.
        :returns: Result with ``tissue`` and ``tissue_categories`` blocks.
        """
        matrix = self._feature_matrix(source, entity_ids)
        return FeaturizerResult(
            entity_ids,
            {
                "tissue": numeric_feature_block(matrix),
                "tissue_categories": global_feature_block(
                    np.asarray(self._encoder.categories, dtype=str),
                ),
            },
            ("tissue",),
        )

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting."""
        return self._encoder.output_dim

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state."""
        return self._encoder.get_state()

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
        """
        self._encoder.set_state(state)
