"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.contracts.hyperparameter_space import TunableComponentMixin
from drevalpy.components.featurizers._declarations import FeaturizerDeclarationsMixin
from drevalpy.components.featurizers._nan_tolerance import NanToleranceMixin
from drevalpy.components.featurizers.storage import FeaturizerStorageMixin
from drevalpy.types.data.batch.feature_block import FeatureBlock
from drevalpy.types.data.feature_source import FeatureSource


class HPOStrategy(Enum):
    """How a featurizer's hyperparameters are searched during HPO."""

    CONTINUOUS = "continuous"
    PRECOMPUTED = "precomputed"


class Featurizer(
    FeaturizerStorageMixin,
    FeaturizerDeclarationsMixin,
    NanToleranceMixin,
    TunableComponentMixin,
    ABC,
):
    """Transform feature tables into per-entity representation payloads.

    Cell-line featurizers consume cell-line features; drug featurizers consume
    drug features. Subclasses must be registered
    to the cell-line or drug featurizer registry using
    ``@register`` (from the cell_line_featurizer or drug_featurizer registry), so that
    they can be discovered and used in models.

    Each subclass declares which raw feature views it reads via ``input_views``
    (or ``requires_view`` / ``entity_id_only`` / a ``resolve_input_views``
    override); registration rejects featurizers that declare nothing.

    ``contract`` may be declared on the class body or passed to ``@register``. When
    both are given the decorator argument wins.

    What is left in this class is the fit/transform contract itself. The concerns
    that reach none of it live in mixins: the class-body declarations in
    ``_declarations.py``, the NaN policy the public methods below bracket their
    subclass hooks with in ``_nan_tolerance.py``, the pre-computed variant store
    in ``storage.py``, and the HPO-space and checkpoint hooks it shares verbatim
    with ``Predictor`` in ``contracts/hyperparameter_space.py``.
    """

    hpo_strategy: ClassVar[HPOStrategy] = HPOStrategy.CONTINUOUS

    # ------------------------------------------------------------------
    # Public fit / transform with NaN tolerance
    # ------------------------------------------------------------------

    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> Featurizer:
        """Fit on valid entities, skipping those with all-NaN feature rows.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.

        :returns: Fitted featurizer instance (usually ``self``).
        """
        ids = entity_ids if entity_ids is not None else source.identifiers
        valid_mask = self._detect_valid(source, ids)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.fit")
        valid_ids = ids[valid_mask] if not valid_mask.all() else ids

        self._fit(
            source,
            entity_ids=valid_ids,
            pair_expanded_ids=pair_expanded_ids,
            pair_expanded_es_ids=pair_expanded_es_ids,
        )
        return self

    def transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Public NaN-safe entry point for block-based transform.

        Detects invalid (all-NaN) entities, transforms only valid ones via
        ``_transform_blocks``, and inserts NaN rows for invalid entities.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform.
        :returns: Mapping of block name to ``FeatureBlock`` payloads aligned with *entity_ids*.
        """
        valid_mask = self._detect_valid(source, entity_ids)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.transform_blocks")
        if valid_mask.all():
            return self._transform_blocks(source, entity_ids)
        valid_ids = entity_ids[valid_mask]
        valid_blocks = self._transform_blocks(source, valid_ids)
        return self._expand_blocks_with_nan(valid_blocks, valid_mask, len(entity_ids))

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Public NaN-safe entry point for matrix-based transform.

        Detects invalid (all-NaN) entities, transforms only valid ones via
        ``_transform``, and inserts NaN rows for invalid entities.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform.
        :returns: Feature matrix aligned with *entity_ids*.
        """
        valid_mask = self._detect_valid(source, entity_ids)
        self._warn_if_above_threshold(valid_mask, f"{type(self).__name__}.transform")
        if valid_mask.all():
            return self._transform(source, entity_ids)
        valid_ids = entity_ids[valid_mask]
        valid_result = self._transform(source, valid_ids)
        result = np.full((len(entity_ids), valid_result.shape[1]), np.nan, dtype=np.float32)
        result[valid_mask] = valid_result
        return result

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> Featurizer:
        """Subclass fitting logic on pre-validated (non-NaN) entity IDs.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.

        :returns: Fitted featurizer instance (usually ``self``).
        """

    @abstractmethod
    def _transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Return named feature blocks for pre-validated (non-NaN) entity IDs.

        Subclasses must implement this. Called by ``transform_blocks`` after
        NaN filtering.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform (only valid ones).
        :returns: Mapping of block name to ``FeatureBlock`` payloads aligned with *entity_ids*.
        """

    def _transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return a flat feature matrix by concatenating numeric blocks.

        Default: derives from ``_transform_blocks``. Subclasses that work
        directly on matrices can override this.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform (only valid ones).
        :returns: Feature matrix aligned with *entity_ids*.
        """
        blocks = self._transform_blocks(source, entity_ids)
        arrays = [b.values for b in blocks.values() if b.entity_aligned and b.format == FeatureFormat.NUMERIC_MATRIX]
        if not arrays:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        return np.concatenate(arrays, axis=1)

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after ``fit``.

        :returns: Result.
        """
