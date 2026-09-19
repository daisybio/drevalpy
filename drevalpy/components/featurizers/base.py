"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.hyperparameter_space import TunableComponentMixin
from drevalpy.components.featurizers._declarations import FeaturizerDeclarationsMixin
from drevalpy.components.featurizers._nan_tolerance import NanToleranceMixin
from drevalpy.components.featurizers.storage import FeaturizerStorageMixin
from drevalpy.types.data.batch.feature_block import FeatureBlock, FeaturizerResult
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

    requires_early_stopping: ClassVar[bool] = False

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

        if pair_expanded_ids is not None:
            pair_expanded_ids = pair_expanded_ids[np.isin(pair_expanded_ids, valid_ids)]
        if pair_expanded_es_ids is not None:
            pair_expanded_es_ids = pair_expanded_es_ids[self._detect_valid(source, pair_expanded_es_ids)]
        if self.requires_early_stopping and (pair_expanded_es_ids is None or len(pair_expanded_es_ids) == 0):
            raise ValueError(f"{type(self).__name__} requires a usable early-stopping holdout")
        self._fit(
            source,
            entity_ids=valid_ids,
            pair_expanded_ids=pair_expanded_ids,
            pair_expanded_es_ids=pair_expanded_es_ids,
        )
        return self

    def transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Compute aligned matrices and blocks once, padding unusable entities.

        :param source: Feature views for this entity side.
        :param entity_ids: Requested identifiers in output order.
        :returns: Both representations, with numeric NaNs or object None for invalid rows.
        """
        valid = self._detect_valid(source, entity_ids)
        self._warn_if_above_threshold(valid, f"{type(self).__name__}.transform_features")
        if not valid.any():
            features = self._empty_transform_features(source)
        else:
            features = self._transform_features(source, entity_ids[valid])
        if valid.all():
            return features
        numeric = features.matrix.dtype.kind != "O"
        matrix = np.full(
            (len(entity_ids), *features.matrix.shape[1:]),
            np.nan if numeric else None,
            dtype=np.result_type(features.matrix.dtype, np.float32) if numeric else object,
        )
        matrix[valid] = features.matrix
        return FeaturizerResult(matrix, self._expand_blocks_with_nan(features.blocks, valid, len(entity_ids)))

    def transform_blocks(self, source: FeatureSource, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Return the named blocks of the combined transformation.

        :param source: Feature source.
        :param entity_ids: Requested identifiers.
        :returns: Aligned feature blocks.
        """
        return self.transform_features(source, entity_ids).blocks

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return the matrix of the combined transformation.

        :param source: Feature source.
        :param entity_ids: Requested identifiers.
        :returns: Aligned numeric or object matrix.
        """
        return self.transform_features(source, entity_ids).matrix

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
    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Compute both representations for valid entities in one pass.

        :param source: Feature source.
        :param entity_ids: Nonempty valid identifiers.
        :returns: Matrix and blocks aligned to these identifiers.
        """

    @abstractmethod
    def _empty_transform_features(self, source: FeatureSource) -> FeaturizerResult:
        """Build zero-row outputs from fitted schema without invoking a backend.

        :param source: Feature names and global metadata, when needed.
        :returns: Empty representations retaining fitted dimensions and metadata.
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after ``fit``.

        :returns: Result.
        """
