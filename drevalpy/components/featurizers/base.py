"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock
from drevalpy.components.core.contracts.contracts import FeatureContract, featurizer_contract
from drevalpy.components.core.features.feature_source import FeatureSource
from drevalpy.log import get_logger

_logger = get_logger(__name__)


class HPOStrategy(Enum):
    """How a featurizer's hyperparameters are searched during HPO."""

    CONTINUOUS = "continuous"
    PRECOMPUTED = "precomputed"


class Featurizer(ABC):
    """Transform feature tables into per-entity representation payloads.

    Cell-line featurizers consume cell-line features; drug featurizers consume
    drug features. Subclasses must be registered
    to the cell-line or drug featurizer registry using
    ``@register_cell_line_featurizer`` or ``@register_drug_featurizer``, so that
    they can be discovered and used in models.

    Each subclass declares which raw feature views it reads via ``input_views``
    (or ``requires_view`` / ``entity_id_only`` / a ``resolve_input_views``
    override); registration rejects featurizers that declare nothing.
    """

    contract: ClassVar[FeatureContract]
    storage_key: ClassVar[str] = ""
    side: ClassVar[str] = ""
    learned: ClassVar[bool] = False
    requires_view: ClassVar[bool] = False
    entity_id_only: ClassVar[bool] = False
    input_views: ClassVar[tuple[str, ...] | None] = None
    hpo_strategy: ClassVar[HPOStrategy] = HPOStrategy.CONTINUOUS
    nan_threshold: ClassVar[float] = 0.2

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject class-body ``contract`` assignments; registration sets it later.

        :param kwargs: Forwarded to ``ABC.__init_subclass__``.
        :raises TypeError: If ``contract`` is assigned on the subclass body.
        """
        super().__init_subclass__(**kwargs)
        if "contract" in cls.__dict__:
            msg = (
                f"{cls.__name__}: do not set contract on the class body; "
                "pass contract= to @register_cell_line_featurizer / @register_drug_featurizer"
            )
            raise TypeError(msg)

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
        from drevalpy.components.core.contracts.contracts import FeatureFormat

        blocks = self._transform_blocks(source, entity_ids)
        arrays = [b.values for b in blocks.values() if b.entity_aligned and b.format == FeatureFormat.NUMERIC_MATRIX]
        if not arrays:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        return np.concatenate(arrays, axis=1)

    # ------------------------------------------------------------------
    # NaN detection and expansion helpers
    # ------------------------------------------------------------------

    def _expand_blocks_with_nan(
        self,
        valid_blocks: dict[str, FeatureBlock],
        valid_mask: np.ndarray,
        n_total: int,
    ) -> dict[str, FeatureBlock]:
        """Expand valid-only blocks back to full size, inserting NaN for invalid rows.

        Non-entity-aligned blocks are passed through unchanged.

        :param valid_blocks: Blocks computed on only valid entity IDs.
        :param valid_mask: Boolean mask of shape ``(n_total,)`` (True = valid).
        :param n_total: Total number of entities (valid + invalid).
        :returns: Blocks aligned to the full set of entity IDs.
        """
        from drevalpy.components.core.contracts.contracts import FeatureFormat

        expanded: dict[str, FeatureBlock] = {}
        for name, block in valid_blocks.items():
            if not block.entity_aligned:
                expanded[name] = block
                continue
            if block.format == FeatureFormat.NUMERIC_MATRIX:
                full = np.full((n_total, block.values.shape[1]), np.nan, dtype=np.float32)
                full[valid_mask] = block.values
                expanded[name] = FeatureBlock(
                    values=full,
                    format=block.format,
                    feature_names=block.feature_names,
                    metadata=block.metadata,
                    entity_aligned=True,
                )
            else:
                full = np.empty(n_total, dtype=object)
                full[:] = None
                full[valid_mask] = block.values
                expanded[name] = FeatureBlock(
                    values=full,
                    format=block.format,
                    feature_names=block.feature_names,
                    metadata=block.metadata,
                    entity_aligned=True,
                )
        return expanded

    def _detect_valid(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return a boolean mask indicating which entities have non-NaN features.

        Default: entity_id_only featurizers treat all as valid; view-based
        featurizers check the first input view for all-NaN rows.

        :param source: Feature source.
        :param entity_ids: Entity IDs to check.
        :returns: Boolean array of shape ``(len(entity_ids),)``.
        """
        if self.entity_id_only:
            return np.ones(len(entity_ids), dtype=bool)

        view = getattr(self, "_view", None)
        if view is None and self.input_views:
            view = self.input_views[0]
        if view is None:
            return np.ones(len(entity_ids), dtype=bool)

        try:
            matrix = source.get_view_matrix(view, entity_ids)
        except (KeyError, TypeError, ValueError):
            return np.ones(len(entity_ids), dtype=bool)

        if matrix.ndim != 2 or matrix.dtype.kind not in ("f", "i", "u"):
            return np.ones(len(entity_ids), dtype=bool)

        return ~np.all(np.isnan(matrix), axis=1)

    def _warn_if_above_threshold(self, valid_mask: np.ndarray, context: str) -> None:
        """Log a warning when the fraction of invalid entities exceeds the threshold.

        :param valid_mask: Boolean array (True = valid).
        :param context: Human-readable label for the warning message.
        """
        if len(valid_mask) == 0:
            return
        invalid_frac = 1.0 - valid_mask.mean()
        if invalid_frac > self.nan_threshold:
            _logger.warning(
                "%s: %.0f%% of inputs are invalid (threshold: %.0f%%)",
                context,
                invalid_frac * 100,
                self.nan_threshold * 100,
            )

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Feature dimension after ``fit``.

        :returns: Result.
        """

    @classmethod
    def output_block_specs_for_config(cls, config: Any) -> tuple[BlockSpec, ...]:
        """Return named output blocks for a featurizer config node.

        Declared ``output_block_specs`` win when present; otherwise a single
        block named after the configured (or single declared input) view is emitted.

        :param config: Featurizer config with optional ``view`` / ``hyperparameters``.
        :returns: Block specs emitted by this featurizer under *config*.
        """
        declared = getattr(cls, "output_block_specs", ())
        if declared:
            return tuple(spec for spec in declared if isinstance(spec, BlockSpec))
        view = getattr(config, "view", None)
        if not isinstance(view, str):
            view = cls.input_views[0] if cls.input_views else None
        if isinstance(view, str):
            return (BlockSpec(view, featurizer_contract(cls).format),)
        return ()

    @classmethod
    def resolve_input_views(cls, **kwargs: Any) -> tuple[str, ...]:
        """Return the raw feature views this featurizer reads under *kwargs*.

        An explicit ``view`` kwarg always wins, which covers view-parameterized
        featurizers such as ``raw`` and ``pca``. Otherwise the declared
        ``input_views`` are used. Featurizers whose input depends on other
        hyperparameters override this hook.

        :param kwargs: Featurizer construction / loader kwargs from the model config.
        :returns: Raw view names required from disk, empty when only entity ids are needed.
        :raises TypeError: If the views cannot be determined from *kwargs* and the class body.
        """
        view = kwargs.get("view")
        if isinstance(view, str) and view.strip():
            return (view,)
        if cls.input_views is not None:
            return cls.input_views
        if cls.entity_id_only:
            return ()
        if cls.requires_view:
            msg = f"{cls.__name__} requires an explicit view; pass view= to resolve_input_views"
            raise TypeError(msg)
        msg = (
            f"{cls.__name__}: declare input_views on the class body, set requires_view/entity_id_only, "
            "or override resolve_input_views"
        )
        raise TypeError(msg)

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Return tunable hyperparameter specs for HPO.

        :returns: Mapping of parameter name to Ray Tune-style spec dicts.
        """
        return {}

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, object]:
        """Return default hyperparameter values from the HP space.

        :returns: Parameter names mapped to their declared ``default`` values.
        """
        from drevalpy.components.core.contracts.hyperparameter_space import validate_hyperparameter_space

        space = cls.get_hyperparameter_space()
        validate_hyperparameter_space(space, context=f"{cls.__name__}.get_hyperparameter_space()")
        return {key: spec["default"] for key, spec in space.items()}

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state for legacy save/load bridges.

        :returns: JSON-serializable mapping of fitted attributes.
        """
        return {}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore fitted state produced by ``get_state``.

        :param state: Mapping previously returned by ``get_state``.
        """
        _ = state

    def fetch(
        self, mdata: Any, entity_ids: np.ndarray, hyperparameters: dict[str, Any] | None = None
    ) -> np.ndarray | None:
        """Fetch pre-computed representations from MuData for the given HPs.

        :param mdata: MuData object.
        :param entity_ids: Entity IDs to fetch for.
        :param hyperparameters: HP setting to match. None matches default (empty params).
        :returns: Feature matrix or None if not pre-computed for these HPs.
        """
        from drevalpy.components.featurizers.storage import find_variant_key

        key = find_variant_key(mdata, self.storage_key, hyperparameters, side=self.side)
        if key is None:
            return None
        return self._fetch_by_key(mdata, key, entity_ids)

    def _fetch_by_key(self, mdata: Any, key: str, entity_ids: np.ndarray) -> np.ndarray | None:
        """Fetch data from MuData by resolved key. Override for custom storage.

        :param mdata: MuData object.
        :param key: Resolved storage key (e.g., "pca_expression_0").
        :param entity_ids: Entity IDs to align to.
        :returns: Feature matrix or None.
        """
        from drevalpy.components.featurizers.storage import fetch_from_modality, fetch_from_obsm, fetch_from_varm

        result = fetch_from_modality(mdata, key, entity_ids)
        if result is not None:
            return result
        if self.side == "drug":
            return fetch_from_varm(mdata, key, entity_ids)
        return fetch_from_obsm(mdata, key, entity_ids)

    def store(
        self, mdata: Any, entity_ids: np.ndarray, data: np.ndarray, hyperparameters: dict[str, Any] | None = None
    ) -> None:
        """Store computed representations into MuData with HP metadata.

        :param mdata: MuData object.
        :param entity_ids: Entity IDs the data is aligned to.
        :param data: Feature matrix to store.
        :param hyperparameters: HP settings this data was computed with.
        """
        from drevalpy.components.featurizers.storage import make_variant_key, next_variant_index, register_variant

        index = next_variant_index(mdata, self.storage_key, side=self.side)
        actual_key = make_variant_key(self.storage_key, index)
        self._store_by_key(mdata, actual_key, entity_ids, data)
        register_variant(mdata, self.storage_key, actual_key, hyperparameters, side=self.side)

    def _store_by_key(self, mdata: Any, key: str, entity_ids: np.ndarray, data: np.ndarray) -> None:
        """Write data to MuData under the given key. Override for custom storage.

        Default stores in response.obsm for cell-line-side, varm for drug-side.

        :param mdata: MuData object.
        :param key: Storage key.
        :param entity_ids: Entity IDs.
        :param data: Data matrix.
        """
        response = mdata.mod["response"]
        if self.side == "drug":
            response.varm[key] = data
        else:
            response.obsm[key] = data

    @classmethod
    def list_stored_variants(cls, mdata: Any) -> dict[str, dict[str, Any]]:
        """Return available pre-computed HP settings for this featurizer.

        :param mdata: MuData object.
        :returns: Dict of {mudata_key: params_dict}.
        """
        from drevalpy.components.featurizers.storage import list_variants

        return list_variants(mdata, cls.storage_key, side=cls.side)
