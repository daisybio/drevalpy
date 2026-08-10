"""Base classes for featurizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import BlockSpec, FeatureBlock
from drevalpy.components.core.contracts.contracts import FeatureContract, featurizer_contract
from drevalpy.components.core.features.feature_source import FeatureSource


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
    requires_view: ClassVar[bool] = False
    entity_id_only: ClassVar[bool] = False
    input_views: ClassVar[tuple[str, ...] | None] = None
    hpo_strategy: ClassVar[HPOStrategy] = HPOStrategy.CONTINUOUS

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

    @abstractmethod
    def fit(
        self,
        source: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> Featurizer:
        """Fit on the entities given by *entity_ids* (or all entities when ``None``).

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.

        :returns: Fitted featurizer instance (usually ``self``).
        """

    @abstractmethod
    def transform_blocks(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
    ) -> dict[str, FeatureBlock]:
        """Return named feature blocks aligned with *entity_ids*.

        This is the primary output method. Subclasses must implement this.

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform.

        :returns: Mapping of block name to ``FeatureBlock`` payloads aligned with *entity_ids*.
        """

    def transform(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Return a flat feature matrix by concatenating numeric blocks.

        Default implementation derives the matrix from transform_blocks.
        Override for custom flat-matrix behavior (e.g. multi-omics featurizers
        that return a subset of blocks as the flat matrix).

        :param source: Feature source providing views for the entity type.
        :param entity_ids: Entity identifiers to transform.

        :returns: Feature payloads aligned with *entity_ids*.
        """
        from drevalpy.components.core.contracts.contracts import FeatureFormat

        blocks = self.transform_blocks(source, entity_ids)
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

        key = find_variant_key(mdata, self.storage_key, hyperparameters)
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
        from drevalpy.components.featurizers.storage import fetch_from_modality, fetch_from_varm

        result = fetch_from_modality(mdata, key, entity_ids)
        if result is not None:
            return result
        return fetch_from_varm(mdata, key, entity_ids)

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

        index = next_variant_index(mdata, self.storage_key)
        actual_key = make_variant_key(self.storage_key, index)
        self._store_by_key(mdata, actual_key, entity_ids, data)
        register_variant(mdata, self.storage_key, actual_key, hyperparameters)

    def _store_by_key(self, mdata: Any, key: str, entity_ids: np.ndarray, data: np.ndarray) -> None:
        """Write data to MuData under the given key. Override for custom storage.

        Default stores in response.varm for drug-side, obsm for cell-line-side.

        :param mdata: MuData object.
        :param key: Storage key.
        :param entity_ids: Entity IDs.
        :param data: Data matrix.
        """
        response = mdata.mod["response"]
        response.varm[key] = data

    @classmethod
    def list_stored_variants(cls, mdata: Any) -> list[dict[str, Any]]:
        """Return available pre-computed HP settings for this featurizer.

        :param mdata: MuData object.
        :returns: List of variant dicts with "index", "params", "key".
        """
        from drevalpy.components.featurizers.storage import list_variants

        return list_variants(mdata, cls.storage_key)
