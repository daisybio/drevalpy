"""Shared dense concatenation logic for cell-line and drug featurizers."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.core.batch.feature_block import FeatureBlock, merge_feature_blocks
from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.core.fitting.featurizer_label import featurizer_config_block_label
from drevalpy.components.featurizers.base import Featurizer


class ConcatFeaturizersMixin:
    """Fit child featurizers independently and concatenate their dense outputs."""

    _not_fitted_msg: ClassVar[str] = "ConcatFeaturizers must be fit before transform"

    @classmethod
    def resolve_input_views(cls, **kwargs: Any) -> tuple[str, ...]:
        """Reject direct resolution; input views come from the child configs.

        :param kwargs: Unused featurizer kwargs.
        :raises TypeError: Always; use ``views_from_featurizer_config`` on the tree instead.
        """
        _ = kwargs
        msg = (
            f"{cls.__name__} has no input views of its own; resolve them from the child configs "
            "via drevalpy.components.core.data_loading.view_resolution.views_from_featurizer_config"
        )
        raise TypeError(msg)

    def _init_concat(
        self,
        *,
        featurizers: list[Any] | None,
        registry: str,
    ) -> None:
        # Local import avoids models↔featurizer package cycles during module import.
        from drevalpy.models.config.featurizer import FeaturizerConfig

        if not featurizers:
            msg = "featurizers must be a non-empty list"
            raise ValueError(msg)
        self._registry = registry
        self._children: list[tuple[str, Featurizer]] = []
        self._child_configs: list[FeaturizerConfig] = []
        self._block_dims: dict[str, int] = {}
        self._output_dim = 0
        self._is_fitted = False

        # Accept either already-built Featurizer instances or template configs/dicts.
        if all(isinstance(item, Featurizer) for item in featurizers):
            for item in featurizers:
                label = getattr(item, "registry_name", type(item).__name__)
                view = getattr(item, "_view", None)
                self._children.append(
                    (featurizer_config_block_label(str(label), view if isinstance(view, str) else None), item)
                )
            return

        parent = FeaturizerConfig.model_validate(
            {
                "name": "concatFeaturizers",
                "registry": registry,
                "featurizers": [
                    item.model_dump(mode="python") if isinstance(item, FeaturizerConfig) else item
                    for item in featurizers
                ],
            },
        )
        self._child_configs = list(parent.featurizers or ())
        self._materialize_children()

    def _materialize_children(self) -> None:
        if self._children:
            return
        if not self._child_configs:
            return
        children: list[tuple[str, Featurizer]] = []
        for config in self._child_configs:
            label = featurizer_config_block_label(config.name, config.view)
            children.append((label, config.create_instance()))
        self._children = children

    @staticmethod
    def _reject_non_numeric_children(children: list[tuple[str, Featurizer]]) -> None:
        for label, child in children:
            if child.contract.format != FeatureFormat.NUMERIC_MATRIX:
                msg = (
                    f"concat featurizer child {label!r} emits {child.contract.format.value}; "
                    "only numeric_matrix children are supported"
                )
                raise ValueError(msg)

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ):
        """Fit on training data.

        :param features: features.
        :param entity_ids: entity ids.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: Result.
        """
        self._materialize_children()
        self._reject_non_numeric_children(self._children)
        self._block_dims = {}
        for name, child in self._children:
            child.fit(
                features,
                entity_ids=entity_ids,
                pair_expanded_ids=pair_expanded_ids,
                pair_expanded_es_ids=pair_expanded_es_ids,
            )
            self._block_dims[name] = child.output_dim
        self._output_dim = sum(self._block_dims.values())
        self._is_fitted = True
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        """Transform inputs into feature payloads.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        """
        blocks = self.transform_blocks(features, entity_ids)
        numeric_blocks = [
            block.values.astype(np.float32)
            for block in blocks.values()
            if block.entity_aligned and block.format == FeatureFormat.NUMERIC_MATRIX
        ]
        if not numeric_blocks:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        return np.concatenate(numeric_blocks, axis=1)

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, FeatureBlock]:
        """Transform blocks.

        :param features: features.
        :param entity_ids: entity ids.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if not self._is_fitted:
            raise RuntimeError(self._not_fitted_msg)
        child_blocks = [child.transform_blocks(features, entity_ids) for _, child in self._children]
        return merge_feature_blocks(*child_blocks)

    @property
    def output_dim(self) -> int:
        """Return output feature dimension after fitting.

        :returns: Result.
        """
        return self._output_dim

    @property
    def block_dims(self) -> dict[str, int]:
        """Return per-child output dimensions after fitting.

        :returns: Result.
        """
        return dict(self._block_dims)

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {
            "child_states": {name: child.get_state() for name, child in self._children},
            "block_dims": dict(self._block_dims),
            "output_dim": self._output_dim,
            "fitted": self._is_fitted,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        self._materialize_children()
        child_states = state.get("child_states")
        if isinstance(child_states, dict):
            for name, child in self._children:
                child_state = child_states.get(name)
                if isinstance(child_state, dict):
                    child.set_state(child_state)
        block_dims = state.get("block_dims")
        if isinstance(block_dims, dict):
            self._block_dims = {str(key): int(value) for key, value in block_dims.items()}
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            self._output_dim = output_dim
        if state.get("fitted"):
            self._is_fitted = True
