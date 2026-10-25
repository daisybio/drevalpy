"""Shared dense concatenation logic for sample and drug featurizers."""

from __future__ import annotations

from typing import Any, NoReturn, Self

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._featurizer_label import featurizer_config_block_label
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.types.data.batch.feature_block import FeaturizerResult, merge_feature_blocks
from drevalpy.types.data.feature_source import FeatureSource


class ConcatFeaturizersMixin:
    """Fit child featurizers independently and concatenate their dense outputs."""

    def _detect_valid(self, source: FeatureSource, entity_ids: np.ndarray) -> np.ndarray:
        """Require usable rows in every configured child view."""
        return np.logical_and.reduce([child._detect_valid(source, entity_ids) for _, child in self._children])

    @classmethod
    def resolve_input_views(cls, **kwargs: Any) -> NoReturn:
        r"""Reject direct resolution; input views come from the child configs.

        :param \*\*kwargs: Unused featurizer kwargs.
        :raises TypeError: Always; use ``views_from_featurizer_config`` on the tree instead.
        """
        _ = kwargs
        msg = (
            f"{cls.__name__} has no input views of its own; resolve them from the child configs "
            "via drevalpy.models.config.view_resolution.views_from_featurizer_config"
        )
        raise TypeError(msg)

    def _init_concat(
        self,
        *,
        featurizers: list[Any] | None,
        registry: str,
    ) -> None:
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
            if child.output_format != FeatureFormat.NUMERIC_MATRIX:
                msg = (
                    f"concat featurizer child {label!r} emits {child.output_format.value}; "
                    "only numeric_matrix children are supported"
                )
                raise ValueError(msg)

    def _fit(
        self,
        features: FeatureSource,
        *,
        entity_ids: np.ndarray | None = None,
        pair_expanded_ids: np.ndarray | None = None,
        pair_expanded_es_ids: np.ndarray | None = None,
    ) -> Self:
        """Fit every child featurizer on the same inputs and record their output widths.

        A child that does not emit a numeric matrix is rejected with ``ValueError``.

        :param features: Feature source providing views for the entity type.
        :param entity_ids: Subset of entity identifiers to fit on; ``None`` uses all.
        :param pair_expanded_ids: Training entity IDs with duplicates per response pair.
        :param pair_expanded_es_ids: Early-stopping entity IDs with duplicates.
        :returns: This featurizer, fitted.
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

    def _transform_features(self, features: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Transform with every child and merge their blocks.

        :param features: Feature source providing views for the entity type.
        :param entity_ids: Nonempty valid identifiers, in output order.
        :returns: The children's feature blocks merged into one result.
        :raises RuntimeError: If called before the featurizer is fitted.
        """
        if not self._is_fitted:
            msg = f"{type(self).__name__} must be fit before transform"
            raise RuntimeError(msg)
        child_blocks = [child.transform_features(features, entity_ids).blocks for _, child in self._children]
        return FeaturizerResult.from_blocks(merge_feature_blocks(*child_blocks), entity_ids)

    @property
    def output_dim(self) -> int:
        """Return the summed output dimension of all children after fitting."""
        return self._output_dim

    @property
    def block_dims(self) -> dict[str, int]:
        """Return per-child output dimensions after fitting, keyed by block label."""
        return dict(self._block_dims)

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state, including each child's own state."""
        return {
            "child_states": {name: child.get_state() for name, child in self._children},
            "block_dims": dict(self._block_dims),
            "output_dim": self._output_dim,
            "fitted": self._is_fitted,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: Mapping previously returned by ``get_state``.
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
