"""Shared dense concatenation logic for cell-line and drug featurizers."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock, merge_feature_blocks
from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizer_label import featurizer_config_block_label
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.models.config import FeaturizerConfig


class ConcatFeaturizersMixin:
    """Fit child featurizers independently and concatenate their dense outputs."""

    _not_fitted_msg: ClassVar[str] = "ConcatFeaturizers must be fit before transform"

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
        # Normalize through FeaturizerConfig so uniqueness is checked once, after construction.
        child_payloads = [item.model_dump() if isinstance(item, FeaturizerConfig) else item for item in featurizers]
        parent = FeaturizerConfig.model_validate(
            {
                "name": "concatFeaturizers",
                "registry": registry,
                "hyperparameters": {"featurizers": child_payloads},
            },
        )
        children = parent.hyperparameters.get("featurizers", [])
        self._child_configs = [
            child if isinstance(child, FeaturizerConfig) else FeaturizerConfig.model_validate(child)
            for child in children
        ]
        self._children: list[tuple[str, Featurizer]] = []
        self._block_dims: dict[str, int] = {}
        self._output_dim = 0
        self._is_fitted = False
        self._materialize_children()

    def _materialize_children(self) -> None:
        if len(self._children) == len(self._child_configs):
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
        context: FeaturizerFitContext | None = None,
    ):
        self._materialize_children()
        self._reject_non_numeric_children(self._children)
        self._block_dims = {}
        for name, child in self._children:
            child.fit(features, entity_ids=entity_ids, context=context)
            self._block_dims[name] = child.output_dim
        self._output_dim = sum(self._block_dims.values())
        self._is_fitted = True
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
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
        if not self._is_fitted:
            raise RuntimeError(self._not_fitted_msg)
        child_blocks = [child.transform_blocks(features, entity_ids) for _, child in self._children]
        return merge_feature_blocks(*child_blocks)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def block_dims(self) -> dict[str, int]:
        return dict(self._block_dims)

    def get_state(self) -> dict[str, object]:
        return {
            "child_states": {name: child.get_state() for name, child in self._children},
            "block_dims": dict(self._block_dims),
            "output_dim": self._output_dim,
            "fitted": self._is_fitted,
        }

    def set_state(self, state: dict[str, object]) -> None:
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
