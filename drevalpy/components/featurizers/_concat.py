"""Shared dense concatenation logic for cell-line and drug featurizers."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.config import FeaturizerConfig
from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizers.base import Featurizer


class ConcatFeaturizersMixin:
    """Fit child featurizers independently and concatenate their dense outputs."""

    output_contract: ClassVar[FeatureContract] = FeatureContract(
        kind=FeatureKind.DENSE,
        scope="multi_view",
    )
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
        self._child_configs = [
            FeaturizerConfig.model_validate(
                normalize_featurizer_config(item, default_registry=registry),
            )
            if not isinstance(item, FeaturizerConfig)
            else item
            for item in featurizers
        ]
        self._children: list[tuple[str, Featurizer]] = []
        self._block_dims: dict[str, int] = {}
        self._output_dim = 0
        self._is_fitted = False

    def fit(
        self,
        features,
        *,
        entity_ids: np.ndarray | None = None,
    ):
        self._children = []
        self._block_dims = {}
        for config in self._child_configs:
            child = config.create_instance()
            child.fit(features, entity_ids=entity_ids)
            self._children.append((config.name, child))
            self._block_dims[config.name] = child.output_dim
        self._output_dim = sum(self._block_dims.values())
        self._is_fitted = True
        return self

    def transform(self, features, entity_ids: np.ndarray) -> np.ndarray:
        blocks = self.transform_blocks(features, entity_ids)
        if not blocks:
            return np.empty((len(entity_ids), 0), dtype=np.float32)
        return np.concatenate([blocks[name] for name, _ in self._children], axis=1).astype(np.float32)

    def transform_blocks(self, features, entity_ids: np.ndarray) -> dict[str, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError(self._not_fitted_msg)
        return {
            name: child.transform(features, entity_ids).astype(np.float32) for name, child in self._children
        }

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
