"""Walk and transform featurizer config trees."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import FeaturizerConfig


def iter_featurizer_leaves(
    featurizer: FeaturizerConfig,
    registry: str,
) -> Iterator[FeaturizerConfig]:
    """Yield leaf featurizer configs from a tree (concat parents are expanded)."""
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            yield from iter_featurizer_leaves(child_cfg, registry)
        return
    yield featurizer


def map_featurizer_tree(
    featurizer: FeaturizerConfig,
    registry: str,
    transform_leaf: Callable[[FeaturizerConfig], FeaturizerConfig],
) -> FeaturizerConfig:
    """Return a copy of ``featurizer`` with ``transform_leaf`` applied at each leaf."""
    if featurizer.name == "concatFeaturizers":
        children = []
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            children.append(map_featurizer_tree(child_cfg, registry, transform_leaf).model_dump())
        return featurizer.model_copy(
            update={"hyperparameters": {**featurizer.hyperparameters, "featurizers": children}},
            deep=True,
        )
    return transform_leaf(featurizer)
