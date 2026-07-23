"""Walk featurizer config trees for tuning helpers."""

from __future__ import annotations

from collections.abc import Iterator

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import FeaturizerConfig, ModelConfig


def iter_featurizer_configs(config: ModelConfig) -> Iterator[FeaturizerConfig]:
    """Yield every leaf featurizer config in a model config.

    :param config: Model config whose featurizer trees are walked.
    :yields: Leaf ``FeaturizerConfig`` nodes from cell-line and drug trees.
    """
    for featurizer in (config.cell_line_featurizer, config.drug_featurizer):
        if featurizer is None:
            continue
        yield from walk_featurizer_configs(featurizer, str(featurizer.registry))


def walk_featurizer_configs(
    featurizer: FeaturizerConfig,
    registry: str,
) -> Iterator[FeaturizerConfig]:
    """Yield leaf featurizer configs from a featurizer tree.

    :param featurizer: Root or intermediate featurizer config.
    :param registry: Default registry used when normalizing nested children.
    :yields: Leaf ``FeaturizerConfig`` nodes (concat parents are expanded).
    """
    if featurizer.name == "concatFeaturizers":
        for child in featurizer.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(
                normalize_featurizer_config(child, default_registry=registry),
            )
            yield from walk_featurizer_configs(child_cfg, registry)
        return
    yield featurizer
