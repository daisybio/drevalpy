"""Resolve view names from featurizer configs."""

from __future__ import annotations

from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.registry.featurizer import get_featurizer


def views_from_featurizer_config(config: FeaturizerConfig) -> list[str]:
    """Ask each featurizer in the tree which raw views it reads.

    :param config: Featurizer config node, possibly a concat parent.
    :returns: Raw view names required by the featurizer tree, with default hyperparameters.
    """
    if config.name == "concatFeaturizers":
        return [view for child in config.featurizers or () for view in views_from_featurizer_config(child)]
    return list(get_featurizer(config.name).views(featurizer_leaf_kwargs(config)))
