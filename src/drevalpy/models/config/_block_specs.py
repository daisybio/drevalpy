"""Derive output block specs from featurizer config trees."""

from __future__ import annotations

from drevalpy.components.featurizers._leaf_kwargs import featurizer_leaf_kwargs
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.registry.featurizer import get_featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec


def resolve_output_block_specs(config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    """Resolve the named blocks emitted by a configured featurizer tree.

    A concat node emits the blocks of its children; a leaf emits
    ``Featurizer.output_blocks`` of its construction kwargs.

    :param config: Featurizer config node to inspect.
    :returns: Block specs emitted by the featurizer tree.
    """
    if config.name == "concatFeaturizers":
        return tuple(spec for child in config.featurizers or () for spec in resolve_output_block_specs(child))
    return tuple(get_featurizer(config.name).output_blocks(featurizer_leaf_kwargs(config)))
