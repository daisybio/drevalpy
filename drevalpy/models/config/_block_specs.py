"""Derive output block specs from featurizer config trees."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import featurizer_contract
from drevalpy.components.feature_block import BlockSpec
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer
from drevalpy.models.config.featurizer import FeaturizerConfig


def _lookup_featurizer_class(config: FeaturizerConfig) -> type[Any]:
    if config.registry == "cell_line":
        return get_cell_line_featurizer(config.name)
    return get_drug_featurizer(config.name)


def _fallback_block_specs(cls: type[Any], config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    """Resolve declared specs or a single view-named block for non-base classes.

    :param cls: Registered featurizer class.
    :param config: Featurizer config node being resolved.
    :returns: Block specs emitted by *config*.
    """
    declared = getattr(cls, "output_block_specs", ())
    if declared:
        return tuple(spec for spec in declared if isinstance(spec, BlockSpec))
    view = config.view or getattr(cls, "_default_view", None)
    if isinstance(view, str):
        return (BlockSpec(view, featurizer_contract(cls).format),)
    return ()


def resolve_output_block_specs(config: FeaturizerConfig) -> tuple[BlockSpec, ...]:
    """Resolve the named blocks emitted by a configured featurizer tree.

    :param config: Featurizer config node to inspect.
    :returns: Block specs emitted by the featurizer tree.
    """
    # Local import: ConcatFeaturizersMixin imports FeaturizerConfig, which can load this module.
    from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin

    cls = _lookup_featurizer_class(config)
    if issubclass(cls, ConcatFeaturizersMixin):
        specs: list[BlockSpec] = []
        for child in config.featurizers or ():
            specs.extend(resolve_output_block_specs(child))
        return tuple(specs)

    hook = getattr(cls, "output_block_specs_for_config", None)
    if callable(hook):
        return tuple(hook(config))
    return _fallback_block_specs(cls, config)
