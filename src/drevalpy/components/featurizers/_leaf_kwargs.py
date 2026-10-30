"""Resolve featurizer leaf kwargs from options and hyperparameter defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.registry.featurizer import get_featurizer


def featurizer_leaf_kwargs(leaf: FeaturizerConfig) -> dict[str, Any]:
    """Build featurizer kwargs from options and hyperparameter defaults.

    These are the kwargs passed to featurizer construction, and the config that
    ``Featurizer.output_blocks`` and ``Featurizer.views`` read.

    :param leaf: Featurizer leaf configuration.
    :returns: Keyword arguments for featurizer construction.
    """
    kwargs: dict[str, Any] = dict(leaf.options or {})
    space = dict(leaf.hyperparameter_space or {})
    if not space:
        space = dict(get_featurizer(leaf.name).get_hyperparameter_space())
    for key, spec in space.items():
        if isinstance(spec, Mapping) and "default" in spec:
            kwargs.setdefault(key, spec["default"])
    if leaf.view is not None:
        kwargs.setdefault("view", leaf.view)
    return kwargs
