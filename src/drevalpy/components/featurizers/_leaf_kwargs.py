"""Resolve featurizer leaf kwargs from options, HP defaults, and resolved values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from drevalpy.components.featurizers._featurizer_label import qualified_featurizer_selector
from drevalpy.models.config.featurizer import FeaturizerConfig
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer

if TYPE_CHECKING:
    from drevalpy.models.config.resolved import ResolvedModelConfig


def featurizer_leaf_kwargs(
    leaf: FeaturizerConfig,
    *,
    registry: Literal["sample", "drug"],
    resolved: ResolvedModelConfig | None,
) -> dict[str, Any]:
    """Build featurizer kwargs from options, hyperparameter defaults, and resolved values.

    These are the kwargs passed to featurizer construction, and
    the same kwargs that ``Featurizer.resolve_input_views`` interprets.

    :param leaf: Featurizer leaf configuration.
    :param registry: ``sample`` or ``drug``.
    :param resolved: Optional resolved instance values for tunable kwargs.
    :returns: Keyword arguments for featurizer construction.
    """
    kwargs: dict[str, Any] = dict(leaf.options or {})
    space = dict(leaf.hyperparameter_space or {})
    if not space:
        cls = get_sample_featurizer(leaf.name) if registry == "sample" else get_drug_featurizer(leaf.name)
        space = dict(cls.get_hyperparameter_space())
    for key, spec in space.items():
        if isinstance(spec, Mapping) and "default" in spec:
            kwargs.setdefault(key, spec["default"])
    if resolved is not None:
        selector = qualified_featurizer_selector(leaf.name, leaf.view)
        kwargs.update(resolved.featurizer_values(registry, selector))
    if leaf.view is not None:
        kwargs.setdefault("view", leaf.view)
    return kwargs
