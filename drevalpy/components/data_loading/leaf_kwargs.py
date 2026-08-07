"""Resolve featurizer leaf kwargs from options, HP defaults, and resolved values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from drevalpy.models.config import FeaturizerConfig, ResolvedModelConfig


def featurizer_leaf_kwargs(
    leaf: FeaturizerConfig,
    *,
    registry: Literal["cell_line", "drug"],
    resolved: ResolvedModelConfig | None,
) -> dict[str, Any]:
    """Build featurizer kwargs from options, hyperparameter defaults, and resolved values.

    These are the kwargs passed to ``load_features`` / featurizer construction, and
    the same kwargs that ``Featurizer.resolve_input_views`` interprets.

    :param leaf: Featurizer leaf configuration.
    :param registry: ``cell_line`` or ``drug``.
    :param resolved: Optional resolved instance values for tunable kwargs.
    :returns: Keyword arguments for ``load_features`` / featurizer construction.
    """
    from drevalpy.components.featurizer_label import qualified_featurizer_selector
    from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

    kwargs: dict[str, Any] = dict(leaf.options or {})
    space = dict(leaf.hyperparameter_space or {})
    if not space:
        cls = get_cell_line_featurizer(leaf.name) if registry == "cell_line" else get_drug_featurizer(leaf.name)
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
