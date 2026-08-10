"""Featurizer fitting context, labels, and tree utilities."""

from .featurizer_fit_context import FeaturizerFitContext
from .featurizer_label import (
    featurizer_config_block_label,
    qualified_featurizer_selector,
    requires_explicit_view,
)
from .featurizer_tree import (
    ensure_unique_qualified_featurizers,
    iter_featurizer_leaves,
)

__all__ = [
    "FeaturizerFitContext",
    "ensure_unique_qualified_featurizers",
    "featurizer_config_block_label",
    "iter_featurizer_leaves",
    "qualified_featurizer_selector",
    "requires_explicit_view",
]
