"""Shared featurizer registry infrastructure used by cell_line and drug sub-packages."""

from ._base import FeaturizerRegistry
from ._validate import validate_featurizer_input_views

__all__ = ["FeaturizerRegistry", "validate_featurizer_input_views"]
