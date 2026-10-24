"""Shared metadata infrastructure for component registries (predictor, featurizer)."""

from ._base import ComponentRegistry
from ._metadata import base_component_metadata, featurizer_component_metadata, predictor_component_metadata
from ._metadata_validate import validate_literature_reference, validate_registered_class
from ._registration_metadata import apply_registration_metadata, normalize_registration_metadata

__all__ = [
    "ComponentRegistry",
    "apply_registration_metadata",
    "base_component_metadata",
    "featurizer_component_metadata",
    "normalize_registration_metadata",
    "predictor_component_metadata",
    "validate_literature_reference",
    "validate_registered_class",
]
