"""Shared metadata and registration helpers for the predictor and featurizer registries."""

from ._metadata import base_component_metadata, featurizer_component_metadata, predictor_component_metadata
from ._metadata_validate import validate_literature_reference
from ._register import register_component
from ._registration_metadata import apply_registration_metadata, normalize_registration_metadata

__all__ = [
    "apply_registration_metadata",
    "base_component_metadata",
    "featurizer_component_metadata",
    "normalize_registration_metadata",
    "predictor_component_metadata",
    "register_component",
    "validate_literature_reference",
]
