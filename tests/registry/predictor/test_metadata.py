"""Tests for the predictor catalog metadata builder.

``drevalpy/registry/predictor/_metadata.py`` delegates to
``registry/components/_metadata.base_component_metadata`` and adds a single field, so
these tests pin the delegation and the added field. The shared base fields are
covered in ``tests/registry/components/test_metadata.py``.
"""

from __future__ import annotations

from drevalpy.registry.components._metadata import base_component_metadata
from drevalpy.registry.predictor._metadata import predictor_component_metadata


class _FeatureFreeLike:
    description = "feature free predictor"
    tags = frozenset({"baseline"})
    input_interface = "feature_free"


def test_metadata_adds_the_input_interface() -> None:
    meta = predictor_component_metadata("predictors", "featureFree", _FeatureFreeLike)

    assert meta["input_interface"] == "feature_free"


def test_metadata_defaults_the_input_interface_to_empty() -> None:
    class NoInterface:
        description = "no declared interface"

    meta = predictor_component_metadata("predictors", "bare", NoInterface)

    assert meta["input_interface"] == ""


def test_metadata_is_the_base_metadata_plus_the_input_interface() -> None:
    meta = predictor_component_metadata("predictors", "featureFree", _FeatureFreeLike)

    expected = base_component_metadata("predictors", "featureFree", _FeatureFreeLike)
    expected["input_interface"] = "feature_free"
    assert meta == expected


def test_metadata_omits_the_legacy_capability_fields() -> None:
    meta = predictor_component_metadata("predictors", "featureFree", _FeatureFreeLike)

    for dropped in (
        "cell_line_format",
        "drug_format",
        "supported_modes",
        "scope",
        "supports_early_stopping",
        "required_cell_line_views",
        "required_drug_views",
    ):
        assert dropped not in meta
