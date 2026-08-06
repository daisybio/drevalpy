"""Tests for drevalpy.models.config._from_dict."""

from __future__ import annotations

import pytest

from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models.config._from_dict import from_dict


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_from_dict_accepts_recipe_strings_in_slots() -> None:
    """Slots may hold recipe strings, which is what lets ``from_spec`` reuse this."""
    config = from_dict(
        {
            "cell_line_featurizer": "raw[expression]+raw[mutations]",
            "drug_featurizer": "fingerprints",
            "predictor": "randomForest",
        }
    )
    assert config.cell_line_featurizer is not None
    assert config.cell_line_featurizer.name == "concatFeaturizers"
    assert config.predictor.name == "randomForest"


def test_source_label_is_included_in_the_error() -> None:
    with pytest.raises(ValueError, match=r"Invalid model config in my-label:"):
        from_dict({"predictor": "naiveMean", "unknown_key": True}, source="my-label")


def test_error_without_source_omits_the_location() -> None:
    with pytest.raises(ValueError, match=r"Invalid model config: "):
        from_dict({"predictor": "naiveMean", "unknown_key": True})


def test_field_level_error_names_the_field() -> None:
    with pytest.raises(ValueError, match=r"predictor: "):
        from_dict({"predictor": 123})


def test_model_level_error_has_no_empty_field_prefix() -> None:
    """Whole-model errors carry an empty ``loc``, which must not render as a bare colon."""
    with pytest.raises(ValueError, match=r"Invalid model config: Value error, Predictor 'elasticNet' requires"):
        from_dict({"predictor": "elasticNet"})
