"""Tests for model identifier helpers."""

from __future__ import annotations

import pytest

from drevalpy.components.model_id import format_model_id, parse_model_id


def test_format_predictor_only_id() -> None:
    assert format_model_id(None, None, "naiveMean") == "naiveMean"


def test_format_full_triple_id() -> None:
    assert format_model_id("scaledGeneExpression", "fingerprints", "elasticNet") == (
        "scaledGeneExpression:fingerprints:elasticNet"
    )


def test_parse_predictor_only_id() -> None:
    assert parse_model_id("naiveMean") == (None, None, "naiveMean")


def test_parse_full_triple_id() -> None:
    assert parse_model_id("scaledGeneExpression:fingerprints:elasticNet") == (
        "scaledGeneExpression",
        "fingerprints",
        "elasticNet",
    )


@pytest.mark.parametrize(
    "model_id",
    ["", "a:b", "a:b:c:d", " :b:c"],
)
def test_parse_rejects_malformed_ids(model_id: str) -> None:
    with pytest.raises(ValueError):
        parse_model_id(model_id)
