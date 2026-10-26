"""Tests for ``FeatureFormat`` and ``check_format``."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat, check_format


@pytest.mark.parametrize("fmt", list(FeatureFormat))
def test_check_format_returns_a_declared_format(fmt: FeatureFormat) -> None:
    class Declares:
        output_format = fmt

    assert check_format(Declares, "output_format") is fmt


def test_check_format_reads_an_inherited_declaration() -> None:
    class Base:
        sample_input_format = FeatureFormat.GRAPH

    class Child(Base):
        pass

    assert check_format(Child, "sample_input_format") is FeatureFormat.GRAPH


@pytest.mark.parametrize("value", [None, "graph", 3, object()], ids=["none", "plain-string", "int", "object"])
def test_check_format_rejects_anything_but_a_feature_format(value: object) -> None:
    class Bad:
        drug_input_format = value

    with pytest.raises(TypeError, match=r"Bad\.drug_input_format must be a FeatureFormat.*pass drug_input_format="):
        check_format(Bad, "drug_input_format")


def test_check_format_rejects_a_missing_attribute_with_the_same_message() -> None:
    class Missing:
        pass

    with pytest.raises(TypeError, match=r"Missing\.output_format must be a FeatureFormat.*class body"):
        check_format(Missing, "output_format")
