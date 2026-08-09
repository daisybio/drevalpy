"""Tests for Featurizer base class contract policy."""

from __future__ import annotations

import pytest

from drevalpy.components.core.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.featurizers.base import Featurizer


def test_featurizer_rejects_class_body_contract() -> None:
    with pytest.raises(TypeError, match="do not set contract on the class body"):

        class BadFeaturizer(Featurizer):  # noqa: B903
            contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
