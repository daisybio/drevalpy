"""Tests for entity feature preparation helpers used by the model pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.models._feature_preparation import transform_entities, usable_responses
from drevalpy.types.data.batch.response_batch import ResponseBatch


def test_transform_without_a_featurizer_returns_empty_result() -> None:
    result = transform_entities(None, None, np.array(["a", "b"]))

    np.testing.assert_array_equal(result.entity_ids, ["a", "b"])
    assert result.matrix.shape == (2, 0)
    assert result.blocks == {}
    assert result.projection == ()


def test_configured_featurizer_requires_a_feature_source() -> None:
    featurizer = object()
    with pytest.raises(ValueError, match="Feature source is required"):
        transform_entities(featurizer, None, np.array(["a"]))  # type: ignore[arg-type]


def test_usable_responses_removes_missing_targets() -> None:
    response = ResponseBatch(
        response=np.array([1.0, np.nan, 3.0]),
        sample_ids=np.array(["a", "a", "b"]),
        drug_ids=np.array(["x", "y", "x"]),
    )

    usable = usable_responses(response, None, None, None, None)

    np.testing.assert_array_equal(usable.response, [1.0, 3.0])
    np.testing.assert_array_equal(usable.sample_ids, ["a", "b"])
