"""Tests for the pipeline_function decorator."""

from __future__ import annotations

from drevalpy.utils import pipeline_function
from drevalpy.utils._pipeline_function import pipeline_function as pipeline_function_impl


def test_pipeline_function_marks_callable() -> None:
    @pipeline_function
    def sample() -> int:
        return 1

    assert sample.is_pipeline_function is True
    assert sample() == 1
    assert pipeline_function is pipeline_function_impl
