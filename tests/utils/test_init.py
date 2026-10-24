"""Tests for init."""

from __future__ import annotations

from drevalpy import utils
from drevalpy.utils.response_transform import get_response_transformation


def test_surface() -> None:
    assert utils.__all__ == ["get_response_transformation"]
    assert utils.get_response_transformation is get_response_transformation
