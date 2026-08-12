"""Tests for the public surface of the utils package."""

from __future__ import annotations

from drevalpy import utils
from drevalpy.utils.response_transform import fit_response_transformation, get_response_transformation


def test_all_lists_the_documented_surface() -> None:
    assert utils.__all__ == ["fit_response_transformation", "get_response_transformation"]


def test_re_export_is_the_defining_function() -> None:
    assert utils.get_response_transformation is get_response_transformation


def test_the_fitting_helper_is_re_exported() -> None:
    assert utils.fit_response_transformation is fit_response_transformation
