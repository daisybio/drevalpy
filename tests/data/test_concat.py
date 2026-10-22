"""Tests for :mod:`drevalpy.data._concat`.

``concat`` is a stub until W1-P1 lands; the real tests replace these.
"""

from __future__ import annotations

import inspect

import pytest

from drevalpy import data
from drevalpy.data._concat import concat


def test_signature_is_the_planned_interface() -> None:
    parameters = inspect.signature(concat).parameters

    assert list(parameters) == ["datasets", "join", "name"]
    assert parameters["join"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["join"].default == "exact"
    assert parameters["name"].default is None


def test_is_exported_from_the_data_package() -> None:
    assert data.concat is concat


def test_is_not_implemented_yet() -> None:
    with pytest.raises(NotImplementedError, match="W1-P1"):
        concat([])
