"""Fixtures shared by the splitter test modules."""

from __future__ import annotations

import pytest

from tests._fake_dataset import FakeResponseDataset


@pytest.fixture
def mock_dataset() -> FakeResponseDataset:
    """A 10x8 response matrix over three tissues with ~30% missing entries."""
    return FakeResponseDataset()
