"""Fixtures shared by the splitter test modules."""

from __future__ import annotations

import pytest

from tests.data.splitters._helpers import MockDataset


@pytest.fixture
def mock_dataset() -> MockDataset:
    """A 10x8 response matrix over three tissues with ~30% missing entries."""
    return MockDataset()
