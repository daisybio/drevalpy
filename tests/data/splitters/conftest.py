"""Fixtures shared by the splitter test modules."""

from __future__ import annotations

import pytest

from tests.data.splitters._helpers import MockMuDataset


@pytest.fixture
def mock_dataset() -> MockMuDataset:
    """A 10x8 response matrix over three tissues with ~30% missing entries."""
    return MockMuDataset()
