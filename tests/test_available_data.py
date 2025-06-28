"""Tests for the available datasets."""

import tempfile

import pytest
from flaky import flaky

from drevalpy.datasets import AVAILABLE_DATASETS


def test_factory() -> None:
    """Test the dataset factory."""
    assert "GDSC1" in AVAILABLE_DATASETS
    assert "GDSC2" in AVAILABLE_DATASETS
    assert "CCLE" in AVAILABLE_DATASETS
    assert "TOYv1" in AVAILABLE_DATASETS
    assert "TOYv2" in AVAILABLE_DATASETS
    assert "CTRPv1" in AVAILABLE_DATASETS
    assert "CTRPv2" in AVAILABLE_DATASETS
    assert len(AVAILABLE_DATASETS) == 7


@pytest.mark.parametrize(
    "name,expected_len",
    [
        ("GDSC1", 316506),
        ("GDSC2", 234436),
        ("CCLE", 11670),
        ("CTRPv1", 60757),
        ("CTRPv2", 395024),
        ("TOYv1", 2711),
        ("TOYv2", 2784),
    ],
)
@flaky(max_runs=3, min_passes=1)
def test_datasets(name, expected_len):
    """Test the datasets.

    :param name: Name of the dataset to test.
    :param expected_len: Expected length of the dataset.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        ds = AVAILABLE_DATASETS[name](path_data=tempdir)
        assert len(ds) == expected_len
