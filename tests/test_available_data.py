"""Tests for the available datasets."""

import tempfile

from drevalpy.datasets import AVAILABLE_DATASETS


def test_factory() -> None:
    """Test the dataset factory."""
    assert "GDSC1" in AVAILABLE_DATASETS
    assert "GDSC2" in AVAILABLE_DATASETS
    assert "CCLE" in AVAILABLE_DATASETS
    assert "Toy_Data" in AVAILABLE_DATASETS
    assert len(AVAILABLE_DATASETS) == 4


def test_gdsc1() -> None:
    """Test the GDSC1 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    gdsc1 = AVAILABLE_DATASETS["GDSC1"](path_data=tempdir.name)
    assert len(gdsc1) == 292849


def test_gdsc2():
    """Test the GDSC2 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    gdsc2 = AVAILABLE_DATASETS["GDSC2"](path_data=tempdir.name)
    assert len(gdsc2) == 131108


def test_ccle():
    """Test the CCLE dataset."""
    tempdir = tempfile.TemporaryDirectory()
    ccle = AVAILABLE_DATASETS["CCLE"](path_data=tempdir.name)
    assert len(ccle) == 8478
