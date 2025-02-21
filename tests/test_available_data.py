"""Tests for the available datasets."""

import tempfile

from drevalpy.datasets import AVAILABLE_DATASETS


def test_factory() -> None:
    """Test the dataset factory."""
    assert "GDSC1" in AVAILABLE_DATASETS
    assert "GDSC2" in AVAILABLE_DATASETS
    assert "CCLE" in AVAILABLE_DATASETS
    assert "Toy_Data" in AVAILABLE_DATASETS
    assert "CTRPv1" in AVAILABLE_DATASETS
    assert "CTRPv2" in AVAILABLE_DATASETS
    assert len(AVAILABLE_DATASETS) == 6


def test_gdsc1() -> None:
    """Test the GDSC1 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    gdsc1 = AVAILABLE_DATASETS["GDSC1"](path_data=tempdir.name)
    assert len(gdsc1) == 316506


def test_gdsc2():
    """Test the GDSC2 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    gdsc2 = AVAILABLE_DATASETS["GDSC2"](path_data=tempdir.name)
    assert len(gdsc2) == 234436


def test_ccle():
    """Test the CCLE dataset."""
    tempdir = tempfile.TemporaryDirectory()
    ccle = AVAILABLE_DATASETS["CCLE"](path_data=tempdir.name)
    assert len(ccle) == 11670


def test_ctrpv1():
    """Test the CTRPv1 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    ctrpv1 = AVAILABLE_DATASETS["CTRPv1"](path_data=tempdir.name)
    assert len(ctrpv1) == 60757


def test_ctrpv2():
    """Test the CTRPv2 dataset."""
    tempdir = tempfile.TemporaryDirectory()
    ctrpv2 = AVAILABLE_DATASETS["CTRPv2"](path_data=tempdir.name)
    assert len(ctrpv2) == 395024


def test_toy_data():
    """Test the Toy_Data dataset."""
    tempdir = tempfile.TemporaryDirectory()
    toy_data = AVAILABLE_DATASETS["Toy_Data"](path_data=tempdir.name)
    assert len(toy_data) == 3426
