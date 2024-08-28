import numpy as np
import pytest
import tempfile

from drevalpy.datasets import RESPONSE_DATASET_FACTORY


def test_factory():
    assert "GDSC1" in RESPONSE_DATASET_FACTORY
    assert "GDSC2" in RESPONSE_DATASET_FACTORY
    assert "CCLE" in RESPONSE_DATASET_FACTORY
    assert "Toy_Data" in RESPONSE_DATASET_FACTORY
    assert len(RESPONSE_DATASET_FACTORY) == 4


def test_gdsc1():
    tempdir = tempfile.TemporaryDirectory()
    gdsc1 = RESPONSE_DATASET_FACTORY["GDSC1"](path_data=tempdir.name)
    assert len(gdsc1) == 310904
    assert np.all(gdsc1.cell_line_ids[0:3] == np.array(["MC/CAR", "ES3", "ES5"]))
    assert np.all(
        np.array([gdsc1.drug_ids[0], gdsc1.drug_ids[-1]])
        == np.array(["Erlotinib", "PFI-3"])
    )
    assert np.allclose(gdsc1.response[0:3], np.array([2.395685, 3.140923, 3.968757]))


def test_gdsc2():
    tempdir = tempfile.TemporaryDirectory()
    gdsc2 = RESPONSE_DATASET_FACTORY["GDSC2"](path_data=tempdir.name)
    assert len(gdsc2) == 135242
    assert np.all(
        gdsc2.cell_line_ids[0:3] == np.array(["HCC1954", "HCC1143", "HCC1187"])
    )
    assert np.all(
        np.array([gdsc2.drug_ids[0], gdsc2.drug_ids[-1]])
        == np.array(["Camptothecin", "JQ1"])
    )
    assert np.allclose(gdsc2.response[0:3], np.array([-0.251083, 1.343315, 1.736985]))


def test_ccle():
    tempdir = tempfile.TemporaryDirectory()
    ccle = RESPONSE_DATASET_FACTORY["CCLE"](path_data=tempdir.name)
    assert len(ccle) == 8478
    assert np.all(
        np.array([ccle.cell_line_ids[0], ccle.cell_line_ids[-1]])
        == np.array(["U-937", "TEN"])
    )
    assert np.all(
        ccle.drug_ids[0:3] == np.array(["Tanespimycin", "Saracatinib", "Crizotinib"])
    )
    assert np.allclose(
        ccle.response[0:3], np.array([7.1485739, 8.34797496, 5.75592623])
    )


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
