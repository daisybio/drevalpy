"""Tests for the DrugResponseDataset class."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_response_dataset
from drevalpy.utils import get_response_transformation


def test_response_dataset_load() -> None:
    """Test if the dataset loads correctly from CSV files."""
    data = {
        "cell_line_id": np.array(["1", "2", "3"]),
        "drug_id": np.array(["A", "B", "C"]),
        "response": np.array([0.1, 0.2, 0.3]),
    }
    dataset = DrugResponseDataset(
        cell_line_ids=data["cell_line_id"],
        drug_ids=data["drug_id"],
        response=data["response"],
    )
    dataset_path = Path("dataset.csv")
    dataset.to_csv(dataset_path)
    del dataset
    dataset = DrugResponseDataset.from_csv(dataset_path)

    dataset_path.unlink()

    assert np.array_equal(dataset.cell_line_ids, data["cell_line_id"])
    assert np.array_equal(dataset.drug_ids, data["drug_id"])
    assert np.allclose(dataset.response, data["response"])


def test_fitting_and_loading_custom_dataset(sample_dataset: DrugResponseDataset, data_dir):
    """Test CurveCurator fitting of raw viability dataset and loading it.

    :param sample_dataset: sample viability dataset
    :param data_dir: path to the data directory
    """
    assert sample_dataset.dataset_name == "TOYv1"
    dataset_name = "CTRPv2_sample_test"
    load_response_dataset(
        dataset_name=dataset_name,
        measure="IC50",
        curve_curator=True,
        cores=200,
    )
    for item in (data_dir / dataset_name).iterdir():
        if item.name == f"{dataset_name}_raw.csv":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _curve_function(x, wanted_ec50, front, back, slope):
    return (front - back) / (1 + (x / wanted_ec50) ** slope) + back


def test_curvecurator_measures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests if CurveCurator computes the response measures correctly.

    :param monkeypatch: pytest fixture to set the cache directory for this test
    """
    temp_dir = tempfile.TemporaryDirectory()
    path_to_temp_dir = Path(temp_dir.name)
    Path.mkdir(path_to_temp_dir / "toy_curves", exist_ok=True)
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(path_to_temp_dir))

    expected_ec50 = 6
    front = 1.0
    back = 0.3
    slope = 1.5
    xvals = 10 ** np.linspace(np.log10(0.001) - 2, np.log10(1000) + 2, 50)
    yvals = _curve_function(xvals, expected_ec50, front, back, slope)
    expected_ic50 = expected_ec50 * (((front - back) / (0.5 - back)) - 1) ** (1 / slope)
    df = pd.DataFrame({"dose": xvals, "response": yvals, "sample": "cell_line_1", "drug": "drug_1", "replicate": "1"})
    df.to_csv(path_to_temp_dir / "toy_curves" / "toy_curves_raw.csv", index=False)
    load_response_dataset(
        dataset_name="toy_curves",
        measure="IC50",
        curve_curator=True,
        cores=200,
    )
    assert Path(path_to_temp_dir / "toy_curves" / "toy_curves.csv").exists()
    df_processed = pd.read_csv(path_to_temp_dir / "toy_curves" / "toy_curves.csv", index_col=0)
    assert np.isclose(df_processed.loc["cell_line_1|drug_1"]["EC50_curvecurator"], expected_ec50, atol=0.1)
    assert np.isclose(df_processed.loc["cell_line_1|drug_1"]["IC50_curvecurator"], expected_ic50, atol=0.1)
    assert round(np.log(df_processed.loc["cell_line_1|drug_1"]["IC50_curvecurator"]), 4) == round(
        df_processed.loc["cell_line_1|drug_1"]["LN_IC50_curvecurator"], 4
    )
    assert round(-np.log10(df_processed.loc["cell_line_1|drug_1"]["EC50_curvecurator"] * 10**-6), 4) == round(
        df_processed.loc["cell_line_1|drug_1"]["pEC50_curvecurator"], 4
    )


def test_response_dataset_add_rows() -> None:
    """Test if the add_rows method works correctly."""
    dataset1 = DrugResponseDataset(
        response=np.array([1, 2, 3]),
        cell_line_ids=np.array(["101", "102", "103"]),
        drug_ids=np.array(["A", "B", "C"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3"]),
    )
    dataset2 = DrugResponseDataset(
        response=np.array([4, 5, 6]),
        cell_line_ids=np.array(["104", "105", "106"]),
        drug_ids=np.array(["D", "E", "F"]),
        tissues=np.array(["Tissue4", "Tissue5", "Tissue6"]),
    )
    dataset1.add_rows(dataset2)

    assert np.array_equal(dataset1.response, np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(dataset1.cell_line_ids, np.array(["101", "102", "103", "104", "105", "106"]))
    assert np.array_equal(dataset1.drug_ids, np.array(["A", "B", "C", "D", "E", "F"]))
    assert np.array_equal(dataset1.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]))


def test_remove_nan_responses() -> None:
    """Test if the remove_nan_responses method works correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, np.nan, 5, 6]),
        cell_line_ids=np.array(["101", "102", "103", "104", "105", "106"]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]),
    )
    dataset.remove_nan_responses()
    assert np.array_equal(dataset.response, np.array([1, 2, 3, 5, 6]))
    assert np.array_equal(dataset.cell_line_ids, np.array(["101", "102", "103", "105", "106"]))
    assert np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "E", "F"]))
    assert np.array_equal(dataset.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue5", "Tissue6"]))


def test_response_dataset_shuffle():
    """Test if the shuffle method works correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5, 6]),
        cell_line_ids=np.array(["101", "102", "103", "104", "105", "106"]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]),
    )

    dataset.shuffle(random_state=42)

    assert len(dataset.response) == 6
    assert len(dataset.cell_line_ids) == 6
    assert len(dataset.drug_ids) == 6
    assert len(dataset.tissue) == 6

    assert not np.array_equal(dataset.response, np.array([1, 2, 3, 4, 5, 6]))
    assert not np.array_equal(dataset.cell_line_ids, np.array(["101", "102", "103", "104", "105", "106"]))
    assert not np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "D", "E", "F"]))
    assert not np.array_equal(
        dataset.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"])
    )


def test_response_data_remove_drugs_and_cell_lines():
    """Test if the remove_drugs and remove_cell_lines methods work correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array(["101", "102", "103", "104", "105"]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )

    dataset._remove_drugs(["A", "C"])
    dataset._remove_cell_lines(["101", "103"])

    assert "A" not in dataset.drug_ids
    assert "C" not in dataset.drug_ids
    assert "101" not in dataset.cell_line_ids
    assert "103" not in dataset.cell_line_ids

    assert len(dataset.response) == 3
    assert len(dataset.cell_line_ids) == 3
    assert len(dataset.drug_ids) == 3
    assert len(dataset.tissue) == 3


def test_remove_rows():
    """Test if the remove_rows method works correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array(["101", "102", "103", "104", "105"]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )
    dataset.remove_rows(np.array([0, 2, 4]))
    assert np.array_equal(dataset.response, np.array([2, 4]))
    assert np.array_equal(dataset.cell_line_ids, np.array(["102", "104"]))
    assert np.array_equal(dataset.drug_ids, np.array(["B", "D"]))
    assert np.array_equal(dataset.tissue, np.array(["Tissue2", "Tissue4"]))


def test_response_dataset_reduce_to():
    """Test if the reduce_to method works correctly and handles edge cases."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )

    dataset.reduce_to(cell_line_ids=np.array([102, 104]), drug_ids=np.array(["B", "D"]))

    assert all(cell_line_id in ["102", "104"] for cell_line_id in dataset.cell_line_ids)
    assert all(drug_id in ["B", "D"] for drug_id in dataset.drug_ids)
    assert len(dataset.response) == 2
    assert len(dataset.cell_line_ids) == 2
    assert len(dataset.drug_ids) == 2
    assert len(dataset.tissue) == 2

    # reduce_to(None, None) does nothing
    dataset = DrugResponseDataset(
        response=np.array([1, 2]),
        cell_line_ids=np.array(["201", "202"]),
        drug_ids=np.array(["X", "Y"]),
        tissues=np.array(["T1", "T2"]),
    )

    dataset.reduce_to(cell_line_ids=None, drug_ids=None)

    assert len(dataset.response) == 2
    assert set(dataset.cell_line_ids) == {"201", "202"}
    assert set(dataset.drug_ids) == {"X", "Y"}

    # reduce_to with empty lists removes all
    dataset = DrugResponseDataset(
        response=np.array([1, 2]),
        cell_line_ids=np.array(["301", "302"]),
        drug_ids=np.array(["M", "N"]),
        tissues=np.array(["T1", "T2"]),
    )

    dataset.reduce_to(cell_line_ids=np.array([]), drug_ids=np.array([]))

    assert len(dataset.response) == 0
    assert len(dataset.cell_line_ids) == 0
    assert len(dataset.drug_ids) == 0
    assert len(dataset.tissue) == 0


_VALIDATION_SPLIT_KEYS = ("validation", "validation_es", "early_stopping")


def _assert_disjoint_validation(mode: str, split: dict, test_set: set) -> None:
    for val_es in _VALIDATION_SPLIT_KEYS:
        if mode == "LCO":
            validation_set = set(split[val_es].cell_line_ids)
        elif mode == "LDO":
            validation_set = set(split[val_es].drug_ids)
        else:
            validation_set = set(zip(split[val_es].cell_line_ids, split[val_es].drug_ids, strict=True))
        assert validation_set.isdisjoint(test_set)


def _assert_cv_split_disjointness(mode: str, split: dict, split_validation: bool) -> None:
    if mode == "LCO":
        test_set = set(split["test"].cell_line_ids)
        assert set(split["train"].cell_line_ids).isdisjoint(test_set)
    elif mode == "LDO":
        test_set = set(split["test"].drug_ids)
        assert set(split["train"].drug_ids).isdisjoint(test_set)
    elif mode == "LPO":
        test_set = set(zip(split["test"].cell_line_ids, split["test"].drug_ids, strict=True))
        train_pairs = set(zip(split["train"].cell_line_ids, split["train"].drug_ids, strict=True))
        assert train_pairs.isdisjoint(test_set)
    else:
        return

    if split_validation:
        _assert_disjoint_validation(mode, split, test_set)


@pytest.mark.parametrize("mode", ["LPO", "LCO", "LDO", "LTO"])
@pytest.mark.parametrize("split_validation", [True, False])
def test_split_response_dataset(mode: str, split_validation: bool) -> None:
    """Test if the split_dataset method works correctly.

    :param mode: test_mode, either LPO, LCO, or LDO
    :param split_validation: whether to split the dataset into validation and early stopping sets
    """
    dataset = DrugResponseDataset(
        response=np.random.random(100),
        cell_line_ids=np.repeat([f"CL-{i}" for i in range(1, 11)], 10),
        drug_ids=np.tile([f"Drug-{i}" for i in range(1, 11)], 10),
        tissues=np.array(
            ["Breast", "Breast", "Lung", "Kidney", "Small intestine", "Brain", "Heart", "Pancreas", "Prostate", "Colon"]
            * 10
        ),
    )

    cv_splits = dataset.split_dataset(
        n_cv_splits=5,
        mode=mode,
        split_validation=split_validation,
        validation_ratio=0.5,
        random_state=42,
    )
    assert isinstance(cv_splits, list)
    assert len(cv_splits) == 5
    for split in cv_splits:
        assert isinstance(split["train"], DrugResponseDataset)
        assert isinstance(split["test"], DrugResponseDataset)
        _assert_cv_split_disjointness(mode, split, split_validation)

    tempdir = tempfile.TemporaryDirectory()
    dataset.save_splits(path=tempdir.name)
    dataset.load_splits(path=tempdir.name)


@pytest.mark.parametrize("resp_transform", ["standard", "minmax", "robust"])
def test_transform(resp_transform: str):
    """Test if the fit_transform and inverse_transform methods work correctly.

    :param resp_transform: response transformation method
    :raises ValueError: if an invalid response transformation method is provided
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array(["101", "102", "103", "104", "105"]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )
    transform = get_response_transformation(resp_transform)
    dataset.fit_transform(transform)
    if resp_transform == "standard":
        scaler = StandardScaler()
    elif resp_transform == "minmax":
        scaler = MinMaxScaler()
    elif resp_transform == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid response transformation method.")
    vals = scaler.fit_transform(np.array([1, 2, 3, 4, 5]).reshape(-1, 1))
    assert np.allclose(dataset.response, vals.flatten())

    dataset.inverse_transform(transform)
    assert np.allclose(dataset.response, np.array([1, 2, 3, 4, 5]))
