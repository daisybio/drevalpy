"""Tests for the DrugResponseDataset and the FeatureDataset class."""

import shutil
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from flaky import flaky

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.loader import load_dataset
from drevalpy.utils import get_response_transformation

# Tests for the DrugResponseDataset class


def test_response_dataset_load() -> None:
    """Test if the dataset loads correctly from CSV files."""
    # Create a temporary CSV file with mock data
    data = {
        "cell_line_id": np.array([1, 2, 3]),
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
    # Load the dataset
    dataset = DrugResponseDataset.from_csv(dataset_path)

    dataset_path.unlink()

    # Check if the dataset loaded correctly
    assert np.array_equal(dataset.cell_line_ids, data["cell_line_id"])
    assert np.array_equal(dataset.drug_ids, data["drug_id"])
    assert np.allclose(dataset.response, data["response"])


def test_fitting_and_loading_custom_dataset():
    """Test CurveCurator fitting of raw viability dataset and loading it."""
    dataset_name = "CTRPv2_sample_test"
    load_dataset(
        dataset_name=dataset_name,
        path_data=str(Path(__file__).parent),
        measure="IC50",
        curve_curator=True,
        cores=200,
    )
    for item in (Path(__file__).parent / dataset_name).iterdir():
        if item.name == f"{dataset_name}_raw.csv":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def test_response_dataset_add_rows() -> None:
    """Test if the add_rows method works correctly."""
    dataset1 = DrugResponseDataset(
        response=np.array([1, 2, 3]),
        cell_line_ids=np.array([101, 102, 103]),
        drug_ids=np.array(["A", "B", "C"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3"]),
    )
    dataset2 = DrugResponseDataset(
        response=np.array([4, 5, 6]),
        cell_line_ids=np.array([104, 105, 106]),
        drug_ids=np.array(["D", "E", "F"]),
        tissues=np.array(["Tissue4", "Tissue5", "Tissue6"]),
    )
    dataset1.add_rows(dataset2)

    assert np.array_equal(dataset1.response, np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(dataset1.cell_line_ids, np.array([101, 102, 103, 104, 105, 106]))
    assert np.array_equal(dataset1.drug_ids, np.array(["A", "B", "C", "D", "E", "F"]))
    assert np.array_equal(dataset1.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]))


def test_remove_nan_responses() -> None:
    """Test if the remove_nan_responses method works correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, np.nan, 5, 6]),
        cell_line_ids=np.array([101, 102, 103, 104, 105, 106]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]),
    )
    dataset.remove_nan_responses()
    assert np.array_equal(dataset.response, np.array([1, 2, 3, 5, 6]))
    assert np.array_equal(dataset.cell_line_ids, np.array([101, 102, 103, 105, 106]))
    assert np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "E", "F"]))
    assert np.array_equal(dataset.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue5", "Tissue6"]))


def test_response_dataset_shuffle():
    """Test if the shuffle method works correctly."""
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5, 6]),
        cell_line_ids=np.array([101, 102, 103, 104, 105, 106]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"]),
    )

    # Shuffle the dataset
    dataset.shuffle(random_state=42)

    # Check if the length remains the same
    assert len(dataset.response) == 6
    assert len(dataset.cell_line_ids) == 6
    assert len(dataset.drug_ids) == 6
    assert len(dataset.tissue) == 6

    # Check if the response, cell_line_ids, and drug_ids arrays are shuffled
    assert not np.array_equal(dataset.response, np.array([1, 2, 3, 4, 5, 6]))
    assert not np.array_equal(dataset.cell_line_ids, np.array([101, 102, 103, 104, 105, 106]))
    assert not np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "D", "E", "F"]))
    assert not np.array_equal(
        dataset.tissue, np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5", "Tissue6"])
    )


def test_response_data_remove_drugs_and_cell_lines():
    """Test if the remove_drugs and remove_cell_lines methods work correctly."""
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )

    # Remove specific drugs and cell lines
    dataset._remove_drugs(["A", "C"])
    dataset._remove_cell_lines([101, 103])

    # Check if the removed drugs and cell lines are not present in the dataset
    assert "A" not in dataset.drug_ids
    assert "C" not in dataset.drug_ids
    assert 101 not in dataset.cell_line_ids
    assert 103 not in dataset.cell_line_ids

    # Check if the length of response, cell_line_ids, and drug_ids arrays is reduced accordingly
    assert len(dataset.response) == 3
    assert len(dataset.cell_line_ids) == 3
    assert len(dataset.drug_ids) == 3
    assert len(dataset.tissue) == 3


def test_remove_rows():
    """Test if the remove_rows method works correctly."""
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )
    dataset.remove_rows(np.array([0, 2, 4]))
    assert np.array_equal(dataset.response, np.array([2, 4]))
    assert np.array_equal(dataset.cell_line_ids, np.array([102, 104]))
    assert np.array_equal(dataset.drug_ids, np.array(["B", "D"]))
    assert np.array_equal(dataset.tissue, np.array(["Tissue2", "Tissue4"]))


def test_response_dataset_reduce_to():
    """Test if the reduce_to method works correctly and handles edge cases."""
    # Case 1: Standard reduction
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
        tissues=np.array(["Tissue1", "Tissue2", "Tissue3", "Tissue4", "Tissue5"]),
    )

    dataset.reduce_to(cell_line_ids=np.array([102, 104]), drug_ids=np.array(["B", "D"]))

    assert all(cell_line_id in [102, 104] for cell_line_id in dataset.cell_line_ids)
    assert all(drug_id in ["B", "D"] for drug_id in dataset.drug_ids)
    assert len(dataset.response) == 2
    assert len(dataset.cell_line_ids) == 2
    assert len(dataset.drug_ids) == 2
    assert len(dataset.tissue) == 2

    # Case 2: reduce_to(None, None) does nothing
    dataset = DrugResponseDataset(
        response=np.array([1, 2]),
        cell_line_ids=np.array([201, 202]),
        drug_ids=np.array(["X", "Y"]),
        tissues=np.array(["T1", "T2"]),
    )

    dataset.reduce_to(cell_line_ids=None, drug_ids=None)

    assert len(dataset.response) == 2
    assert set(dataset.cell_line_ids) == {201, 202}
    assert set(dataset.drug_ids) == {"X", "Y"}

    # Case 3: reduce_to with empty lists removes all
    dataset = DrugResponseDataset(
        response=np.array([1, 2]),
        cell_line_ids=np.array([301, 302]),
        drug_ids=np.array(["M", "N"]),
        tissues=np.array(["T1", "T2"]),
    )

    dataset.reduce_to(cell_line_ids=np.array([]), drug_ids=np.array([]))

    assert len(dataset.response) == 0
    assert len(dataset.cell_line_ids) == 0
    assert len(dataset.drug_ids) == 0
    assert len(dataset.tissue) == 0


@pytest.mark.parametrize("mode", ["LPO", "LCO", "LDO", "LTO"])
@pytest.mark.parametrize("split_validation", [True, False])
def test_split_response_dataset(mode: str, split_validation: bool) -> None:
    """
    Test if the split_dataset method works correctly.

    :param mode: test_mode, either LPO, LCO, or LDO
    :param split_validation: whether to split the dataset into validation and early stopping sets
    """
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.random.random(100),
        cell_line_ids=np.repeat([f"CL-{i}" for i in range(1, 11)], 10),
        drug_ids=np.tile([f"Drug-{i}" for i in range(1, 11)], 10),
        tissues=np.array(
            ["Breast", "Breast", "Lung", "Kidney", "Small intestine", "Brain", "Heart", "Pancreas", "Prostate", "Colon"]
            * 10
        ),
    )
    # 100 datapoints, 10 cell lines, 10 drugs
    # LPO: With 10% validation, 5 folds -> in 1 fold: 20 samples in test,
    # 80 in train+val -> 40 in train,
    # 40 samples in validation -> 30 in val_es, 10 in early stopping
    # LCO: With 10% validation, 5 folds ->
    # in 1 fold: 2 cell lines in test, 8 in train + val ->
    # 4 in train, 4 samples in validation -> 3 in val_es, 1 in early stopping
    # LDO: With 10% validation, 5 folds ->
    # in 1 fold: 2 drugs in test, 8 in train + val ->
    # 4 in train, 4 samples in validation -> 3 in val_es, 1 in early stopping

    # Test splitting the dataset with the specified mode and validation split
    cv_splits = dataset.split_dataset(
        n_cv_splits=5,
        mode=mode,
        split_validation=split_validation,
        validation_ratio=0.5,
        random_state=42,
    )
    assert isinstance(cv_splits, list)
    assert len(cv_splits) == 5  # Check if the correct number of splits is returned
    for split in cv_splits:
        assert isinstance(split["train"], DrugResponseDataset)
        assert isinstance(split["test"], DrugResponseDataset)

        # Check that drugs/cell lines in the training data are not present in the test data
        if mode == "LCO":
            train_cell_lines = set(split["train"].cell_line_ids)
            test_cell_lines = set(split["test"].cell_line_ids)

            assert train_cell_lines.isdisjoint(test_cell_lines)

            if split_validation:  # Only check if validation split is enabled
                for val_es in [
                    "validation",
                    "validation_es",
                    "early_stopping",
                ]:
                    validation_cell_lines = set(split[val_es].cell_line_ids)
                    assert validation_cell_lines.isdisjoint(
                        test_cell_lines
                    )  # Check for disjointness between validation and test cell lines

        elif mode == "LDO":
            train_drugs = set(split["train"].drug_ids)
            test_drugs = set(split["test"].drug_ids)

            assert train_drugs.isdisjoint(test_drugs)

            if split_validation:  # Only check if validation split is enabled
                for val_es in [
                    "validation",
                    "validation_es",
                    "early_stopping",
                ]:
                    validation_drugs = set(split[val_es].drug_ids)
                    assert validation_drugs.isdisjoint(
                        test_drugs
                    )  # Check for disjointness between validation and test drugs

        elif mode == "LPO":
            train_pairs = set(zip(split["train"].cell_line_ids, split["train"].drug_ids, strict=True))
            test_pairs = set(zip(split["test"].cell_line_ids, split["test"].drug_ids, strict=True))

            assert train_pairs.isdisjoint(test_pairs)

            if split_validation:  # Only check if validation split is enabled
                for val_es in [
                    "validation",
                    "validation_es",
                    "early_stopping",
                ]:
                    validation_pairs = set(zip(split[val_es].cell_line_ids, split[val_es].drug_ids, strict=True))
                    assert validation_pairs.isdisjoint(
                        test_pairs
                    )  # Check for disjointness between validation and test pairs

    tempdir = tempfile.TemporaryDirectory()
    dataset.save_splits(path=tempdir.name)
    dataset.load_splits(path=tempdir.name)


@pytest.mark.parametrize("resp_transform", ["standard", "minmax", "robust"])
def test_transform(resp_transform: str):
    """
    Test if the fit_transform and inverse_transform methods work correctly.

    :param resp_transform: response transformation method
    :raises ValueError: if an invalid response transformation method is provided
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
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


# Tests for the FeatureDataset class


@pytest.fixture
def sample_dataset() -> FeatureDataset:
    """
    Create a sample FeatureDataset for testing.

    :returns: a sample FeatureDataset
    """
    features = {
        "drug1": {
            "fingerprints": np.random.rand(5),
            "chemical_features": np.random.rand(5),
        },
        "drug2": {
            "fingerprints": np.random.rand(5),
            "chemical_features": np.random.rand(5),
        },
        "drug3": {
            "fingerprints": np.random.rand(5),
            "chemical_features": np.random.rand(5),
        },
        "drug4": {
            "fingerprints": np.random.rand(5),
            "chemical_features": np.random.rand(5),
        },
        "drug5": {
            "fingerprints": np.random.rand(5),
            "chemical_features": np.random.rand(5),
        },
    }
    meta_info = {
        "fingerprints": ["Dim1", "Dim2", "Dim3", "Dim4", "Dim5"],
        "chemical_features": [
            "Feature1",
            "Feature2",
            "Feature3",
            "Feature4",
            "Feature5",
        ],
    }
    return FeatureDataset(features=features, meta_info=meta_info)


def random_power_law_graph(size: int = 20) -> nx.Graph:
    """
    Create a random graph with power law degree distribution.

    :param size: size of the graph
    :returns: a random graph with power law degree distribution
    """
    # make a graph with degrees distributed as a power law
    graph = nx.Graph()
    degrees = np.round(nx.utils.powerlaw_sequence(size, 2.5))
    graph.add_nodes_from(range(size))
    graph = nx.expected_degree_graph(degrees, selfloops=False)
    # only extract largest connected component
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc).copy()
    # assign edge attributes
    for u, v in graph.edges():
        graph[u][v]["original_edge"] = f"({u}_{v})"
    return graph


@pytest.fixture
def graph_dataset() -> FeatureDataset:
    """
    Create a sample FeatureDataset with molecular graphs for testing.

    :returns: a sample FeatureDataset with molecular graphs
    """
    features = {
        "drug1": {
            "molecular_graph": random_power_law_graph(),
        },
        "drug2": {
            "molecular_graph": random_power_law_graph(),
        },
        "drug3": {
            "molecular_graph": random_power_law_graph(),
        },
        "drug4": {
            "molecular_graph": random_power_law_graph(),
        },
        "drug5": {
            "molecular_graph": random_power_law_graph(),
        },
    }
    meta_info = {
        "molecular_graph": "Atom graph created with power law",
    }
    return FeatureDataset(features=features, meta_info=meta_info)


def test_feature_dataset_get_ids(sample_dataset: FeatureDataset) -> None:
    """
    Test if the get_ids method works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    assert np.all(sample_dataset.identifiers == ["drug1", "drug2", "drug3", "drug4", "drug5"])


def test_feature_dataset_get_view_names(sample_dataset: FeatureDataset) -> None:
    """
    Test if the get_view_names method works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    assert sample_dataset.view_names == [
        "fingerprints",
        "chemical_features",
    ]


def test_feature_dataset_get_feature_matrix(sample_dataset: FeatureDataset) -> None:
    """
    Test if the get_feature_matrix method works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    feature_matrix = sample_dataset.get_feature_matrix("fingerprints", np.array(["drug1", "drug2"]))
    assert feature_matrix.shape == (2, 5)
    assert np.allclose(
        feature_matrix,
        np.array(
            [
                sample_dataset.features["drug1"]["fingerprints"],
                sample_dataset.features["drug2"]["fingerprints"],
            ]
        ),
    )
    assert isinstance(feature_matrix, np.ndarray)


def test_feature_dataset_copy(sample_dataset: FeatureDataset) -> None:
    """
    Test if the copy method works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    copied_dataset = sample_dataset.copy()
    assert copied_dataset.features["drug1"]["fingerprints"] is not sample_dataset.features["drug1"]["fingerprints"]
    assert np.allclose(
        copied_dataset.features["drug1"]["fingerprints"],
        sample_dataset.features["drug1"]["fingerprints"],
    )
    assert copied_dataset.features is not sample_dataset.features
    copied_dataset.features["drug1"]["fingerprints"] = np.zeros(5)
    assert not np.allclose(
        copied_dataset.features["drug1"]["fingerprints"],
        sample_dataset.features["drug1"]["fingerprints"],
    )


@flaky(max_runs=25)  # permutation randomization might map to the same feature vector for some tries
def test_permutation_randomization(sample_dataset: FeatureDataset) -> None:
    """
    Test if the permutation randomization works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    views_to_randomize, randomization_type = "fingerprints", "permutation"
    start_sample_dataset = sample_dataset.copy()
    sample_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in sample_dataset.features.items():
        assert not np.allclose(
            features[views_to_randomize],
            start_sample_dataset.features[drug][views_to_randomize],
        )


@flaky(max_runs=25)  # permutation randomization might map to the same feature vector for some tries
def test_permutation_randomization_graph(graph_dataset: FeatureDataset) -> None:
    """
    Test if the permutation randomization works correctly for molecular graphs.

    :param graph_dataset: sample FeatureDataset with molecular graphs
    """
    views_to_randomize, randomization_type = "molecular_graph", "permutation"
    start_graph_dataset = graph_dataset.copy()
    graph_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in graph_dataset.features.items():
        # assert that drugs have different molecular graphs now
        assert not nx.is_isomorphic(
            features[views_to_randomize],
            start_graph_dataset.features[drug][views_to_randomize],
        )


def test_invariant_randomization_array(sample_dataset: FeatureDataset) -> None:
    """
    Test if the invariant randomization works correctly.

    :param sample_dataset: sample FeatureDataset
    """
    views_to_randomize, randomization_type = "chemical_features", "invariant"
    start_sample_dataset = sample_dataset.copy()
    sample_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in sample_dataset.features.items():
        assert not np.allclose(
            features[views_to_randomize],
            start_sample_dataset.features[drug][views_to_randomize],
        )


@flaky(max_runs=5)  # expected degree randomization might produce the same graph
def test_invariant_randomization_graph(graph_dataset: FeatureDataset) -> None:
    """
    Test if the invariant randomization works correctly for molecular graphs.

    :param graph_dataset: sample FeatureDataset with molecular graphs
    """
    views_to_randomize, randomization_type = "molecular_graph", "invariant"
    start_graph_dataset = graph_dataset.copy()
    graph_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in graph_dataset.features.items():
        assert not nx.is_isomorphic(
            features[views_to_randomize],
            start_graph_dataset.features[drug][views_to_randomize],
        )


def test_add_features(sample_dataset: FeatureDataset, graph_dataset: FeatureDataset) -> None:
    """
    Test if the add_features method works correctly.

    :param sample_dataset: sample FeatureDataset
    :param graph_dataset: sample FeatureDataset with molecular graphs
    """
    sample_dataset.add_features(graph_dataset)
    assert sample_dataset.meta_info is not None
    assert "molecular_graph" in sample_dataset.meta_info
    assert "molecular_graph" in sample_dataset.view_names


def test_feature_dataset_csv_meta_handling():
    """Test `from_csv` and `to_csv` methods with and without meta_info handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # ------------------------------------
        # 0. Create initial test DataFrame/CSV
        # ------------------------------------
        df_with_named_cols = pd.DataFrame(
            {
                "id": ["A", "B", "C"],
                "feature_1": [1.0, 2.0, 3.0],
                "feature_2": [4.0, 5.0, 6.0],
            }
        )
        csv_with_meta = temp_dir / "input_with_meta.csv"
        df_with_named_cols.to_csv(csv_with_meta, index=False)

        view_name = "example_view"

        # ------------------------------------
        # 1. Load from CSV → should extract meta_info
        # ------------------------------------
        dataset = FeatureDataset.from_csv(
            path_to_csv=csv_with_meta,
            id_column="id",
            view_name=view_name,
        )

        assert dataset.meta_info == {view_name: ["feature_1", "feature_2"]}
        assert set(dataset.identifiers) == {"A", "B", "C"}
        assert dataset.view_names == [view_name]

        # ------------------------------------
        # 2. Save with meta_info → column names should be preserved
        # ------------------------------------
        csv_out_with_meta = temp_dir / "saved_with_meta.csv"
        dataset.to_csv(csv_out_with_meta, id_column="id", view_name=view_name)

        saved_df = pd.read_csv(csv_out_with_meta)
        pd.testing.assert_frame_equal(saved_df, df_with_named_cols, check_dtype=False)

        # ------------------------------------
        # 3. Save without meta_info → fallback to generic feature_0, feature_1
        # ------------------------------------
        dataset._meta_info = {}  # simulate no meta info
        csv_out_no_meta = temp_dir / "saved_no_meta.csv"
        dataset.to_csv(csv_out_no_meta, id_column="id", view_name=view_name)

        df_fallback = pd.DataFrame(
            {
                "id": ["A", "B", "C"],
                "feature_0": [1.0, 2.0, 3.0],
                "feature_1": [4.0, 5.0, 6.0],
            }
        )
        saved_fallback_df = pd.read_csv(csv_out_no_meta)
        pd.testing.assert_frame_equal(saved_fallback_df, df_fallback, check_dtype=False)

        # ------------------------------------
        # 4. Load fallback CSV → should reconstruct generic meta_info
        # ------------------------------------
        dataset_fallback = FeatureDataset.from_csv(
            path_to_csv=csv_out_no_meta,
            id_column="id",
            view_name=view_name,
        )

        assert dataset_fallback.meta_info == {view_name: ["feature_0", "feature_1"]}
        np.testing.assert_array_equal(
            dataset_fallback.features["B"][view_name],
            np.array([2.0, 5.0]),
        )
