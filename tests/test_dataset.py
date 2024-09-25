import os
import numpy as np
import tempfile
import networkx as nx
import pytest
from flaky import flaky

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.utils import get_response_transformation

#### Tests for the DrugResponseDataset class


# Test if the dataset loads correctly from CSV files
def test_response_dataset_load():
    # Create a temporary CSV file with mock data
    data = {
        "cell_line_id": [1, 2, 3],
        "drug_id": ["A", "B", "C"],
        "response": [0.1, 0.2, 0.3],
    }
    dataset = DrugResponseDataset(
        cell_line_ids=data["cell_line_id"],
        drug_ids=data["drug_id"],
        response=data["response"],
    )
    dataset.save("dataset.csv")
    del dataset
    # Load the dataset
    dataset = DrugResponseDataset()
    dataset.load("dataset.csv")

    os.remove("dataset.csv")

    # Check if the dataset loaded correctly
    assert np.array_equal(dataset.cell_line_ids, data["cell_line_id"])
    assert np.array_equal(dataset.drug_ids, data["drug_id"])
    assert np.allclose(dataset.response, data["response"])


def test_response_dataset_add_rows():
    dataset1 = DrugResponseDataset(
        response=np.array([1, 2, 3]),
        cell_line_ids=np.array([101, 102, 103]),
        drug_ids=np.array(["A", "B", "C"]),
    )
    dataset2 = DrugResponseDataset(
        response=np.array([4, 5, 6]),
        cell_line_ids=np.array([104, 105, 106]),
        drug_ids=np.array(["D", "E", "F"]),
    )
    dataset1.add_rows(dataset2)

    assert np.array_equal(dataset1.response, np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(
        dataset1.cell_line_ids, np.array([101, 102, 103, 104, 105, 106])
    )
    assert np.array_equal(dataset1.drug_ids, np.array(["A", "B", "C", "D", "E", "F"]))


def test_remove_nan_responses():
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, np.nan, 5, 6]),
        cell_line_ids=np.array([101, 102, 103, 104, 105, 106]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
    )
    dataset.remove_nan_responses()
    assert np.array_equal(dataset.response, np.array([1, 2, 3, 5, 6]))
    assert np.array_equal(dataset.cell_line_ids, np.array([101, 102, 103, 105, 106]))
    assert np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "E", "F"]))


def test_response_dataset_shuffle():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5, 6]),
        cell_line_ids=np.array([101, 102, 103, 104, 105, 106]),
        drug_ids=np.array(["A", "B", "C", "D", "E", "F"]),
    )

    # Shuffle the dataset
    dataset.shuffle(random_state=42)

    # Check if the length remains the same
    assert len(dataset.response) == 6
    assert len(dataset.cell_line_ids) == 6
    assert len(dataset.drug_ids) == 6

    # Check if the response, cell_line_ids, and drug_ids arrays are shuffled
    assert not np.array_equal(dataset.response, np.array([1, 2, 3, 4, 5]))
    assert not np.array_equal(
        dataset.cell_line_ids, np.array([101, 102, 103, 104, 105])
    )
    assert not np.array_equal(dataset.drug_ids, np.array(["A", "B", "C", "D", "E"]))


def test_response_data_remove_drugs_and_cell_lines():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )

    # Remove specific drugs and cell lines
    dataset.remove_drugs(["A", "C"])
    dataset.remove_cell_lines([101, 103])

    # Check if the removed drugs and cell lines are not present in the dataset
    assert "A" not in dataset.drug_ids
    assert "C" not in dataset.drug_ids
    assert 101 not in dataset.cell_line_ids
    assert 103 not in dataset.cell_line_ids

    # Check if the length of response, cell_line_ids, and drug_ids arrays is reduced accordingly
    assert len(dataset.response) == 3
    assert len(dataset.cell_line_ids) == 3
    assert len(dataset.drug_ids) == 3


def test_remove_rows():
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )
    dataset.remove_rows([0, 2, 4])
    assert np.array_equal(dataset.response, np.array([2, 4]))
    assert np.array_equal(dataset.cell_line_ids, np.array([102, 104]))
    assert np.array_equal(dataset.drug_ids, np.array(["B", "D"]))


def test_response_dataset_reduce_to():
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )

    # Reduce the dataset to a subset of cell line IDs and drug IDs
    dataset.reduce_to(cell_line_ids=[102, 104], drug_ids=["B", "D"])

    # Check if only the rows corresponding to the specified cell line IDs and drug IDs remain
    assert all(cell_line_id in [102, 104] for cell_line_id in dataset.cell_line_ids)
    assert all(drug_id in ["B", "D"] for drug_id in dataset.drug_ids)

    # Check if the length of response, cell_line_ids, and drug_ids arrays is reduced accordingly
    assert len(dataset.response) == 2
    assert len(dataset.cell_line_ids) == 2
    assert len(dataset.drug_ids) == 2


@pytest.mark.parametrize("mode", ["LPO", "LCO", "LDO"])
@pytest.mark.parametrize("split_validation", [True, False])
def test_split_response_dataset(mode, split_validation):
    # Create a dataset with known values
    dataset = DrugResponseDataset(
        response=np.random.random(100),
        cell_line_ids=["CL-1"] * 10
        + ["CL-2"] * 10
        + ["CL-3"] * 10
        + ["CL-4"] * 10
        + ["CL-5"] * 10
        + ["CL-6"] * 10
        + ["CL-7"] * 10
        + ["CL-8"] * 10
        + ["CL-9"] * 10
        + ["CL-10"] * 10,
        drug_ids=[
            "Drug-1",
            "Drug-2",
            "Drug-3",
            "Drug-4",
            "Drug-5",
            "Drug-6",
            "Drug-7",
            "Drug-8",
            "Drug-9",
            "Drug-10",
        ]
        * 10,
    )
    # 100 datapoints, 10 cell lines, 10 drugs
    # LPO: With 10% validation, 5 folds -> in 1 fold: 20 samples in test, 80 in train+val -> 40 in train, 40 samples in validation -> 30 in val_es, 10 in early stopping
    # LCO: With 10% validation, 5 folds -> in 1 fold: 2 cell lines in test, 8 in train + val -> 4 in train, 4 samples in validation -> 3 in val_es, 1 in early stopping
    # LDO: With 10% validation, 5 folds -> in 1 fold: 2 drugs in test, 8 in train + val -> 4 in train, 4 samples in validation -> 3 in val_es, 1 in early stopping

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
                for val_es in ["validation", "validation_es", "early_stopping"]:
                    validation_cell_lines = set(split[val_es].cell_line_ids)
                    assert validation_cell_lines.isdisjoint(
                        test_cell_lines
                    )  # Check for disjointness between validation and test cell lines

        elif mode == "LDO":
            train_drugs = set(split["train"].drug_ids)
            test_drugs = set(split["test"].drug_ids)

            assert train_drugs.isdisjoint(test_drugs)

            if split_validation:  # Only check if validation split is enabled
                for val_es in ["validation", "validation_es", "early_stopping"]:
                    validation_drugs = set(split[val_es].drug_ids)
                    assert validation_drugs.isdisjoint(
                        test_drugs
                    )  # Check for disjointness between validation and test drugs

        elif mode == "LPO":
            train_pairs = set(
                zip(split["train"].cell_line_ids, split["train"].drug_ids)
            )
            test_pairs = set(zip(split["test"].cell_line_ids, split["test"].drug_ids))

            assert train_pairs.isdisjoint(test_pairs)

            if split_validation:  # Only check if validation split is enabled
                for val_es in ["validation", "validation_es", "early_stopping"]:
                    validation_pairs = set(
                        zip(split[val_es].cell_line_ids, split[val_es].drug_ids)
                    )
                    assert validation_pairs.isdisjoint(
                        test_pairs
                    )  # Check for disjointness between validation and test pairs

    tempdir = tempfile.TemporaryDirectory()
    dataset.save_splits(path=tempdir.name)
    dataset.load_splits(path=tempdir.name)


@pytest.mark.parametrize("resp_transform", ["standard", "minmax", "robust"])
def test_transform(resp_transform):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    dataset = DrugResponseDataset(
        response=np.array([1, 2, 3, 4, 5]),
        cell_line_ids=np.array([101, 102, 103, 104, 105]),
        drug_ids=np.array(["A", "B", "C", "D", "E"]),
    )
    transform = get_response_transformation(resp_transform)
    dataset.fit_transform(transform)
    if resp_transform == "standard":
        scaler = StandardScaler()
    elif resp_transform == "minmax":
        scaler = MinMaxScaler()
    elif resp_transform == "robust":
        scaler = RobustScaler()
    vals = scaler.fit_transform(np.array([1, 2, 3, 4, 5]).reshape(-1, 1))
    assert np.allclose(dataset.response, vals.flatten())

    dataset.inverse_transform(transform)
    assert np.allclose(dataset.response, np.array([1, 2, 3, 4, 5]))


#### Tests for the FeatureDataset class


@pytest.fixture
def sample_dataset():
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


def random_power_law_graph(size=20):
    # make a graph with degrees distributed as a power law
    G = nx.Graph()
    degrees = np.round(nx.utils.powerlaw_sequence(size, 2.5))
    G.add_nodes_from(range(size))
    G = nx.expected_degree_graph(degrees, selfloops=False)
    # only extract largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    # assign edge attributes
    for u, v in G.edges():
        G[u][v]["original_edge"] = f"({u}_{v})"
    return G


@pytest.fixture
def graph_dataset():
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


def test_feature_dataset_get_ids(sample_dataset):
    assert sample_dataset.get_ids() == ["drug1", "drug2", "drug3", "drug4", "drug5"]


def test_feature_dataset_get_view_names(sample_dataset):
    assert sample_dataset.get_view_names() == ["fingerprints", "chemical_features"]


def test_feature_dataset_get_feature_matrix(sample_dataset):
    feature_matrix = sample_dataset.get_feature_matrix(
        "fingerprints", ["drug1", "drug2"]
    )
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
    assert type(feature_matrix) == np.ndarray


def test_feature_dataset_copy(sample_dataset):
    copied_dataset = sample_dataset.copy()
    assert (
        copied_dataset.features["drug1"]["fingerprints"]
        is not sample_dataset.features["drug1"]["fingerprints"]
    )
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


@flaky(
    max_runs=25
)  # permutation randomization might map to the same feature vector for some tries
def test_permutation_randomization(sample_dataset):
    views_to_randomize, randomization_type = "fingerprints", "permutation"
    start_sample_dataset = sample_dataset.copy()
    sample_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in sample_dataset.features.items():
        assert not np.allclose(
            features[views_to_randomize],
            start_sample_dataset.features[drug][views_to_randomize],
        )


@flaky(
    max_runs=25
)  # permutation randomization might map to the same feature vector for some tries
def test_permutation_randomization_graph(graph_dataset):
    views_to_randomize, randomization_type = "molecular_graph", "permutation"
    start_graph_dataset = graph_dataset.copy()
    graph_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in graph_dataset.features.items():
        # assert that drugs have different molecular graphs now
        assert not nx.is_isomorphic(
            features[views_to_randomize],
            start_graph_dataset.features[drug][views_to_randomize],
        )


def test_invariant_randomization_array(sample_dataset):
    views_to_randomize, randomization_type = "chemical_features", "invariant"
    start_sample_dataset = sample_dataset.copy()
    sample_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in sample_dataset.features.items():
        assert not np.allclose(
            features[views_to_randomize],
            start_sample_dataset.features[drug][views_to_randomize],
        )


@flaky(max_runs=5)  # expected degree randomization might produce the same graph
def test_invariant_randomization_graph(graph_dataset):
    views_to_randomize, randomization_type = "molecular_graph", "invariant"
    start_graph_dataset = graph_dataset.copy()
    graph_dataset.randomize_features(views_to_randomize, randomization_type)
    for drug, features in graph_dataset.features.items():
        assert not nx.is_isomorphic(
            features[views_to_randomize],
            start_graph_dataset.features[drug][views_to_randomize],
        )


def test_feature_dataset_save_and_load(sample_dataset):
    tmp = tempfile.NamedTemporaryFile()
    with pytest.raises(NotImplementedError):
        sample_dataset.save(path=tmp.name)

    with pytest.raises(NotImplementedError):
        sample_dataset.load(path=tmp.name)


def test_add_features(sample_dataset, graph_dataset):
    sample_dataset.add_features(graph_dataset)
    assert "molecular_graph" in sample_dataset.meta_info
    assert "molecular_graph" in sample_dataset.get_view_names()


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
