"""Tests for the target mutation feature of the sklearn models."""

from typing import Any

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.baselines.sklearn_models import ElasticNetModel, RandomForest, SklearnModel
from drevalpy.models.baselines.target_mutation import FEATURE_NAMES, TargetMutationFeatures

_HYPERPARAMETERS: dict[type, dict[str, Any]] = {
    ElasticNetModel: {"l1_ratio": 0.5, "alpha": 1.0},
    RandomForest: {"n_estimators": 5, "criterion": "squared_error", "max_samples": 0.5, "n_jobs": 1, "max_depth": 3},
}


def _features(standardize: bool = False) -> TargetMutationFeatures:
    """
    Build a tiny feature builder: BRAF inhibitor 1, TP53 reactivator 2 (wildtype is sensitive), no entry for drug 3.

    :param standardize: whether to standardize the block
    :returns: the feature builder
    """
    return TargetMutationFeatures(
        drug_to_gene={"1": "BRAF", "2": "TP53"},
        drug_to_sensitive_status={"1": "activating", "2": "wildtype"},
        status={
            ("CVCL_A", "BRAF"): "activating",
            ("CVCL_B", "BRAF"): "wildtype",
            ("CVCL_A", "TP53"): "LoF",
            ("CVCL_B", "TP53"): "wildtype",
        },
        cell_line_to_cellosaurus={"A": "CVCL_A", "B": "CVCL_B"},
        standardize=standardize,
    )


def test_block_encodes_the_status_of_the_target_gene() -> None:
    """The block holds the one-hot status of the gene the drug acts on and the sensitizing flag."""
    block = _features().transform(np.array(["A", "B", "A", "B", "C", "A"]), np.array(["1", "1", "2", "2", "1", "3"]))
    assert block.shape == (6, len(FEATURE_NAMES)) == (6, 6)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 1],  # BRAF activating, sensitive
            [0, 0, 0, 1, 0, 0],  # BRAF wildtype
            [0, 1, 0, 0, 0, 0],  # TP53 LoF
            [0, 0, 0, 1, 0, 1],  # TP53 wildtype is the sensitive status of this drug
            [0, 0, 0, 0, 1, 0],  # cell line not in the status matrix
            [0, 0, 0, 0, 0, 0],  # drug without curated entry
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(block, expected)


def test_status_depends_on_the_pair() -> None:
    """The same cell line gets a different status for drugs with different target genes."""
    block = _features().transform(np.array(["A", "A"]), np.array(["1", "2"]))
    assert not np.array_equal(block[0], block[1])


def test_standardization_is_fitted_on_the_training_rows() -> None:
    """After fit, the columns of the training block have mean 0, and predict reuses the training statistics."""
    features = _features(standardize=True)
    cell_lines = np.array(["A", "B", "A", "B"])
    drugs = np.array(["1", "1", "2", "2"])
    features.fit(cell_lines, drugs)
    np.testing.assert_allclose(features.transform(cell_lines, drugs).mean(axis=0), 0, atol=1e-6)
    # a single row is transformed with the training mean, not its own
    single = features.transform(np.array(["A"]), np.array(["1"]))
    assert np.abs(single).max() > 0


def test_missing_table_raises(tmp_path) -> None:
    """A missing table is reported with its path.

    :param tmp_path: pytest temporary path fixture
    """
    with pytest.raises(FileNotFoundError, match="curated_drug_target_variant.csv"):
        TargetMutationFeatures.from_directory(str(tmp_path), str(tmp_path / "cell_line_names.csv"))


def _prepare(model_class: type[SklearnModel], data_dir, sample_dataset: DrugResponseDataset, target_mutation: bool):
    """
    Build the model, load the TOYv1 features and return the training data.

    :param model_class: sklearn model class
    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    :param target_mutation: value of the hyperparameter target_mutation
    :returns: model, training dataset, cell line input, drug input
    """
    model = model_class()
    model.build_model({**_HYPERPARAMETERS[model_class], "target_mutation": target_mutation})
    cell_line_input = model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path=str(data_dir), dataset_name="TOYv1")
    train = DrugResponseDataset(
        response=sample_dataset.response,
        cell_line_ids=sample_dataset.cell_line_ids,
        drug_ids=sample_dataset.drug_ids,
        dataset_name="TOYv1",
    )
    assert cell_line_input is not None and drug_input is not None
    train.reduce_to(cell_line_ids=cell_line_input.identifiers, drug_ids=drug_input.identifiers)
    return model, train, cell_line_input, drug_input


@pytest.mark.parametrize("model_class", [ElasticNetModel, RandomForest])
def test_model_runs_with_target_mutation(model_class, data_dir, sample_dataset) -> None:
    """
    The model trains and predicts with the extra block, which adds 6 columns to the feature matrix.

    :param model_class: sklearn model class
    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    """
    model, train, cell_line_input, drug_input = _prepare(model_class, data_dir, sample_dataset, True)
    plain, _, _, _ = _prepare(model_class, data_dir, sample_dataset, False)
    kwargs = dict(
        cell_line_view=model.cell_line_views[0],
        drug_view=model.drug_views[0],
        cell_line_ids_output=train.cell_line_ids,
        drug_ids_output=train.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    with_block = model.get_concatenated_features(**kwargs)
    without_block = plain.get_concatenated_features(**kwargs)
    assert with_block.shape[1] == without_block.shape[1] + len(FEATURE_NAMES)
    np.testing.assert_array_equal(with_block[:, :-6], without_block)
    assert np.abs(with_block[:, -6:]).sum() > 0, "the synthetic tables should give at least one curated pair"

    model.train(output=train, cell_line_input=cell_line_input, drug_input=drug_input)
    predictions = model.predict(
        cell_line_ids=train.cell_line_ids,
        drug_ids=train.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    assert predictions.shape == (len(train),)
    assert np.isfinite(predictions).all()


def test_target_mutation_is_off_by_default(data_dir, sample_dataset) -> None:
    """
    Without the hyperparameter, no tables are loaded and the feature matrix is unchanged.

    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    """
    model = RandomForest()
    model.build_model(_HYPERPARAMETERS[RandomForest])
    model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")
    assert model.target_mutation is None


def test_missing_tables_are_reported_when_target_mutation_is_set(data_dir, sample_dataset, tmp_path) -> None:
    """
    A wrong table directory fails when the features are loaded, not silently during training.

    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    :param tmp_path: pytest temporary path fixture
    """
    model = RandomForest()
    model.build_model(
        {**_HYPERPARAMETERS[RandomForest], "target_mutation": True, "target_mutation_path": str(tmp_path)}
    )
    with pytest.raises(FileNotFoundError):
        model.load_cell_line_features(data_path=str(data_dir), dataset_name="TOYv1")


def test_features_before_loading_raise(data_dir, sample_dataset) -> None:
    """
    Calling get_concatenated_features before the tables are loaded gives a clear error.

    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    """
    model, train, cell_line_input, drug_input = _prepare(RandomForest, data_dir, sample_dataset, True)
    model.target_mutation = None
    with pytest.raises(RuntimeError, match="not loaded"):
        model.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=train.cell_line_ids,
            drug_ids_output=train.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )


@pytest.mark.parametrize("model_class", [ElasticNetModel, RandomForest])
def test_save_and_load_keeps_the_feature(model_class, data_dir, sample_dataset, tmp_path) -> None:
    """
    A restored model predicts the same values and does not need the tables again.

    :param model_class: sklearn model class
    :param data_dir: path to the data directory
    :param sample_dataset: TOYv1 response data
    :param tmp_path: pytest temporary path fixture
    """
    model, train, cell_line_input, drug_input = _prepare(model_class, data_dir, sample_dataset, True)
    model.train(output=train, cell_line_input=cell_line_input, drug_input=drug_input)
    kwargs = dict(
        cell_line_ids=train.cell_line_ids,
        drug_ids=train.drug_ids,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )
    expected = model.predict(**kwargs)
    model.save(str(tmp_path))
    restored = model_class.load(str(tmp_path))
    assert restored.target_mutation is not None
    np.testing.assert_allclose(restored.predict(**kwargs), expected)
