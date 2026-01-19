"""Tests for CV split integrity - ensuring no data leakage and proper tissue column handling.

These tests verify the fixes for:

- GitHub Issue #349: Validation data accumulation bug - where validation data was
  inadvertently accumulated into training data across sequential model runs.
- Tissue column preservation when loading splits from CSV files.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.experiment import get_datasets_from_cv_split
from drevalpy.models import MODEL_FACTORY


class TestTissueColumnPreservation:
    """Tests for tissue column preservation when saving/loading splits."""

    def test_tissue_column_preserved_in_csv_roundtrip(self):
        """Test that tissue column is preserved when saving and loading from CSV."""
        # Create dataset with tissue information
        dataset = DrugResponseDataset(
            response=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            cell_line_ids=np.array(["CL1", "CL2", "CL3", "CL4", "CL5"]),
            drug_ids=np.array(["D1", "D2", "D3", "D4", "D5"]),
            tissues=np.array(["Breast", "Lung", "Kidney", "Brain", "Liver"]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_dataset.csv"
            dataset.to_csv(csv_path)

            # Load the dataset back - tissue_column now defaults to "tissue"
            loaded_dataset = DrugResponseDataset.from_csv(csv_path)

            assert loaded_dataset.tissue is not None, "Tissue column should be loaded"
            np.testing.assert_array_equal(
                loaded_dataset.tissue,
                dataset.tissue,
                err_msg="Tissue values should match after roundtrip",
            )

    def test_tissue_column_preserved_in_split_roundtrip(self):
        """Test that tissue column is preserved when saving and loading CV splits."""
        # Create dataset with tissue information
        dataset = DrugResponseDataset(
            response=np.random.random(100),
            cell_line_ids=np.repeat([f"CL-{i}" for i in range(10)], 10),
            drug_ids=np.tile([f"Drug-{i}" for i in range(10)], 10),
            tissues=np.array(
                ["Breast", "Lung", "Kidney", "Brain", "Liver", "Pancreas", "Colon", "Skin", "Bone", "Blood"] * 10
            ),
        )

        # Create splits
        dataset.split_dataset(n_cv_splits=3, mode="LPO", split_validation=True, validation_ratio=0.5, random_state=42)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save splits
            dataset.save_splits(path=temp_dir)

            # Create new dataset and load splits
            new_dataset = DrugResponseDataset(
                response=np.array([]),
                cell_line_ids=np.array([]),
                drug_ids=np.array([]),
                dataset_name=dataset.dataset_name,
            )
            new_dataset.load_splits(path=temp_dir)

            # Check that tissue is preserved in all splits
            for i, split in enumerate(new_dataset.cv_splits):
                for split_name in ["train", "test", "validation", "validation_es", "early_stopping"]:
                    if split_name in split:
                        assert (
                            split[split_name].tissue is not None
                        ), f"Tissue column should be loaded for split {i} {split_name}"
                        assert len(split[split_name].tissue) == len(
                            split[split_name].response
                        ), f"Tissue array length should match response length for split {i} {split_name}"

    def test_from_csv_default_tissue_column(self):
        """Test that from_csv defaults to loading tissue column when present."""
        dataset = DrugResponseDataset(
            response=np.array([1.0, 2.0, 3.0]),
            cell_line_ids=np.array(["CL1", "CL2", "CL3"]),
            drug_ids=np.array(["D1", "D2", "D3"]),
            tissues=np.array(["Breast", "Lung", "Kidney"]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            dataset.to_csv(csv_path)

            # Load without explicitly specifying tissue_column - should still load tissue
            loaded = DrugResponseDataset.from_csv(csv_path)
            assert loaded.tissue is not None
            np.testing.assert_array_equal(loaded.tissue, dataset.tissue)

    def test_from_csv_missing_tissue_column(self):
        """Test that from_csv handles missing tissue column gracefully."""
        # Create dataset without tissue
        dataset = DrugResponseDataset(
            response=np.array([1.0, 2.0, 3.0]),
            cell_line_ids=np.array(["CL1", "CL2", "CL3"]),
            drug_ids=np.array(["D1", "D2", "D3"]),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test.csv"
            dataset.to_csv(csv_path)

            # Load - should handle missing tissue column gracefully
            loaded = DrugResponseDataset.from_csv(csv_path)
            assert loaded.tissue is None


class TestCVSplitDataLeakage:
    """Tests for ensuring no data leakage between models when using CV splits."""

    @pytest.fixture
    def sample_cv_split(self):
        """Create a sample CV split with all dataset types.

        :returns: dictionary with train, validation, validation_es, early_stopping, and test datasets
        """
        np.random.seed(42)
        n_samples = 100

        train = DrugResponseDataset(
            response=np.random.random(n_samples),
            cell_line_ids=np.array([f"CL-{i}" for i in range(n_samples)]),
            drug_ids=np.array([f"Drug-{i % 10}" for i in range(n_samples)]),
            tissues=np.array([f"Tissue-{i % 5}" for i in range(n_samples)]),
        )

        validation = DrugResponseDataset(
            response=np.random.random(30),
            cell_line_ids=np.array([f"CL-V{i}" for i in range(30)]),
            drug_ids=np.array([f"Drug-{i % 10}" for i in range(30)]),
            tissues=np.array([f"Tissue-{i % 5}" for i in range(30)]),
        )

        validation_es = DrugResponseDataset(
            response=np.random.random(20),
            cell_line_ids=np.array([f"CL-VE{i}" for i in range(20)]),
            drug_ids=np.array([f"Drug-{i % 10}" for i in range(20)]),
            tissues=np.array([f"Tissue-{i % 5}" for i in range(20)]),
        )

        early_stopping = DrugResponseDataset(
            response=np.random.random(10),
            cell_line_ids=np.array([f"CL-ES{i}" for i in range(10)]),
            drug_ids=np.array([f"Drug-{i % 10}" for i in range(10)]),
            tissues=np.array([f"Tissue-{i % 5}" for i in range(10)]),
        )

        test = DrugResponseDataset(
            response=np.random.random(25),
            cell_line_ids=np.array([f"CL-T{i}" for i in range(25)]),
            drug_ids=np.array([f"Drug-{i % 10}" for i in range(25)]),
            tissues=np.array([f"Tissue-{i % 5}" for i in range(25)]),
        )

        return {
            "train": train,
            "validation": validation,
            "validation_es": validation_es,
            "early_stopping": early_stopping,
            "test": test,
        }

    def test_get_datasets_returns_copies(self, sample_cv_split):
        """Test that get_datasets_from_cv_split returns copies, not references.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        model_class = MODEL_FACTORY["ElasticNet"]

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split, model_class=model_class, model_name="ElasticNet"
        )

        # Verify that returned datasets are copies, not references
        assert train is not sample_cv_split["train"]
        assert val is not sample_cv_split["validation"]
        assert test is not sample_cv_split["test"]

        # Verify that modifying returned datasets doesn't affect originals
        original_train_len = len(sample_cv_split["train"].response)
        train.add_rows(val)

        assert (
            len(sample_cv_split["train"].response) == original_train_len
        ), "Original train dataset should not be modified when adding rows to the copy"

    def test_get_datasets_returns_copies_with_early_stopping(self, sample_cv_split):
        """Test that early stopping datasets are also copies.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        model_class = MODEL_FACTORY["SimpleNeuralNetwork"]  # Uses early stopping

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split, model_class=model_class, model_name="SimpleNeuralNetwork"
        )

        # For early stopping models, validation should be validation_es
        assert val is not sample_cv_split["validation_es"]
        assert es is not sample_cv_split["early_stopping"]

        # Verify modifications don't affect originals
        original_es_len = len(sample_cv_split["early_stopping"].response)
        es.remove_rows(np.array([0, 1, 2]))

        assert (
            len(sample_cv_split["early_stopping"].response) == original_es_len
        ), "Original early_stopping dataset should not be modified"

    def test_no_validation_accumulation_across_models(self, sample_cv_split):
        """Test that validation data is not accumulated into training data across models.

        This is the core test for the bug fix in GitHub Issue #349.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        original_train_len = len(sample_cv_split["train"].response)
        original_val_len = len(sample_cv_split["validation"].response)

        # Simulate running multiple models sequentially
        models_to_test = ["ElasticNet", "RandomForest", "ElasticNet"]

        for model_name in models_to_test:
            model_class = MODEL_FACTORY[model_name]

            train, val, es, test = get_datasets_from_cv_split(
                split=sample_cv_split, model_class=model_class, model_name=model_name
            )

            # Simulate what happens in experiment.py: train.add_rows(val)
            train.add_rows(val)

            # After each model, verify original split is unchanged
            assert len(sample_cv_split["train"].response) == original_train_len, (
                f"Original train dataset should remain {original_train_len} samples "
                f"after processing {model_name}, but got {len(sample_cv_split['train'].response)}"
            )
            assert len(sample_cv_split["validation"].response) == original_val_len, (
                f"Original validation dataset should remain {original_val_len} samples "
                f"after processing {model_name}"
            )

    def test_no_test_data_leakage(self, sample_cv_split):
        """Test that test data is never added to training data.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        original_test_cell_lines = set(sample_cv_split["test"].cell_line_ids)

        for model_name in ["ElasticNet", "SimpleNeuralNetwork"]:
            model_class = MODEL_FACTORY[model_name]

            train, val, es, test = get_datasets_from_cv_split(
                split=sample_cv_split, model_class=model_class, model_name=model_name
            )

            # Simulate adding validation to train (as done in experiment.py)
            train.add_rows(val)

            # Verify no test cell lines leaked into training
            train_cell_lines = set(train.cell_line_ids)

            # Check for intersection (should be empty for our test data setup)
            leaked_cell_lines = train_cell_lines & original_test_cell_lines
            assert len(leaked_cell_lines) == 0, f"Test cell lines leaked into training: {leaked_cell_lines}"

    def test_datasets_have_correct_sizes(self, sample_cv_split):
        """Test that returned datasets have the expected sizes.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        model_class = MODEL_FACTORY["ElasticNet"]

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split, model_class=model_class, model_name="ElasticNet"
        )

        # For non-early-stopping models, validation should be from "validation" key
        assert len(train.response) == len(sample_cv_split["train"].response)
        assert len(val.response) == len(sample_cv_split["validation"].response)
        assert len(test.response) == len(sample_cv_split["test"].response)
        assert es is None  # ElasticNet doesn't use early stopping

    def test_datasets_have_correct_sizes_early_stopping(self, sample_cv_split):
        """Test that early stopping models get correct dataset sizes.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        model_class = MODEL_FACTORY["SimpleNeuralNetwork"]

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split, model_class=model_class, model_name="SimpleNeuralNetwork"
        )

        # For early-stopping models, validation should be from "validation_es" key
        assert len(train.response) == len(sample_cv_split["train"].response)
        assert len(val.response) == len(sample_cv_split["validation_es"].response)
        assert len(es.response) == len(sample_cv_split["early_stopping"].response)
        assert len(test.response) == len(sample_cv_split["test"].response)

    def test_tissue_preserved_in_get_datasets(self, sample_cv_split):
        """Test that tissue information is preserved when getting datasets from split.

        :param sample_cv_split: pytest fixture providing sample CV split data
        """
        model_class = MODEL_FACTORY["ElasticNet"]

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split, model_class=model_class, model_name="ElasticNet"
        )

        # Verify tissue is preserved in all returned datasets
        assert train.tissue is not None
        assert val.tissue is not None
        assert test.tissue is not None

        # Verify tissue values match originals
        np.testing.assert_array_equal(train.tissue, sample_cv_split["train"].tissue)
        np.testing.assert_array_equal(val.tissue, sample_cv_split["validation"].tissue)
        np.testing.assert_array_equal(test.tissue, sample_cv_split["test"].tissue)


class TestSingleDrugModelSplits:
    """Tests specific to single-drug model handling in CV splits."""

    @pytest.fixture
    def sample_cv_split_multi_drug(self):
        """Create a sample CV split with multiple drugs for single-drug model testing.

        :returns: dictionary with train, validation, and test datasets containing multiple drugs
        """
        np.random.seed(42)

        # Create data with multiple drugs
        drugs = ["DrugA", "DrugB", "DrugC"]
        n_per_drug = 20

        train = DrugResponseDataset(
            response=np.random.random(n_per_drug * len(drugs)),
            cell_line_ids=np.array([f"CL-{i}" for i in range(n_per_drug * len(drugs))]),
            drug_ids=np.array(drugs * n_per_drug),
            tissues=np.array([f"Tissue-{i % 3}" for i in range(n_per_drug * len(drugs))]),
        )

        validation = DrugResponseDataset(
            response=np.random.random(10 * len(drugs)),
            cell_line_ids=np.array([f"CL-V{i}" for i in range(10 * len(drugs))]),
            drug_ids=np.array(drugs * 10),
            tissues=np.array([f"Tissue-{i % 3}" for i in range(10 * len(drugs))]),
        )

        test = DrugResponseDataset(
            response=np.random.random(5 * len(drugs)),
            cell_line_ids=np.array([f"CL-T{i}" for i in range(5 * len(drugs))]),
            drug_ids=np.array(drugs * 5),
            tissues=np.array([f"Tissue-{i % 3}" for i in range(5 * len(drugs))]),
        )

        return {
            "train": train,
            "validation": validation,
            "test": test,
        }

    def test_single_drug_model_masks_correctly(self, sample_cv_split_multi_drug):
        """Test that single-drug models only get data for their specific drug.

        :param sample_cv_split_multi_drug: pytest fixture providing sample CV split with multiple drugs
        """
        from drevalpy.models import SINGLE_DRUG_MODEL_FACTORY

        if len(SINGLE_DRUG_MODEL_FACTORY) == 0:
            pytest.skip("No single-drug models available")

        # Get the first available single-drug model
        model_name = list(SINGLE_DRUG_MODEL_FACTORY.keys())[0]
        model_class = SINGLE_DRUG_MODEL_FACTORY[model_name]

        target_drug = "DrugA"

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split_multi_drug, model_class=model_class, model_name=model_name, drug_id=target_drug
        )

        # Verify only target drug data is returned
        assert all(drug == target_drug for drug in train.drug_ids)
        assert all(drug == target_drug for drug in val.drug_ids)
        assert all(drug == target_drug for drug in test.drug_ids)

    def test_single_drug_model_doesnt_modify_original(self, sample_cv_split_multi_drug):
        """Test that single-drug model masking doesn't modify original split.

        :param sample_cv_split_multi_drug: pytest fixture providing sample CV split with multiple drugs
        """
        from drevalpy.models import SINGLE_DRUG_MODEL_FACTORY

        if len(SINGLE_DRUG_MODEL_FACTORY) == 0:
            pytest.skip("No single-drug models available")

        model_name = list(SINGLE_DRUG_MODEL_FACTORY.keys())[0]
        model_class = SINGLE_DRUG_MODEL_FACTORY[model_name]

        original_train_len = len(sample_cv_split_multi_drug["train"].response)
        original_drugs = set(sample_cv_split_multi_drug["train"].drug_ids)

        train, val, es, test = get_datasets_from_cv_split(
            split=sample_cv_split_multi_drug, model_class=model_class, model_name=model_name, drug_id="DrugA"
        )

        # Original should be unchanged
        assert len(sample_cv_split_multi_drug["train"].response) == original_train_len
        assert set(sample_cv_split_multi_drug["train"].drug_ids) == original_drugs
