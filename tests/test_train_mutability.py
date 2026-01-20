"""Test for train_and_predict dataset mutation fix."""

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.experiment import train_and_predict
from drevalpy.models import MODEL_FACTORY


def test_train_and_predict_does_not_mutate_with_reduce_to():
    """Test that reduce_to etc. doesn't mutate the original datasets.

    Before the fix: reduce_to was called directly on input datasets.
    After the fix: train_and_predict copies datasets first.
    """
    np.random.seed(42)

    train = DrugResponseDataset(
        response=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        cell_line_ids=np.array(["CL-1", "CL-2", "CL-3", "CL-4", "CL-5"]),
        drug_ids=np.array(["Drug-1", "Drug-1", "Drug-1", "Drug-1", "Drug-1"]),
        dataset_name="Toy_Data",
    )

    test = DrugResponseDataset(
        response=np.array([1.5, 2.5, 3.5]),
        cell_line_ids=np.array(["CL-1", "CL-2", "CL-3"]),
        drug_ids=np.array(["Drug-1", "Drug-1", "Drug-1"]),
        dataset_name="Toy_Data",
    )

    original_train_len = len(train)
    original_test_len = len(test)

    model = MODEL_FACTORY["NaivePredictor"]()

    # Create FeatureDataset with only some cell lines, forces reduce_to to remove rows
    cl_features = FeatureDataset(
        features={
            "CL-1": {"cell_line_id": np.array(["CL-1"])},
            "CL-2": {"cell_line_id": np.array(["CL-2"])},
        }
    )
    drug_features = FeatureDataset(features={"Drug-1": {"drug_id": np.array(["Drug-1"])}})

    train_and_predict(
        model=model,
        hpams={},
        path_data="data",
        train_dataset=train,
        prediction_dataset=test,
        cl_features=cl_features,
        drug_features=drug_features,
    )

    # Before fix: these fail because reduce_to removes rows from original datasets
    # After fix: these pass because train_and_predict copies datasets first
    assert (
        len(train) == original_train_len
    ), f"train_dataset was mutated by reduce_to: {original_train_len} -> {len(train)}"
    assert len(test) == original_test_len, f"test_dataset was mutated by reduce_to: {original_test_len} -> {len(test)}"
