"""Tests for shared CV-fold preparation helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.experiment_fold import (
    merge_train_validation,
    prepare_final_fold_training_data,
    prepare_fold_datasets,
)
from drevalpy.experiment_paths import consolidate_results_path, experiment_result_path
from drevalpy.models import construct_model


def _toy_split(*, include_es: bool = False) -> dict[str, DrugResponseDataset]:
    def _ds(responses: list[float], drugs: list[str]) -> DrugResponseDataset:
        return DrugResponseDataset(
            response=np.array(responses, dtype=float),
            cell_line_ids=np.array([f"cl{i}" for i in range(len(responses))]),
            drug_ids=np.array(drugs),
            dataset_name="TOYv1",
        )

    split = {
        "train": _ds([1.0, 2.0, 3.0], ["d1", "d2", "d1"]),
        "validation": _ds([4.0], ["d1"]),
        "test": _ds([5.0, 6.0], ["d1", "d2"]),
    }
    if include_es:
        split["validation_es"] = _ds([4.0], ["d1"])
        split["early_stopping"] = _ds([7.0], ["d1"])
    return split


def test_prepare_fold_datasets_uses_early_stopping_partitions() -> None:
    split = _toy_split(include_es=True)
    nn = construct_model("SimpleNeuralNetwork")
    fold = prepare_fold_datasets(split, nn, "SimpleNeuralNetwork", None)
    assert len(fold.train) == 3
    assert fold.early_stopping is not None
    assert len(fold.early_stopping) == 1
    assert set(fold.early_stopping.drug_ids) == {"d1"}


def test_prepare_final_fold_merges_train_validation_without_raw_es_override() -> None:
    model_class = construct_model("ElasticNet")
    split = _toy_split(include_es=False)
    fold = prepare_final_fold_training_data(split, model_class, "ElasticNet", None)
    assert len(fold.train) == 4
    # Original split objects must remain unchanged.
    assert len(split["train"]) == 3
    assert len(split["validation"]) == 1


def test_merge_train_validation_returns_copy() -> None:
    split = _toy_split()
    merged = merge_train_validation(split["train"], split["validation"])
    assert len(merged) == 4
    assert len(split["train"]) == 3


def test_result_paths_include_dataset() -> None:
    path = experiment_result_path("results", "run1", "TOYv1", "LCO")
    assert path.as_posix().endswith("results/run1/TOYv1/LCO")
    assert consolidate_results_path("out", "run1", "TOYv1", "LPO").as_posix().endswith("out/run1/TOYv1/LPO")


def test_single_drug_masking_applies_to_all_partitions() -> None:
    model_class = construct_model("SingleDrugElasticNet")
    split = _toy_split(include_es=False)
    fold = prepare_fold_datasets(split, model_class, "SingleDrugElasticNet", "d1")
    assert set(fold.train.drug_ids) == {"d1"}
    assert set(fold.validation.drug_ids) == {"d1"}
    assert set(fold.test.drug_ids) == {"d1"}
