"""Shared fixtures for the drug featurizer tests."""

from __future__ import annotations

import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource


@pytest.fixture
def drug_source(synthetic_dataset: Dataset) -> DrugFeatureSource:
    return DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)
