"""Shared fixtures for the drug featurizer tests."""

from __future__ import annotations

import pytest

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource
from tests.components.featurizers.drug._helpers import DrugSource


@pytest.fixture
def drug_source(synthetic_dataset: Dataset) -> FeatureSource:
    return FeatureSource(synthetic_dataset, "drug", synthetic_dataset.drug_ids)


@pytest.fixture
def smiles_only(synthetic_dataset: Dataset) -> DrugSource:
    """The synthetic drugs' SMILES without any stored view."""
    smiles = synthetic_dataset.mdata.mod["response"].var["canonical_smiles"]
    return DrugSource(smiles={str(drug): smiles[drug] for drug in synthetic_dataset.drug_ids})
