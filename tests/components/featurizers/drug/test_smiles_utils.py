"""Tests for SMILES lookup through a ``FeatureSource``.

Mirrors :mod:`drevalpy.components.featurizers.drug._smiles_utils`.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug._smiles_utils import get_smiles_for_entities
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests.conftest import MockFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints


def test_returns_none_without_a_mudata_backing() -> None:
    source = MockFeatureSource(features={"d1": {}})

    assert get_smiles_for_entities(source, np.array(["d1"])) is None


def test_returns_none_when_the_response_var_has_no_smiles_column() -> None:
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    source = DrugFeatureSource(dataset, dataset.drug_ids)

    assert get_smiles_for_entities(source, dataset.drug_ids) is None


def test_returns_smiles_indexed_by_the_requested_drug_ids(synthetic_dataset: Dataset) -> None:
    drug_ids = synthetic_dataset.drug_ids[:3]
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    smiles = get_smiles_for_entities(source, drug_ids)

    assert smiles is not None
    assert list(smiles.index) == list(drug_ids)
    assert all(isinstance(value, str) and value for value in smiles)


def test_unknown_drug_ids_reindex_to_nan(synthetic_dataset: Dataset) -> None:
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    smiles = get_smiles_for_entities(source, np.array(["not-a-drug"]))

    assert smiles is not None
    assert smiles.isna().all()
