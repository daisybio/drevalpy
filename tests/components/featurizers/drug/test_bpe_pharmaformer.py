"""Tests for the BPE PharmaFormer drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.bpe_pharmaformer`. ``subword-nmt``
is a hard dependency and BPE codes are learned from the fixture's own SMILES, so
nothing here touches the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.bpe_pharmaformer import BpePharmaformerDrugFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests.conftest import MockFeatureSource


@pytest.fixture
def drug_source(synthetic_dataset: Dataset) -> DrugFeatureSource:
    """Dataset-backed drug source with real, rdkit-parseable SMILES."""
    return DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)


def test_output_dim_defaults_to_max_length() -> None:
    assert BpePharmaformerDrugFeaturizer(max_length=64).output_dim == 64


def test_hyperparameter_space_exposes_num_symbols_and_max_length() -> None:
    assert set(BpePharmaformerDrugFeaturizer.get_hyperparameter_space()) == {"num_symbols", "max_length"}


def test_transform_before_fit_raises() -> None:
    featurizer = BpePharmaformerDrugFeaturizer()
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(RuntimeError, match="must be fit before transform"):
        featurizer._transform(source, np.array(["d1"]))


def test_fit_without_smiles_raises() -> None:
    featurizer = BpePharmaformerDrugFeaturizer()
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="Cannot learn BPE codes"):
        featurizer.fit(source, entity_ids=np.array(["d1"]))


def test_fit_then_transform_emits_a_padded_token_matrix(
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    drug_ids = synthetic_dataset.drug_ids[:2]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=32)

    featurizer.fit(drug_source, entity_ids=drug_ids)
    matrix = featurizer.transform(drug_source, drug_ids)

    assert matrix.shape == (2, 32)
    assert matrix.dtype == np.float32
    assert featurizer.output_dim == 32


def test_transform_truncates_sequences_longer_than_max_length(
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    drug_ids = synthetic_dataset.drug_ids[:2]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=4)
    featurizer.fit(drug_source, entity_ids=drug_ids)

    matrix = featurizer.transform(drug_source, drug_ids)

    assert matrix.shape == (2, 4)
    assert np.all(matrix > 0)


def test_transform_leaves_unknown_drugs_as_zero_rows(
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=16)
    featurizer.fit(drug_source, entity_ids=synthetic_dataset.drug_ids[:2])

    # ``_transform`` rather than ``transform``: the public wrapper's NaN detection
    # sees no ``bpe_smiles`` row for an unknown drug and overwrites it with NaN.
    matrix = featurizer._transform(drug_source, np.array(["not-a-drug"]))

    np.testing.assert_allclose(matrix, np.zeros((1, 16), dtype=np.float32))


def test_transform_blocks_emit_a_single_bpe_smiles_block(
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    drug_ids = synthetic_dataset.drug_ids[:2]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=16)
    featurizer.fit(drug_source, entity_ids=drug_ids)

    blocks = featurizer.transform_blocks(drug_source, drug_ids)

    assert set(blocks) == {"bpe_smiles"}
    assert blocks["bpe_smiles"].feature_names is None


def test_transform_pads_sequences_shorter_than_max_length(
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    drug_ids = synthetic_dataset.drug_ids[:2]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=512)
    featurizer.fit(drug_source, entity_ids=drug_ids)

    matrix = featurizer._transform(drug_source, drug_ids)

    assert matrix.shape == (2, 512)
    assert np.any(matrix[0] == 0.0)


def test_transform_without_smiles_raises(drug_source: DrugFeatureSource, synthetic_dataset: Dataset) -> None:
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, max_length=16)
    featurizer.fit(drug_source, entity_ids=synthetic_dataset.drug_ids[:2])

    with pytest.raises(ValueError, match="Cannot encode BPE"):
        featurizer._transform(MockFeatureSource(features={"d1": {}}), np.array(["d1"]))


def test_learn_bpe_reports_missing_subword_nmt(
    monkeypatch: pytest.MonkeyPatch,
    drug_source: DrugFeatureSource,
    synthetic_dataset: Dataset,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _fail_on_subword_nmt(name, *args, **kwargs):
        if name.startswith("subword_nmt"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_on_subword_nmt)

    with pytest.raises(ImportError, match="subword-nmt is required"):
        BpePharmaformerDrugFeaturizer().fit(drug_source, entity_ids=synthetic_dataset.drug_ids[:2])
