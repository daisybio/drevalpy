"""Tests for the BPE PharmaFormer drug featurizer; BPE codes are learned from the fixture SMILES."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.bpe_pharmaformer import BpePharmaformerDrugFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource


@pytest.fixture
def drug_source(synthetic_dataset: Dataset) -> DrugFeatureSource:
    return DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)


_IDS_SLICE = slice(0, 2)


def _fitted(drug_source: DrugFeatureSource, **kwargs) -> tuple[BpePharmaformerDrugFeaturizer, np.ndarray]:
    drug_ids = drug_source.identifiers[_IDS_SLICE]
    featurizer = BpePharmaformerDrugFeaturizer(num_symbols=50, **kwargs)
    return featurizer.fit(drug_source, entity_ids=drug_ids), drug_ids


def test_defaults_and_hyperparameter_space() -> None:
    assert BpePharmaformerDrugFeaturizer(max_length=64).output_dim == 64
    assert set(BpePharmaformerDrugFeaturizer.get_hyperparameter_space()) == {"num_symbols", "max_length"}


def test_transform_before_fit_and_fit_without_smiles_raise() -> None:
    source = MockFeatureSource(features={"d1": {}})
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        BpePharmaformerDrugFeaturizer()._transform_features(source, np.array(["d1"]))
    with pytest.raises(ValueError, match="Cannot learn BPE codes"):
        BpePharmaformerDrugFeaturizer().fit(source, entity_ids=np.array(["d1"]))


def test_transform_without_smiles_raises(drug_source: DrugFeatureSource) -> None:
    featurizer, _ = _fitted(drug_source, max_length=16)
    with pytest.raises(ValueError, match="Cannot encode BPE"):
        featurizer._transform_features(MockFeatureSource(features={"d1": {}}), np.array(["d1"]))


def test_fit_then_transform_emits_a_padded_token_matrix_with_one_block(drug_source: DrugFeatureSource) -> None:
    featurizer, drug_ids = _fitted(drug_source, max_length=32)

    matrix = featurizer.transform(drug_source, drug_ids)

    assert matrix.shape == (2, 32)
    assert matrix.dtype == np.float32
    blocks = featurizer.transform_blocks(drug_source, drug_ids)
    assert set(blocks) == {"bpe_smiles"}
    assert blocks["bpe_smiles"].feature_names is None


def test_transform_truncates_long_sequences_and_pads_short_ones(drug_source: DrugFeatureSource) -> None:
    short, drug_ids = _fitted(drug_source, max_length=4)
    assert np.all(short.transform(drug_source, drug_ids) > 0)
    padded, _ = _fitted(drug_source, max_length=512)
    assert np.any(padded._transform_features(drug_source, drug_ids).matrix[0] == 0.0)


def test_transform_leaves_unknown_drugs_as_zero_rows(drug_source: DrugFeatureSource) -> None:
    featurizer, _ = _fitted(drug_source, max_length=16)

    # ``_transform_features``: the public wrapper's NaN detection would overwrite the unknown drug's row with NaN.
    matrix = featurizer._transform_features(drug_source, np.array(["not-a-drug"])).matrix

    np.testing.assert_allclose(matrix, np.zeros((1, 16), dtype=np.float32))


def test_learn_bpe_reports_missing_subword_nmt(monkeypatch: pytest.MonkeyPatch, drug_source: DrugFeatureSource) -> None:
    block_imports(monkeypatch, "subword_nmt")

    with pytest.raises(ImportError, match="subword-nmt is required"):
        BpePharmaformerDrugFeaturizer().fit(drug_source, entity_ids=drug_source.identifiers[_IDS_SLICE])


def test_learning_bpe_silences_progress_and_restores_the_module_binding(
    drug_source: DrugFeatureSource, capsys: pytest.CaptureFixture[str]
) -> None:
    from subword_nmt import learn_bpe as learn_bpe_module

    original = learn_bpe_module.tqdm

    _fitted(drug_source, max_length=16)

    assert learn_bpe_module.tqdm is original
    assert "it/s" not in capsys.readouterr().err
