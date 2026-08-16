"""Tests for the Morgan fingerprint drug featurizer.

Mirrors :mod:`drevalpy.components.featurizers.drug.fingerprints`. The happy path
through ``_compute_from_source`` is already smoke-tested in
``test_precompute_smoke.py``; this file pins the residual error and NaN paths of
``_fingerprint_for_smiles``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource
from tests._import_shims import block_imports
from tests.conftest import MockFeatureSource


def _generator(n_bits: int = 16):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)


def test_hyperparameter_space_declares_the_four_tunables() -> None:
    space = FingerprintsFeaturizer.get_hyperparameter_space()

    assert set(space) == {"radius", "n_bits", "use_chirality", "use_counts"}


def test_compute_from_source_without_smiles_raises() -> None:
    featurizer = FingerprintsFeaturizer(n_bits=16)
    source = MockFeatureSource(features={"d1": {}})

    with pytest.raises(ValueError, match="no SMILES available"):
        featurizer._compute_from_source(source, np.array(["d1"]))


def test_compute_from_source_reports_a_missing_rdkit(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_dataset: Dataset,
) -> None:
    block_imports(monkeypatch, "rdkit")
    featurizer = FingerprintsFeaturizer(n_bits=16)
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)

    with pytest.raises(ImportError, match="rdkit is required"):
        featurizer._compute_from_source(source, synthetic_dataset.drug_ids[:1])


@pytest.mark.parametrize(
    "smiles",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty-string"),
        pytest.param(float("nan"), id="not-a-string"),
        pytest.param("this is not a molecule", id="unparseable"),
    ],
)
def test_unusable_smiles_produce_a_nan_row(smiles: object) -> None:
    from drevalpy.components.featurizers.drug.fingerprints import _fingerprint_for_smiles

    row = _fingerprint_for_smiles(smiles, _generator(), 16, False)

    assert row.shape == (16,)
    assert np.all(np.isnan(row))


def test_binary_fingerprints_are_zero_or_one() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import _fingerprint_for_smiles

    row = _fingerprint_for_smiles("CCO", _generator(), 16, False)

    assert set(np.unique(row)) <= {0.0, 1.0}


def test_count_fingerprints_can_exceed_one() -> None:
    from drevalpy.components.featurizers.drug.fingerprints import _fingerprint_for_smiles

    counts = _fingerprint_for_smiles("CCCCCCCCCC", _generator(8), 8, True)
    binary = _fingerprint_for_smiles("CCCCCCCCCC", _generator(8), 8, False)

    assert counts.sum() > binary.sum()


def test_compute_from_source_emits_one_row_per_drug(synthetic_dataset: Dataset) -> None:
    featurizer = FingerprintsFeaturizer(n_bits=32, use_counts=True)
    source = DrugFeatureSource(synthetic_dataset, synthetic_dataset.drug_ids)
    drug_ids = synthetic_dataset.drug_ids[:2]

    matrix = featurizer._compute_from_source(source, drug_ids)

    assert matrix.shape == (2, 32)


def test_transform_blocks_are_named_fingerprints() -> None:
    source = MockFeatureSource(
        features={"d1": {"morgan_fingerprint": np.array([1.0, 0.0])}},
        meta_info={"morgan_fingerprint": ["fp1", "fp2"]},
    )
    ids = np.array(["d1"], dtype=str)
    featurizer = FingerprintsFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    assert set(blocks) == {"fingerprints"}
    assert blocks["fingerprints"].feature_names == ("fp1", "fp2")
