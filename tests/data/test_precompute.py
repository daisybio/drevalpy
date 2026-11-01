"""Pre-computing featurizer outputs into a dataset as views and reading them back."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pytest
from upath import UPath

from drevalpy.components.featurizers._dense import PrecomputableFeaturizer
from drevalpy.components.featurizers.drug.fingerprints import FingerprintsFeaturizer
from drevalpy.data.precompute import precompute, precompute_all
from drevalpy.models._feature_preparation import FitIds, prepare_features
from drevalpy.registry.featurizer import register
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, precomputed_views
from tests.synthetic import N_DRUGS, build_synthetic_dataset


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def dataset(h5mu_path: UPath) -> Dataset:
    return Dataset.load(h5mu_path)


class _Scaled(PrecomputableFeaturizer):
    """Precomputable stub: ``scale`` times the first column of ``gene_expression``, as the ``scaled`` view."""

    input_views: ClassVar[tuple[str, ...]] = ("scaled",)
    source_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    computed: ClassVar[list[float]] = []

    def __init__(self, scale: float = 1.0) -> None:
        self._scale = scale
        super().__init__(self.input_views[0], {"scale": scale})

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {"scale": {"type": "float", "low": 0.5, "high": 4.0, "default": 1.0}}

    def compute(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        type(self).computed.append(self._scale)
        return (source.view(self.source_views[0], ids)[:, :1] * self._scale).astype(np.float32)


def _register(name: str, side: str = "sample", **attrs: Any) -> type[_Scaled]:
    cls = type(f"_Scaled_{name}", (_Scaled,), {"computed": [], **attrs})
    return register(name, side=side, description="precompute stub")(cls)


def test_a_sample_view_is_written_with_its_settings_and_read_only_at_those_settings(dataset: Dataset) -> None:
    cls = _register("scaledStub")

    assert precompute(dataset, "scaledStub", "sample") == "scaled"

    assert cls.computed == [1.0]
    assert precomputed_views(dataset.mdata) == {"sample": {"scaled": {"scale": 1.0}}}
    ids = dataset.sample_ids
    source = FeatureSource(dataset, "sample")
    cls.computed.clear()
    read = prepare_features(cls(), source, ids, fit=FitIds(ids))
    other = prepare_features(cls(scale=2.0), source, ids, fit=FitIds(ids))
    assert cls.computed == [2.0]
    expected = FeatureSource(dataset, "sample").view("gene_expression", ids)[:, :1]
    np.testing.assert_allclose(read.matrix, expected, rtol=1e-6)
    np.testing.assert_allclose(other.matrix, expected * 2.0, rtol=1e-6)


def test_the_record_survives_saving(dataset: Dataset, tmp_path: UPath) -> None:
    _register("scaledStub")
    precompute(dataset, "scaledStub", "sample")
    path = UPath(tmp_path) / "with_view.h5mu"

    dataset.save(path)
    reloaded = Dataset.load(path)

    assert FeatureSource(reloaded, "sample").view_settings("scaled") == {"scale": 1.0}
    assert _Scaled().reads_view(FeatureSource(reloaded, "sample"))


def test_a_drug_view_lands_in_varm(dataset: Dataset) -> None:
    def compute(self: _Scaled, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        return source.view("chemberta", ids)[:, :1]

    cls = _register(
        "drugScaled",
        side="drug",
        input_views=("drug_scaled",),
        source_views=("chemberta",),
        compute=compute,
    )

    assert precompute(dataset, "drugScaled", "drug") == "drug_scaled"

    assert dataset.mdata.mod["response"].varm["drug_scaled"].shape == (N_DRUGS, 1)
    assert cls().reads_view(FeatureSource(dataset, "drug"))


def test_a_view_the_featurizer_already_reads_is_not_computed_again(dataset: Dataset) -> None:
    cls = _register("scaledStub")
    precompute(dataset, "scaledStub", "sample")
    cls.computed.clear()

    assert precompute(dataset, "scaledStub", "sample") is None
    assert cls.computed == []


def test_an_unsupported_side_or_a_featurizer_that_needs_fitting_raises(dataset: Dataset) -> None:
    _register("scaledStub")

    with pytest.raises(ValueError, match="'scaledStub' is a sample featurizer and cannot be used on the drug side"):
        precompute(dataset, "scaledStub", "drug")
    with pytest.raises(ValueError, match="cannot be precomputed"):
        precompute(dataset, "pca", "sample")


def test_precompute_all_covers_eligible_featurizers_and_skips_the_rest(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from drevalpy.registry.featurizer import featurizer_registry

    featurizers = featurizer_registry("sample")
    eligible = _register("eligibleStub", input_views=("eligible",))
    absent = _register("absentStub", input_views=("absent",), source_views=("not_a_view",))

    def broken(self: _Scaled, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        raise ValueError("boom")

    _register("brokenStub", input_views=("broken",), compute=broken)
    original = dict(featurizers.items())
    monkeypatch.setattr(featurizers, "items", lambda: tuple((n, c) for n, c in original.items() if "Stub" in n))

    with caplog.at_level(logging.WARNING, logger="drevalpy.data.precompute"):
        written = precompute_all(dataset)

    assert "eligible" in written
    assert eligible.computed == [1.0]
    assert absent.computed == []
    assert any("Failed to precompute brokenStub: boom" in record.getMessage() for record in caplog.records)


def test_fingerprints_are_written_at_their_defaults_and_other_bit_counts_compute(dataset: Dataset) -> None:
    pytest.importorskip("rdkit")
    del dataset.mdata.mod["response"].varm["morgan_fingerprint"]

    assert precompute(dataset, "fingerprints", "drug") == "morgan_fingerprint"

    source = FeatureSource(dataset, "drug")
    assert dataset.mdata.mod["response"].varm["morgan_fingerprint"].shape == (N_DRUGS, 2048)
    assert source.view_settings("morgan_fingerprint") == FingerprintsFeaturizer().settings
    assert FingerprintsFeaturizer().reads_view(source)
    assert not FingerprintsFeaturizer(n_bits=512).reads_view(source)


def test_an_unrecorded_shipped_view_is_read_whatever_the_settings(dataset: Dataset) -> None:
    source = FeatureSource(dataset, "drug")

    assert source.view_settings("morgan_fingerprint") is None
    assert FingerprintsFeaturizer(n_bits=512).reads_view(source)
    assert precompute(dataset, "fingerprints", "drug") is None
