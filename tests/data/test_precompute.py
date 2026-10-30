"""Pre-computing featurizer outputs into a dataset and reading them back as stored variants."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from upath import UPath

from drevalpy.components.featurizers._dense import dense_result
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.data.precompute import precompute, precompute_all
from drevalpy.models._feature_preparation import FitIds, prepare_features
from drevalpy.registry.featurizer import get_featurizer, register
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource
from tests.synthetic import N_DRUGS, N_SAMPLES, build_synthetic_dataset


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def dataset(h5mu_path: UPath) -> Dataset:
    return Dataset.load(h5mu_path)


class _Scaled(Featurizer):
    """Stored-capable stub: ``scale`` times the first column of its view, one column per entity."""

    sides: ClassVar[frozenset[str]] = frozenset({"sample"})
    input_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, scale: float = 1.0, view: str | None = None) -> None:
        self._scale = scale
        self._view = view or self.input_views[0]

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        return {"scale": {"type": "float", "low": 0.5, "high": 4.0, "default": 1.0}}

    def stored_params(self) -> Mapping[str, Any] | None:
        return {"scale": self._scale}

    def from_stored(self, source: FeatureSource, ids: np.ndarray, matrix: np.ndarray) -> FeaturizerResult:
        return dense_result(ids, self._view, matrix, ("first",))

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        return np.ones(len(ids), dtype=bool)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        type(self).calls.append({"view": self._view, "scale": self._scale, "n": len(train_ids)})
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        matrix = source.get_view_matrix(self._view, ids)[:, :1] * self._scale
        return dense_result(ids, self._view, matrix.astype(np.float32), ("first",))


def _register(name: str, **attrs: Any) -> type[_Scaled]:
    cls = type(f"_Scaled_{name}", (_Scaled,), {"calls": [], **attrs})
    return register(name, description="precompute stub")(cls)


def test_the_default_and_the_given_configs_are_stored_and_read_back(dataset: Dataset) -> None:
    cls = _register("scaledStub")

    keys = precompute(dataset, "scaledStub", "sample", [{"scale": 2.0}])

    assert keys == ["scaledStub_0", "scaledStub_1"]
    assert [call["scale"] for call in cls.calls] == [1.0, 2.0]
    assert all(call["n"] == N_SAMPLES for call in cls.calls)
    ids = dataset.sample_ids[[3, 0]]
    stored = FeatureSource(dataset, "sample", ids).stored("scaledStub", {"scale": 2.0}, ids)
    expected = dataset.get_sample_features("gene_expression", ids)[:, :1] * 2.0
    np.testing.assert_allclose(stored, expected, rtol=1e-6)


def test_prepare_features_uses_the_stored_variant_instead_of_fitting(dataset: Dataset) -> None:
    cls = _register("scaledStub")
    ids = dataset.sample_ids
    computed = prepare_features(cls(scale=3.0), FeatureSource(dataset, "sample", ids), ids, fit=FitIds(ids))
    precompute(dataset, "scaledStub", "sample", [{"scale": 3.0}])
    cls.calls.clear()

    stored = prepare_features(cls(scale=3.0), FeatureSource(dataset, "sample", ids), ids, fit=FitIds(ids))

    assert cls.calls == []
    np.testing.assert_array_equal(stored.matrix, computed.matrix)


def test_a_drug_side_variant_lands_in_varm(dataset: Dataset) -> None:
    _register("drugScaled", sides=frozenset({"drug"}), input_views=("morgan_fingerprint",))

    (key,) = precompute(dataset, "drugScaled", "drug", [])

    assert dataset.mdata.mod["response"].varm[key].shape == (N_DRUGS, 1)


def test_a_variant_already_stored_is_not_computed_again(dataset: Dataset) -> None:
    cls = _register("scaledStub")
    precompute(dataset, "scaledStub", "sample", [])
    cls.calls.clear()

    assert precompute(dataset, "scaledStub", "sample", [{"scale": 1.0}]) == []
    assert cls.calls == []


def test_the_view_is_forwarded_and_int_hyperparameters_are_drawn(dataset: Dataset) -> None:
    cls = _register("scaledStub")

    keys = precompute(dataset, "scaledStub", "sample", 2, view="methylation")

    assert len(keys) == 3
    assert {call["view"] for call in cls.calls} == {"methylation"}
    assert cls.calls[0]["scale"] == 1.0


def test_an_unsupported_side_or_a_featurizer_without_stored_outputs_raises(dataset: Dataset) -> None:
    _register("scaledStub")
    _register("unstored", stored_params=lambda self: None)

    with pytest.raises(ValueError, match="cannot be used on the drug side"):
        precompute(dataset, "scaledStub", "drug", [])
    with pytest.raises(ValueError, match="never looks up stored outputs"):
        precompute(dataset, "unstored", "sample", [])


def test_precompute_all_covers_eligible_featurizers_and_skips_the_rest(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from drevalpy.registry.featurizer import featurizers

    eligible = _register("eligibleStub")
    unstored = _register("unstoredStub", stored_params=lambda self: None)
    absent = _register("absentStub", input_views=("not_a_view",))

    def broken_transform(self: _Scaled, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        raise ValueError("boom")

    _register("brokenStub", transform=broken_transform)
    original = {name: cls for name, cls in featurizers.items()}
    monkeypatch.setattr(featurizers, "items", lambda: tuple((n, c) for n, c in original.items() if "Stub" in n))

    with caplog.at_level(logging.WARNING, logger="drevalpy.data.precompute"):
        precompute_all(dataset, n_variants=1)

    assert len(eligible.calls) == 2
    assert unstored.calls == [] and absent.calls == []
    assert any("Failed to precompute brokenStub: boom" in record.getMessage() for record in caplog.records)


def test_a_builtin_featurizer_round_trips_through_storage(dataset: Dataset) -> None:
    ids = dataset.sample_ids
    pca = get_featurizer("pca", "sample")
    computed = prepare_features(
        pca(view="gene_expression", n_components=2), FeatureSource(dataset, "sample", ids), ids, fit=FitIds(ids)
    )

    precompute(dataset, "pca", "sample", [{"n_components": 2}], view="gene_expression")
    stored = prepare_features(
        pca(view="gene_expression", n_components=2), FeatureSource(dataset, "sample", ids), ids, fit=FitIds(ids)
    )

    np.testing.assert_allclose(stored.matrix, computed.matrix, rtol=1e-5, atol=1e-6)
