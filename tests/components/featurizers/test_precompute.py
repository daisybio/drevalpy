"""Pre-computing featurizer representations into a dataset."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

from drevalpy.components.featurizers import precompute as precompute_module
from drevalpy.components.featurizers.precompute import _has_source_data, _precompute_single, precompute, precompute_all
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource, SampleFeatureSource
from tests.components.featurizers._precompute_stubs import stub_featurizer, stub_fit_transform_featurizer
from tests.synthetic import N_DRUGS, N_SAMPLES, build_synthetic_dataset


@pytest.fixture(scope="module")
def h5mu_path(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    path = UPath(tmp_path_factory.mktemp("dataset")) / "synthetic.h5mu"
    build_synthetic_dataset().save(path)
    return path


@pytest.fixture()
def dataset(h5mu_path: UPath) -> Dataset:
    return Dataset.load(h5mu_path)


@pytest.fixture()
def records() -> list[dict]:
    return []


class TestPrecompute:
    def test_default_config_is_prepended_to_explicit_configs(self, dataset: Dataset, records: list[dict]):
        precompute(dataset, stub_featurizer(records), [{"n_components": 2}])

        assert [call["hyperparameters"] for call in records] == [{}, {"n_components": 2}]

    @pytest.mark.parametrize(("side", "n"), [("sample", N_SAMPLES), ("drug", N_DRUGS)])
    def test_every_entity_of_the_side_is_precomputed(self, dataset: Dataset, records: list[dict], side, n):
        precompute(dataset, stub_featurizer(records, side=side), [])

        assert records[0]["n_entities"] == n

    def test_featurizer_without_compute_from_source_is_fitted_first(self, dataset: Dataset, records: list[dict]):
        precompute(dataset, stub_fit_transform_featurizer(records), [])

        assert records[0] == {"call": "fit"}
        assert records[1]["shape"] == (N_SAMPLES, 3)

    def test_view_is_forwarded_to_the_featurizer_constructor(self, dataset: Dataset, records: list[dict]):
        seen: list[dict] = []
        featurizer_cls = stub_featurizer(records)
        original_init = featurizer_cls.__init__

        def spy_init(self, **kwargs):
            seen.append(kwargs)
            original_init(self, **kwargs)

        featurizer_cls.__init__ = spy_init
        precompute(dataset, featurizer_cls, [], view="methylation")

        assert seen == [{"view": "methylation"}]

    def test_int_hyperparameters_are_drawn(self, dataset: Dataset, records: list[dict]):
        precompute(dataset, stub_featurizer(records), 2)

        assert len(records) == 2


class TestPrecomputeSingle:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"precompute": False},
            {"entity_id_only": True},
            {"source_views": ("absent_modality",)},
            {"requires_view": True, "input_views": None},
        ],
        ids=["not-marked", "entity-id-only", "missing-source", "view-without-input-views"],
    )
    def test_ineligible_featurizers_are_skipped(self, dataset: Dataset, records: list[dict], kwargs):
        _precompute_single(dataset, stub_featurizer(records, **kwargs), 1)

        assert records == []

    def test_eligible_featurizer_is_precomputed(self, dataset: Dataset, records: list[dict]):
        _precompute_single(dataset, stub_featurizer(records, source_views=("gene_expression",)), 1)

        assert len(records) == 1

    def test_featurizer_errors_are_swallowed(self, dataset: Dataset, records: list[dict]):
        def raise_value_error(source, entity_ids):
            raise ValueError("no features")

        _precompute_single(dataset, stub_featurizer(records, compute=raise_value_error), 1)

        assert records == []


def test_precompute_all_visits_every_registered_featurizer(dataset: Dataset, monkeypatch):
    visited: list[str] = []
    monkeypatch.setattr(
        precompute_module, "_precompute_single", lambda dataset, cls, n_variants: visited.append(cls.__name__)
    )

    precompute_all(dataset, n_variants=1)

    assert len(visited) == len(sample_featurizer_registry.list_names()) + len(drug_featurizer_registry.list_names())


@pytest.mark.parametrize(
    ("side", "source", "expected"),
    [
        ("drug", "canonical_smiles", True),
        ("sample", "copy_number_variation_gistic", True),
        ("sample", "absent_modality", False),
        ("drug", "morgan_fingerprint", True),
        ("drug", "absent_view", False),
    ],
)
def test_has_source_data(dataset: Dataset, side, source, expected):
    assert _has_source_data(dataset, (source,), side) is expected


#: Featurizers that need a pretrained-weight or remote-annotation download, so
#: they cannot be derived from the raw data the fixture carries. Marked ``network``
#: rather than skipped unconditionally, so ``-m "not network"`` keeps CI hermetic
#: while a developer with connectivity can still run them.
_NETWORK_FEATURIZERS = frozenset({"chemberta", "molgnet", "smilesvec"})

#: Featurizers the fixture cannot serve at all, mapped to why.
#:
#: ``bionic`` wants ``uns['dipk']``, which the real data builder writes but no
#: library code reads, and ``bionic.py`` reaches instead for an 83 MB S3 artifact
#: behind a named AWS profile CI can never assume. ``molgnet`` wants a ragged
#: ``response.varm['molgnet_features']`` that ``h5py`` cannot store, and
#: ``FeatureSource.get_entity_view`` raises ``KeyError`` when the varm key is
#: absent instead of returning ``None``, so its on-the-fly fallback is dead code.
_UNSUPPORTED = {
    "bionic": "uns['dipk'], which the synthetic fixture omits",
    "molgnet": "a ragged response.varm['molgnet_features'], which cannot be stored in a .h5mu",
}


def _collect_precomputable_featurizers():
    items = []
    for registry, side in [
        (sample_featurizer_registry, "sample"),
        (drug_featurizer_registry, "drug"),
    ]:
        for name in registry.list_names():
            cls = registry.get(name)
            if not cls.precompute:
                continue
            marks = [pytest.mark.network] if name in _NETWORK_FEATURIZERS else []
            items.append(pytest.param(name, cls, side, marks=marks, id=f"{side}/{name}"))
    return items


def _instantiate_featurizer(name, cls):
    try:
        default_hps = cls.get_default_hyperparameters()
    except Exception:
        default_hps = {}

    if cls.requires_view and "view" not in default_hps:
        views = cls.input_views
        if views:
            default_hps["view"] = views[0]
        else:
            pytest.skip(f"{name} requires an explicit view but declares none")

    try:
        return cls(**default_hps)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency missing: {exc}")
    except TypeError:
        return cls()


def _make_source_and_ids(side, ds):
    if side == "sample":
        return SampleFeatureSource(ds, ds.sample_ids), ds.sample_ids[:2]
    return DrugFeatureSource(ds, ds.drug_ids), ds.drug_ids[:2]


@pytest.mark.parametrize("name, cls, side", _collect_precomputable_featurizers())
def test_precompute_fit_transform(name, cls, side, synthetic_dataset: Dataset) -> None:
    """Use ``_compute_from_source`` (the ``precompute`` path); fit/transform would read the stored view."""
    if name in _UNSUPPORTED:
        pytest.skip(f"{name} requires {_UNSUPPORTED[name]}")

    featurizer = _instantiate_featurizer(name, cls)
    source, ids = _make_source_and_ids(side, synthetic_dataset)

    try:
        result = featurizer._compute_from_source(source, ids)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency missing: {exc}")

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 2
