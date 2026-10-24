"""Smoke tests verifying all precomputable featurizers can compute from raw data."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.registry.sample_featurizer import sample_featurizer_registry
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource, SampleFeatureSource

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
    """Use ``_compute_from_source`` (the ``Dataset.precompute`` path); fit/transform would read the stored view."""
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
