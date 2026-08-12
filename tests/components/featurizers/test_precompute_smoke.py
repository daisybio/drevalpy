"""Smoke tests verifying all precomputable featurizers can compute from raw data."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import CellLineFeatureSource, DrugFeatureSource

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
    """Collect all registered featurizers with precompute=True."""
    items = []
    for registry, side in [
        (cell_line_featurizer_registry, "cell_line"),
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
    """Create a featurizer instance with default hyperparameters."""
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
    """Build the appropriate FeatureSource and pick 2 entity IDs."""
    if side == "cell_line":
        return CellLineFeatureSource(ds, ds.cell_line_ids), ds.cell_line_ids[:2]
    return DrugFeatureSource(ds, ds.drug_ids), ds.drug_ids[:2]


@pytest.mark.parametrize("name, cls, side", _collect_precomputable_featurizers())
def test_precompute_fit_transform(name, cls, side, synthetic_dataset: Dataset) -> None:
    """Each precomputable featurizer derives 2 entities' features from raw data.

    Goes through ``_compute_from_source``, which is the path
    ``Dataset.precompute`` takes for independent featurizers. Calling
    ``fit``/``transform`` instead would silently read the already-stored view and
    prove nothing about computation.

    :param name: Registry name of the featurizer.
    :param cls: Featurizer class.
    :param side: ``cell_line`` or ``drug``.
    :param synthetic_dataset: Session-scoped synthetic raw-omics dataset.
    """
    if name in _UNSUPPORTED:
        pytest.skip(f"{name} requires {_UNSUPPORTED[name]}")

    featurizer = _instantiate_featurizer(name, cls)
    source, ids = _make_source_and_ids(side, synthetic_dataset)

    try:
        result = featurizer._compute_from_source(source, ids)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency missing: {exc}")

    assert isinstance(result, np.ndarray), f"{name}: expected ndarray, got {type(result)}"
    assert result.shape[0] == 2, f"{name}: expected 2 rows, got {result.shape[0]}"
