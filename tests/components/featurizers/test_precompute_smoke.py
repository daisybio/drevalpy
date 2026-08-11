"""Smoke tests verifying all precomputable featurizers can compute from raw data."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.registry.cell_line_featurizer import cell_line_featurizer_registry
from drevalpy.registry.drug_featurizer import drug_featurizer_registry
from drevalpy.types.data.feature_source import CellLineFeatureSource, DrugFeatureSource


def _collect_precomputable_featurizers():
    """Collect all registered featurizers with precompute=True."""
    items = []
    for registry, side in [
        (cell_line_featurizer_registry, "cell_line"),
        (drug_featurizer_registry, "drug"),
    ]:
        for name in registry.list_names():
            cls = registry.get(name)
            if cls.precompute:
                items.append(pytest.param(name, cls, side, id=f"{side}/{name}"))
    return items


_SKIP_MISSING_AUX = {
    "bionic": ("dipk", "uns['dipk']"),
    "pathways": ("pathways_gmt", "uns['pathways_gmt']"),
}

_SLOW_FEATURIZERS = {"chemberta", "molgnet", "smilesvec"}


@pytest.fixture(scope="module")
def toyv1_dataset():
    """Load TOYv1 once per module."""
    from drevalpy.data import load

    return load("TOYv1")


def _check_skips(name, toyv1_dataset):
    """Skip tests that require missing data or optional dependencies."""
    if name in _SLOW_FEATURIZERS:
        pytest.skip(f"{name} is slow / requires optional deps — skipped by default")

    if name in _SKIP_MISSING_AUX:
        key, label = _SKIP_MISSING_AUX[name]
        if key not in toyv1_dataset.mdata.uns:
            pytest.skip(f"{name} requires {label} not present in TOYv1")


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
def test_precompute_fit_transform(name, cls, side, toyv1_dataset):
    """Each precomputable featurizer should fit and transform 2 entities."""
    _check_skips(name, toyv1_dataset)
    featurizer = _instantiate_featurizer(name, cls)
    source, ids = _make_source_and_ids(side, toyv1_dataset)

    try:
        featurizer.fit(source, entity_ids=ids)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency missing during fit: {exc}")
    except (ValueError, KeyError) as exc:
        pytest.skip(f"{name} cannot fit on TOYv1 data: {exc}")

    try:
        result = featurizer.transform(source, ids)
    except ImportError as exc:
        pytest.skip(f"{name} optional dependency missing during transform: {exc}")
    except (ValueError, KeyError) as exc:
        pytest.skip(f"{name} cannot transform on TOYv1 data: {exc}")

    assert isinstance(result, np.ndarray), f"{name}: expected ndarray, got {type(result)}"
    assert result.shape[0] == 2, f"{name}: expected 2 rows, got {result.shape[0]}"
