"""Shared helpers for sample featurizer tests."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.components.featurizers.sample import gene_lists
from drevalpy.types.data.feature_source import FeatureSource
from tests.components.featurizers._helpers import Views, mock_source

F = TypeVar("F", bound=Featurizer)


def pickled(featurizer: F) -> F:
    """Return a copy of *featurizer* that went through a pickle round trip, as a saved model does."""
    return pickle.loads(pickle.dumps(featurizer))  # noqa: S301 -- pickled on this line


def _reversed_with_extra(views: Views) -> Views:
    return {
        view: (np.hstack([matrix[:, ::-1], np.full((matrix.shape[0], 1), 99.0)]), [*list(names)[::-1], "EXTRA"])
        for view, (matrix, names) in views.items()
    }


def _without_first_feature(views: Views, view: str) -> Views:
    matrix, names = views[view]
    return {**views, view: (matrix[:, 1:], list(names)[1:])}


def assert_reads_by_name(make_featurizer: Callable[[], Featurizer], views: Views, *, drop_view: str) -> None:
    """Assert transforms ignore column order and extra columns, and a missing fitted feature raises.

    Covers the fitted featurizer and a pickled copy of it.

    :param make_featurizer: Factory for an unfitted featurizer.
    :param views: Views to fit on, with every column named.
    :param drop_view: View from which the first feature is removed for the missing-feature check.
    """
    source = mock_source(views)
    ids = source.identifiers
    fitted = make_featurizer().fit(source, ids, train_pair_ids=ids, es_pair_ids=ids)

    def read(featurizer: Featurizer, other: FeatureSource) -> dict[str, Any]:
        return {name: block.values for name, block in featurizer.transform(other, ids).blocks.items()}

    expected = read(fitted, source)
    shuffled = mock_source(_reversed_with_extra(views))
    missing = mock_source(_without_first_feature(views, drop_view))
    for featurizer in (fitted, pickled(fitted)):
        actual = read(featurizer, shuffled)
        assert actual.keys() == expected.keys()
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values, rtol=1e-5, atol=1e-6)
        with pytest.raises(FeatureMismatchError, match=rf"view '{drop_view}' is missing 1 of"):
            featurizer.transform(missing, ids)


def use_gene_lists(monkeypatch: pytest.MonkeyPatch, directory: Path, lists: Mapping[str, Sequence[str]]) -> None:
    """Replace the shipped gene lists by small stand-in files written to *directory*."""
    for name, symbols in lists.items():
        pd.DataFrame({"Symbol": list(symbols)}).to_parquet(directory / f"{name}.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(directory))
