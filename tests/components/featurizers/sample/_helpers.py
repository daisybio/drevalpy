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
from drevalpy.types.data.feature_source import FeatureSource, store_variant
from tests.components.featurizers._helpers import Views, mock_source
from tests.models.synthetic_fixtures import synthetic_model_dataset

PRECOMPUTED = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)

F = TypeVar("F", bound=Featurizer)


def pickled(featurizer: F) -> F:
    """Return a copy of *featurizer* that went through a pickle round trip, as a saved model does."""
    return pickle.loads(pickle.dumps(featurizer))  # noqa: S301 -- pickled on this line


def precomputed_source(
    featurizer_cls: type[Featurizer],
    *,
    matrix: np.ndarray | None = None,
    hyperparameters: dict[str, object] | None = None,
) -> FeatureSource:
    """Dataset-backed source storing a variant of *featurizer_cls* for its first two samples."""
    values = PRECOMPUTED if matrix is None else matrix
    dataset = synthetic_model_dataset(extra_views=("mutations", "copy_number_variation_gistic"))
    n_samples = dataset.mdata.mod["response"].n_obs
    padded = np.zeros((n_samples, values.shape[1]), dtype=values.dtype)
    padded[: len(values)] = values
    assert featurizer_cls.storage_key is not None
    store_variant(dataset, "sample", featurizer_cls.storage_key, padded, hyperparameters)
    ids = np.asarray(dataset.mdata.mod["response"].obs_names[: len(values)], dtype=str)
    return FeatureSource(dataset, "sample", ids)


def assert_uses_precomputed_variant(
    featurizer: Featurizer,
    *,
    hyperparameters: dict[str, object] | None = None,
    expected_blocks: tuple[str, ...] | None = None,
) -> None:
    """Assert *featurizer* looks up the stored variant and serves it through ``from_stored``."""
    source = precomputed_source(type(featurizer), hyperparameters=hyperparameters)
    ids = source.identifiers
    params = featurizer.stored_params()
    assert params is not None
    assert featurizer.storage_key is not None
    stored = source.stored(featurizer.storage_key, params, ids)
    assert stored is not None

    result = featurizer.from_stored(source, ids, stored)

    np.testing.assert_allclose(result.matrix, PRECOMPUTED)
    if expected_blocks is not None:
        assert set(result.blocks) == set(expected_blocks)
        for name in expected_blocks:
            np.testing.assert_allclose(result.blocks[name].values, PRECOMPUTED)


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
