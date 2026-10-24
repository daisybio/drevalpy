"""Shared check that a fitted featurizer reads its view by feature name.

Plain module (no ``__init__.py``) imported by dotted path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from drevalpy.components.featurizers import FeatureMismatchError
from tests.conftest import MockFeatureSource

Views = Mapping[str, tuple[np.ndarray, Sequence[str]]]


def random_views(names_by_view: Mapping[str, Sequence[str]], n_samples: int = 6, seed: int = 0) -> Views:
    """Return positive random matrices, one per view, with the given feature names."""
    rng = np.random.default_rng(seed)
    return {view: (rng.uniform(1.0, 10.0, (n_samples, len(names))), names) for view, names in names_by_view.items()}


def mock_source(views: Views) -> MockFeatureSource:
    """Build a feature source holding *views* for samples ``s0..s{n-1}``."""
    n_samples = next(iter(views.values()))[0].shape[0]
    return MockFeatureSource(
        {f"s{i}": {view: matrix[i] for view, (matrix, _) in views.items()} for i in range(n_samples)},
        meta_info={view: list(names) for view, (_, names) in views.items()},
    )


def _reversed_with_extra(views: Views) -> Views:
    return {
        view: (np.hstack([matrix[:, ::-1], np.full((matrix.shape[0], 1), 99.0)]), [*list(names)[::-1], "EXTRA"])
        for view, (matrix, names) in views.items()
    }


def _without_first_feature(views: Views, view: str) -> Views:
    matrix, names = views[view]
    return {**views, view: (matrix[:, 1:], list(names)[1:])}


def assert_aligns_by_name(
    make_featurizer: Callable[[], Any],
    views: Views,
    *,
    drop_view: str = "gene_expression",
    fit_kwargs: Callable[[np.ndarray], dict[str, Any]] = lambda ids: {"entity_ids": ids},
) -> None:
    """Assert predictions ignore column order and extra columns, and a missing feature raises.

    Covers the fitted featurizer and a copy restored through ``set_state``.

    :param make_featurizer: Factory for an unfitted featurizer.
    :param views: Views to fit on, with every column named.
    :param drop_view: View from which the first feature is removed for the missing-feature check.
    :param fit_kwargs: Maps the sample IDs to the keyword arguments of ``fit``.
    """
    source = mock_source(views)
    ids = source.identifiers
    fitted = make_featurizer().fit(source, **fit_kwargs(ids))
    restored = make_featurizer()
    restored.set_state(fitted.get_state())

    def read(featurizer: Any, other: MockFeatureSource) -> dict[str, np.ndarray]:
        return {name: block.values for name, block in featurizer.transform_blocks(other, ids).items()}

    expected = read(fitted, source)
    shuffled = mock_source(_reversed_with_extra(views))
    missing = mock_source(_without_first_feature(views, drop_view))
    for featurizer in (fitted, restored):
        actual = read(featurizer, shuffled)
        assert actual.keys() == expected.keys()
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values, rtol=1e-5, atol=1e-6)
        with pytest.raises(FeatureMismatchError, match=rf"view '{drop_view}' is missing 1 of"):
            featurizer.transform_blocks(missing, ids)
