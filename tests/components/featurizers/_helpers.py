"""Shared feature-source builders for featurizer tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

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
