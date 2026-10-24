"""Shared stubs and name-alignment checks for featurizer tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from drevalpy.components.contracts.contracts import FeatureContract, FeatureFormat
from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers.base import Featurizer
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.feature_source import FeatureSource
from tests.conftest import MockFeatureSource


class StubSource(FeatureSource):
    """Minimal feature source serving one view matrix, NaN rows included."""

    def __init__(self, view_matrix: np.ndarray, identifiers: np.ndarray) -> None:
        """Store the backing matrix and its row identifiers."""
        self._view_matrix = view_matrix
        self._identifiers = identifiers

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return self._identifiers

    @property
    def mdata(self) -> None:
        """No MuData backing for stubs."""
        return None

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return the rows of the backing matrix for *entity_ids*."""
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        indices = [idx_map[eid] for eid in entity_ids]
        return self._view_matrix[indices]

    def get_entity_view(self, entity_id: str, view: str) -> np.ndarray | None:
        """Return one row, or ``None`` for an unknown entity."""
        idx_map = {eid: i for i, eid in enumerate(self._identifiers)}
        if entity_id not in idx_map:
            return None
        return self._view_matrix[idx_map[entity_id]]

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """No feature names for stubs."""
        return None


class SmilesSource(MockFeatureSource):
    """Feature source that serves *smiles* (by drug id) through a stubbed ``mdata``."""

    def __init__(self, smiles: Mapping[str, Any]) -> None:
        """Serve *smiles* by drug id."""
        super().__init__({drug: {} for drug in smiles})
        frame = pd.DataFrame({"canonical_smiles": pd.Series(dict(smiles))})
        self._mdata = SimpleNamespace(mod={"response": SimpleNamespace(var=frame)})

    @property
    def mdata(self) -> Any:
        """The stubbed MuData holding the SMILES column."""
        return self._mdata


class StatelessFeaturizer(Featurizer):
    """Featurizer whose fit is a no-op; subclasses supply the transform and ``output_dim``."""

    def _fit(self, source, *, entity_ids=None, pair_expanded_ids=None, pair_expanded_es_ids=None):
        """Fitting is a no-op; the transform is stateless."""
        return self


class DoublingFeaturizer(StatelessFeaturizer):
    """Featurizer that doubles the values of ``test_view``."""

    input_views = ("test_view",)

    def _transform_features(self, source: FeatureSource, entity_ids: np.ndarray) -> FeaturizerResult:
        """Return the doubled matrix and its named block together."""
        matrix = source.get_view_matrix("test_view", entity_ids)
        doubled = (matrix * 2).astype(np.float32)
        return FeaturizerResult(entity_ids, {"test_view": numeric_feature_block(doubled)}, ("test_view",))

    @property
    def output_dim(self) -> int:
        """Fixed width of the stub view."""
        return 3


# Registration normally injects the contract; these stubs are never registered.
DoublingFeaturizer.contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)


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
