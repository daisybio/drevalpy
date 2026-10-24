"""Shared helpers for sample featurizer tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.sample import gene_lists
from drevalpy.components.featurizers.storage import register_variant
from drevalpy.types.data.feature_source import SampleFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset

PRECOMPUTED = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)


def precomputed_source(
    featurizer_cls: type,
    *,
    matrix: np.ndarray | None = None,
    hyperparameters: dict[str, object] | None = None,
) -> SampleFeatureSource:
    """Dataset-backed source carrying a registered variant for *featurizer_cls* (the ``fetch``-hit branch)."""
    values = PRECOMPUTED if matrix is None else matrix
    dataset = synthetic_mudataset(extra_views=("mutations", "copy_number_variation_gistic"))
    key = f"{featurizer_cls.__name__}_precomputed"
    dataset.mdata.mod["response"].obsm[key] = values
    register_variant(
        dataset.mdata,
        featurizer_cls.storage_key,
        key,
        hyperparameters,
        side=featurizer_cls.side,
    )
    return SampleFeatureSource(dataset, dataset.sample_ids)


def assert_uses_precomputed_variant(
    featurizer,
    *,
    ids_kwarg: str = "entity_ids",
    hyperparameters: dict[str, object] | None = None,
    expect_output_dim: bool = True,
    expected_blocks: tuple[str, ...] | None = None,
) -> None:
    """Assert *featurizer* serves a stored matrix (and, if *expected_blocks*, those blocks) rather than recomputing."""
    source = precomputed_source(type(featurizer), hyperparameters=hyperparameters)
    ids = source.identifiers

    fit_kwargs = {ids_kwarg: ids}
    if featurizer.requires_early_stopping:
        fit_kwargs["pair_expanded_es_ids"] = ids
    featurizer.fit(source, **fit_kwargs)

    if expect_output_dim:
        assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)

    if expected_blocks is None:
        return
    blocks = featurizer.transform_blocks(source, ids)
    assert set(blocks) == set(expected_blocks)
    for name in expected_blocks:
        np.testing.assert_allclose(blocks[name].values, PRECOMPUTED)


def use_gene_lists(monkeypatch: pytest.MonkeyPatch, directory: Path, lists: Mapping[str, Sequence[str]]) -> None:
    """Replace the shipped gene lists by small stand-in CSVs written to *directory*."""
    for name, symbols in lists.items():
        pd.DataFrame({"Symbol": list(symbols)}).to_csv(directory / f"{name}.csv", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(directory))
