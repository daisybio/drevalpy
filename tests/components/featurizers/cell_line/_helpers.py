"""Shared helpers for cell-line featurizer tests.

Plain module (no ``__init__.py``) imported by dotted path, per the test layout
rules in ``AGENTS.md``.
"""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.storage import register_variant
from drevalpy.types.data.feature_source import CellLineFeatureSource
from tests.models.synthetic_fixtures import synthetic_mudataset_gene_expression_fingerprints

PRECOMPUTED = np.array([[9.0, 8.0], [7.0, 6.0]], dtype=np.float32)


def precomputed_source(
    featurizer_cls: type,
    *,
    matrix: np.ndarray | None = None,
    hyperparameters: dict[str, object] | None = None,
) -> CellLineFeatureSource:
    """Return a dataset-backed source carrying a registered variant for *featurizer_cls*.

    Exercises the ``fetch``-hit branch every dense cell-line featurizer takes when
    ``Dataset.precompute()`` has already written its matrix.

    :param featurizer_cls: Featurizer whose ``storage_key`` / ``side`` to register under.
    :param matrix: Values to store; defaults to :data:`PRECOMPUTED`.
    :param hyperparameters: HP setting the variant was computed under; featurizers
        that pass their own HPs to ``fetch`` (``pca``) need these to match.
    :returns: Cell-line feature source over a 2x2 synthetic dataset.
    """
    values = PRECOMPUTED if matrix is None else matrix
    dataset = synthetic_mudataset_gene_expression_fingerprints()
    key = f"{featurizer_cls.__name__}_precomputed"
    dataset.mdata.mod["response"].obsm[key] = values
    register_variant(
        dataset.mdata,
        featurizer_cls.storage_key,
        key,
        hyperparameters,
        side=featurizer_cls.side,
    )
    return CellLineFeatureSource(dataset, dataset.cell_line_ids)


def assert_uses_precomputed_variant(
    featurizer,
    *,
    ids_kwarg: str = "entity_ids",
    hyperparameters: dict[str, object] | None = None,
    expect_output_dim: bool = True,
    expected_blocks: tuple[str, ...] | None = None,
) -> None:
    """Assert *featurizer* serves a stored matrix rather than recomputing one.

    Every dense cell-line featurizer takes the identical ``fetch``-hit path, so
    this assertion was written out once per featurizer. What actually differs per
    featurizer is captured in the keyword arguments: which keyword its ``fit``
    names the entity IDs with, whether it forwards its own hyperparameters to
    ``fetch``, whether ``output_dim`` follows the stored width, and which blocks
    ``transform_blocks`` is expected to emit.

    :param featurizer: Unfitted featurizer instance to exercise.
    :param ids_kwarg: ``fit`` keyword carrying the entity IDs.
    :param hyperparameters: HP setting the stored variant is registered under.
    :param expect_output_dim: Assert ``output_dim`` equals the stored width.
    :param expected_blocks: When given, also assert ``transform_blocks`` returns
        exactly these block names, each carrying the stored matrix.
    """
    source = precomputed_source(type(featurizer), hyperparameters=hyperparameters)
    ids = source.identifiers

    featurizer.fit(source, **{ids_kwarg: ids})

    if expect_output_dim:
        assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)

    if expected_blocks is None:
        return
    blocks = featurizer.transform_blocks(source, ids)
    assert set(blocks) == set(expected_blocks)
    for name in expected_blocks:
        np.testing.assert_allclose(blocks[name].values, PRECOMPUTED)
