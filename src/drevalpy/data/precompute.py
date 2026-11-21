"""Pre-compute featurizer outputs and store them in a dataset as views.

:func:`precompute` computes the output of one
:class:`~drevalpy.components.featurizers._dense.PrecomputableFeaturizer` (one that needs no
fitting, such as ``fingerprints`` or ``chemberta``) for every entity of its side at its default
settings and stores it as the featurizer's view: a ``response.varm`` key on the drug side, a
modality on the sample side. The settings are recorded under
:data:`~drevalpy.types.data.feature_source.PRECOMPUTED_VIEWS_KEY` in ``mdata.uns``, so a model
whose featurizer has the same settings reads the view, and one with other settings computes
its own. :func:`precompute_all` does this for every registered precomputable featurizer whose
source views the dataset holds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from drevalpy.log import get_logger
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import PRECOMPUTED_VIEWS_KEY, FeatureSource, Side, precomputed_views

logger = get_logger(__name__)


def _store_view(
    dataset: Dataset, side: Side, view: str, ids: np.ndarray, matrix: np.ndarray, settings: dict[str, Any]
) -> None:
    """Store *matrix* as *view* of *side* and record the settings it was computed with.

    :param dataset: Dataset that receives the view.
    :param side: Side of the view.
    :param view: View name.
    :param ids: Entity IDs, one per row of *matrix*, in the order of the dataset's entities.
    :param matrix: Output with one row per entity.
    :param settings: Settings *matrix* was computed with.
    """
    mdata = dataset.mdata
    if side == "drug":
        mdata.mod["response"].varm[view] = matrix
    else:
        import anndata as ad
        import mudata as md
        import pandas as pd

        mdata.mod[view] = ad.AnnData(
            matrix, obs=pd.DataFrame(index=pd.Index(ids, dtype=str)), var=pd.DataFrame(index=_column_names(matrix))
        )
        with md.set_options(pull_on_update=False):
            mdata.update()
    record = precomputed_views(mdata)
    record.setdefault(side, {})[view] = settings
    mdata.uns[PRECOMPUTED_VIEWS_KEY] = json.dumps(record)


def _column_names(matrix: np.ndarray) -> list[str]:
    return [str(column) for column in range(matrix.shape[1])]


def precompute(dataset: Dataset, featurizer_name: str, side: Side) -> str | None:
    """Compute one featurizer's output at its default settings and store it as its view.

    Nothing is computed when the dataset already holds the view and the featurizer would read it.

    :param dataset: Dataset whose entities are featurized and which receives the view.
    :param featurizer_name: Registry name of the featurizer.
    :param side: Side to featurize, whose registry holds *featurizer_name*.
    :returns: The view written, or ``None`` when the dataset already held it.
    :raises ValueError: If the featurizer cannot be precomputed (it is not a ``PrecomputableFeaturizer``).
    """
    from drevalpy.components.featurizers._dense import PrecomputableFeaturizer
    from drevalpy.registry.featurizer import get_featurizer

    cls = get_featurizer(featurizer_name, side)
    if not issubclass(cls, PrecomputableFeaturizer):
        msg = f"featurizer {featurizer_name!r} needs fitting or reads no view of its own, so it cannot be precomputed"
        raise ValueError(msg)
    featurizer = cls(**cls.default_hyperparameters())
    ids = np.asarray(dataset.sample_ids if side == Side.SAMPLE else dataset.drug_ids, dtype=str)
    source = FeatureSource(dataset, side)
    if featurizer.reads_view(source):
        return None
    logger.info("Precomputing %s into view %r", featurizer_name, featurizer.view)
    matrix = featurizer.fit(source, ids).transform(source, ids).matrix
    _store_view(dataset, side, featurizer.view, ids, matrix, featurizer.settings)
    return featurizer.view


def precompute_all(dataset: Dataset) -> list[str]:
    """Pre-compute every registered precomputable featurizer, on its side.

    A featurizer is skipped when the dataset lacks its source views; one that fails is logged
    and skipped.

    :param dataset: Dataset whose entities are featurized and which receives the views.
    :returns: The views written.
    """
    from drevalpy.components.featurizers._dense import PrecomputableFeaturizer
    from drevalpy.registry.featurizer import featurizer_registries

    written: list[str] = []
    for side, registry in featurizer_registries.items():
        for name, cls in registry.items():
            if not issubclass(cls, PrecomputableFeaturizer):
                continue
            source = FeatureSource(dataset, side)
            if not all(source.has_view(view) for view in cls.source_views):
                logger.debug("Skipping %s: source views %s not available", name, cls.source_views)
                continue
            try:
                view = precompute(dataset, name, side)
            except (ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
                logger.warning("Failed to precompute %s: %s", name, exc)
                continue
            if view is not None:
                written.append(view)
    return written
