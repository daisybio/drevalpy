"""Pre-compute featurizer outputs and store them in a dataset.

:func:`precompute` fits one featurizer on every entity of a side and transforms them, once per
hyperparameter configuration, and stores each output with :func:`~drevalpy.types.data.feature_source.store_variant`
under the featurizer's ``storage_key`` and ``stored_params()``. A later model run whose featurizer
returns the same ``stored_params()`` reads the matrix back through
:meth:`~drevalpy.types.data.feature_source.FeatureSource.stored` and is neither fitted nor
transformed. :func:`precompute_all` does this for every registered featurizer that looks up
stored outputs and finds its input data in the dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from rich.progress import Progress

from drevalpy.log import get_logger
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import FeatureSource, Side, store_variant

logger = get_logger(__name__)


def _configurations(cls: type[Any], hyperparameters: list[dict[str, Any]] | int) -> list[dict[str, Any]]:
    """Return the configurations to compute: the given or drawn ones, with the default first.

    :param cls: Featurizer class.
    :param hyperparameters: Explicit configurations, or how many to draw from the class's space.
    :returns: Configurations, the default included once.
    """
    if isinstance(hyperparameters, int):
        from drevalpy.models.tuning.search_space import sample_configurations

        configs = sample_configurations(cls.get_hyperparameter_space(), hyperparameters)
    else:
        configs = [dict(config) for config in hyperparameters]
    default = cls.default_hyperparameters()
    if default not in configs:
        configs.insert(0, default)
    return configs


def _stored_matrix(name: str, blocks: Mapping[str, Any]) -> np.ndarray:
    """Return the matrix to store: the first entity-aligned block, which ``from_stored`` wraps.

    :param name: Featurizer name, for the message.
    :param blocks: Blocks of every entity of the side.
    :returns: The block's values.
    :raises TypeError: If there is no entity-aligned numeric block.
    """
    for block in blocks.values():
        if block.entity_aligned:
            values = np.asarray(block.values)
            if values.dtype.kind not in "biuf" or values.ndim != 2:
                break
            return values
    msg = f"featurizer {name!r} emits no entity-aligned numeric matrix, so its output cannot be stored"
    raise TypeError(msg)


def precompute(
    dataset: Dataset,
    featurizer_name: str,
    side: Side,
    hyperparameters: list[dict[str, Any]] | int,
    view: str | None = None,
) -> list[str]:
    """Compute and store the output of one featurizer for several hyperparameter configurations.

    The default configuration is always included. A configuration whose output the dataset
    already stores is skipped.

    :param dataset: Dataset whose entities are featurized and which receives the matrices.
    :param featurizer_name: Registry name of the featurizer.
    :param side: Side to featurize; the featurizer must support it.
    :param hyperparameters: Explicit configurations, or how many to draw from the featurizer's space.
    :param view: View for a featurizer that takes one, e.g. ``"gene_expression"``.
    :returns: The keys the new matrices are stored under.
    :raises ValueError: If the featurizer does not support *side* or never looks up stored outputs.
    """
    from drevalpy.models._feature_preparation import FitIds, prepare_features
    from drevalpy.registry.featurizer import get_featurizer

    cls = get_featurizer(featurizer_name, side)
    entity_ids = np.asarray(dataset.sample_ids if side == "sample" else dataset.drug_ids)
    source = FeatureSource(dataset, side, entity_ids)
    base: dict[str, Any] = {} if view is None else {"view": view}
    configs = _configurations(cls, hyperparameters)
    stored_keys: list[str] = []
    with Progress() as progress:
        task = progress.add_task(f"Precomputing {featurizer_name}", total=len(configs))
        for config in configs:
            featurizer = cls(**{**base, **config})
            params = featurizer.stored_params()
            if featurizer.storage_key is None or params is None:
                msg = f"featurizer {featurizer_name!r} never looks up stored outputs, so there is nothing to precompute"
                raise ValueError(msg)
            if source.stored(featurizer.storage_key, params, entity_ids) is None:
                result = prepare_features(featurizer, source, entity_ids, fit=FitIds(entity_ids))
                matrix = _stored_matrix(featurizer_name, result.blocks)
                stored_keys.append(store_variant(dataset, side, featurizer.storage_key, matrix, params))
            progress.advance(task)
    return stored_keys


def precompute_all(dataset: Dataset, n_variants: int = 10) -> None:
    """Pre-compute every registered featurizer that looks up stored outputs, on each side it supports.

    A featurizer is skipped when its default instance's ``stored_params()`` is ``None``, when it
    needs an explicit view and declares no input view to use, or when the dataset holds neither
    its input views nor its source views. A featurizer with a tunable space
    gets *n_variants* drawn configurations plus the default; one without gets the default only.
    A featurizer that fails is logged and skipped.

    :param dataset: Dataset whose entities are featurized and which receives the matrices.
    :param n_variants: Configurations to draw for a featurizer with a tunable space.
    """
    from drevalpy.registry.featurizer import featurizers

    for name, cls in featurizers.items():
        for side in ("sample", "drug"):
            if side in cls.sides:
                _precompute_single(dataset, name, cls, side, n_variants)


def _view_for(name: str, cls: type[Any]) -> tuple[bool, str | None]:
    """Return whether *cls* can be built without a config, and the view to build it with.

    :param name: Registry name.
    :param cls: Featurizer class.
    :returns: ``(usable, view)``; *view* is ``None`` when the featurizer takes none.
    """
    from drevalpy.components.featurizers._featurizer_label import requires_explicit_view

    if not requires_explicit_view(name):
        return True, None
    if cls.input_views:
        return True, cls.input_views[0]
    return False, None


def _precompute_single(dataset: Dataset, name: str, cls: type[Any], side: Side, n_variants: int) -> None:
    """Pre-compute one featurizer on one side when it is eligible.

    :param dataset: Dataset to featurize.
    :param name: Registry name.
    :param cls: Featurizer class.
    :param side: Side to featurize.
    :param n_variants: Configurations to draw for a tunable featurizer.
    """
    buildable, view = _view_for(name, cls)
    if not buildable:
        logger.debug("Skipping %s: needs a view and declares no input view", name)
        return
    config: dict[str, Any] = {**cls.default_hyperparameters(), **({} if view is None else {"view": view})}
    try:
        featurizer = cls(**config)
    except (TypeError, ValueError) as exc:
        logger.debug("Skipping %s: cannot be built from its defaults: %s", name, exc)
        return
    stored_params = getattr(featurizer, "stored_params", None)
    if featurizer.storage_key is None or stored_params is None or stored_params() is None:
        return
    source = FeatureSource(dataset, side, np.asarray(dataset.sample_ids if side == "sample" else dataset.drug_ids))
    inputs = all(source.has_view(required) for required in cls.views(config))
    sources = bool(cls.source_views) and all(source.has_view(required) for required in cls.source_views)
    if not (inputs or sources):
        logger.debug("Skipping %s: required source data not available", name)
        return
    effective_n = n_variants if cls.get_hyperparameter_space() else 1
    try:
        logger.info("Precomputing %s (%d variants)", name, effective_n)
        precompute(dataset, name, side, effective_n, view=view)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        logger.warning("Failed to precompute %s: %s", name, exc)
