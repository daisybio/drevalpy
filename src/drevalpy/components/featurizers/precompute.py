"""Pre-compute featurizer representations and store them in a dataset.

:func:`precompute` runs one featurizer over every entity of its side for a list of
hyperparameter configurations; :func:`precompute_all` does so for every registered
featurizer that opts in and finds its source data in the dataset. The matrices land
in the dataset's MuData through :meth:`~drevalpy.components.featurizers.storage.FeaturizerStorageMixin.store`,
so a model run that asks for the same hyperparameters reads them back instead of computing.
"""

from __future__ import annotations

from rich.progress import Progress

from drevalpy.log import get_logger
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_source import DrugFeatureSource, SampleFeatureSource
from drevalpy.types.data.views import stored_view_name

logger = get_logger(__name__)


def precompute(
    dataset: Dataset,
    featurizer_cls: type,
    hyperparameters: list[dict] | int,
    view: str | None = None,
) -> None:
    """Pre-compute and store featurizer representations for given HP configs.

    For independent featurizers (those with ``_compute_from_source``), calls
    that method directly, bypassing fit/transform. Always includes the
    default HP config in addition to drawn variants.

    :param dataset: Dataset whose entities are featurized and which receives the matrices.
    :param featurizer_cls: Registered featurizer class (knows its own side).
    :param hyperparameters: Either a list of explicit HP dicts,
        or an int N to draw N configs from the featurizer's HP space.
    :param view: View name for view-parameterized featurizers (e.g., "gene_expression").
    """
    if isinstance(hyperparameters, int):
        from drevalpy.models.tuning.search_space import sample_configurations

        configs = sample_configurations(featurizer_cls.get_hyperparameter_space(), hyperparameters)
    else:
        configs = list(hyperparameters)

    default_config = featurizer_cls.get_default_hyperparameters()
    if default_config not in configs:
        configs.insert(0, default_config)

    if getattr(featurizer_cls, "side", "sample") == "sample":
        entity_ids = dataset.sample_ids
        source = SampleFeatureSource(dataset, entity_ids)
    else:
        entity_ids = dataset.drug_ids
        source = DrugFeatureSource(dataset, entity_ids)

    base_kwargs: dict = {} if view is None else {"view": view}

    with Progress() as progress:
        task = progress.add_task(f"Precomputing {featurizer_cls.storage_key}", total=len(configs))
        for config in configs:
            featurizer = featurizer_cls(**{**base_kwargs, **config})
            if hasattr(featurizer, "_compute_from_source"):
                matrix = featurizer._compute_from_source(source, entity_ids)
            else:
                featurizer.fit(source, entity_ids=entity_ids)
                matrix = featurizer.transform(source, entity_ids)
            featurizer.store(dataset.mdata, entity_ids, matrix, hyperparameters=config)
            progress.advance(task)


def precompute_all(dataset: Dataset, n_variants: int = 10) -> None:
    """Pre-compute all fixed featurizers with N HP variants each.

    Iterates all registered featurizers (sample + drug), skips those
    not marked for precomputation, and pre-computes N drawn HP
    configurations for the rest. Featurizers without any HP space get a
    single default-params variant.

    :param dataset: Dataset whose entities are featurized and which receives the matrices.
    :param n_variants: Number of HP configurations to draw for featurizers
        that have a tunable HP space.
    """
    from drevalpy.registry.drug_featurizer import drug_featurizer_registry
    from drevalpy.registry.sample_featurizer import sample_featurizer_registry

    for registry in (sample_featurizer_registry, drug_featurizer_registry):
        for name in registry.list_names():
            _precompute_single(dataset, registry.get(name), n_variants)


def _precompute_single(dataset: Dataset, cls: type, n_variants: int) -> None:
    """Pre-compute one featurizer class if eligible."""
    name = getattr(cls, "registry_name", cls.__name__)
    if not cls.precompute:
        return
    if cls.entity_id_only:
        logger.debug("Skipping %s: entity_id_only", name)
        return
    source_views = getattr(cls, "source_views", None)
    if source_views and not _has_source_data(dataset, source_views, getattr(cls, "side", "sample")):
        logger.debug("Skipping %s: required source data not available", name)
        return
    effective_n = n_variants if cls.get_hyperparameter_space() else 1
    view: str | None = None
    if cls.requires_view:
        if cls.input_views:
            view = cls.input_views[0]
        else:
            logger.debug("Skipping %s: requires_view but no input_views declared", name)
            return
    try:
        logger.info("Precomputing %s (%d variants)", name, effective_n)
        precompute(dataset, cls, effective_n, view=view)
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        logger.warning("Failed to precompute %s: %s", name, exc)


def _has_source_data(dataset: Dataset, source_views: tuple[str, ...], side: str) -> bool:
    """Check if the dataset has the raw source data every view in *source_views* needs."""
    return all(_has_view(dataset, view, side) for view in source_views)


def _has_view(dataset: Dataset, view: str, side: str) -> bool:
    """Check if the dataset stores *view* for entities of *side*."""
    response = dataset.response
    if view == "canonical_smiles":
        return "canonical_smiles" in response.var.columns
    if side == "sample":
        return stored_view_name(view) in dataset.mdata.mod
    return view in response.varm
