"""Register the built-in components by importing the modules that define them.

:data:`BUILTIN_MODULES` lists every module whose import registers a built-in predictor,
featurizer, splitter or visualization, in the order they register: the registries keep
registration order, so this tuple fixes the order of every listing. A new built-in
component module is added here by hand.

:func:`register_builtins` imports them eagerly and is idempotent, since importing a module
twice registers nothing new. Registries are never re-populated after they are emptied:
tests that change a registry restore it from :meth:`Registry.snapshot
<drevalpy.registry._registry.Registry.snapshot>` instead.
"""

from __future__ import annotations

import importlib
import traceback
from typing import Any

from drevalpy.log import get_logger
from drevalpy.registry._registry import Registry

logger = get_logger(__name__)

_FEATURIZERS = "drevalpy.components.featurizers"
_PREDICTORS = "drevalpy.components.predictors"

#: Modules whose import registers the built-in components, in registration order.
BUILTIN_MODULES: tuple[str, ...] = (
    f"{_FEATURIZERS}.shared.constant",
    f"{_FEATURIZERS}.shared.one_hot",
    f"{_FEATURIZERS}.shared.concat",
    f"{_FEATURIZERS}.sample.ae_gene_expression",
    f"{_FEATURIZERS}.sample.bionic",
    f"{_FEATURIZERS}.sample.dipk_gene_expression",
    f"{_FEATURIZERS}.sample.landmark",
    f"{_FEATURIZERS}.sample.molir_omics",
    f"{_FEATURIZERS}.sample.normalized_proteomics",
    f"{_FEATURIZERS}.sample.pathways",
    f"{_FEATURIZERS}.sample.pca",
    f"{_FEATURIZERS}.sample.pharmaformer_gene_expression",
    f"{_FEATURIZERS}.sample.scaled_gene_expression",
    f"{_FEATURIZERS}.sample.sparsego_ontology",
    f"{_FEATURIZERS}.sample.superfeltr_omics",
    f"{_FEATURIZERS}.sample.tissue",
    f"{_FEATURIZERS}.sample.view",
    f"{_FEATURIZERS}.drug.bpe_pharmaformer",
    f"{_FEATURIZERS}.drug.chemberta",
    f"{_FEATURIZERS}.drug.drug_graph",
    f"{_FEATURIZERS}.drug.fingerprints",
    f"{_FEATURIZERS}.drug.molgnet",
    f"{_FEATURIZERS}.drug.molgnet_pooled",
    f"{_FEATURIZERS}.drug.smiles_tokens",
    f"{_FEATURIZERS}.drug.smilesvec",
    f"{_FEATURIZERS}.drug.view",
    f"{_PREDICTORS}.naive.effects",
    f"{_PREDICTORS}.naive.entity_mean",
    f"{_PREDICTORS}.naive.mean",
    f"{_PREDICTORS}.naive.tissue",
    f"{_PREDICTORS}.lightgbm_pred",
    f"{_PREDICTORS}.sklearn_models",
    f"{_PREDICTORS}.xgboost_pred",
    f"{_PREDICTORS}.literature.dipk.predictor",
    f"{_PREDICTORS}.literature.druggnn.predictor",
    f"{_PREDICTORS}.literature.ensemblemf.predictor",
    f"{_PREDICTORS}.literature.molir.predictor",
    f"{_PREDICTORS}.literature.paccmann.predictor",
    f"{_PREDICTORS}.literature.pharmaformer.predictor",
    f"{_PREDICTORS}.literature.precily.predictor",
    f"{_PREDICTORS}.literature.sparsego.predictor",
    f"{_PREDICTORS}.literature.srmf.predictor",
    f"{_PREDICTORS}.literature.superfeltr.predictor",
    f"{_PREDICTORS}.neural_network.predictor",
    "drevalpy.data.splitters.entity",
    "drevalpy.data.splitters.lpo",
    "drevalpy.data.splitters.lsto",
    "drevalpy.data.splitters.lto",
    "drevalpy.visualization.plots.comparison_scatter",
    "drevalpy.visualization.plots.critical_difference",
    "drevalpy.visualization.plots.heatmap",
    "drevalpy.visualization.plots.leaderboard",
    "drevalpy.visualization.plots.regression_scatter",
    "drevalpy.visualization.plots.violin",
)

_SKIPPED_MODULES: dict[str, str] = {}


def get_skipped_builtin_modules() -> dict[str, str]:
    """Return modules that could not be imported during built-in registration.

    :returns: Mapping of dotted module name to the formatted traceback of the
        import failure. Empty when every built-in module imported cleanly.
    """
    return dict(_SKIPPED_MODULES)


def _try_import(module_name: str) -> Exception | None:
    """Import *module_name*, returning the failure instead of raising it.

    :param module_name: Dotted name of a built-in component module.
    :returns: The import error, or ``None`` once the module is imported.
    """
    try:
        importlib.import_module(module_name)
    except (ImportError, AttributeError) as exc:
        return exc
    _SKIPPED_MODULES.pop(module_name, None)
    return None


def register_builtins() -> None:
    """Import every module in :data:`BUILTIN_MODULES`, so their decorators register the built-ins.

    A module that fails is retried once right away: under ``import drevalpy``,
    ``shared.concat`` first meets ``models.config`` while that package is still
    initializing. The failed attempt leaves the already-imported ``models.config``
    submodules in :data:`sys.modules`, so the second attempt gets past the cycle. A module that fails
    twice is recorded in :func:`get_skipped_builtin_modules` and reported at WARNING
    level, because a skipped module means its components silently disappear from the
    registries. Calling this again retries only the skipped modules.
    """
    for module_name in BUILTIN_MODULES:
        exc = _try_import(module_name) and _try_import(module_name)
        if exc is None:
            continue
        _SKIPPED_MODULES[module_name] = "".join(traceback.format_exception(exc))
        logger.warning(
            "Skipping built-in component module %s: its components will be unavailable (%s: %s)",
            module_name,
            type(exc).__name__,
            exc,
        )


def builtin_names(registry: Registry[Any]) -> tuple[str, ...]:
    """Return the names in *registry* whose entry is defined in a built-in module.

    :param registry: A component, splitter or visualization registry.
    :returns: Built-in names in registration order; plugin and extension entries are left out.
    """
    modules = frozenset(BUILTIN_MODULES)
    return tuple(name for name, obj in registry.items() if getattr(obj, "__module__", None) in modules)
