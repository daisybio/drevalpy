"""Resolve legacy view names from featurizer configs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
    from drevalpy.models.config.model import ModelConfig
    from drevalpy.models.config.resolved import ResolvedModelConfig

_PCA_METHYLATION_TOKEN = "pca[" + "methylation" + "]"
_NORMALIZED_PROTEOMICS_TOKEN = "normal" + "izedProteomics"

FEATURIZER_NAME_TO_CELL_LINE_VIEW = {
    "scaledGeneExpression": "gene_expression",
    _PCA_METHYLATION_TOKEN: "methylation",
    "raw[mutations]": "mutations",
    "raw[cnv]": "copy_number_variation_gistic",
    _NORMALIZED_PROTEOMICS_TOKEN: "proteomics",
    "bionic": "bionic_features",
    "landmarkGenes": "gene_expression",
    "landmarkGenesReduced": "gene_expression",
    "pathways": "pathways",
    "dipkGeneExpression": "gene_expression",
    "pharmaFormerGeneExpression": "gene_expression",
    "sparsegoOntology": "gene_expression",
    "molirOmics": "gene_expression",
    "superfeltrOmics": "gene_expression",
}

DRUG_FEATURIZER_TO_VIEW = {
    "fingerprints": "fingerprints",
    "smilesvec": "smilesvec",
    "bpePharmaformer": "bpe_smiles",
    "molgnet": "molgnet_features",
    "drugGraph": "drug_graph",
    "identity": "one_hot",
    "view": "fingerprints",
}


def _featurizer_cls(config: FeaturizerConfig, *, registry: str) -> type[Any]:
    from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

    if registry == "cell_line":
        return get_cell_line_featurizer(config.name)
    return get_drug_featurizer(config.name)


def entity_id_only_from_featurizer_config(config: FeaturizerConfig, *, registry: str) -> bool:
    """Return True when the featurizer only needs entity identifiers, not omics or drug views.

    :param config: Featurizer config node to inspect.
    :param registry: ``cell_line`` or ``drug`` registry label.
    :returns: ``True`` when the featurizer tree is entity-id-only.
    """
    if config.name == "concatFeaturizers":
        children = config.featurizers or ()
        if not children:
            return False
        return all(entity_id_only_from_featurizer_config(child, registry=registry) for child in children)
    return bool(getattr(_featurizer_cls(config, registry=registry), "entity_id_only", False))


def cell_line_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured cell-line featurizer only needs entity ids.

    :param config: Model configuration to inspect.
    :returns: ``True`` when no cell-line omics views are required.
    """
    if config.cell_line_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.cell_line_featurizer, registry="cell_line")


def drug_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured drug featurizer only needs entity ids.

    :param config: Model configuration to inspect.
    :returns: ``True`` when no drug feature views are required.
    """
    if config.drug_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.drug_featurizer, registry="drug")


_MULTI_OMICS_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


def _concat_child_views(
    config: FeaturizerConfig,
    *,
    registry: str,
    resolved: ResolvedModelConfig | None,
) -> list[str]:
    views: list[str] = []
    for child in config.featurizers or ():
        views.extend(_views_from_featurizer_config(child, registry=registry, resolved=resolved))
    return views


def _sparsego_input_type(config: FeaturizerConfig, *, resolved: ResolvedModelConfig | None) -> str:
    from drevalpy.components.featurizer_label import qualified_featurizer_selector

    if resolved is not None:
        selector = qualified_featurizer_selector(config.name, config.view)
        values = resolved.featurizer_values("cell_line", selector)
        if "input_type" in values:
            return str(values["input_type"])
    space = dict(config.hyperparameter_space or {})
    if not space:
        from drevalpy.components.registry import get_cell_line_featurizer

        space = dict(get_cell_line_featurizer(config.name).get_hyperparameter_space())
    spec = space.get("input_type")
    if isinstance(spec, Mapping) and "default" in spec:
        return str(spec["default"])
    return "expression"


def _special_cell_line_views(
    config: FeaturizerConfig,
    *,
    resolved: ResolvedModelConfig | None,
) -> list[str] | None:
    if config.name in {"molirOmics", "superfeltrOmics"}:
        return list(_MULTI_OMICS_VIEWS)
    if config.name == "sparsegoOntology":
        return ["mutations" if _sparsego_input_type(config, resolved=resolved) == "mutations" else "gene_expression"]
    mapped = FEATURIZER_NAME_TO_CELL_LINE_VIEW.get(config.name)
    return [mapped] if mapped else None


def _view_featurizer_view_name(
    config: FeaturizerConfig,
    *,
    registry: str,
    resolved: ResolvedModelConfig | None,
) -> str:
    if resolved is not None:
        from drevalpy.components.featurizer_label import qualified_featurizer_selector

        selector = qualified_featurizer_selector(config.name, config.view)
        values = resolved.featurizer_values(registry, selector)
        if "view" in values:
            return str(values["view"])
    if config.view is not None:
        return str(config.view)
    options = config.options or {}
    if "view" in options:
        return str(options["view"])
    return "fingerprints"


def _mapped_leaf_views(
    config: FeaturizerConfig,
    *,
    registry: str,
    resolved: ResolvedModelConfig | None,
) -> list[str]:
    if config.name in ("raw", "pca"):
        return [str(config.view)]
    if config.name == "view":
        return [_view_featurizer_view_name(config, registry=registry, resolved=resolved)]
    if registry == "cell_line":
        special = _special_cell_line_views(config, resolved=resolved)
        if special is not None:
            return special
    if registry == "drug":
        drug_view = DRUG_FEATURIZER_TO_VIEW.get(config.name)
        if drug_view:
            return [drug_view]
    return [str(config.view)] if config.view else []


def _views_from_featurizer_config(
    config: FeaturizerConfig,
    *,
    registry: str,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    if config.name == "concatFeaturizers":
        return _concat_child_views(config, registry=registry, resolved=resolved)
    return _mapped_leaf_views(config, registry=registry, resolved=resolved)


def cell_line_views_from_model_config(
    config: ModelConfig,
    *,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Resolve legacy cell-line view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Legacy cell-line view names required by the config.
    """
    if config.cell_line_featurizer is None:
        return []
    if cell_line_entity_id_only_from_model_config(config):
        return []
    if config.cell_line_featurizer.name == "tissue":
        return []
    views = _views_from_featurizer_config(
        config.cell_line_featurizer,
        registry="cell_line",
        resolved=resolved,
    )
    return views or ["gene_expression"]


def drug_views_from_model_config(
    config: ModelConfig,
    *,
    resolved: ResolvedModelConfig | None = None,
) -> list[str]:
    """Resolve legacy drug view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :param resolved: Optional resolved values that can affect view selection.
    :returns: Legacy drug view names required by the config.
    """
    if config.drug_featurizer is None:
        return []
    if drug_entity_id_only_from_model_config(config):
        return []
    views = _views_from_featurizer_config(
        config.drug_featurizer,
        registry="drug",
        resolved=resolved,
    )
    return views or ["fingerprints"]


def cell_line_views_from_resolved(resolved: ResolvedModelConfig) -> list[str]:
    """Resolve cell-line views from a resolved instance config.

    :param resolved: Resolved model configuration.
    :returns: Legacy cell-line view names.
    """
    return cell_line_views_from_model_config(resolved.template, resolved=resolved)


def drug_views_from_resolved(resolved: ResolvedModelConfig) -> list[str]:
    """Resolve drug views from a resolved instance config.

    :param resolved: Resolved model configuration.
    :returns: Legacy drug view names.
    """
    return drug_views_from_model_config(resolved.template, resolved=resolved)
