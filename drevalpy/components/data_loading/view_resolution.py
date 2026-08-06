"""Resolve legacy view names from featurizer configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config

if TYPE_CHECKING:
    from drevalpy.models.config.featurizer import FeaturizerConfig
    from drevalpy.models.config.model import ModelConfig

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


def _featurizer_config_cls() -> type[FeaturizerConfig]:
    from drevalpy.models.config.featurizer import FeaturizerConfig

    return FeaturizerConfig


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
    featurizer_config_cls = _featurizer_config_cls()
    if config.name == "concatFeaturizers":
        children = config.hyperparameters.get("featurizers", [])
        if not children:
            return False
        return all(
            entity_id_only_from_featurizer_config(
                featurizer_config_cls.model_validate(normalize_featurizer_config(child, default_registry=registry)),
                registry=registry,
            )
            for child in children
        )
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


def _concat_child_views(config: FeaturizerConfig, *, registry: str) -> list[str]:
    featurizer_config_cls = _featurizer_config_cls()
    views: list[str] = []
    for child in config.hyperparameters.get("featurizers", []):
        child_cfg = featurizer_config_cls.model_validate(normalize_featurizer_config(child, default_registry=registry))
        views.extend(_views_from_featurizer_config(child_cfg, registry=registry))
    return views


def _special_cell_line_views(config: FeaturizerConfig) -> list[str] | None:
    if config.name in {"molirOmics", "superfeltrOmics"}:
        return list(_MULTI_OMICS_VIEWS)
    if config.name == "sparsegoOntology":
        return ["mutations" if config.hyperparameters.get("input_type") == "mutations" else "gene_expression"]
    mapped = FEATURIZER_NAME_TO_CELL_LINE_VIEW.get(config.name)
    return [mapped] if mapped else None


def _mapped_leaf_views(config: FeaturizerConfig, *, registry: str) -> list[str]:
    if config.name in ("raw", "pca"):
        return [str(config.view or config.hyperparameters.get("view"))]
    if config.name == "view":
        return [str(config.hyperparameters.get("view", "fingerprints"))]
    if registry == "cell_line":
        special = _special_cell_line_views(config)
        if special is not None:
            return special
    if registry == "drug":
        drug_view = DRUG_FEATURIZER_TO_VIEW.get(config.name)
        if drug_view:
            return [drug_view]
    return [str(config.view)] if config.view else []


def _views_from_featurizer_config(config: FeaturizerConfig, *, registry: str) -> list[str]:
    if config.name == "concatFeaturizers":
        return _concat_child_views(config, registry=registry)
    return _mapped_leaf_views(config, registry=registry)


def cell_line_views_from_model_config(config: ModelConfig) -> list[str]:
    """Resolve legacy cell-line view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :returns: Legacy cell-line view names required by the config.
    """
    if config.cell_line_featurizer is None:
        return []
    if cell_line_entity_id_only_from_model_config(config):
        return []
    if config.cell_line_featurizer.name == "tissue":
        return []
    views = _views_from_featurizer_config(config.cell_line_featurizer, registry="cell_line")
    return views or ["gene_expression"]


def drug_views_from_model_config(config: ModelConfig) -> list[str]:
    """Resolve legacy drug view names from a zoo-backed model config.

    :param config: Model configuration to resolve.
    :returns: Legacy drug view names required by the config.
    """
    if config.drug_featurizer is None:
        return []
    if drug_entity_id_only_from_model_config(config):
        return []
    views = _views_from_featurizer_config(config.drug_featurizer, registry="drug")
    return views or ["fingerprints"]
