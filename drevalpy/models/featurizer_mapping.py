"""Map legacy cell-line view names to explicit featurizer configs."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import FeaturizerConfig, ModelConfig

CELL_LINE_VIEW_TO_FEATURIZER = {
    "gene_expression": "scaledGeneExpression",
    "methylation": "pca[methylation]",
    "mutations": "raw[mutations]",
    "copy_number_variation_gistic": "raw[cnv]",
    "proteomics": "proteomics",
    "bionic_features": "bionic",
}

PROTEOMICS_HP_KEYS = (
    "proteomics_feature_threshold",
    "proteomics_n_features",
    "proteomics_normalization_width",
    "proteomics_normalization_downshift",
)


def _child_config_for_view(view: str, hyperparameters: dict[str, Any]) -> str | dict[str, Any]:
    if view not in CELL_LINE_VIEW_TO_FEATURIZER:
        return f"raw[{view}]"
    token = CELL_LINE_VIEW_TO_FEATURIZER[view]
    if token == "pca[methylation]":
        n_components = hyperparameters.get("methylation_n_components")
        if n_components is None:
            n_components = hyperparameters.get("methylation_pca_components", 100)
        return {token: {"n_components": int(n_components)}}
    if token == "proteomics":
        proteomics_hp = {key: hyperparameters[key] for key in PROTEOMICS_HP_KEYS if key in hyperparameters}
        return {"proteomics": proteomics_hp} if proteomics_hp else "proteomics"
    return token


def cell_line_featurizer_from_views(views: list[str], hyperparameters: dict[str, Any]) -> FeaturizerConfig:
    """Build a compact featurizer config from legacy cell-line view names."""
    if len(views) == 1:
        return FeaturizerConfig.model_validate(
            normalize_featurizer_config(
                _child_config_for_view(views[0], hyperparameters),
                default_registry="cell_line",
            )
        )
    children = [_child_config_for_view(view, hyperparameters) for view in views]
    return FeaturizerConfig.model_validate(
        normalize_featurizer_config(
            {"concatFeaturizers": {"featurizers": children}},
            default_registry="cell_line",
        )
    )


def drug_featurizer_from_view(view: str) -> FeaturizerConfig:
    """Build a drug featurizer config from a legacy drug view name."""
    if view == "fingerprints":
        return FeaturizerConfig.model_validate(
            normalize_featurizer_config("fingerprints", default_registry="drug"),
        )
    named = {
        "smilesvec": "smilesvec",
        "bpe_smiles": "bpePharmaformer",
        "molgnet_features": "molgnet",
        "drug_graph": "drugGraph",
        "one_hot": "oneHot",
    }
    if view in named:
        return FeaturizerConfig.model_validate(
            normalize_featurizer_config(named[view], default_registry="drug"),
        )
    return FeaturizerConfig.model_validate(
        {
            "name": "view",
            "hyperparameters": {"view": view},
            "registry": "drug",
        },
    )


FEATURIZER_NAME_TO_CELL_LINE_VIEW = {value: key for key, value in CELL_LINE_VIEW_TO_FEATURIZER.items()}
FEATURIZER_NAME_TO_CELL_LINE_VIEW.update(
    {
        "geneExpression": "gene_expression",
        "landmarkGeneExpression": "gene_expression",
        "pathways": "pathways",
        "bionic": "bionic_features",
    }
)

DRUG_FEATURIZER_TO_VIEW = {
    "fingerprints": "fingerprints",
    "smilesvec": "smilesvec",
    "bpePharmaformer": "bpe_smiles",
    "molgnet": "molgnet_features",
    "drugGraph": "drug_graph",
    "oneHot": "one_hot",
    "view": "fingerprints",
}


def _featurizer_cls(config: FeaturizerConfig, *, registry: str) -> type[Any]:
    from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer

    if registry == "cell_line":
        return get_cell_line_featurizer(config.name)
    return get_drug_featurizer(config.name)


def entity_id_only_from_featurizer_config(config: FeaturizerConfig, *, registry: str) -> bool:
    """Return True when the featurizer only needs entity identifiers, not omics or drug views."""
    if config.name == "concatFeaturizers":
        children = config.hyperparameters.get("featurizers", [])
        if not children:
            return False
        return all(
            entity_id_only_from_featurizer_config(
                FeaturizerConfig.model_validate(normalize_featurizer_config(child, default_registry=registry)),
                registry=registry,
            )
            for child in children
        )
    return bool(getattr(_featurizer_cls(config, registry=registry), "entity_id_only", False))


def cell_line_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured cell-line featurizer only needs entity ids."""
    if config.cell_line_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.cell_line_featurizer, registry="cell_line")


def drug_entity_id_only_from_model_config(config: ModelConfig) -> bool:
    """Return True when the configured drug featurizer only needs entity ids."""
    if config.drug_featurizer is None:
        return False
    return entity_id_only_from_featurizer_config(config.drug_featurizer, registry="drug")


def _views_from_featurizer_config(config: FeaturizerConfig, *, registry: str) -> list[str]:
    if config.name == "concatFeaturizers":
        views: list[str] = []
        for child in config.hyperparameters.get("featurizers", []):
            child_cfg = FeaturizerConfig.model_validate(normalize_featurizer_config(child, default_registry=registry))
            views.extend(_views_from_featurizer_config(child_cfg, registry=registry))
        return views
    if config.name in ("raw", "pca"):
        return [str(config.view or config.hyperparameters.get("view"))]
    if config.name == "geneExpression":
        return [str(config.view or config.hyperparameters.get("view", "gene_expression"))]
    if config.name == "view":
        return [str(config.hyperparameters.get("view", "fingerprints"))]
    mapped = FEATURIZER_NAME_TO_CELL_LINE_VIEW.get(config.name) if registry == "cell_line" else None
    if mapped:
        return [mapped]
    if registry == "drug":
        drug_view = DRUG_FEATURIZER_TO_VIEW.get(config.name)
        if drug_view:
            return [drug_view]
    if config.view:
        return [str(config.view)]
    return []


def cell_line_views_from_model_config(config: ModelConfig) -> list[str]:
    """Resolve legacy cell-line view names from a zoo-backed model config."""
    if config.cell_line_featurizer is None:
        return ["gene_expression"]
    if cell_line_entity_id_only_from_model_config(config):
        return []
    views = _views_from_featurizer_config(config.cell_line_featurizer, registry="cell_line")
    return views or ["gene_expression"]


def drug_views_from_model_config(config: ModelConfig) -> list[str]:
    """Resolve legacy drug view names from a zoo-backed model config."""
    if config.drug_featurizer is None:
        return []
    if drug_entity_id_only_from_model_config(config):
        return []
    views = _views_from_featurizer_config(config.drug_featurizer, registry="drug")
    return views or ["fingerprints"]
