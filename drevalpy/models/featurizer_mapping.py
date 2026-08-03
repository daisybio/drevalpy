"""Map legacy cell-line view names to explicit featurizer configs."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import (
    CellLineFeaturizerConfig,
    DrugFeaturizerConfig,
    FeaturizerConfig,
    ModelConfig,
)

CELL_LINE_VIEW_TO_FEATURIZER = {
    "gene_expression": "scaledGeneExpression",
    "methylation": "pca[methylation]",
    "mutations": "raw[mutations]",
    "copy_number_variation_gistic": "raw[cnv]",
    "proteomics": "normalizedProteomics",
    "bionic_features": "bionic",
}

PROTEOMICS_HP_KEYS = (
    "proteomics_feature_threshold",
    "proteomics_n_features",
    "proteomics_normalization_width",
    "proteomics_normalization_downshift",
)


def view_to_concat_block_label(view: str) -> str:
    """Map a legacy omics view name to a concat block label."""
    return CELL_LINE_VIEW_TO_FEATURIZER.get(view, f"raw[{view}]")


def _child_config_for_view(view: str, hyperparameters: dict[str, Any]) -> str | dict[str, Any]:
    if view not in CELL_LINE_VIEW_TO_FEATURIZER:
        return {"name": "raw", "view": view, "hyperparameters": {}}
    token = CELL_LINE_VIEW_TO_FEATURIZER[view]
    if token == "pca[methylation]":  # noqa: S105
        n_components = hyperparameters.get("methylation_n_components")
        if n_components is None:
            n_components = hyperparameters.get("methylation_pca_components", 100)
        return {token: {"n_components": int(n_components)}}
    if token == "normalizedProteomics":  # noqa: S105
        proteomics_hp = {key: hyperparameters[key] for key in PROTEOMICS_HP_KEYS if key in hyperparameters}
        return {"normalizedProteomics": proteomics_hp} if proteomics_hp else "normalizedProteomics"
    return token


def cell_line_featurizer_from_views(views: list[str], hyperparameters: dict[str, Any]) -> CellLineFeaturizerConfig:
    """Build a compact featurizer config from legacy cell-line view names."""
    if len(views) == 1:
        return CellLineFeaturizerConfig.model_validate(
            normalize_featurizer_config(
                _child_config_for_view(views[0], hyperparameters),
                default_registry="cell_line",
            )
        )
    children = [_child_config_for_view(view, hyperparameters) for view in views]
    return CellLineFeaturizerConfig.model_validate(
        normalize_featurizer_config(
            {"concatFeaturizers": {"featurizers": children}},
            default_registry="cell_line",
        )
    )


def drug_featurizer_from_view(view: str) -> DrugFeaturizerConfig:
    """Build a drug featurizer config from a legacy drug view name."""
    if view == "fingerprints":
        return DrugFeaturizerConfig.model_validate(
            normalize_featurizer_config("fingerprints", default_registry="drug"),
        )
    named = {
        "smilesvec": "smilesvec",
        "bpe_smiles": "bpePharmaformer",
        "molgnet_features": "molgnet",
        "drug_graph": "drugGraph",
        "one_hot": "identity",
    }
    if view in named:
        return DrugFeaturizerConfig.model_validate(
            normalize_featurizer_config(named[view], default_registry="drug"),
        )
    return DrugFeaturizerConfig.model_validate(
        {
            "name": "view",
            "hyperparameters": {"view": view},
        },
    )


FEATURIZER_NAME_TO_CELL_LINE_VIEW = {value: key for key, value in CELL_LINE_VIEW_TO_FEATURIZER.items()}
FEATURIZER_NAME_TO_CELL_LINE_VIEW.update(
    {
        "landmarkGenes": "gene_expression",
        "landmarkGenesReduced": "gene_expression",
        "pathways": "pathways",
        "bionic": "bionic_features",
        "normalizedProteomics": "proteomics",
        "dipkGeneExpression": "gene_expression",
        "pharmaFormerGeneExpression": "gene_expression",
        "sparsegoOntology": "gene_expression",
        "molirOmics": "gene_expression",
        "superfeltrOmics": "gene_expression",
    }
)

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


_MULTI_OMICS_VIEWS = ("gene_expression", "mutations", "copy_number_variation_gistic")


def _concat_child_views(config: FeaturizerConfig, *, registry: str) -> list[str]:
    views: list[str] = []
    for child in config.hyperparameters.get("featurizers", []):
        child_cfg = FeaturizerConfig.model_validate(normalize_featurizer_config(child, default_registry=registry))
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
    """Resolve legacy cell-line view names from a zoo-backed model config."""
    if config.cell_line_featurizer is None:
        return []
    if cell_line_entity_id_only_from_model_config(config):
        return []
    if config.cell_line_featurizer.name == "tissue":
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
