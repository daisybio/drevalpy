"""Map legacy cell-line view names to explicit featurizer configs."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.models.config import FeaturizerConfig

CELL_LINE_VIEW_TO_FEATURIZER = {
    "gene_expression": "scaledGeneExpression",
    "methylation": "methylationPCA",
    "mutations": "mutations",
    "copy_number_variation_gistic": "copyNumberVariationGistic",
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
        return {"geneExpression": {"view": view}}
    name = CELL_LINE_VIEW_TO_FEATURIZER[view]
    if name == "methylationPCA":
        n_components = hyperparameters.get("methylation_n_components")
        if n_components is None:
            n_components = hyperparameters.get("methylation_pca_components", 100)
        return {"methylationPCA": {"n_components": int(n_components)}}
    if name == "proteomics":
        proteomics_hp = {key: hyperparameters[key] for key in PROTEOMICS_HP_KEYS if key in hyperparameters}
        return {"proteomics": proteomics_hp} if proteomics_hp else "proteomics"
    return name


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
