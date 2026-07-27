"""Tests for featurizer config tree helpers."""

from __future__ import annotations

from drevalpy.components.featurizer_config_parse import normalize_featurizer_config
from drevalpy.components.featurizer_tree import iter_featurizer_leaves, map_featurizer_tree
from drevalpy.models.config import FeaturizerConfig


def test_iter_featurizer_leaves_expands_concat() -> None:
    root = FeaturizerConfig.model_validate(
        normalize_featurizer_config("raw[expression]+pca[methylation]", default_registry="cell_line"),
    )
    names = [leaf.name for leaf in iter_featurizer_leaves(root, "cell_line")]
    assert names == ["raw", "pca"]


def test_map_featurizer_tree_patches_matching_leaf() -> None:
    root = FeaturizerConfig.model_validate(
        normalize_featurizer_config("raw[expression]+pca[methylation]", default_registry="cell_line"),
    )

    def bump_n_components(child: FeaturizerConfig) -> FeaturizerConfig:
        if child.name == "pca" and child.view == "methylation":
            return child.model_copy(
                update={"hyperparameters": {**child.hyperparameters, "n_components": 7}},
                deep=True,
            )
        return child

    updated = map_featurizer_tree(root, "cell_line", bump_n_components)
    for leaf in iter_featurizer_leaves(updated, "cell_line"):
        if leaf.name == "pca":
            assert leaf.hyperparameters.get("n_components") == 7
            return
    raise AssertionError("pca leaf missing")
