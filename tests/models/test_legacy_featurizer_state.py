"""Tests for translating legacy flat featurizer state."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.models._legacy_featurizer_state import (
    collect_legacy_concat_state,
    restore_legacy_concat_state,
)


def test_concat_legacy_state_round_trip() -> None:
    """Legacy preprocessing attributes map to concat children and back."""
    register_builtin_components()
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            "scaledGeneExpression",
            {"name": "pca", "view": "methylation", "hyperparameters": {"n_components": 1}},
        ]
    )
    scaler = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    pca = PCA(n_components=1).fit([[1.0, 2.0], [3.0, 4.0]])

    restore_legacy_concat_state(
        featurizer,
        {
            "gene_expression_scaler": scaler,
            "methylation_pca": pca,
            "view_dims": {"gene_expression": 2, "methylation": 1},
            "output_dim": 3,
            "fitted": True,
        },
    )

    restored = collect_legacy_concat_state(featurizer)
    assert restored["gene_expression_scaler"] is scaler
    assert restored["methylation_pca"] is pca
    assert restored["view_dims"] == {"gene_expression": 2, "methylation": 1}
    assert restored["output_dim"] == 3
    assert restored["fitted"] is True
