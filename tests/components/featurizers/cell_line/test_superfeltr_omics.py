"""Tests for SuperFELTR omics preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.superfeltr_omics import SuperFELTROmicsFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def test_superfeltr_omics_selects_each_view_and_round_trips_state() -> None:
    features = FeatureDataset(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            view: [f"{view}{i}" for i in range(3)]
            for view in (
                "gene_expression",
                "mutations",
                "copy_number_variation_gistic",
            )
        },
    )
    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features, entity_ids=np.array(["cl0", "cl1"]))
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert all(block.values.shape == (2, 2) for block in blocks.values())
    restored = SuperFELTROmicsFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)
