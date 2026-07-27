"""Tests for MOLIR utility helpers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.literature.impl.molir.utils import filter_and_sort_omics
from drevalpy.datasets.dataset import FeatureDataset


class _OmicModelStub:
    def __init__(self) -> None:
        self.gene_expression_features = np.array(["g1", "g2"])
        self.mutations_features = np.array(["m1"])
        self.copy_number_variation_features = np.array(["c2", "c1"])


def test_filter_and_sort_omics_realigns_columns_and_fills_missing() -> None:
    model = _OmicModelStub()
    gene_expression = np.array([[1.0, 2.0, 3.0]])
    mutations = np.array([[4.0, 5.0]])
    cnvs = np.array([[6.0, 7.0, 8.0]])

    cell_line_input = FeatureDataset(
        features={
            "cl1": {
                "gene_expression": gene_expression[0],
                "mutations": mutations[0],
                "copy_number_variation_gistic": cnvs[0],
            }
        },
        meta_info={
            "gene_expression": np.array(["g0", "g1", "g2"]),
            "mutations": np.array(["m0", "m1"]),
            "copy_number_variation_gistic": np.array(["c0", "c1", "c2"]),
        },
    )

    gex, mut, cnv = filter_and_sort_omics(
        model=model,
        gene_expression=gene_expression,
        mutations=mutations,
        cnvs=cnvs,
        cell_line_input=cell_line_input,
    )

    assert gex.shape == (1, 2)
    np.testing.assert_allclose(gex, [[2.0, 3.0]])
    assert mut.shape == (1, 1)
    np.testing.assert_allclose(mut, [[5.0]])
    assert cnv.shape == (1, 2)
    np.testing.assert_allclose(cnv, [[8.0, 7.0]])
