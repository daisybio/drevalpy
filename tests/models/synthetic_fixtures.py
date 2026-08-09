"""Tiny in-memory fixtures for model execution gates."""

from __future__ import annotations

import numpy as np

from drevalpy.data.structures import SplitMask, SplitMasks
from drevalpy.data.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.types.response_batch import ResponseBatch
from tests.conftest import MockFeatureSource


def multi_drug_response() -> ResponseBatch:
    return ResponseBatch(
        response=np.array([1.0, 2.0, 3.0, 4.0]),
        cell_line_ids=np.array(["cl1", "cl1", "cl2", "cl2"]),
        drug_ids=np.array(["d1", "d2", "d1", "d2"]),
    )


def cell_line_gene_expression() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.1, 0.2, 0.3])},
            "cl2": {"gene_expression": np.array([0.4, 0.5, 0.6])},
        }
    )


def drug_fingerprints() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "d1": {"fingerprints": np.array([1.0, 0.0])},
            "d2": {"fingerprints": np.array([0.0, 1.0])},
        }
    )


def identity_cell_line_features(*, with_tissue: bool = False) -> MockFeatureSource:
    features = {
        "cl1": {CELL_LINE_IDENTIFIER: np.array(["cl1"])},
        "cl2": {CELL_LINE_IDENTIFIER: np.array(["cl2"])},
    }
    if with_tissue:
        features["cl1"][TISSUE_IDENTIFIER] = np.array(["Lung"])
        features["cl2"][TISSUE_IDENTIFIER] = np.array(["Blood"])
    return MockFeatureSource(features=features)


def identity_drug_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "d1": {DRUG_IDENTIFIER: np.array(["d1"])},
            "d2": {DRUG_IDENTIFIER: np.array(["d2"])},
        }
    )


def synthetic_mudataset_gene_expression_fingerprints():
    """Build a minimal Dataset with gene_expression + fingerprints for 2 cell lines and 2 drugs."""
    import anndata as ad
    import pandas as pd

    import mudata as md
    from drevalpy.types.dataset import Dataset

    # Response matrix: 2 cell lines x 2 drugs
    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])

    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    # Gene expression modality: 2 cell lines x 3 genes
    ge_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    gene_expression_ad = ad.AnnData(
        X=ge_matrix,
        obs=pd.DataFrame(index=cl_ids),
        var=pd.DataFrame(index=[f"gene{i}" for i in range(3)]),
    )
    # Fingerprints in response.varm
    fingerprints = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    response_ad.varm["fingerprints"] = fingerprints

    mdata = md.MuData({"response": response_ad, "gene_expression": gene_expression_ad})
    return Dataset(mdata, name="test")


def synthetic_mudataset_identity():
    """Build a minimal Dataset for identity (cell_line_id + drug_id) models."""
    import anndata as ad
    import pandas as pd

    import mudata as md
    from drevalpy.types.dataset import Dataset

    response_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])

    response_ad = ad.AnnData(
        X=response_matrix,
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    mdata = md.MuData({"response": response_ad})
    return Dataset(mdata, name="test")


def _mask_2x2(*positions: tuple[int, int]) -> SplitMask:
    """Build a 2x2 SplitMask with True at the given (row, col) positions."""
    mask = np.zeros((2, 2), dtype=bool)
    for r, c in positions:
        mask[r, c] = True
    return SplitMask(mask)


def lpo_split_masks_all_train() -> SplitMasks:
    """LPO-style masks: all 4 pairs as train, (0,0) as test, none as val."""
    return SplitMasks(
        train=_mask_2x2((0, 0), (0, 1), (1, 0), (1, 1)),
        test=_mask_2x2((0, 0)),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )


def lco_split_masks() -> SplitMasks:
    """LCO-style masks: cl0 pairs in train, cl1 pairs in test, no val."""
    return SplitMasks(
        train=_mask_2x2((0, 0), (0, 1)),
        test=_mask_2x2((1, 0), (1, 1)),
        val=SplitMask(np.zeros((2, 2), dtype=bool)),
    )
