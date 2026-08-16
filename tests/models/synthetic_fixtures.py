"""Tiny in-memory fixtures for model execution gates."""

from __future__ import annotations

import numpy as np

from drevalpy.data.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.batch.response_batch import ResponseBatch
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


#: Column-name prefix and value offset (in tenths) for each cell-line view. The
#: offset only has to make the views distinguishable; ``gene_expression``'s value
#: of 1 is what reproduces the matrix this module has always built.
_VIEW_SPECS = {
    "gene_expression": ("gene", 1),
    "methylation": ("cpg", 2),
    "mutations": ("mut", 3),
    "copy_number_variation_gistic": ("cnv", 4),
    "proteomics": ("prot", 5),
}


def _view_matrix(width: int, offset: int) -> np.ndarray:
    """Build a deterministic 2-row matrix, shifted by *offset* tenths.

    :param width: Number of columns.
    :param offset: Added to every element before scaling, in tenths.
    :returns: ``(2, width)`` float32 matrix.
    """
    return ((np.arange(2 * width, dtype=np.float32) + offset) / 10.0).reshape(2, width)


def synthetic_mudataset(
    *,
    n_features_per_view: int = 3,
    fingerprint_width: int = 2,
    extra_views: tuple[str, ...] = (),
):
    """Build a two-cell-line, two-drug ``Dataset`` with the requested views.

    The one builder behind every synthetic ``Dataset`` in the suite. Callers that
    needed a wider gene-expression matrix or additional omics modalities used to
    write out their own AnnData/MuData assembly, which is what made those test
    files read as clones of each other.

    :param n_features_per_view: Columns in each cell-line modality.
    :param fingerprint_width: Columns of the ``morgan_fingerprint`` in ``response.varm``.
    :param extra_views: Further cell-line modalities to attach, named after the
        views a featurizer reads; see :data:`_VIEW_SPECS`.
    :returns: The assembled ``Dataset``.
    """
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset

    cl_ids = np.array(["cl1", "cl2"])
    drug_ids = np.array(["d1", "d2"])
    response_ad = ad.AnnData(
        X=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        obs=pd.DataFrame({"cell_line_name": cl_ids, "tissue": ["Lung", "Blood"]}, index=cl_ids),
        var=pd.DataFrame(index=drug_ids),
    )
    response_ad.varm["morgan_fingerprint"] = np.eye(2, fingerprint_width, dtype=np.float32)

    modalities = {"response": response_ad}
    for view in ("gene_expression", *extra_views):
        prefix, offset = _VIEW_SPECS[view]
        modalities[view] = ad.AnnData(
            X=_view_matrix(n_features_per_view, offset),
            obs=pd.DataFrame(index=cl_ids),
            var=pd.DataFrame(index=[f"{prefix}{i}" for i in range(n_features_per_view)]),
        )
    return Dataset(md.MuData(modalities), name="test")


def synthetic_mudataset_gene_expression_fingerprints():
    """Build a minimal Dataset with gene_expression + fingerprints for 2 cell lines and 2 drugs.

    :returns: A ``Dataset`` with a 3-gene expression modality and 2-wide fingerprints.
    """
    return synthetic_mudataset()


def synthetic_mudataset_identity():
    """Build a minimal Dataset for identity (cell_line_id + drug_id) models."""
    import anndata as ad
    import mudata as md
    import pandas as pd

    from drevalpy.types.data.dataset import Dataset

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
