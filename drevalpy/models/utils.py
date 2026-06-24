"""Utility functions for loading and processing data (compatibility re-exports).

Prefer :mod:`drevalpy.data.features` and :mod:`drevalpy.data.preprocessing` for new code.
"""

from drevalpy.data.features import (
    _get_view_as_list,
    get_multiomics_feature_dataset,
    iterate_features,
    load_and_select_gene_features,
    load_cl_ids_and_tissues_from_csv,
    load_cl_ids_from_csv,
    load_drug_fingerprint_features,
    load_drug_ids_from_csv,
    load_generic_csv,
    load_multi_cell_line_view,
    load_single_cell_line_view,
    load_single_drug_view,
    load_tissues_from_csv,
    unique,
)
from drevalpy.data.preprocessing import (
    ProteomicsMedianCenterAndImputeTransformer,
    VarianceFeatureSelector,
    log10_and_set_na,
    prepare_expression_and_methylation,
    prepare_proteomics,
    scale_gene_expression,
)
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER

__all__ = [
    "CELL_LINE_IDENTIFIER",
    "DRUG_IDENTIFIER",
    "TISSUE_IDENTIFIER",
    "ProteomicsMedianCenterAndImputeTransformer",
    "VarianceFeatureSelector",
    "_get_view_as_list",
    "get_multiomics_feature_dataset",
    "iterate_features",
    "load_and_select_gene_features",
    "load_cl_ids_and_tissues_from_csv",
    "load_cl_ids_from_csv",
    "load_drug_fingerprint_features",
    "load_drug_ids_from_csv",
    "load_generic_csv",
    "load_multi_cell_line_view",
    "load_single_cell_line_view",
    "load_single_drug_view",
    "load_tissues_from_csv",
    "log10_and_set_na",
    "prepare_expression_and_methylation",
    "prepare_proteomics",
    "scale_gene_expression",
    "unique",
]
