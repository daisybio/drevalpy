"""Legacy view-string loaders for cell-line and drug feature tables."""

from __future__ import annotations

import logging

from drevalpy.components.data_loading.multiomics import get_multiomics_feature_dataset, load_and_select_gene_features
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import (
    load_drug_fingerprint_features,
    load_drug_ids_from_csv,
    load_generic_csv,
)
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER

logger = logging.getLogger(__name__)


def load_cell_line_feature_views(
    views: list[str],
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset:
    """Load cell-line features for the configured cell-line views."""
    if len(views) == 1:
        return load_single_cell_line_view(views, data_path, dataset_name, model_name)
    return load_multi_cell_line_view(views, data_path, dataset_name, model_name)


def load_drug_feature_views(
    views: list[str],
    data_path: str,
    dataset_name: str,
    *,
    model_name: str = "DRPModel",
) -> FeatureDataset | None:
    """Load drug features for the configured drug views."""
    if not views:
        return None
    return load_single_drug_view(views, data_path, dataset_name, model_name)


def load_single_cell_line_view(
    cell_line_views: list[str],
    data_path: str,
    dataset_name: str,
    model_name: str,
) -> FeatureDataset:
    """
    Load cell line features for a single-view model.

    If the view is "gene_expression", the landmark_genes_reduced list is used for subsetting.
    Otherwise, the whole CSV is loaded.

    :param cell_line_views: list of cell line views (must have exactly one element)
    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC1
    :param model_name: name of the model, used for error messages
    :returns: FeatureDataset containing the cell line features
    :raises ValueError: if cell_line_views is empty or has more than one element
    """
    if len(cell_line_views) == 0:
        raise ValueError(
            "cell_line_views is empty. Construct the model (Model() or "
            "Model(hyperparameters)) before load_cell_line_features() so the "
            "model knows which omics to load."
        )
    if len(cell_line_views) > 1:
        raise ValueError(f"Only one cell line view is supported for {model_name}.")
    logger.debug("Loading a %s with the following cell line views: %s", model_name, cell_line_views)

    if "gene_expression" in cell_line_views:
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )
    else:
        return load_generic_csv(
            path=data_path,
            dataset_name=dataset_name,
            feature_name=cell_line_views[0],
            index_col=CELL_LINE_IDENTIFIER,
        )


def load_multi_cell_line_view(
    cell_line_views: list[str],
    data_path: str,
    dataset_name: str,
    model_name: str,
) -> FeatureDataset:
    """
    Load cell line features for a multi-view model.

    Known omics types use specific gene lists for subsetting. Unknown types are loaded in full.

    :param cell_line_views: list of cell line views
    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC1
    :param model_name: name of the model, used for error messages
    :returns: FeatureDataset containing the cell line features
    :raises ValueError: if cell_line_views is empty
    """
    if len(cell_line_views) == 0:
        raise ValueError(
            "cell_line_views is empty. Construct the model (Model() or "
            "Model(hyperparameters)) before load_cell_line_features() so the "
            "model knows which omics to load."
        )
    logger.debug("Loading a %s with the following cell line views: %s", model_name, cell_line_views)

    gene_list_defaults = {
        "gene_expression": "drug_target_genes_all_drugs",
        "methylation": "methylation_intersection",
        "mutations": "drug_target_genes_all_drugs",
        "copy_number_variation_gistic": "drug_target_genes_all_drugs",
        "proteomics": "drug_target_genes_all_drugs_proteomics",
    }
    gene_lists = {feature_name: gene_list_defaults.get(feature_name, None) for feature_name in cell_line_views}

    return get_multiomics_feature_dataset(
        data_path=data_path, gene_lists=gene_lists, dataset_name=dataset_name, omics=cell_line_views
    )


def load_single_drug_view(
    drug_views: list[str],
    data_path: str,
    dataset_name: str,
    model_name: str,
) -> FeatureDataset | None:
    """
    Load drug features for a single-view model.

    If drug_views is empty, drug IDs are loaded. If "fingerprints", fingerprints are loaded.
    Otherwise, the CSV is loaded generically.

    :param drug_views: list of drug views (at most one element)
    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC1
    :param model_name: name of the model, used for error messages
    :returns: FeatureDataset containing the drug features
    :raises ValueError: if more than one drug view is specified
    """
    if len(drug_views) > 1:
        raise ValueError(f"Only one drug view is supported for {model_name}.")
    logger.debug("Loading a %s with the following drug views: %s", model_name, drug_views)

    if len(drug_views) == 0:
        return load_drug_ids_from_csv(data_path, dataset_name)
    elif drug_views[0] == "fingerprints":
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)
    else:
        return load_generic_csv(
            path=data_path, dataset_name=dataset_name, feature_name=drug_views[0], index_col=DRUG_IDENTIFIER
        )
