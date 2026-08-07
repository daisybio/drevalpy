"""Multi-omics feature loading with optional gene-list subsetting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import iterate_features
from drevalpy.datasets.gene_lists import gene_names_from_list_csv, resolve_gene_list_path
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


def load_and_select_gene_features(
    feature_type: str,
    gene_list: str | None,
    data_path: str | Path,
    dataset_name: str,
) -> FeatureDataset:
    """Load and reduce features of a single feature type.

    When *gene_list* is ``None``, all features are loaded, which can be
    problematic for cross-study prediction.

    :param feature_type: Feature view name, for example ``gene_expression``.
    :param gene_list: Optional gene-list CSV name used for subsetting and ordering.
    :param data_path: Root directory containing dataset feature tables.
    :param dataset_name: Dataset subdirectory or registry name.

    :returns: ``FeatureDataset`` with the selected features.

    :raises ValueError: If genes from *gene_list* are missing in the dataset.
    """
    ge = pd.read_csv(Path(data_path) / dataset_name / f"{feature_type}.csv", index_col=CELL_LINE_IDENTIFIER)
    ge.index = ge.index.astype(str)
    if "cellosaurus_id" in ge.columns:
        ge = ge.drop(columns=["cellosaurus_id"])

    cl_features = FeatureDataset(
        features=iterate_features(df=ge, feature_type=feature_type),
        meta_info={feature_type: ge.columns.values},
    )
    if gene_list is None:
        return cl_features

    ordered_genes = gene_names_from_list_csv(resolve_gene_list_path(gene_list, data_path=data_path))

    genes_in_features = set(cl_features.meta_info[feature_type])
    missing_genes = [gene for gene in ordered_genes if gene not in genes_in_features]

    if missing_genes:
        missing_str = (
            f"{', '.join(missing_genes[:10])}, ... ({len(missing_genes)} genes in total)"
            if len(missing_genes) > 10
            else ", ".join(missing_genes)
        )
        raise ValueError(
            f"The following genes are missing from the dataset {dataset_name} for {feature_type}: {missing_str}"
        )

    gene_to_idx = {str(gene): index for index, gene in enumerate(cl_features.meta_info[feature_type])}
    indices_to_keep = [gene_to_idx[str(gene)] for gene in ordered_genes]

    cl_features.meta_info[feature_type] = np.array(ordered_genes)

    for cell_line in cl_features.features.keys():
        cl_features.features[cell_line][feature_type] = cl_features.features[cell_line][feature_type][indices_to_keep]

    return cl_features


def get_multiomics_feature_dataset(
    data_path: str | Path,
    dataset_name: str,
    gene_lists: dict | None = None,
    omics: list[str] | None = None,
) -> FeatureDataset:
    """Get multiomics feature dataset for the given list of OMICs.

    :param data_path: Root directory containing dataset feature tables.
    :param dataset_name: Dataset subdirectory or registry name.
    :param gene_lists: Optional per-omics gene-list names; ``None`` loads all features for that omics type.
    :param omics: Omics view names to include.

    :returns: Combined ``FeatureDataset`` with every requested omics view.

    :raises ValueError: If gene-list keys do not match *omics* or no views load.
    """
    if omics is None:
        omics = ["gene_expression", "methylation", "mutations", "copy_number_variation_gistic", "proteomics"]

    if gene_lists is None:
        gene_lists = {o: None for o in omics}

    if not np.all([k in omics for k in gene_lists.keys()]):
        raise ValueError("Gene lists must be provided for all omics types.")

    feature_dataset = None
    for omic in omics:
        if feature_dataset is None:
            feature_dataset = load_and_select_gene_features(
                feature_type=omic,
                gene_list=gene_lists[omic],
                data_path=data_path,
                dataset_name=dataset_name,
            )
        else:
            feature_dataset.add_features(
                load_and_select_gene_features(
                    feature_type=omic,
                    gene_list=gene_lists[omic],
                    data_path=data_path,
                    dataset_name=dataset_name,
                )
            )
    if feature_dataset is None:
        raise ValueError("No omics features found.")
    return feature_dataset
