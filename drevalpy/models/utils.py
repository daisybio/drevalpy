"""Utility functions for loading and processing data."""

import os.path
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset


def load_cl_ids_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load cell line ids from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the cell line ids
    """
    cl_names = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=1)
    return FeatureDataset(features={cl: {"cell_line_id": np.array([cl])} for cl in cl_names.index})


def load_and_reduce_gene_features(
    feature_type: str,
    gene_list: Optional[str],
    data_path: str,
    dataset_name: str,
) -> FeatureDataset:
    """
    Load and reduce features of a single feature type.

    :param feature_type: type of feature, e.g., gene_expression, methylation, etc.
    :param gene_list: list of genes to include, e.g., landmark_genes
    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the reduced features
    :raises ValueError: if genes from gene_list are missing in the dataset
    """
    ge = pd.read_csv(f"{data_path}/{dataset_name}/{feature_type}.csv", index_col=1)
    # remove column
    ge = ge.drop(columns=["cellosaurus_id"])
    cl_features = FeatureDataset(
        features=iterate_features(df=ge, feature_type=feature_type),
        meta_info={feature_type: ge.columns.values},
    )
    if gene_list is None:
        return cl_features

    gene_info = pd.read_csv(
        f"{data_path}/{dataset_name}/gene_lists/{gene_list}.csv",
        sep=",",
    )

    genes_in_list = set(gene_info["Symbol"])
    if cl_features.meta_info is None:
        raise ValueError("No meta information available in the dataset.")

    genes_in_features = set(cl_features.meta_info[feature_type])
    # Ensure that all genes from gene_list are in the dataset
    missing_genes = genes_in_list - genes_in_features
    if missing_genes:
        missing_genes_list = list(missing_genes)
        if len(missing_genes_list) > 10:
            raise ValueError(
                f"The following genes are missing from the dataset {dataset_name} for {feature_type}: "
                f"{', '.join(missing_genes_list[:10])}, ... ({len(missing_genes)} genes in total)"
            )
        else:
            raise ValueError(
                f"The following genes are missing from the dataset {dataset_name} for {feature_type}: "
                f"{', '.join(missing_genes_list)}"
            )

    # Only proceed with genes that are available
    gene_mask = np.array([gene in genes_in_list for gene in cl_features.meta_info[feature_type]])
    cl_features.meta_info[feature_type] = cl_features.meta_info[feature_type][gene_mask]
    for cell_line in cl_features.features.keys():
        cl_features.features[cell_line][feature_type] = cl_features.features[cell_line][feature_type][gene_mask]
    return cl_features


def iterate_features(df: pd.DataFrame, feature_type: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Iterate over features.

    :param df: DataFrame with the features
    :param feature_type: type of feature, e.g., gene_expression, methylation, etc.
    :returns: dictionary with the features
    """
    features = {}
    for cl in df.index:
        rows = df.loc[cl]
        if len(rows.shape) > 1 and rows.shape[0] > 1:  # multiple rows returned
            warnings.warn(
                f"Multiple rows returned for {cl} in feature {feature_type}, taking the first one.", stacklevel=2
            )
            rows = rows.iloc[0]
        # convert to float values
        rows = rows.astype(float)
        features[cl] = {feature_type: rows.values}
    return features


def load_drug_ids_from_csv(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug ids from csv file.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the drug ids
    """
    drug_names = pd.read_csv(f"{data_path}/{dataset_name}/drug_names.csv", index_col=0)
    return FeatureDataset(features={drug: {"drug_id": np.array([drug])} for drug in drug_names.index})


def load_drug_fingerprint_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug features from fingerprints.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the drug fingerprints
    """
    if dataset_name == "Toy_Data":
        fingerprints = pd.read_csv(os.path.join(data_path, dataset_name, "fingerprints.csv"), index_col=0)
    else:
        fingerprints = pd.read_csv(
            os.path.join(data_path, dataset_name, "drug_fingerprints", "drug_name_to_demorgan_128_map.csv"),
            index_col=0,
        ).T
    return FeatureDataset(
        features={drug: {"fingerprints": fingerprints.loc[drug].values} for drug in fingerprints.index}
    )


def get_multiomics_feature_dataset(
    data_path: str,
    dataset_name: str,
    gene_list: Optional[str] = "drug_target_genes_all_drugs",
    omics: Optional[list[str]] = None,
) -> FeatureDataset:
    """
    Get multiomics feature dataset for the given list of OMICs.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param gene_list: list of genes to include, e.g., landmark_genes
    :param omics: list of omics to include, e.g., ["gene_expression", "methylation"]
    :returns: FeatureDataset with the multiomics features
    :raises ValueError: if no omics features are found
    """
    if omics is None:
        omics = ["gene_expression", "methylation", "mutations", "copy_number_variation_gistic"]
    feature_dataset = None
    for omic in omics:
        if feature_dataset is None:
            feature_dataset = load_and_reduce_gene_features(
                feature_type=omic,
                gene_list=None if omic == "methylation" else gene_list,
                data_path=data_path,
                dataset_name=dataset_name,
            )
        else:
            feature_dataset.add_features(
                load_and_reduce_gene_features(
                    feature_type=omic,
                    gene_list=None if omic == "methylation" else gene_list,
                    data_path=data_path,
                    dataset_name=dataset_name,
                )
            )
    if feature_dataset is None:
        raise ValueError("No omics features found.")
    return feature_dataset


def unique(array):
    """
    Get unique values ordered by first occurrence.

    :param array: array of values
    :returns: unique values ordered by first occurrence
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
