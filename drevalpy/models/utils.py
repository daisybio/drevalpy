"""
Utility functions for loading and processing data.
"""

import os.path
import warnings
from typing import Optional
import pickle
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import TransformerMixin
from drevalpy.datasets.dataset import FeatureDataset


def load_cl_ids_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load cell line ids from csv file.
    :param path:
    :param dataset_name:
    :return:
    """
    if dataset_name == "Toy_Data":
        return load_toy_features(path, dataset_name, "cell_line")

    cl_names = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=0)
    return FeatureDataset(
        features={cl: {"cell_line_id": np.array([cl])} for cl in cl_names.index}
    )


def load_and_reduce_gene_features(
    feature_type: str, gene_list: Optional[str], data_path: str, dataset_name: str
) -> FeatureDataset:
    """
    Load and reduce gene features.
    :param feature_type:
    :param gene_list:
    :param data_path:
    :param dataset_name:
    :return:
    """
    if dataset_name == "Toy_Data":
        cl_features = load_toy_features(data_path, dataset_name, "cell_line")
        dataset_name = "GDSC1"
    else:
        ge = pd.read_csv(f"{data_path}/{dataset_name}/{feature_type}.csv", index_col=0)
        cl_features = FeatureDataset(
            features=iterate_features(df=ge, feature_type=feature_type),
            meta_info={feature_type: ge.columns.values},
        )
    if gene_list is None:
        return cl_features

    gene_info = pd.read_csv(
        f"{data_path}/{dataset_name}/gene_lists/{gene_list}.csv",
        sep=("\t" if gene_list == "landmark_genes" else ","),
    )
    genes_to_use = set(gene_info["Symbol"]) & set(cl_features.meta_info[feature_type])
    gene_mask = np.array(
        [gene in genes_to_use for gene in cl_features.meta_info[feature_type]]
    )
    cl_features.meta_info[feature_type] = cl_features.meta_info[feature_type][gene_mask]
    for cell_line in cl_features.features.keys():
        cl_features.features[cell_line][feature_type] = cl_features.features[cell_line][
            feature_type
        ][gene_mask]
    return cl_features


def iterate_features(df: pd.DataFrame, feature_type: str):
    """
    Iterate over features.
    :param df:
    :param feature_type:
    :return:
    """
    features = {}
    for cl in df.index:
        rows = df.loc[cl]
        if len(rows.shape) > 1 and rows.shape[0] > 1:  # multiple rows returned
            warnings.warn(
                f"Multiple rows returned for {cl} in feature {feature_type}, taking the first one."
            )
            features[cl] = {feature_type: rows.iloc[0].values}
        else:
            features[cl] = {feature_type: rows.values}
    return features


def load_drug_ids_from_csv(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug ids from csv file.
    :param data_path:
    :param dataset_name:
    :return:
    """
    if dataset_name == "Toy_Data":
        return load_toy_features(data_path, dataset_name, "drug")
    drug_names = pd.read_csv(f"{data_path}/{dataset_name}/drug_names.csv", index_col=0)
    return FeatureDataset(
        features={drug: {"drug_id": np.array([drug])} for drug in drug_names.index}
    )


def load_drug_fingerprint_features(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug features from fingerprints.
    :param data_path:
    :param dataset_name:
    :return:
    """
    if dataset_name == "Toy_Data":
        return load_toy_features(data_path, dataset_name, "drug")
    fingerprints = pd.read_csv(
        f"{data_path}/{dataset_name}/drug_fingerprints/drug_name_to_demorgan_128_map.csv",
        index_col=0,
    ).T
    return FeatureDataset(
        features={
            drug: {"fingerprints": fingerprints.loc[drug].values}
            for drug in fingerprints.index
        }
    )


def get_multiomics_feature_dataset(
    data_path: str, dataset_name: str, gene_list: str = "drug_target_genes_all_drugs"
) -> FeatureDataset:
    """
    Get multiomics feature dataset.
    :param data_path:
    :param dataset_name:
    :param gene_list:
    :return:
    """
    if dataset_name == "Toy_Data":
        return load_toy_features(data_path, dataset_name, "cell_line")

    ge_dataset = load_and_reduce_gene_features(
        feature_type="gene_expression",
        gene_list=gene_list,
        data_path=data_path,
        dataset_name=dataset_name,
    )
    me_dataset = load_and_reduce_gene_features(
        feature_type="methylation",
        gene_list=None,
        data_path=data_path,
        dataset_name=dataset_name,
    )
    mu_dataset = load_and_reduce_gene_features(
        feature_type="mutations",
        gene_list=gene_list,
        data_path=data_path,
        dataset_name=dataset_name,
    )
    cnv_dataset = load_and_reduce_gene_features(
        feature_type="copy_number_variation_gistic",
        gene_list=gene_list,
        data_path=data_path,
        dataset_name=dataset_name,
    )
    for fd in [me_dataset, mu_dataset, cnv_dataset]:
        ge_dataset.add_features(fd)
    return ge_dataset


def unique(array):
    """
    Get unique values ordered by first occurence.
    :param array:
    :return:
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def load_toy_features(
    data_path: str, dataset_name: str, feature: str
) -> FeatureDataset:
    """
    Load toy features.
    :param data_path: path to data passed via args
    :param dataset_name: should be Toy_Data
    :param feature: cell_line or drug
    :return:
    """
    assert dataset_name == "Toy_Data"
    assert feature in ["cell_line", "drug"]
    if feature == "cell_line":
        path_to_features = os.path.join(
            data_path, dataset_name, "toy_data_cl_features.pkl"
        )
    else:
        path_to_features = os.path.join(
            data_path, dataset_name, "toy_data_drug_features.pkl"
        )
    with open(path_to_features, "rb") as f:
        features = pickle.load(f)
    return features
