"""Load feature tables from CSV into FeatureDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER


def load_generic_csv(
    path: str | Path, dataset_name: str, feature_name: str, index_col=CELL_LINE_IDENTIFIER
) -> FeatureDataset:
    """Loads a generic CSV file with cell line IDs as index and features as columns.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param feature_name: name of the feature, e.g., gene_expression
    :param index_col: name of the index column, e.g., cell_line_id
    :returns: FeatureDataset with the features
    """
    feature_csv = pd.read_csv(Path(path) / dataset_name / f"{feature_name}.csv", index_col=index_col)
    feature_csv.index = feature_csv.index.astype(str)
    if "cellosaurus_id" in feature_csv.columns:
        feature_csv = feature_csv.drop(columns=["cellosaurus_id"])
    return FeatureDataset(
        features=iterate_features(df=feature_csv, feature_type=feature_name),
        meta_info={feature_name: feature_csv.columns.values},
    )


def iterate_features(df: pd.DataFrame, feature_type: str) -> dict[str, dict[str, np.ndarray]]:
    """Iterate over features.

    :param df: DataFrame with the features
    :param feature_type: type of feature, e.g., gene_expression, methylation, etc.
    :returns: dictionary with the features
    """
    features: dict[str, dict[str, np.ndarray]] = {}
    for cl in df.index:
        if cl in features.keys():
            continue
        rows = df.loc[cl]
        rows = rows.astype(float).to_numpy()
        if (len(rows.shape) > 1) and (rows.shape[0] > 1):  # multiple rows returned
            # take mean
            rows = np.mean(rows, axis=0)
        features[cl] = {feature_type: rows}
    return features


def load_cl_ids_from_csv(path: str | Path, dataset_name: str) -> FeatureDataset:
    """Load cell line ids from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the cell line ids
    """
    cl_names = pd.read_csv(Path(path) / dataset_name / "cell_line_names.csv", index_col=CELL_LINE_IDENTIFIER)
    cl_names.index = cl_names.index.astype(str)
    return FeatureDataset(features={cl: {CELL_LINE_IDENTIFIER: np.array([cl])} for cl in cl_names.index})


def load_tissues_from_csv(path: str | Path, dataset_name: str) -> FeatureDataset:
    """Load tissues from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the tissues
    """
    tissues = pd.read_csv(
        Path(path) / dataset_name / "cell_line_names.csv", index_col=CELL_LINE_IDENTIFIER
    ).drop_duplicates()
    return FeatureDataset(
        features={cl: {TISSUE_IDENTIFIER: np.array([tissues.loc[cl, TISSUE_IDENTIFIER]])} for cl in tissues.index}
    )


def load_cl_ids_and_tissues_from_csv(path: str | Path, dataset_name: str) -> FeatureDataset:
    """Load cell line ids and optional tissue annotations from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with cell line ids and tissue annotations, if available
    """
    cl_ids = load_cl_ids_from_csv(path, dataset_name)
    try:
        cl_ids.add_features(load_tissues_from_csv(path, dataset_name))
    except KeyError:
        pass
    return cl_ids


def load_drug_ids_from_csv(data_path: str | Path, dataset_name: str) -> FeatureDataset:
    """Load drug ids from csv file.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the drug ids
    """
    drug_names = pd.read_csv(
        Path(data_path) / dataset_name / "drug_names.csv",
        index_col=DRUG_IDENTIFIER,
        dtype={"pubchem_id": str},
        low_memory=False,
    )
    drug_names.index = drug_names.index.astype(str)
    return FeatureDataset(features={drug: {DRUG_IDENTIFIER: np.array([drug])} for drug in drug_names.index})


def load_drug_fingerprint_features(
    data_path: str | Path, dataset_name: str, fill_na=True, n_bits=128
) -> FeatureDataset:
    """Load drug features from fingerprints.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param fill_na: whether to use default pubchemid-hashed fingerprints if fingerprint is not available
    :param n_bits: number of bits in the fingerprint
    :returns: FeatureDataset with the drug fingerprints
    """
    fingerprints = pd.read_csv(
        Path(data_path) / dataset_name / "drug_fingerprints" / f"pubchem_id_to_demorgan_{n_bits}_map.csv",
        index_col=None,
    ).T
    if fill_na:
        for drug in fingerprints.index:
            if (
                not fingerprints.loc[drug].isna().all()
            ):  # if all values are NaN, replace with random fingerprint for the drug
                continue
            # Create random fingerprint for the drug, which is based on a hash of the pubchemid
            rng = np.random.default_rng(hash(drug) % (2**32))
            fingerprints.loc[drug] = rng.integers(0, 2, size=fingerprints.loc[drug].shape)

    return FeatureDataset(
        features={drug: {"fingerprints": fingerprints.loc[drug].values} for drug in fingerprints.index}
    )


def unique(array):
    """Get unique values ordered by first occurrence.

    :param array: array of values
    :returns: unique values ordered by first occurrence
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
