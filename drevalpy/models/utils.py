"""Utility functions for loading and processing data."""

import os.path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER


def load_cl_ids_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load cell line ids from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the cell line ids
    """
    cl_names = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=1)
    return FeatureDataset(features={cl: {CELL_LINE_IDENTIFIER: np.array([cl])} for cl in cl_names.index})


def load_tissues_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load tissues from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the tissues
    """
    tissues = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=1).drop_duplicates()
    return FeatureDataset(
        features={cl: {TISSUE_IDENTIFIER: np.array([tissues.loc[cl, "tissue"]])} for cl in tissues.index}
    )


def load_and_select_gene_features(
    feature_type: str,
    gene_list: str | None,
    data_path: str,
    dataset_name: str,
) -> FeatureDataset:
    """
    Load and reduce features of a single feature type, ensuring selection and ordering based on the gene list.

    Attention: if gene_list is None, all features are loaded, which can be problematic for cross study prediction.

    :param feature_type: type of feature, e.g., gene_expression, methylation, etc.
    :param gene_list: list of genes to include, e.g., landmark_genes
    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the reduced features
    :raises ValueError: if genes from gene_list are missing in the dataset
    """
    ge = pd.read_csv(f"{data_path}/{dataset_name}/{feature_type}.csv", index_col=1)
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
    ordered_genes = gene_info["Symbol"].tolist()

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

    indices_to_keep = [i for i, gene in enumerate(cl_features.meta_info[feature_type]) if gene in ordered_genes]

    cl_features.meta_info[feature_type] = np.array(ordered_genes)

    for cell_line in cl_features.features.keys():
        cl_features.features[cell_line][feature_type] = cl_features.features[cell_line][feature_type][indices_to_keep]

    return cl_features


def iterate_features(df: pd.DataFrame, feature_type: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Iterate over features.

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


def load_drug_ids_from_csv(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug ids from csv file.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the drug ids
    """
    drug_names = pd.read_csv(f"{data_path}/{dataset_name}/drug_names.csv", index_col=0)
    drug_names.index = drug_names.index.astype(str)
    return FeatureDataset(features={drug: {DRUG_IDENTIFIER: np.array([drug])} for drug in drug_names.index})


def load_drug_fingerprint_features(data_path: str, dataset_name: str, fill_na=True, n_bits=128) -> FeatureDataset:
    """
    Load drug features from fingerprints.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param fill_na: whether to use default pubchemid-hashed fingerprints if fingerprint is not available
    :param n_bits: number of bits in the fingerprint
    :returns: FeatureDataset with the drug fingerprints
    """
    fingerprints = pd.read_csv(
        os.path.join(data_path, dataset_name, "drug_fingerprints", f"pubchem_id_to_demorgan_{n_bits}_map.csv"),
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


def get_multiomics_feature_dataset(
    data_path: str,
    dataset_name: str,
    gene_lists: dict | None = None,
    omics: list[str] | None = None,
) -> FeatureDataset:
    """
    Get multiomics feature dataset for the given list of OMICs.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param gene_lists: dictionary of names of lists of genes to include, for each omics type,
                e.g., {"gene_expression": "landmark_genes"}, if None, all features are not reduced
    :param omics: list of omics to include, e.g., ["gene_expression", "methylation"]
    :returns: FeatureDataset with the multiomics features
    :raises ValueError: if no omics features are found
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


def unique(array):
    """
    Get unique values ordered by first occurrence.

    :param array: array of values
    :returns: unique values ordered by first occurrence
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def log10_and_set_na(x):
    """
    Log10 transform and set NaN for infinite values.

    :param x: input array
    :returns: log10 transformed array with NaN for infinite values
    """
    x = np.log10(x)
    x[np.isinf(x)] = np.nan
    return x


class ProteomicsMedianCenterAndImputeTransformer(BaseEstimator, TransformerMixin):
    """Performs median centering and imputation of proteomics data."""

    def __init__(self, n_features=1000, normalization_downshift=1.8, normalization_width=0.3):
        """
        Hyperparameters for the normalization.

        :param n_features: number of features to select
        :param normalization_downshift: downshift factor for the mean
        :param normalization_width: width factor for the standard deviation
        """
        self.n_features = n_features
        self.normalization_downshift = normalization_downshift
        self.normalization_width = normalization_width
        self.protein_indices = np.array([])
        self.mean_median = 0

    def fit(self, X, y=None):
        """
        Learns the top n_feature complete proteins and calculates the mean median of the train cell lines.

        :param X: input proteomics data
        :param y: not used
        :returns: self
        """
        # identify the top 1000 complete columns
        completeness = np.sum(~np.isnan(X), axis=0)
        # sort by completeness
        sorted_indices = np.argsort(completeness)[::-1]
        self.protein_indices = sorted_indices[: self.n_features]
        X = X[:, self.protein_indices]
        # calculate mean of sample medians
        medians = np.nanmedian(X, axis=1)
        self.mean_median = np.nanmean(medians)
        return self

    def transform(self, X):
        """
        Median center the data and impute missing values with downshifted normal distribution.

        :param X: input proteomics data
        :returns: transformed proteomics data
        """
        X = X[0]
        X = X[self.protein_indices]
        correction_factor = self.mean_median / np.nanmedian(X)
        X = X * correction_factor
        # downshifted mean
        np.random.seed(seed=100)
        cell_line_mean = np.nanmean(X)
        cell_line_sd = np.nanstd(X)
        downshifted_mean = cell_line_mean - (self.normalization_downshift * cell_line_sd)
        shrinked_sd = self.normalization_width * cell_line_sd
        n_missing = np.count_nonzero(np.isnan(X))
        X[np.isnan(X)] = np.random.normal(loc=downshifted_mean, scale=shrinked_sd, size=n_missing)
        return [X]
