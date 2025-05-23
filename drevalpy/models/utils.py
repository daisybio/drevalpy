"""Utility functions for loading and processing data."""

import os.path

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
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


def prepare_expression_and_methylation(
    cell_line_input: FeatureDataset,
    cell_line_ids: np.ndarray,
    training: bool,
    gene_expression_scaler: TransformerMixin | None = None,
    methylation_scaler: TransformerMixin | None = None,
    methylation_pca: PCA | None = None,
):
    """
    Applies preprocessing to gene expression and optionally methylation views.

    - Applies arcsinh + scaling to gene expression if a scaler is provided.
    - Applies scaling + PCA to methylation if both a scaler and PCA are provided.
    - Applies to all cell lines in `cell_line_input`, using fitting only on the given IDs if training=True.

    :param cell_line_input: FeatureDataset with the cell line features
    :param cell_line_ids: IDs of the cell lines used for training or transformation
    :param training: Whether to fit the scalers/PCA (True) or just apply transformation (False)
    :param gene_expression_scaler: Optional fitted or to-be-fitted scaler for gene expression
    :param methylation_scaler: Optional fitted or to-be-fitted scaler for methylation
    :param methylation_pca: Optional PCA transformer for methylation
    """
    if gene_expression_scaler is not None:
        cell_line_input.apply(function=np.arcsinh, view="gene_expression")
        if training:
            cell_line_input.fit_transform_features(
                train_ids=cell_line_ids,
                transformer=gene_expression_scaler,
                view="gene_expression",
            )
        else:
            cell_line_input.transform_features(
                ids=cell_line_ids,
                transformer=gene_expression_scaler,
                view="gene_expression",
            )

    if methylation_scaler is not None and methylation_pca is not None:
        if training:
            cell_line_input.fit_transform_features(
                train_ids=cell_line_ids,
                transformer=methylation_scaler,
                view="methylation",
            )
            # Ensure the number of PCA components does not exceed the number of unique cell lines.
            methylation_pca.n_components = min(methylation_pca.n_components, len(np.unique(cell_line_ids)))
            cell_line_input.fit_transform_features(
                train_ids=cell_line_ids,
                transformer=methylation_pca,
                view="methylation",
            )
        else:
            cell_line_input.transform_features(
                ids=cell_line_ids,
                transformer=methylation_scaler,
                view="methylation",
            )
            cell_line_input.transform_features(
                ids=cell_line_ids,
                transformer=methylation_pca,
                view="methylation",
            )


def scale_gene_expression(
    cell_line_input: FeatureDataset,
    cell_line_ids: np.ndarray,
    training: bool,
    gene_expression_scaler: TransformerMixin,
):
    """
    Scales gene expression inplace using arcsinh transformation and a provided scaler.

    :param cell_line_input: FeatureDataset with the cell line features
    :param cell_line_ids: IDs of cell lines to use for fitting or transformation
    :param training: whether to fit or transform
    :param gene_expression_scaler: sklearn transformer for gene expression
    """
    prepare_expression_and_methylation(
        cell_line_input=cell_line_input,
        cell_line_ids=cell_line_ids,
        training=training,
        gene_expression_scaler=gene_expression_scaler,
    )


class VarianceFeatureSelector:
    """
    Selects the top-k features with highest variance for a specific omics view.

    Stores a boolean mask after fitting on training data and applies it
    consistently to other datasets.
    """

    def __init__(self, view: str, k: int = 1000):
        """
        Initialize the selector.

        :param view: omics view to select from, e.g., "gene_expression"
        :param k: number of top-variance features to retain
        """
        self.view = view
        self.k = k
        self.mask: np.ndarray = np.array([])
        self.selected_meta_info: list[str] = []

    def fit(self, cell_line_input: FeatureDataset, output: DrugResponseDataset) -> None:
        """
        Fit the selector to the training data by computing a variance-based mask.

        :param cell_line_input: FeatureDataset containing omics features
        :param output: DrugResponseDataset with the training cell line IDs
        """
        train_features = np.vstack(
            [cell_line_input.features[identifier][self.view] for identifier in np.unique(output.cell_line_ids)]
        )
        variances = np.var(train_features, axis=0)
        self.mask = np.zeros(len(variances), dtype=bool)
        self.mask[np.argsort(variances)[::-1][: self.k]] = True
        self.selected_meta_info = list(np.array(cell_line_input.meta_info[self.view])[self.mask])

    def transform(self, cell_line_input: FeatureDataset) -> FeatureDataset:
        """
        Apply the feature mask to reduce the dataset to selected features.

        :param cell_line_input: FeatureDataset to transform
        :returns: reduced FeatureDataset
        :raises RuntimeError: if selector was not fitted
        """
        if self.mask.size == 0:
            raise RuntimeError("VarianceFeatureSelector must be fitted before transform()")

        for identifier in cell_line_input.features:
            cell_line_input.features[identifier][self.view] = cell_line_input.features[identifier][self.view][self.mask]
        cell_line_input.meta_info[self.view] = self.selected_meta_info
        return cell_line_input
