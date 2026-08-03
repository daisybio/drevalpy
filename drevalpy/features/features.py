"""Load feature tables for drevalpy models and components."""

from __future__ import annotations

import logging
import os.path

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER

logger = logging.getLogger(__name__)


def load_generic_csv(path: str, dataset_name: str, feature_name: str, index_col=CELL_LINE_IDENTIFIER) -> FeatureDataset:
    """
    Loads a generic CSV file with cell line IDs as index and features as columns.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :param feature_name: name of the feature, e.g., gene_expression
    :param index_col: name of the index column, e.g., cell_line_id
    :returns: FeatureDataset with the features
    """
    feature_csv = pd.read_csv(f"{path}/{dataset_name}/{feature_name}.csv", index_col=index_col)
    feature_csv.index = feature_csv.index.astype(str)
    if "cellosaurus_id" in feature_csv.columns:
        feature_csv = feature_csv.drop(columns=["cellosaurus_id"])
    return FeatureDataset(
        features=iterate_features(df=feature_csv, feature_type=feature_name),
        meta_info={feature_name: feature_csv.columns.values},
    )


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


def load_cl_ids_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load cell line ids from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the cell line ids
    """
    cl_names = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=CELL_LINE_IDENTIFIER)
    cl_names.index = cl_names.index.astype(str)
    return FeatureDataset(features={cl: {CELL_LINE_IDENTIFIER: np.array([cl])} for cl in cl_names.index})


def load_tissues_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load tissues from csv file.

    :param path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the tissues
    """
    tissues = pd.read_csv(
        f"{path}/{dataset_name}/cell_line_names.csv", index_col=CELL_LINE_IDENTIFIER
    ).drop_duplicates()
    return FeatureDataset(
        features={cl: {TISSUE_IDENTIFIER: np.array([tissues.loc[cl, TISSUE_IDENTIFIER]])} for cl in tissues.index}
    )


def load_cl_ids_and_tissues_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    """
    Load cell line ids and optional tissue annotations from csv file.

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
    ge = pd.read_csv(f"{data_path}/{dataset_name}/{feature_type}.csv", index_col=CELL_LINE_IDENTIFIER)
    ge.index = ge.index.astype(str)
    if "cellosaurus_id" in ge.columns:
        ge = ge.drop(columns=["cellosaurus_id"])

    cl_features = FeatureDataset(
        features=iterate_features(df=ge, feature_type=feature_type),
        meta_info={feature_type: ge.columns.values},
    )
    if gene_list is None:
        return cl_features

    from drevalpy.features.gene_lists import gene_names_from_list_csv, resolve_gene_list_path

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


def load_drug_ids_from_csv(data_path: str, dataset_name: str) -> FeatureDataset:
    """
    Load drug ids from csv file.

    :param data_path: path to the data, e.g., data/
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with the drug ids
    """
    drug_names = pd.read_csv(
        f"{data_path}/{dataset_name}/drug_names.csv",
        index_col=DRUG_IDENTIFIER,
        dtype={"pubchem_id": str},
        low_memory=False,
    )
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
                e.g., {"gene_expression": "landmark_genes_reduced"}, if None, all features are not reduced
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


def _get_view_as_list(value):
    return [value] if isinstance(value, str) else value


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
