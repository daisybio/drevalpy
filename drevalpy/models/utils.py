"""Utility functions for loading and processing data."""

import os.path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER


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
    return FeatureDataset(features=iterate_features(df=feature_csv, feature_type=feature_name))


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

    gene_info = pd.read_csv(
        f"{data_path}/meta/gene_lists/{gene_list}.csv",
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


def prepare_expression_and_methylation(
    cell_line_input: FeatureDataset,
    cell_line_ids: np.ndarray,
    training: bool,
    gene_expression_scaler: TransformerMixin | None = None,
    methylation_scaler: TransformerMixin | None = None,
    methylation_pca: PCA | None = None,
) -> FeatureDataset:
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
    :returns: FeatureDataset with the transformed features
    """
    cell_line_input = cell_line_input.copy()
    first_feature = next(iter(cell_line_input.features.values()))
    if ("gene_expression" in first_feature.keys()) and (gene_expression_scaler is not None):
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

    if ("methylation" in first_feature.keys()) and (methylation_scaler is not None) and (methylation_pca is not None):
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
    return cell_line_input


def scale_gene_expression(
    cell_line_input: FeatureDataset,
    cell_line_ids: np.ndarray,
    training: bool,
    gene_expression_scaler: TransformerMixin,
) -> FeatureDataset:
    """
    Scales gene expression inplace using arcsinh transformation and a provided scaler.

    :param cell_line_input: FeatureDataset with the cell line features
    :param cell_line_ids: IDs of cell lines to use for fitting or transformation
    :param training: whether to fit or transform
    :param gene_expression_scaler: sklearn transformer for gene expression
    :returns: FeatureDataset with the transformed features
    """
    cell_line_input = prepare_expression_and_methylation(
        cell_line_input=cell_line_input,
        cell_line_ids=cell_line_ids,
        training=training,
        gene_expression_scaler=gene_expression_scaler,
    )
    return cell_line_input


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

    def __init__(
        self,
        feature_threshold=0.7,
        n_features=1000,
        normalization_downshift=1.8,
        normalization_width=0.3,
        imputation_seed=100,
    ):
        """
        Hyperparameters for the normalization.

        :param feature_threshold: Require that, e.g., 70% of the proteins are measured without NAs
            over all cell lines -> n_complete_features = number of proteins with at least 70% of the cell lines
        :param n_features: fallback for feature selection. Take top n complete features.
            Select max(n_complete_features, n_features) features.
        :param normalization_downshift: downshift factor for the mean
        :param normalization_width: width factor for the standard deviation
        :param imputation_seed: seed for the per-call RNG used to impute missing values; kept
            here (rather than mutating np.random globally) so the transformer stays reproducible
            without touching the global RNG state.
        """
        self.feature_threshold = feature_threshold
        self.n_features = n_features
        self.normalization_downshift = normalization_downshift
        self.normalization_width = normalization_width
        self.imputation_seed = imputation_seed
        self.protein_indices = np.array([])
        self.mean_median = 0

    def fit(self, X, y=None):
        """
        Learns the top n_feature complete proteins and calculates the mean median of the train cell lines.

        :param X: input proteomics data
        :param y: not used
        :returns: self
        """
        required_proteins = int(X.shape[0] * self.feature_threshold)
        # identify the complete columns
        completeness = np.sum(~np.isnan(X), axis=0)
        n_complete_features = np.count_nonzero(completeness >= required_proteins)
        if n_complete_features < self.n_features:
            # select top 1000 complete features
            # sort by completeness
            sorted_indices = np.argsort(completeness)[::-1]
            self.protein_indices = sorted_indices[: self.n_features]
        else:
            # select the features meeting the required threshold
            self.protein_indices = np.where(completeness >= required_proteins)[0]
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
        cell_line_mean = np.nanmean(X)
        cell_line_sd = np.nanstd(X)
        downshifted_mean = cell_line_mean - (self.normalization_downshift * cell_line_sd)
        shrinked_sd = self.normalization_width * cell_line_sd
        n_missing = np.count_nonzero(np.isnan(X))
        # local RNG keeps imputation deterministic without poisoning the global np.random state
        rng = np.random.default_rng(self.imputation_seed)
        X[np.isnan(X)] = rng.normal(loc=downshifted_mean, scale=shrinked_sd, size=n_missing)
        return [X]


def prepare_proteomics(
    cell_line_input: FeatureDataset,
    cell_line_ids: np.ndarray,
    training: bool,
    transformer: ProteomicsMedianCenterAndImputeTransformer,
) -> FeatureDataset:
    """
    Applies log10 transform and proteomics normalization (centering + imputation) to proteomics view.

    :param cell_line_input: FeatureDataset with proteomics features
    :param cell_line_ids: cell line IDs for training or transformation
    :param training: whether to fit or only transform
    :param transformer: Proteomics transformer
    :returns: transformed FeatureDataset
    """
    cell_line_input = cell_line_input.copy()
    cell_line_input.apply(log10_and_set_na, view="proteomics")
    if training:
        cell_line_input.fit_transform_features(
            train_ids=cell_line_ids,
            transformer=transformer,
            view="proteomics",
        )
    else:
        cell_line_input.transform_features(
            ids=cell_line_ids,
            transformer=transformer,
            view="proteomics",
        )
    return cell_line_input


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
            "cell_line_views is empty. Call build_model() before load_cell_line_features() "
            "so the model knows which omics to load."
        )
    if len(cell_line_views) > 1:
        raise ValueError(f"Only one cell line view is supported for {model_name}.")
    print(f"Loading a {model_name} with the following cell line views: {cell_line_views}")

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
            "cell_line_views is empty. Call build_model() before load_cell_line_features() "
            "so the model knows which omics to load."
        )
    print(f"Loading a {model_name} with the following cell line views: {cell_line_views}")

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
    print(f"Loading a {model_name} with the following drug views: {drug_views}")

    if len(drug_views) == 0:
        return load_drug_ids_from_csv(data_path, dataset_name)
    elif drug_views[0] == "fingerprints":
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)
    else:
        return load_generic_csv(
            path=data_path, dataset_name=dataset_name, feature_name=drug_views[0], index_col=DRUG_IDENTIFIER
        )
