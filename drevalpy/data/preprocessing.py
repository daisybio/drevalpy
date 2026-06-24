"""Preprocessing helpers for cell-line feature views."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

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

