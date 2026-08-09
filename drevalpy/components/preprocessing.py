"""Preprocessing helpers for cell-line feature views."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def log10_and_set_na(x: np.ndarray) -> np.ndarray:
    """Log10 transform and set NaN for infinite values.

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
        """Hyperparameters for the normalization.

        :param feature_threshold: Minimum fraction of non-missing protein values per feature.
        :param n_features: Fallback feature count when thresholding leaves too few features.
        :param normalization_downshift: Downshift factor for the mean.
        :param normalization_width: Width factor for the standard deviation.
        :param imputation_seed: Seed for per-call missing-value imputation without touching the
            global NumPy RNG state.
        """
        self.feature_threshold = feature_threshold
        self.n_features = n_features
        self.normalization_downshift = normalization_downshift
        self.normalization_width = normalization_width
        self.imputation_seed = imputation_seed
        self.protein_indices = np.array([])
        self.mean_median = 0

    def fit(self, X, y=None):  # noqa: N803  # sklearn API
        """Learns the top n_feature complete proteins and calculates the mean median of the train cell lines.

        :param X: input proteomics data
        :param y: not used
        :returns: self
        """
        required_proteins = int(X.shape[0] * self.feature_threshold)
        completeness = np.sum(~np.isnan(X), axis=0)
        n_complete_features = np.count_nonzero(completeness >= required_proteins)
        if n_complete_features < self.n_features:
            sorted_indices = np.argsort(completeness)[::-1]
            self.protein_indices = sorted_indices[: self.n_features]
        else:
            self.protein_indices = np.where(completeness >= required_proteins)[0]
        selected_proteins = X[:, self.protein_indices]
        medians = np.nanmedian(selected_proteins, axis=1)
        self.mean_median = np.nanmean(medians)
        return self

    def transform(self, X):  # noqa: N803  # sklearn API
        """Median center the data and impute missing values with downshifted normal distribution.

        :param X: input proteomics data
        :returns: transformed proteomics data
        """
        proteomics_vector = X[0][self.protein_indices]

        correction_factor = self.mean_median / np.nanmedian(proteomics_vector)
        proteomics_vector = proteomics_vector * correction_factor
        cell_line_mean = np.nanmean(proteomics_vector)
        cell_line_sd = np.nanstd(proteomics_vector)
        downshifted_mean = cell_line_mean - (self.normalization_downshift * cell_line_sd)
        shrinked_sd = self.normalization_width * cell_line_sd
        n_missing = np.count_nonzero(np.isnan(proteomics_vector))
        rng = np.random.default_rng(self.imputation_seed)
        proteomics_vector[np.isnan(proteomics_vector)] = rng.normal(
            loc=downshifted_mean, scale=shrinked_sd, size=n_missing
        )
        return [proteomics_vector]
