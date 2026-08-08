"""Plots a heatmap of the evaluation metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from .vioheat import VioHeat


class Heatmap(VioHeat):
    """Plots a heatmap of the evaluation metrics."""

    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        """Initialize heatmap from evaluation results.

        :param df: Predictions for all algorithms or all tests for one algorithm.
        :param normalized_metrics: Whether to show only normalized metric columns.
        :param whole_name: Whether to display full algorithm setting labels.

        :raises ValueError: If the DataFrame is empty or lacks required metrics.
        """
        super().__init__(df, normalized_metrics, whole_name)
        if normalized_metrics and not any(["normalized" in col for col in self.df.columns]):
            raise ValueError(
                "The DataFrame does not contain normalized metrics. Please provide a DataFrame with normalized metrics."
            )
        if self.df.empty:
            raise ValueError("The DataFrame is empty. Please provide a valid DataFrame with metrics.")

        self.df = self.df[[col for col in self.df.columns if col in self.all_metrics]]
        if self.df.empty:
            raise ValueError("The DataFrame does not contain any valid metrics. Please check the columns.")
        self.n_models = len(self.df.index)

        if self.normalized_metrics:
            titles = [
                "Mean R^2: normalized",
                "Mean Correlations: normalized",
            ]
            nr_subplots = 3
            self.plot_settings = ["r2", "correlations"]
        else:
            titles = [
                "Mean R^2",
                "Mean Correlations",
                "Mean Errors",
                "Strictly Standardized Mean Difference for R^2",
                "Strictly Standardized Mean Difference for MSE",
            ]
            self.plot_settings = [
                "r2",
                "correlations",
                "errors",
                "ssmd_R^2",
                "ssmd_MSE",
            ]
            nr_subplots = len(self.plot_settings)

        self.fig = make_subplots(
            rows=nr_subplots,
            cols=1,
            subplot_titles=tuple(titles),
            vertical_spacing=0.1,
        )

    def draw_and_save(self, out_prefix: str | Path, out_suffix: str) -> None:
        """Draw heatmap and save as HTML.

        :param out_prefix: Output directory (for example ``results/my_run/heatmaps``).
        :param out_suffix: Filename suffix (for example ``algorithms_normalized``).
        """
        self._draw()
        path_out = Path(out_prefix) / f"heatmap_{out_suffix}.html"
        self.fig.write_html(path_out)

    def _draw(self) -> None:
        """Draw the heatmap."""
        print("Drawing heatmaps ...")
        for plot_setting in self.plot_settings:
            self._draw_subplots(plot_setting)

        # Dynamically adjust figure height based on number of models
        num_models = self.n_models
        height_per_model = 35  # Increase spacing for each model
        max_height = 5000  # Increase max height if needed
        new_height = min(500 + num_models * height_per_model, max_height)
        self.fig.update_layout(
            height=new_height,
            width=1300,
            title_text="Heatmap of the evaluation metrics",
        )
        self.fig.update_traces(showscale=False)

    def _draw_subplots(self, plot_setting: str) -> None:
        """Draw the subplots of the heatmap.

        :param plot_setting: One of ``r2``, ``correlations``, ``errors``, or ``ssmd``.
        """
        from .heatmap_subplots import add_heatmap_subplot

        add_heatmap_subplot(self, plot_setting)

    def _compute_ssmd(self, metric: str) -> pd.DataFrame:
        """Compute Strictly Standardized Mean Difference (SSMD) for a given metric across splits.

        :param metric: Evaluation metric to compute SSMD for (for example ``R^2``, ``RMSE``).

        :returns: SSMD heatmap matrix (models × models) as a DataFrame.
        """
        if metric not in self.df.columns:
            print(f"Warning: '{metric}' metric not found in DataFrame. Skipping SSMD heatmap.")
            return pd.DataFrame()

        # Extract only the base model name (remove _predictions_testmode_split_X)
        self.df["model_name"] = self.df.index.to_series().apply(lambda x: x.split("_predictions")[0])

        models = self.df["model_name"].unique()
        ssmd_matrix = pd.DataFrame(index=models, columns=models)

        for m1 in models:
            for m2 in models:
                if m1 == m2:
                    ssmd_matrix.loc[m1, m2] = 0  # No self-comparison
                    continue

                # Get metric values across splits for both models
                values_m1 = self.df[self.df["model_name"] == m1][metric]
                values_m2 = self.df[self.df["model_name"] == m2][metric]

                # Compute SSMD
                mu1, mu2 = values_m1.mean(), values_m2.mean()
                sigma1_sq, sigma2_sq = values_m1.var(ddof=1), values_m2.var(ddof=1)
                ssmd = (mu1 - mu2) / np.sqrt(sigma1_sq + sigma2_sq) if sigma1_sq + sigma2_sq > 0 else np.nan

                ssmd_matrix.loc[m1, m2] = ssmd

        return ssmd_matrix.astype(float)

    @staticmethod
    def _calc_summary_metric(x: pd.DataFrame, std_error: bool = False):
        """Calculate the mean or standard error of the metrics.

        :param x: DataFrame containing the metrics.
        :param std_error: Whether to calculate standard error instead of mean.

        :returns: Series containing the mean or standard error of the metrics.
        """
        results = pd.Series(index=x.columns)
        for col in x.columns:
            if np.count_nonzero(np.isnan(x[col])) == len(x[col]):
                results[col] = np.nan
            elif std_error:
                results[col] = np.nanstd(x[col]) / np.sqrt(x.shape[0])
            else:
                results[col] = np.nanmean(x[col])
        return results
