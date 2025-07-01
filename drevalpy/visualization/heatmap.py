"""Plots a heatmap of the evaluation metrics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .vioheat import VioHeat


class Heatmap(VioHeat):
    """Plots a heatmap of the evaluation metrics."""

    def __init__(self, df: pd.DataFrame, normalized_metrics=False, whole_name=False):
        """
        Initialize the Heatmap class.

        :param df: either containing all predictions for all algorithms or all tests for one algorithm (including
            robustness, randomization, … tests then)
        :param normalized_metrics: whether the metrics are normalized
        :param whole_name: whether the whole name should be displayed
        :raises ValueError: If the DataFrame is empty or does not contain the required metrics.
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

    def draw_and_save(self, out_prefix: str, out_suffix: str) -> None:
        """
        Draw the heatmap and save it to a file.

        :param out_prefix: e.g., results/my_run/heatmaps/
        :param out_suffix: e.g., algorithms_normalized
        """
        self._draw()
        path_out = f"{out_prefix}heatmap_{out_suffix}.html"
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
        """
        Draw the subplots of the heatmap.

        :param plot_setting: Either  "r2", "correlations", "errors", or "ssmd"
        :raises ValueError: If an unknown plot setting is given
        """
        idx_split = self.df.index.to_series().str.split("_")
        setting = idx_split.str[0:3].str.join("_")
        if plot_setting.startswith("ssmd_"):
            metric_name = plot_setting.split("_")[1]  # Extract metric name (e.g., "ssmd_r2" → "r2")
            dt = self._compute_ssmd(metric_name)
            dt["sort_key"] = dt.max(axis=1)
            dt = dt.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
            dt = dt[dt.index]  # Ensure columns match sorted rows

            if dt.empty:
                print(f"Warning: SSMD heatmap for {metric_name} is empty. Skipping.")
                return
            row_idx = self.plot_settings.index(plot_setting) + 1

            colorscale = "RdBu"
            text_labels = dt.round(3).astype(str)
        else:
            dt_std_errs = self.df.groupby(setting).apply(lambda x: self._calc_summary_metric(x, std_error=True))

            if plot_setting == "r2":
                r2_columns = [col for col in self.df.columns if "R^2" in col]

                dt = self.df[r2_columns].groupby(setting).apply(lambda x: self._calc_summary_metric(x))
                dt = dt.sort_values(by=r2_columns[0], ascending=True)
                dt_std_errs = dt_std_errs[r2_columns]
                dt_std_errs = dt_std_errs.loc[dt.index]

                row_idx = 1
                colorscale = "Blues"
            elif plot_setting == "correlations":
                corr_columns = [
                    col for col in self.df.columns if "Pearson" in col or "Spearman" in col or "Kendall" in col
                ]
                dt = self.df[corr_columns].groupby(setting).apply(lambda x: self._calc_summary_metric(x))
                dt = dt.sort_values(by=corr_columns[0], ascending=True)
                dt_std_errs = dt_std_errs[corr_columns]
                dt_std_errs = dt_std_errs.loc[dt.index]

                row_idx = 2
                colorscale = "Viridis"
            elif plot_setting == "errors":
                error_columns = [col for col in self.df.columns if col in ["MSE", "RMSE", "MAE"]]
                if not error_columns:
                    print("Warning: No error metric columns found. Skipping error heatmap.")
                    return
                dt = self.df[error_columns].groupby(setting).apply(lambda x: self._calc_summary_metric(x))
                dt = dt.sort_values(by=error_columns[0], ascending=False)
                dt_std_errs = dt_std_errs[error_columns]
                dt_std_errs = dt_std_errs.loc[dt.index]

                row_idx = 3
                colorscale = "hot"
            else:
                raise ValueError(f"Unknown plot setting: {plot_setting}")
            text_labels = dt.round(3).astype(str) + " ± " + dt_std_errs.round(3).astype(str)

        labels = [i.replace("_", " ") if self.whole_name else i.split("_")[0] for i in dt.index]
        self.fig.add_trace(
            go.Heatmap(
                z=dt.values,
                x=dt.columns,
                y=labels,
                colorscale=colorscale,
                texttemplate="%{text}",
                text=text_labels,
                textfont={"size": 16},  # size of labels of pixels of the heatmap
            ),
            row=row_idx,
            col=1,
        )

        # Force all y-ticks to be displayed
        self.fig.update_yaxes(
            row=row_idx,
            col=1,
            tickmode="array",
            tickvals=list(range(len(dt.index))),  # Force showing all ticks
            ticktext=labels,
            automargin=True,  # Prevent cutoff
            tickfont=dict(size=15),  # Adjust text size
        )

    def _compute_ssmd(self, metric: str) -> pd.DataFrame:
        """
        Compute Strictly Standardized Mean Difference (SSMD) for a given metric across splits.

        :param metric: The evaluation metric to compute SSMD for (e.g., "R^2", "RMSE", "MAE", "Pearson").
        :return: SSMD heatmap matrix (models × models) as a DataFrame.
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
        """
        Calculate the mean or standard error of the metrics.

        :param x: DataFrame containing the metrics
        :param std_error: whether to calculate the standard error or the mean
        :returns: Series containing the mean or standard error of the metrics
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
