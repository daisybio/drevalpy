"""Heatmap subplot assembly helpers."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _setting_groups(df: pd.DataFrame) -> pd.Series:
    idx_split = df.index.to_series().str.split("_")
    return idx_split.str[0:3].str.join("_")


def _ssmd_heatmap_data(heatmap, plot_setting: str) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    metric_name = plot_setting.split("_")[1]
    dt = heatmap._compute_ssmd(metric_name)
    if dt.empty:
        return dt, metric_name, None
    dt = dt.copy()
    dt["sort_key"] = dt.max(axis=1)
    dt = dt.sort_values(by="sort_key", ascending=True).drop(columns=["sort_key"])
    dt = dt[dt.index]
    text_labels = dt.round(3).astype(str)
    return dt, metric_name, text_labels


def _columns_for_plot_setting(plot_setting: str, df_columns: list[str]) -> tuple[list[str], int, str, bool] | None:
    if plot_setting == "r2":
        columns = [col for col in df_columns if "R^2" in col]
        return columns, 1, "Blues", True
    if plot_setting == "correlations":
        columns = [col for col in df_columns if "Pearson" in col or "Spearman" in col or "Kendall" in col]
        return columns, 2, "Viridis", True
    if plot_setting == "errors":
        columns = [col for col in df_columns if col in ["MSE", "RMSE", "MAE"]]
        if not columns:
            print("Warning: No error metric columns found. Skipping error heatmap.")
            return None
        return columns, 3, "hot", False
    return None


def _metric_block(
    heatmap,
    plot_setting: str,
    setting: pd.Series,
    dt_std_errs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int, str] | None:
    spec = _columns_for_plot_setting(plot_setting, list(heatmap.df.columns))
    if spec is None:
        return None
    columns, row_idx, colorscale, ascending = spec
    dt = heatmap.df[columns].groupby(setting).apply(lambda x: heatmap._calc_summary_metric(x))
    dt = dt.sort_values(by=columns[0], ascending=ascending)
    std_part = dt_std_errs[columns].loc[dt.index]
    return dt, std_part, row_idx, colorscale


def add_heatmap_subplot(heatmap, plot_setting: str) -> None:
    """Add one heatmap row for ``plot_setting`` to ``heatmap.fig``."""
    setting = _setting_groups(heatmap.df)

    if plot_setting.startswith("ssmd_"):
        dt, metric_name, text_labels = _ssmd_heatmap_data(heatmap, plot_setting)
        if dt.empty:
            print(f"Warning: SSMD heatmap for {metric_name} is empty. Skipping.")
            return
        row_idx = heatmap.plot_settings.index(plot_setting) + 1
        colorscale = "RdBu"
    else:
        dt_std_errs = heatmap.df.groupby(setting).apply(lambda x: heatmap._calc_summary_metric(x, std_error=True))
        block = _metric_block(heatmap, plot_setting, setting, dt_std_errs)
        if block is None:
            raise ValueError(f"Unknown plot setting: {plot_setting}")
        dt, std_part, row_idx, colorscale = block
        text_labels = dt.round(3).astype(str) + " ± " + std_part.round(3).astype(str)

    labels = [i.replace("_", " ") if heatmap.whole_name else i.split("_")[0] for i in dt.index]
    heatmap.fig.add_trace(
        go.Heatmap(
            z=dt.values,
            x=dt.columns,
            y=labels,
            colorscale=colorscale,
            texttemplate="%{text}",
            text=text_labels,
            textfont={"size": 16},
        ),
        row=row_idx,
        col=1,
    )
    heatmap.fig.update_yaxes(
        row=row_idx,
        col=1,
        tickmode="array",
        tickvals=list(range(len(dt.index))),
        ticktext=labels,
        automargin=True,
        tickfont=dict(size=15),
    )
