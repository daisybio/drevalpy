import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_heatmap(df: pd.DataFrame, normalized_metrics=False, whole_name=False):
    print('Drawing Heatmap ...')
    df = df.sort_index()
    # drop the columns that are not needed
    all_metrics = ["R^2", "R^2: drug normalized", "R^2: cell_line normalized",
                   "Pearson", "Pearson: drug normalized", "Pearson: cell_line normalized",
                   "Spearman", "Spearman: drug normalized", "Spearman: cell_line normalized",
                   "Kendall", "Kendall: drug normalized", "Kendall: cell_line normalized",
                   "Partial_Correlation", "Partial_Correlation: drug normalized",
                   "Partial_Correlation: cell_line normalized",
                   "MSE", "RMSE", "MAE"]
    df = df[[col for col in df.columns if col in all_metrics]]
    # remove columns with only NaN values
    df = df.dropna(axis=1, how='all')
    if normalized_metrics:
        df = df[[col for col in df.columns if 'normalized' in col]]
    else:
        df = df[[col for col in df.columns if 'normalized' not in col]]
    corr_columns = [col for col in df.columns if 'Pearson' in col or 'Spearman' in col or 'Kendall' in col or 'Partial_Correlation' in col]
    corr_columns.sort()
    df_corrs = df[corr_columns]
    idx_split = df.index.to_series().str.split('_')
    setting = idx_split.str[0:3].str.join('_')
    if normalized_metrics:
        titles = ['Standard Errors over CV folds', 'Mean R^2: normalized', 'Mean Correlations: normalized']
        nr_subplots = 3
    else:
        titles = ['Standard Errors over CV folds', 'Mean R^2', 'Mean Correlations', 'Mean Errors']
        nr_subplots = 4
        df_errors = df[['MSE', 'RMSE', 'MAE']]
    fig = make_subplots(rows=nr_subplots, cols=1, subplot_titles=tuple(titles))
    for i in range(nr_subplots):
        if i == 0:
            # heatmap for standard errors
            results = df.groupby(setting).apply(lambda x: calc_std_error(x))
            if whole_name:
                labels = [i.replace('_', ' ') for i in list(results.index)]
            else:
                labels = [i.split('_')[0] for i in list(results.index)]
            fig = fig.add_trace(
                go.Heatmap(z=results.values,
                           x=results.columns,
                           y=labels,
                           colorscale='Pinkyl', texttemplate='%{z:.2f}'),
                row=1, col=1
            )

        elif i == 1:
            # heatmap for r^2
            r2_columns = [col for col in df.columns if 'R^2' in col]
            dt = df[r2_columns]
            dt = dt.groupby(setting).apply(lambda x: calc_mean(x))
            dt = dt.sort_values(by=r2_columns[0], ascending=True)
            if whole_name:
                labels = [i.replace('_', ' ') for i in list(dt.index)]
            else:
                labels = [i.split('_')[0] for i in list(dt.index)]
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=dt.columns,
                           y=labels,
                           colorscale='Blues', texttemplate='%{z:.2f}',
                           ),
                row=2, col=1
            )
        elif i == 2:
            # heatmap for correlations
            dt = df_corrs.groupby(setting).apply(lambda x: calc_mean(x))
            dt = dt.sort_values(by=df_corrs.columns[0], ascending=True)
            if whole_name:
                labels = [i.replace('_', ' ') for i in list(dt.index)]
            else:
                labels = [i.split('_')[0] for i in list(dt.index)]
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=[c.replace('_', ' ') for c in list(dt.columns)],
                           y=labels,
                           colorscale='Viridis',
                           texttemplate='%{z:.2f}'),
                row=3, col=1
            )
        else:
            # heatmap for errors
            dt = df_errors.groupby(setting).apply(lambda x: calc_mean(x))
            dt = dt.sort_values(by='MSE', ascending=False)
            if whole_name:
                labels = [i.replace('_', ' ') for i in list(dt.index)]
            else:
                labels = [i.split('_')[0] for i in list(dt.index)]
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=[c.replace('_', ' ') for c in list(dt.columns)],
                           y=labels,
                           colorscale='hot',
                           texttemplate='%{z:.2f}'),
                row=4, col=1
            )
    fig.update_layout(height=1200, width= 1400, title_text="Heatmap of the evaluation metrics")
    fig.update_traces(showscale=False)
    return fig


def calc_std_error(x):
    # make empty results series
    results = pd.Series(index=x.columns)
    # iterate over columns
    for col in x.columns:
        if np.count_nonzero(np.isnan(x[col])) == len(x[col]):
            results[col] = np.nan
        else:
            results[col] = np.nanstd(x[col]) / np.sqrt(x.shape[0])
    return results


def calc_mean(x):
    results = pd.Series(index=x.columns)
    for col in x.columns:
        if np.count_nonzero(np.isnan(x[col])) == len(x[col]):
            results[col] = np.nan
        else:
            results[col] = np.nanmean(x[col])
    return results
