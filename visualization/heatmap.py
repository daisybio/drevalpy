import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_heatmap(df: pd.DataFrame):
    print('Drawing Heatmap ...')
    df.sort_index(inplace=True)
    df_errors = df[['MSE', 'RMSE', 'MAE']]
    corr_columns = [col for col in df.columns if 'Pearson' in col or 'Spearman' in col or 'Kendall' in col or 'Partial_Correlation' in col]
    corr_columns.sort()
    df_corrs = df[corr_columns]
    titles = ['Standard Errors over CV folds', 'R^2', 'Correlations', 'Errors']
    fig = make_subplots(rows=4, cols=1, subplot_titles=tuple(titles))
    for i in range(4):
        if i == 0:
            # heatmap for standard errors
            idx_split = df.index.to_series().str.split('_')
            setting = idx_split.str[0:3].str.join('_')
            results = df.groupby(setting).apply(lambda x: np.std(x, axis=0) / np.sqrt(x.shape[0]))
            fig = fig.add_trace(
                go.Heatmap(z=results.values,
                           x=results.columns,
                           y=[i.replace('_', ' ') for i in list(results.index)],
                           colorscale='Pinkyl', texttemplate='%{z:.2f}'),
                row=1, col=1
            )

        elif i == 1:
            # heatmap for r^2
            r2_columns = [col for col in df.columns if 'R^2' in col]
            dt = df[r2_columns].sort_values(by='R^2', ascending=True)
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=dt.columns,
                           y=[i.replace('_', ' ') for i in list(dt.index)],
                           colorscale='Blues', texttemplate='%{z:.2f}'),
                row=2, col=1
            )
        elif i == 2:
            # heatmap for correlations
            dt = df_corrs.sort_values(by='Pearson', ascending=True)
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=[c.replace('_', ' ') for c in list(dt.columns)],
                           y=[i.replace('_', ' ') for i in list(dt.index)],
                           colorscale='Viridis',
                           texttemplate='%{z:.2f}'),
                row=3, col=1
            )
        else:
            # heatmap for errors
            dt = df_errors.sort_values(by='MSE', ascending=False)
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=[c.replace('_', ' ') for c in list(dt.columns)],
                           y=[i.replace('_', ' ') for i in list(dt.index)],
                           colorscale='hot',
                           texttemplate='%{z:.2f}'),
                row=4, col=1
            )
    fig.update_layout(height=1500, width=1000, title_text="Heatmap of the evaluation metrics")
    fig.update_traces(showscale=False)
    return fig
