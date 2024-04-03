import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_heatmap(df: pd.DataFrame):
    # df = df.fillna(0)
    df.sort_index(inplace=True)
    # drop r^2, mse, rmse
    df_errors = df[['MSE', 'RMSE', 'MAE']]
    df_corrs = df[['Pearson', 'Spearman', 'Kendall', 'Partial_Correlation']]
    titles = ['R^2', 'Correlations', 'Errors']
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(titles))
    for i in range(3):
        if i == 0:
            # heatmap for r^2
            dt = df[['R^2']].sort_values(by='R^2', ascending=True)
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=['R^2'],
                           y=[i.replace('_', ' ') for i in list(dt.index)],
                           colorscale='Blues', texttemplate='%{z:.2f}'),
                row=1, col=1
            )
        elif i == 1:
            # heatmap for correlations
            dt = df_corrs.sort_values(by='Pearson', ascending=True)
            fig = fig.add_trace(
                go.Heatmap(z=dt.values,
                           x=[c.replace('_', ' ') for c in list(dt.columns)],
                           y=[i.replace('_', ' ') for i in list(dt.index)],
                           colorscale='Viridis',
                           texttemplate='%{z:.2f}'),
                row=2, col=1
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
                row=3, col=1
            )
    fig.update_layout(height=1000, width=1000, title_text="Heatmap of the evaluation metrics")
    fig.update_traces(showscale=False)
    return fig
