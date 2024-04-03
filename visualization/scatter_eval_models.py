import pandas as pd
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


def generate_scatter_eval_models_plot(df: pd.DataFrame, metric):
    df.sort_values('model', inplace=True)
    buttons_x = list()
    buttons_y = list()

    models = df["model"].unique()

    fig_overall = make_subplots(rows=len(models), cols=len(models),
                                subplot_titles=[str(model).replace('_', '<br>', 2) for model in models])
    for i in range(len(models)):
        fig_overall['layout']['annotations'][i]['font']['size'] = 6
    fig = go.Figure()
    x = df[df["model"] == models[0]][metric]
    # replace nan with 0
    x = np.nan_to_num(x)
    y = df[df["model"] == models[0]][metric]
    # replace nan with 0
    y = np.nan_to_num(y)
    scatterplot = go.Scatter(x=x,
                             y=y,
                             mode='markers',
                             marker=dict(size=10, showscale=False),
                             name=f'{models[0]} vs {models[0]}',
                             showlegend=False,
                             visible=True
                             )
    fig.add_trace(scatterplot)

    line_corr = go.Line(
            x=[-1, 1],
            y=[-1, 1],
            mode='lines',
            line=dict(color='gold', width=2, dash='dash'),
            showlegend=False,
            visible=True
        )

    for run_idx in range(len(models)):
        run = models[run_idx]
        x = df[df["model"] == run][metric]
        # replace nan with 0
        x = np.nan_to_num(x)
        buttons_x.append(
            dict(label=run,
                 method="update",
                 args=[{"x": [x]},
                       {"xaxis": {"title": run, "range": [-1, 1]}}])
        )
        for run2_idx in range(len(models)):
            run2 = models[run2_idx]
            y = df[df["model"] == run2][metric]
            # replace nan with 0
            y = np.nan_to_num(y)
            density = get_density(x, y)
            scatterplot = go.Scatter(x=x,
                                     y=y,
                                     mode='markers',
                                     marker=dict(size=4, color=density, colorscale='Viridis', showscale=False),
                                     name=f'{run} vs {run2}',
                                     showlegend=False,
                                     visible=True,
                           )
            fig_overall.add_trace(scatterplot, col=run_idx+1, row=run2_idx+1)
            fig_overall.add_trace(line_corr, col=run_idx+1, row=run2_idx+1)
            if run_idx == 0:
                buttons_y.append(
                    dict(label=run2,
                         method="update",
                         args=[{"y": [y]},
                               {"yaxis": {"title": run2, "range": [-1, 1]}}])
                )
                if run2_idx == 0:
                    fig_overall['layout']['yaxis']['title'] = str(run2).replace('_', '<br>', 2)
                    fig_overall['layout']['yaxis']['title']['font']['size'] = 6
                else:
                    y_axis_idx = (run2_idx) * len(models) + 1
                    fig_overall['layout'][f'yaxis{y_axis_idx}']['title'] = str(run2).replace('_', '<br>', 2)
                    fig_overall['layout'][f'yaxis{y_axis_idx}']['title']['font']['size'] = 6

    fig.update_layout(title=f'Scatter plot of {metric} for each model',
                      xaxis_title=models[0], yaxis_title=models[0],
                      showlegend=False)
    fig_overall.update_layout(title=f'Scatter plot of {metric} for each model',
                      showlegend=False)

    fig.update_layout(
        updatemenus=[
            {'buttons': buttons_x,
             'direction': 'down',
             'showactive': True,
             'x': 0.0,
             'xanchor': 'left',
             'y': 1.5,
             'yanchor': 'top'},
            {'buttons': buttons_y,
             'direction': 'down',
             'showactive': True,
             'x': 0.5,
             'xanchor': 'left',
             'y': 1.5,
             'yanchor': 'top'}
        ]
    )
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])
    return fig, fig_overall


def get_density(x:np.ndarray, y:np.ndarray):
    """Get kernal density estimate for each (x, y) point."""
    try:
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        density = kernel(values)
    except:
        density = np.zeros(len(x))
    return density
