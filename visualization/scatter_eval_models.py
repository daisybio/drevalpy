import pandas as pd
import numpy as np
import scipy
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math


def generate_scatter_eval_models_plot(df: pd.DataFrame, metric, color_by):
    df.sort_values('model', inplace=True)
    buttons_x = list()
    buttons_y = list()

    models = df["model"].unique()

    fig_overall = make_subplots(rows=len(models), cols=len(models),
                                subplot_titles=[str(model).replace('_', '<br>', 2) for model in models])
    for i in range(len(models)):
        fig_overall['layout']['annotations'][i]['font']['size'] = 6
    fig = go.Figure()
    # subset the dataframe to have model==models[0] and get the metric and color_by column
    tmp_df = df[df["model"] == models[0]][[metric, color_by]]
    # make color_by the index
    tmp_df.set_index(color_by, inplace=True)
    # sort the dataframe by the index
    tmp_df.sort_index(inplace=True)
    # replace nan with 0
    tmp_df[metric] = tmp_df[metric].fillna(0)
    scatterplot = go.Scatter(x=tmp_df[metric],
                             y=tmp_df[metric],
                             mode='markers',
                             marker=dict(size=6, showscale=False),
                             text=tmp_df.index,
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
        x_df = df[df["model"] == run][[metric, color_by]]
        x_df.set_index(color_by, inplace=True)
        x_df.sort_index(inplace=True)
        x_df[metric] = x_df[metric].fillna(0)
        buttons_x.append(
            dict(label=run,
                 method="update",
                 args=[{"x": [x_df[metric]]},
                       {"xaxis": {"title": run, "range": [-1, 1]}}])
        )
        for run2_idx in range(len(models)):
            run2 = models[run2_idx]
            y_df = df[df["model"] == run2][[metric, color_by]]
            y_df.set_index(color_by, inplace=True)
            y_df.sort_index(inplace=True)
            # replace nan with 0
            y_df[metric] = y_df[metric].fillna(0)
            # only retain the common indices
            common_indices = x_df.index.intersection(y_df.index)
            x_df2 = x_df.loc[common_indices]
            y_df = y_df.loc[common_indices]

            density = get_density(x_df2[metric], y_df[metric])
            scatterplot = go.Scatter(x=x_df2[metric],
                                     y=y_df[metric],
                                     mode='markers',
                                     marker=dict(size=4, color=density, colorscale='Viridis', showscale=False),
                                     name=f'{run} vs {run2}',
                                     showlegend=False,
                                     visible=True,
                                     text=y_df.index
                           )
            fig_overall.add_trace(scatterplot, col=run_idx+1, row=run2_idx+1)
            fig_overall.add_trace(line_corr, col=run_idx+1, row=run2_idx+1)
            if run_idx == 0:
                buttons_y.append(
                    dict(label=run2,
                         method="update",
                         args=[{"y": [y_df[metric]]},
                               {"yaxis": {"title": run2, "range": [-1, 1]}}])
                )
                if run2_idx == 0:
                    fig_overall['layout']['yaxis']['title'] = str(run2).replace('_', '<br>', 2)
                    fig_overall['layout']['yaxis']['title']['font']['size'] = 6
                else:
                    y_axis_idx = (run2_idx) * len(models) + 1
                    fig_overall['layout'][f'yaxis{y_axis_idx}']['title'] = str(run2).replace('_', '<br>', 2)
                    fig_overall['layout'][f'yaxis{y_axis_idx}']['title']['font']['size'] = 6

    fig.update_layout(title=f'{str(color_by).replace("_", " ").capitalize()}-wise scatter plot of {metric} for each model',
                      xaxis_title=models[0], yaxis_title=models[0],
                      showlegend=False)
    fig_overall.update_layout(title=f'{str(color_by).replace("_", " ").capitalize()}-wise scatter plot of {metric} for each model',
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


def get_density(x:pd.Series, y:pd.Series):
    """Get kernal density estimate for each (x, y) point."""
    try:
        values = np.vstack([x, y])
        kernel = stats.gaussian_kde(values)
        density = kernel(values)
    except scipy.linalg.LinAlgError:
        density = np.zeros(len(x))
    return density
