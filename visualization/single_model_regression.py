import plotly.express as px
import scipy.stats
import numpy as np
import pandas as pd
import os


def generate_regression_plots(df: pd.DataFrame, id):
    if not os.path.exists(f'../results/{id}/regression_plots'):
        os.mkdir(f'../results/{id}/regression_plots')
    sccs = df.groupby(['algorithm', 'rand_setting', 'eval_setting', 'drug']).apply(
        lambda x: scipy.stats.pearsonr(x['y_true'], x['y_pred'])[0])
    sccs = sccs.reset_index()
    sccs.columns = ['algorithm', 'rand_setting', 'eval_setting', 'drug', 'scc']
    df = df.merge(sccs, on=['algorithm', 'rand_setting', 'eval_setting', 'drug'])
    df['combination'] = df['algorithm'] + ' ' + df['rand_setting'] + ' ' + df['eval_setting']
    for combination in df['combination'].unique():
        tmp_df = df[df['combination'] == combination]
        fig = make_regression_slider(tmp_df)
        fig.write_html(f'../results/{id}/regression_plots/{combination}_regression_lines.html')


def make_regression_slider(df: pd.DataFrame):
    n_ticks = 21
    # sort df by drugname
    df = df.sort_values("drug")
    fig = px.scatter(df, x="y_true", y="y_pred", color="drug", trendline="ols",
                     hover_name="drug", hover_data=["scc", "cell_line"],
                     title=f"{df['combination'].unique()[0]}: Regression plot")

    # Create and add slider
    steps = []
    # take the range from scc and divide it into 10 equal parts
    scc_parts = np.linspace(-1, 1, n_ticks)
    for i in range(n_ticks):
        # from the fig data, get the hover data and check if it is greater than the scc_parts[i]
        # only iterate over even numbers
        sccs = [0 for _ in range(0, len(fig.data))]
        for j in range(0, len(fig.data)):
            if j % 2 == 0:
                sccs[j] = fig.data[j].customdata[0, 0]
            else:
                sccs[j] = fig.data[j - 1].customdata[0, 0]

        if i == n_ticks - 1:
            visible_traces = sccs >= scc_parts[i]
            title = f"{df['combination'].unique()[0]}: Slider for SCCs >= {str(round(scc_parts[i], 1))} (step {str(i + 1)} of {str(n_ticks)})"
        else:
            visible_traces_gt = sccs >= scc_parts[i]
            visible_traces_lt = sccs < scc_parts[i + 1]
            visible_traces = visible_traces_gt & visible_traces_lt
            title = f"{df['combination'].unique()[0]}: Slider for SCCs between {str(round(scc_parts[i], 1))} and {str(round(scc_parts[i + 1], 1))} (step {str(i + 1)} of {str(n_ticks)})"
        step = dict(
            method="update",
            args=[{"visible": visible_traces},
                  {"title": title}],
            label=str(round(scc_parts[i], 1))
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "SCC="},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.05
        )
    )

    fig.update_xaxes(range=[np.min(df["y_true"]), np.max(df["y_true"])])
    fig.update_yaxes(range=[np.min(df["y_pred"]), np.max(df["y_pred"])])
    return fig
