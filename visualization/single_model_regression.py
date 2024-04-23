import plotly.express as px
import scipy.stats
import numpy as np
import pandas as pd
import os


def generate_regression_plots(df: pd.DataFrame, run_id: str, group_by: str = 'drug', normalize=False):
    print(f'Generating regression plots for {group_by}, normalize={normalize}...')
    if not os.path.exists(f'../results/{run_id}/regression_plots'):
        os.mkdir(f'../results/{run_id}/regression_plots')
    if normalize:
        if group_by == 'drug':
            # kick out all LDO samples
            df = df[~df['eval_setting'].str.contains('LDO')]
            df['y_true'] = df['y_true'] - df['mean_y_true_per_drug']
            df['y_pred'] = df['y_pred'] - df['mean_y_true_per_drug']
        else:
            # kick out all LCO samples
            df = df[~df['eval_setting'].str.contains('LCO')]
            df['y_true'] = df['y_true'] - df['mean_y_true_per_cell_line']
            df['y_pred'] = df['y_pred'] - df['mean_y_true_per_cell_line']
    df['setting'] = df['algorithm'] + ' ' + df['rand_setting'] + ' ' + df['eval_setting'].str.split('_').str[0]
    # kick out all groups with less than 2 samples but keep it as grouped object
    df = df.groupby(['setting', group_by]).filter(lambda x: len(x) > 1)
    sccs = df.groupby(['setting', group_by]).apply(
        lambda x: scipy.stats.pearsonr(x['y_true'], x['y_pred'])[0])
    sccs = sccs.reset_index()
    sccs.columns = ['setting', group_by, 'scc']
    df = df.merge(sccs, on=['setting', group_by])
    for setting in df['setting'].unique():
        tmp_df = df[df['setting'] == setting]
        fig = make_regression_slider(tmp_df, group_by=group_by)
        if normalize:
            fig.write_html(f'../results/{run_id}/regression_plots/{setting}_regression_lines_{group_by}_normalized.html')
        else:
            fig.write_html(f'../results/{run_id}/regression_plots/{setting}_regression_lines_{group_by}.html')


def make_regression_slider(df: pd.DataFrame, group_by: str = 'drug'):
    n_ticks = 21
    # sort df by group name
    df = df.sort_values(group_by)
    fig = px.scatter(df, x="y_true", y="y_pred",
                     color=group_by, trendline="ols",
                     hover_name=group_by, hover_data=["scc", "cell_line", "drug", "eval_setting"],
                     title=f"{df['setting'].unique()[0]}: Regression plot")

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
            title = f"{df['setting'].unique()[0]}: Slider for SCCs >= {str(round(scc_parts[i], 1))} (step {str(i + 1)} of {str(n_ticks)})"
        else:
            visible_traces_gt = sccs >= scc_parts[i]
            visible_traces_lt = sccs < scc_parts[i + 1]
            visible_traces = visible_traces_gt & visible_traces_lt
            title = f"{df['setting'].unique()[0]}: Slider for SCCs between {str(round(scc_parts[i], 1))} and {str(round(scc_parts[i + 1], 1))} (step {str(i + 1)} of {str(n_ticks)})"
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
