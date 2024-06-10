import plotly.express as px
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os


def generate_regression_plots(df: pd.DataFrame, run_id: str, group_by: str = 'drug', normalize=False):
    print(f'Generating regression plots for {group_by}, normalize={normalize}...')
    if not os.path.exists(f'../results/{run_id}/regression_plots'):
        os.mkdir(f'../results/{run_id}/regression_plots')

    df = df[df['rand_setting'] == 'predictions']
    if normalize:
        if group_by == 'cell_line':
            df['y_true'] = df['y_true'] - df['mean_y_true_per_drug']
            df['y_pred'] = df['y_pred'] - df['mean_y_true_per_drug']
        else:
            df['y_true'] = df['y_true'] - df['mean_y_true_per_cell_line']
            df['y_pred'] = df['y_pred'] - df['mean_y_true_per_cell_line']
    # kick out all groups with less than 2 samples but keep it as grouped object
    df = df.groupby(group_by).filter(lambda x: len(x) > 1)
    pccs = df.groupby(group_by).apply(
        lambda x: pearsonr(x['y_true'], x['y_pred'])[0])
    pccs = pccs.reset_index()
    pccs.columns = [group_by, 'pcc']
    df = df.merge(pccs, on=group_by)

    fig = make_regression_slider(df, group_by=group_by, normalize=normalize)
    setting = df['LPO_LCO_LDO'].unique()[0]
    if normalize:
        fig.write_html(f'../results/{run_id}/regression_plots/{setting}_regression_lines_{group_by}_normalized.html')
    else:
        fig.write_html(f'../results/{run_id}/regression_plots/{setting}_regression_lines_{group_by}.html')


def make_regression_slider(df: pd.DataFrame, group_by: str = 'drug', normalize=False):
    n_ticks = 21
    # sort df by group name
    df = df.sort_values(group_by)
    setting_title = df['algorithm'].unique()[0] + ' ' + df['LPO_LCO_LDO'].unique()[0]
    if normalize:
        if group_by == 'cell_line':
            setting_title += f', normalized by drug mean'
            fig = px.scatter(df, x="y_true", y="y_pred",
                             color=group_by, trendline="ols",
                             hover_name=group_by, hover_data=["pcc", "cell_line", "drug", "mean_y_true_per_drug"],
                             title=f"{setting_title}: Regression plot")
        else:
            setting_title += f', normalized by cell line mean'
            fig = px.scatter(df, x="y_true", y="y_pred",
                             color=group_by, trendline="ols",
                             hover_name=group_by, hover_data=["pcc", "cell_line", "drug", "mean_y_true_per_cell_line"],
                             title=f"{setting_title}: Regression plot")
    else:
        fig = px.scatter(df, x="y_true", y="y_pred",
                         color=group_by, trendline="ols",
                         hover_name=group_by, hover_data=["pcc", "cell_line", "drug"],
                         title=f"{setting_title}: Regression plot")

    # Create and add slider
    steps = []
    # take the range from pcc and divide it into 10 equal parts
    pcc_parts = np.linspace(-1, 1, n_ticks)
    for i in range(n_ticks):
        # from the fig data, get the hover data and check if it is greater than the pcc_parts[i]
        # only iterate over even numbers
        pccs = [0 for _ in range(0, len(fig.data))]
        for j in range(0, len(fig.data)):
            if j % 2 == 0:
                pccs[j] = fig.data[j].customdata[0, 0]
            else:
                pccs[j] = fig.data[j - 1].customdata[0, 0]
        if i == n_ticks - 1:
            visible_traces = pccs >= pcc_parts[i]
            title = f"{setting_title}: Slider for PCCs >= {str(round(pcc_parts[i], 1))} (step {str(i + 1)} of {str(n_ticks)})"
        else:
            visible_traces_gt = pccs >= pcc_parts[i]
            visible_traces_lt = pccs < pcc_parts[i + 1]
            visible_traces = visible_traces_gt & visible_traces_lt
            title = f"{setting_title}: Slider for PCCs between {str(round(pcc_parts[i], 1))} and {str(round(pcc_parts[i + 1], 1))} (step {str(i + 1)} of {str(n_ticks)})"
        step = dict(
            method="update",
            args=[{"visible": visible_traces},
                  {"title": title}],
            label=str(round(pcc_parts[i], 1))
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Pearson correlation coefficient="},
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
    min_val = np.min([np.min(df["y_true"]), np.min(df["y_pred"])])
    max_val = np.max([np.max(df["y_true"]), np.max(df["y_pred"])])
    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val])
    return fig
