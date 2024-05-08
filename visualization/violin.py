import pandas as pd
import plotly.graph_objects as go


def create_evaluation_violin(df: pd.DataFrame, normalized_metrics=False, whole_name=False):
    print('Drawing Violin plots ...')
    df = df.sort_index()
    df['box'] = df['algorithm'] + '_' + df['rand_setting'] + '_' + df['LPO_LCO_LDO']
    # remove columns with only NaN values
    df = df.dropna(axis=1, how='all')
    fig = go.Figure()
    all_metrics = ["R^2", "R^2: drug normalized", "R^2: cell_line normalized",
                   "Pearson", "Pearson: drug normalized", "Pearson: cell_line normalized",
                   "Spearman", "Spearman: drug normalized", "Spearman: cell_line normalized",
                   "Kendall", "Kendall: drug normalized", "Kendall: cell_line normalized",
                   "Partial_Correlation", "Partial_Correlation: drug normalized",
                   "Partial_Correlation: cell_line normalized",
                   "MSE", "RMSE", "MAE"]
    if normalized_metrics:
        all_metrics = [metric for metric in all_metrics if 'normalized' in metric]
    else:
        all_metrics = [metric for metric in all_metrics if 'normalized' not in metric]
    occurring_metrics = [metric for metric in all_metrics if metric in df.columns]
    count_r2 = 0
    count_pearson = 0
    count_spearman = 0
    count_kendall = 0
    count_partial_correlation = 0
    count_mse = 0
    count_rmse = 0
    count_mae = 0
    for metric in occurring_metrics:
        if 'R^2' in metric:
            count_r2 += 1 * len(df['box'].unique())
        elif 'Pearson' in metric:
            count_pearson += 1 * len(df['box'].unique())
        elif 'Spearman' in metric:
            count_spearman += 1 * len(df['box'].unique())
        elif 'Kendall' in metric:
            count_kendall += 1 * len(df['box'].unique())
        elif 'Partial_Correlation' in metric:
            count_partial_correlation += 1 * len(df['box'].unique())
        elif 'RMSE' in metric:
            count_rmse += 1 * len(df['box'].unique())
        elif 'MSE' in metric:
            count_mse += 1 * len(df['box'].unique())
        elif 'MAE' in metric:
            count_mae += 1 * len(df['box'].unique())
        fig = add_violin(fig, df, metric, whole_name)

    count_sum = count_r2 + count_pearson + count_spearman + count_kendall + count_partial_correlation + count_mse + count_rmse + count_mae

    if normalized_metrics:
        buttons_update = list([
            dict(label="All Metrics",
                 method="update",
                 args=[{"visible": [True] * count_sum},
                       {"title": "All Metrics"}]),
            dict(label="R^2",
                 method="update",
                 args=[{"visible": [True] * count_r2 + [False] * (count_sum - count_r2)},
                       {"title": "R^2"}]),
            dict(label="Correlations",
                 method="update",
                 args=[{"visible": [False] * count_r2 + [True] * (
                             count_pearson + count_spearman + count_kendall + count_partial_correlation) + [False] * (
                                               count_mse + count_rmse + count_mae)},
                       {"title": "All Correlations"}]),
            dict(label="Pearson",
                 method="update",
                 args=[{"visible": [False] * count_r2 + [True] * count_pearson + [False] * (
                             count_sum - count_r2 - count_pearson)},
                       {"title": "Pearson"},
                       ]),
            dict(label="Spearman",
                 method="update",
                 args=[{"visible": [False] * (count_r2 + count_pearson) + [True] * count_spearman + [False] * (
                             count_sum - count_r2 - count_pearson - count_spearman)},
                       {"title": "Spearman"}]),
            dict(label="Kendall",
                 method="update",
                 args=[{"visible": [False] * (count_r2 + count_pearson + count_spearman) + [True] * count_kendall + [
                     False] * (count_sum - count_r2 - count_pearson - count_spearman - count_kendall)},
                       {"title": "Kendall"}]),
            dict(label="Partial Correlation",
                 method="update",
                 args=[{"visible": [False] * (
                             count_sum - count_partial_correlation - count_mse - count_rmse - count_mae) + [
                                       True] * count_partial_correlation + [False] * (
                                               count_mse + count_rmse + count_mae)},
                       {"title": "Partial Correlation"}])
        ])
    else:
        buttons_update = list([
            dict(label="All Metrics",
                 method="update",
                 args=[{"visible": [True] * count_sum},
                       {"title": "All Metrics"}]),
            dict(label="R^2",
                 method="update",
                 args=[{"visible": [True] * count_r2 + [False] * (count_sum - count_r2)},
                       {"title": "R^2"}]),
            dict(label="Correlations",
                 method="update",
                 args=[{"visible": [False] * count_r2 + [True] * (
                             count_pearson + count_spearman + count_kendall + count_partial_correlation) + [False] * (
                                               count_mse + count_rmse + count_mae)},
                       {"title": "All Correlations"}]),
            dict(label="Errors",
                 method="update",
                 args=[{"visible": [False] * (count_sum - count_mse - count_rmse - count_mae) + [True] * (
                             count_mse + count_rmse + count_mae)},
                       {"title": "All Errors"}]),
            dict(label="Pearson",
                 method="update",
                 args=[{"visible": [False] * count_r2 + [True] * count_pearson + [False] * (
                             count_sum - count_r2 - count_pearson)},
                       {"title": "Pearson"},
                       ]),
            dict(label="Spearman",
                 method="update",
                 args=[{"visible": [False] * (count_r2 + count_pearson) + [True] * count_spearman + [False] * (
                             count_sum - count_r2 - count_pearson - count_spearman)},
                       {"title": "Spearman"}]),
            dict(label="Kendall",
                 method="update",
                 args=[{"visible": [False] * (count_r2 + count_pearson + count_spearman) + [True] * count_kendall + [
                     False] * (count_sum - count_r2 - count_pearson - count_spearman - count_kendall)},
                       {"title": "Kendall"}]),
            dict(label="Partial Correlation",
                 method="update",
                 args=[{"visible": [False] * (
                             count_sum - count_partial_correlation - count_mse - count_rmse - count_mae) + [
                                       True] * count_partial_correlation + [False] * (
                                               count_mse + count_rmse + count_mae)},
                       {"title": "Partial Correlation"}]),
            dict(label="MSE",
                 method="update",
                 args=[{"visible": [False] * (count_sum - count_mse - count_rmse - count_mae) + [True] * count_mse + [
                     False] * (count_rmse + count_mae)},
                       {"title": "MSE"}]),
            dict(label="RMSE",
                 method="update",
                 args=[{"visible": [False] * (count_sum - count_rmse - count_mae) + [True] * count_rmse + [
                     False] * count_mae},
                       {"title": "RMSE"}]),
            dict(label="MAE",
                 method="update",
                 args=[{"visible": [False] * (count_sum - count_mae) + [True] * count_mae},
                       {"title": "MAE"}])
        ])

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons_update,
            )
        ]
    )
    fig.update_layout(title_text="All Metrics", height=600, width=1100)
    return fig


def add_violin(fig, df, metric, whole_name=False):
    for box in df['box'].unique():
        tmp_df = df[df['box'] == box]
        if whole_name:
            label = box + ': ' + metric
        else:
            label = box.split('_')[0] + ': ' + metric
        fig.add_trace(go.Violin(
            y=tmp_df[metric],
            x=[label] * len(tmp_df[metric]),
            name=label,
            box_visible=True,
            meanline_visible=True
        ))
    return fig
