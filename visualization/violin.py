import pandas as pd
import plotly.graph_objects as go


def create_evaluation_violin(df: pd.DataFrame):
    df.sort_index(inplace=True)
    df['algorithm'] = df.index.str.split('_').str[0]
    df['rand_setting'] = df.index.str.split('_').str[1]
    df['eval_setting'] = df.index.str.split('_').str[2]
    df['box'] = df['algorithm'] + '_' + df['rand_setting'] + '_' + df['eval_setting']
    fig = go.Figure()
    all_metrics = ["R^2", "drug normalized R^2", "cell line normalized R^2",
                   "Pearson", "drug normalized Pearson", "cell line normalized Pearson",
                   "Spearman", "drug normalized Spearman", "cell line normalized Spearman",
                   "Kendall", "drug normalized Kendall", "cell line normalized Kendall",
                   "Partial_Correlation", "drug normalized Partial_Correlation", "cell line normalized Partial_Correlation",
                   "MSE", "RMSE", "MAE"]
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
        fig = add_violin(fig, df, metric)

    count_sum = count_r2 + count_pearson + count_spearman + count_kendall + count_partial_correlation + count_mse + count_rmse + count_mae

    fig.update_layout(
        updatemenus = [
            dict(
                active=0,
                buttons=list([
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
                         args=[{"visible": [False] * count_r2 + [True] * (count_pearson + count_spearman + count_kendall + count_partial_correlation) + [False] * (count_mse  + count_rmse + count_mae)},
                               {"title": "All Correlations"}]),
                    dict(label="Errors",
                            method="update",
                            args=[{"visible": [False] * (count_sum - count_mse - count_rmse - count_mae) + [True] * (count_mse + count_rmse + count_mae)},
                                {"title": "All Errors"}]),
                    dict(label="Pearson",
                         method="update",
                         args=[{"visible": [False] * count_r2 + [True] * count_pearson + [False] * (count_sum - count_r2 - count_pearson)},
                               {"title": "Pearson"},
                               ]),
                    dict(label="Spearman",
                         method="update",
                         args=[{"visible": [False] * (count_r2 + count_pearson) + [True] * count_spearman + [False] * (count_sum - count_r2 - count_pearson - count_spearman)},
                               {"title": "Spearman"}]),
                    dict(label="Kendall",
                         method="update",
                         args=[{"visible": [False] * (count_r2 + count_pearson + count_spearman) + [True] * count_kendall + [False] * (count_sum - count_r2 - count_pearson - count_spearman - count_kendall)},
                               {"title": "Kendall"}]),
                    dict(label="Partial Correlation",
                         method="update",
                         args=[{"visible": [False] * (count_sum - count_partial_correlation - count_mse - count_rmse - count_mae) + [True] * count_partial_correlation + [False] * (count_mse + count_rmse + count_mae)},
                               {"title": "Partial Correlation"}]),
                    dict(label="MSE",
                         method="update",
                         args=[{"visible": [False] * (count_sum - count_mse - count_rmse - count_mae) + [True] * count_mse + [False] * (count_rmse + count_mae)},
                               {"title": "MSE"}]),
                    dict(label="RMSE",
                         method="update",
                         args=[{"visible": [False] * (count_sum - count_rmse - count_mae) + [True] * count_rmse + [False] * count_mae},
                               {"title": "RMSE"}]),
                    dict(label="MAE",
                         method="update",
                         args=[{"visible": [False] * (count_sum - count_mae) + [True] * count_mae},
                               {"title": "MAE"}]),
                ]),
            )
        ]
    )
    fig.update_layout(title_text="All Metrics")
    return fig


def add_violin(fig, df, metric):
    for box in df['box'].unique():
        tmp_df = df[df['box'] == box]
        fig.add_trace(go.Violin(
            y=tmp_df[metric],
            x=[label.replace('_', ' ') for label in tmp_df['box']+ '_' + metric],
            name=(box + ': ' + metric).replace('_', ' '),
            box_visible=True,
            meanline_visible=True
        ))
    return fig
