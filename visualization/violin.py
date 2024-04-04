import pandas as pd
import plotly.graph_objects as go


def create_evaluation_violin(df: pd.DataFrame):
    df.sort_index(inplace=True)
    df['algorithm'] = df.index.str.split('_').str[0]
    df['rand_setting'] = df.index.str.split('_').str[1]
    df['eval_setting'] = df.index.str.split('_').str[2]
    df['box'] = df['algorithm'] + '_' + df['rand_setting'] + '_' + df['eval_setting']
    fig = go.Figure()
    for metric in ["R^2", "drug normalized R^2", "cell line normalized R^2",
                   "Pearson", "drug normalized Pearson", "cell line normalized Pearson",
                   "Spearman", "drug normalized Spearman", "cell line normalized Spearman",
                   "Kendall", "drug normalized Kendall", "cell line normalized Kendall",
                   "Partial_Correlation", "drug normalized Partial_Correlation", "cell line normalized Partial_Correlation",
                   "MSE", "RMSE", "MAE"]:
        fig = add_violin(fig, df, metric)
    fig.update_layout(
        updatemenus = [
            dict(
                active=0,
                buttons=list([
                    dict(label="All Metrics",
                         method="update",
                         args=[{"visible": [True]*36},
                                 {"title": "All Metrics"}]),
                    dict(label="R^2",
                         method="update",
                         args=[{"visible": [True] * 6 + [False] * 30},
                               {"title": "R^2"}]),
                    dict(label="Correlations",
                         method="update",
                         args=[{"visible": [False] * 6 + [True] * 24 + [False] * 6},
                               {"title": "All Correlations"}]),
                    dict(label="Errors",
                            method="update",
                            args=[{"visible": [False] * 30 + [True] * 6},
                                {"title": "All Errors"}]),
                    dict(label="Pearson",
                         method="update",
                         args=[{"visible": [False] * 6 + [True] * 6 + [False] * 24},
                               {"title": "Pearson"},
                               ]),
                    dict(label="Spearman",
                         method="update",
                         args=[{"visible": [False] * 12 + [True] * 6 + [False] * 18},
                               {"title": "Spearman"}]),
                    dict(label="Kendall",
                         method="update",
                         args=[{"visible": [False] * 18 + [True] * 6 + [False] * 12},
                               {"title": "Kendall"}]),
                    dict(label="Partial Correlation",
                         method="update",
                         args=[{"visible": [False] * 24 + [True] * 6 + [False] * 6},
                               {"title": "Partial Correlation"}]),
                    dict(label="MSE",
                         method="update",
                         args=[{"visible": [False] * 30 + [True] * 2 + [False] * 4},
                               {"title": "MSE"}]),
                    dict(label="RMSE",
                         method="update",
                         args=[{"visible": [False] * 32 + [True] * 2 + [False] * 2},
                               {"title": "RMSE"}]),
                    dict(label="MAE",
                         method="update",
                         args=[{"visible": [False] * 34 + [True] * 2},
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