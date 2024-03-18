import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
from plotly.subplots import make_subplots
import numpy as np
import pathlib
import pandas as pd
import os
from suite.evaluation import evaluate, visualize_results, AVAILABLE_METRICS
from suite.dataset import DrugResponseDataset


def parse_results(id):
    result_dir = pathlib.Path(f'../results/{id}')
    # recursively find all the files in the result directory
    result_files = list(result_dir.rglob('*.csv'))
    evaluation_results = {}
    true_vs_pred = pd.DataFrame({'algorithm': [], 'rand_setting': [], 'eval_setting': [], 'y_true': [], 'y_pred': []})
    for file in result_files:
        result = pd.read_csv(file)
        dataset = DrugResponseDataset(
            response=result['response'],
            cell_line_ids=result['cell_line_ids'],
            drug_ids=result['drug_ids'],
            predictions=result['predictions']
        )
        file_parts = os.path.normpath(file).split('/')
        algorithm = file_parts[3]
        rand_setting = file_parts[-2].replace('_', '-')
        filename = file_parts[-1]
        eval_setting = f"{filename.split('_')[2]}_split_{filename.split('_')[4].split('.')[0]}"
        evaluation_results[f"{algorithm}_{rand_setting}_{eval_setting}"] = evaluate(dataset, AVAILABLE_METRICS.keys())
        tmp_df = pd.DataFrame({
            'algorithm': [algorithm for _ in range(len(dataset.response))],
            'rand_setting': [rand_setting for _ in range(len(dataset.response))],
            'eval_setting': [eval_setting for _ in range(len(dataset.response))],
            'drug': dataset.drug_ids,
            'cell_line': dataset.cell_line_ids,
            'y_true': dataset.response,
            'y_pred': dataset.predictions})
        true_vs_pred = pd.concat([true_vs_pred, tmp_df])
    evaluation_results = pd.DataFrame.from_dict(evaluation_results, orient='index')
    return evaluation_results, true_vs_pred


def generate_heatmap(df: pd.DataFrame):
    #df = df.fillna(0)
    df.sort_index(inplace=True)
    # drop r^2, mse, rmse
    df_errors = df[['mse', 'rmse', 'mae']]
    df_corrs = df[['pearson', 'spearman', 'kendall', 'partial_correlation']]
    titles = ['R^2', 'Correlations', 'Errors']
    fig = make_subplots(rows=3, cols=1, subplot_titles=tuple(titles))
    for i in range(3):
        if i == 0:
            # heatmap for r^2
            dt = df[['r2']].sort_values(by='r2', ascending=True)
            fig=fig.add_trace(
                go.Heatmap(z=dt.values, x=['R^2'], y=list(dt.index), colorscale='Blues', texttemplate='%{z:.2f}'),
                row=1, col=1
            )
        elif i == 1:
            # heatmap for correlations
            dt = df_corrs.sort_values(by='pearson', ascending=True)
            fig=fig.add_trace(
                go.Heatmap(z=dt.values, x=list(dt.columns), y=list(dt.index), colorscale='Viridis', texttemplate='%{z:.2f}'),
                row=2, col=1
            )
        else:
            # heatmap for errors
            dt = df_errors.sort_values(by='mse', ascending=False)
            fig=fig.add_trace(
                go.Heatmap(z=dt.values, x=list(dt.columns), y=list(dt.index), colorscale='hot', texttemplate='%{z:.2f}'),
                row=3, col=1
            )
    fig.update_layout(height=1000, width=1000, title_text="Heatmap of the evaluation metrics")
    fig.update_traces(showscale=False)
    return fig


def generate_regression_plots(df: pd.DataFrame, id):
    if not os.path.exists(f'../results/{id}/regression_plots'):
        os.mkdir(f'../results/{id}/regression_plots')
    sccs = df.groupby(['algorithm', 'rand_setting', 'eval_setting', 'drug']).apply(lambda x: scipy.stats.pearsonr(x['y_true'], x['y_pred'])[0])
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
    fig = px.scatter(df, x="y_true", y="y_pred", color="drug", trendline="ols",
                     hover_name="drug", hover_data=["scc", "cell_line"], title=f"{df['combination'].unique()[0]}: Regression plot")

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


def create_index_html(id):
    # copy favicon.png to the results directory
    os.system(f'cp favicon.png ../results/{id}')
    with open(f'../results/{id}/index.html', 'w') as f:
        f.write('<html>\n')
        f.write('<head>\n')
        f.write(f'<title>Results for Run {id}</title>\n')
        f.write('<link rel="icon" href="favicon.png">\n')
        f.write('<style>\n')
        f.write('body {font-family: Avenir, sans-serif;}\n')
        f.write('ul {list-style-type: none;}\n')
        f.write('li {padding: 5px;}\n')
        f.write('</style>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write(f'<h1>Results for {id}</h1>\n')
        f.write('<h2>Heatmap</h2>\n')
        f.write(f'<iframe src="heatmap.html" width="1000" height="1000"></iframe>\n')
        f.write('<h2>Regression plots</h2>\n')
        f.write('<ul>\n')
        file_list = os.listdir(f'../results/{id}/regression_plots')
        file_list.sort()
        for file in file_list:
            f.write(f'<li><a href="regression_plots/{file}" target="_blank">{file}</a></li>\n')
        f.write('</ul>\n')
        f.write('</body>\n')
        f.write('</html>\n')


if __name__ == "__main__":
    # Load the dataset
    #evaluation_results, true_vs_pred = parse_results('my_run')
    #fig = generate_heatmap(evaluation_results)
    #fig.write_html('../results/my_run/heatmap.html')
    #generate_regression_plots(true_vs_pred, 'my_run')
    create_index_html('my_run')
