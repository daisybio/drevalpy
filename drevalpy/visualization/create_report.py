import os
import pandas as pd

from utils import parse_layout, parse_results, prep_results
from heatmap import generate_heatmap
from single_model_regression import generate_regression_plots
from violin import create_evaluation_violin
from scatter_eval_models import generate_scatter_eval_models_plot


def write_violins_and_heatmaps(f, setting, plot='Violin'):
    if plot == 'Violin':
        nav_id = 'violin'
        dir_name = 'violin_plots'
        prefix = 'violinplot'
    else:
        nav_id = 'heatmap'
        dir_name = 'heatmaps'
        prefix = 'heatmap'
    f.write(f'<h2 id="{nav_id}">{plot} Plots of Performance Measures over CV runs</h2>\n')
    f.write(f'<h3>{plot} plots comparing all models</h3>\n')
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
    f.write(f'<h3>{plot} plots comparing all models with normalized metrics</h3>\n')
    f.write(f'Before calculating the evaluation metrics, all values were normalized by the mean of the drug or cell line. '
            f'Since this only influences the R^2 and the correlation metrics, the error metrics are not shown. \n')
    f.write(
        f'<iframe src="{dir_name}/{prefix}_algorithms_{setting}_normalized.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
    plot_list = [f for f in os.listdir(f'../results/{run_id}/{dir_name}') if setting in f
                 and f != f'{prefix}_algorithms_{setting}.html'
                 and f != f'{prefix}_algorithms_{setting}_normalized.html']
    f.write(f'<h3>{plot} plots comparing performance measures for tests within each model</h3>\n')
    f.write('<ul>')
    for plot in plot_list:
        f.write(f'<li><a href="{dir_name}/{plot}" target="_blank">{plot}</a></li>\n')
    f.write('</ul>\n')


def write_scatter_eval_models(f, setting, group_by):
    group_comparison_list = [f for f in os.listdir(f'../results/{run_id}/scatter_eval_models') if
                            setting in f and group_by in f]
    if len(group_comparison_list) > 0:
        f.write('<h3 id="corr_comp_drug">Drug-wise comparison</h3>\n')
        f.write('<h4>Overall comparison between models</h4>\n')
        f.write(f'<iframe src="scatter_eval_models/scatter_eval_models_{group_by}_overall_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h4>Comparison between all models, dropdown menu</h4>\n')
        f.write(f'<iframe src="scatter_eval_models/scatter_eval_models_{group_by}_{setting}.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h4>Comparisons per model</h4>\n')
        f.write('<ul>\n')
        group_comparison_list = [elem for elem in group_comparison_list
                                 if elem != f'scatter_eval_models_{group_by}_{setting}.html' and
                                 elem != f'scatter_eval_models_{group_by}_overall_{setting}.html']
        group_comparison_list.sort()
        for group_comparison in group_comparison_list:
            f.write(f'<li><a href="scatter_eval_models/{group_comparison}" target="_blank">{group_comparison}</a></li>\n')
        f.write('</ul>\n')


def create_html(run_id, setting):
    # copy images to the results directory
    os.system(f'cp favicon.png ../results/{run_id}')
    os.system(f'cp nf-core-drugresponseeval_logo_light.png ../results/{run_id}')
    with open(f'../results/{run_id}/{setting}.html', 'w') as f:
        parse_layout(f)
        f.write(f'<h1>Results for {run_id}: {setting}</h1>\n')

        write_violins_and_heatmaps(f, setting, plot='Violin')
        write_violins_and_heatmaps(f, setting, plot='Heatmap')

        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write('<ul>\n')
        file_list = [f for f in os.listdir(f'../results/{run_id}/regression_plots') if setting in f]
        file_list.sort()
        for file in file_list:
            f.write(f'<li><a href="regression_plots/{file}" target="_blank">{file}</a></li>\n')
        f.write('</ul>\n')

        f.write('<h2 id="corr_comp">Comparison of correlation metrics</h2>\n')
        write_scatter_eval_models(f, setting, 'drug')
        write_scatter_eval_models(f, setting, 'cell_line')

        f.write('<h2 id="tables"> Evaluation Results Table</h2>\n')
        with open(f'../results/{run_id}/evaluation_results_{setting}.html', 'r') as eval_f:
            eval_results = eval_f.readlines()
            eval_results[0] = eval_results[0].replace('<table border="1" class="dataframe">', '<table class="display customDataTable" style="width:100%">')
            for line in eval_results:
                f.write(line)
        if setting != 'LCO':
            f.write('<h2> Evaluation Results per Cell Line Table</h2>\n')
            with open(f'../results/{run_id}/evaluation_results_per_cell_line_{setting}.html', 'r') as eval_f:
                eval_results = eval_f.readlines()
                eval_results[0] = eval_results[0].replace('<table border="1" class="dataframe">', '<table class="display customDataTable" style="width:100%">')
                for line in eval_results:
                    f.write(line)
        if setting != 'LDO':
            f.write('<h2> Evaluation Results per Drug Table</h2>\n')
            with open(f'../results/{run_id}/evaluation_results_per_drug_{setting}.html', 'r') as eval_f:
                eval_results = eval_f.readlines()
                eval_results[0] = eval_results[0].replace('<table border="1" class="dataframe">', '<table class="display customDataTable" style="width:100%">')
                for line in eval_results:
                    f.write(line)
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')


def create_index_html(run_id):
    # copy images to the results directory
    os.system(f'cp LPO.png ../results/{run_id}')
    os.system(f'cp LCO.png ../results/{run_id}')
    os.system(f'cp LDO.png ../results/{run_id}')
    with open(f'../results/{run_id}/index.html', 'w') as f:
        parse_layout(f, index=True)
        f.write(f'<h1>Results for {run_id}</h1>\n')
        f.write('<h2>Available settings</h2>\n')
        f.write('Click on the images to open the respective report in a new tab.\n')
        settings = [f.split('.html')[0] for f in os.listdir(f'../results/{run_id}') if f.endswith('.html') and f.startswith('L')]
        settings.sort()
        f.write('<div style="display: inline-block;">\n')
        for setting in settings:
            f.write(f'<a href="{setting}.html" target="_blank"><img src="{setting}.png" style="width:300px;height:300px;"></a>\n')
        f.write('</div>\n')
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')


def draw_violin_and_heatmap(df, run_id, plotname, whole_name=False, normalized_metrics=False):
    if not os.path.exists(f'../results/{run_id}/violin_plots'):
        os.mkdir(f'../results/{run_id}/violin_plots')
    if not os.path.exists(f'../results/{run_id}/heatmaps'):
        os.mkdir(f'../results/{run_id}/heatmaps')
    fig = create_evaluation_violin(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    fig.write_html(f'../results/{run_id}/violin_plots/violinplot_{plotname}.html')
    fig = generate_heatmap(df, normalized_metrics=normalized_metrics, whole_name=whole_name)
    fig.write_html(f'../results/{run_id}/heatmaps/heatmap_{plotname}.html')


def draw_scatter_grids_per_group(eval_res_group, group_by, setting, run_id):
    if not os.path.exists(f'../results/{run_id}/scatter_eval_models'):
        os.mkdir(f'../results/{run_id}/scatter_eval_models')
    eval_res_group_subset = eval_res_group[eval_res_group['LPO_LCO_LDO'] == setting]
    eval_res_group_models = eval_res_group_subset[eval_res_group_subset['rand_setting'] == 'predictions']
    # draw plots for comparison between all models
    fig, fig_overall = generate_scatter_eval_models_plot(eval_res_group_models, metric='Pearson',
                                                         color_by=group_by)
    fig.write_html(f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{setting}.html')
    fig_overall.write_html(
        f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{setting}.html')

    # draw plots per model: compare between original model and models with modification
    for algorithm in eval_res_group_models['algorithm'].unique():
        eval_res_group_algorithm = eval_res_group_subset[eval_res_group_subset['algorithm'] == algorithm]
        fig, fig_overall = generate_scatter_eval_models_plot(eval_res_group_algorithm, metric='Pearson',
                                                             color_by=group_by)
        fig.write_html(
            f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_{algorithm}_{setting}.html')
        fig_overall.write_html(
            f'../results/{run_id}/scatter_eval_models/scatter_eval_models_{group_by}_overall_{algorithm}_{setting}.html')


if __name__ == "__main__":
    # Load the dataset
    run_id = 'test3'
    evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred = prep_results(
        path_to_results=f'../results/{run_id}')

    settings = evaluation_results['LPO_LCO_LDO'].unique()

    for setting in settings:
        print(f'Generating report for {setting} ...')
        eval_results_subset = evaluation_results[evaluation_results['LPO_LCO_LDO'] == setting]
        true_vs_pred_subset = true_vs_pred[true_vs_pred['LPO_LCO_LDO'] == setting]

        # only draw figures for 'real' predictions comparing all models
        eval_results_algorithms = eval_results_subset[eval_results_subset['rand_setting'] == 'predictions']
        draw_violin_and_heatmap(eval_results_algorithms, run_id, f'algorithms_{setting}')

        # draw the same figures but with drug/cell-line normalized metrics
        draw_violin_and_heatmap(eval_results_algorithms, run_id, f'algorithms_{setting}_normalized', normalized_metrics=True)

        # draw figures for each algorithm with all randomizations etc
        for algorithm in eval_results_algorithms['algorithm'].unique():
            eval_results_algorithm = eval_results_subset[eval_results_subset['algorithm'] == algorithm]
            draw_violin_and_heatmap(eval_results_algorithm, run_id, f'{algorithm}_{setting}', whole_name=True)

        if setting == 'LPO' or setting == 'LCO':
            draw_scatter_grids_per_group(eval_res_group=evaluation_results_per_drug, group_by='drug', setting=setting,
                                        run_id=run_id)
            evaluation_results_per_drug_subs = evaluation_results_per_drug[evaluation_results_per_drug['LPO_LCO_LDO'] == setting]
            evaluation_results_per_drug_subs = evaluation_results_per_drug_subs[
                ['drug', 'algorithm', 'rand_setting', 'CV_split',
                'MSE', 'R^2', 'Pearson', 'RMSE', 'MAE', 'Spearman', 'Kendall', 'Partial_Correlation', 'LPO_LCO_LDO',]]
            evaluation_results_per_drug_subs.to_html(f'../results/{run_id}/evaluation_results_per_drug_{setting}.html', index=False)
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='cell_line')
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='cell_line', normalize=True)

        if setting == 'LPO' or setting == 'LDO':
            draw_scatter_grids_per_group(eval_res_group=evaluation_results_per_cell_line, group_by='cell_line',
                                         setting=setting, run_id=run_id)
            evaluation_results_per_cell_line_subs = evaluation_results_per_cell_line[evaluation_results_per_cell_line['LPO_LCO_LDO'] == setting]
            evaluation_results_per_cell_line_subs = evaluation_results_per_cell_line_subs[['cell_line', 'algorithm', 'rand_setting', 'CV_split',
                                                                                  'MSE', 'R^2', 'Pearson', 'RMSE', 'MAE', 'Spearman', 'Kendall', 'Partial_Correlation', 'LPO_LCO_LDO',]]
            evaluation_results_per_cell_line_subs.to_html(f'../results/{run_id}/evaluation_results_per_cell_line_{setting}.html', index=False)
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='drug')
            generate_regression_plots(true_vs_pred_subset, run_id, group_by='drug', normalize=True)
        # reorder columns
        eval_results_subset = eval_results_subset[['algorithm', 'rand_setting', 'CV_split',
                                                   'MSE', 'R^2', 'Pearson',
                                                   'R^2: drug normalized', 'Pearson: drug normalized',
                                                   'R^2: cell_line normalized', 'Pearson: cell_line normalized',
                                                   'RMSE', 'MAE',
                                                   'Spearman', 'Kendall', 'Partial_Correlation', 'Spearman: drug normalized', 'Kendall: drug normalized',
                                                   'Partial_Correlation: drug normalized', 'Spearman: cell_line normalized', 'Kendall: cell_line normalized',
                                                   'Partial_Correlation: cell_line normalized', 'LPO_LCO_LDO']]
        eval_results_subset.to_html(f'../results/{run_id}/evaluation_results_{setting}.html', index=False)
        create_html(run_id, setting)
    create_index_html(run_id)
