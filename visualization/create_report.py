import os

import pandas as pd

from utils import parse_results
from heatmap import generate_heatmap
from single_model_regression import generate_regression_plots
from violin import create_evaluation_violin
from scatter_eval_models import generate_scatter_eval_models_plot


def create_index_html(id):
    # copy images to the results directory
    os.system(f'cp favicon.png ../results/{id}')
    os.system(f'cp nf-core-drugresponseeval_logo_light.png ../results/{id}')
    with open(f'../results/{id}/index.html', 'w') as f:
        f.write('<html>\n')
        f.write('<head>\n')
        f.write(f'<title>Results for Run {id}</title>\n')
        f.write('<link rel="icon" href="favicon.png">\n')
        f.write(
            '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome'
            '.min.css">\n')
        f.write('<style>\n')
        f.write('body {font-family: Avenir, sans-serif;}\n')
        f.write('ul {list-style-type: none;}\n')
        f.write('li {padding: 5px;}\n')
        f.write(
            '.sidenav {\nwidth: 200px;\nposition: fixed;\nz-index: 1;\ntop: 20px;\nleft: 10px;\nbackground: '
            '#eee;\noverflow-x: hidden;\npadding: 8px 0;\n}\n')
        f.write(
            '.sidenav a {\npadding: 6px 8px 6px 16px;\ntext-decoration: none;\nfont-size: 20px;\ncolor: '
            '#2196F3;\ndisplay: block;\n}\n')
        f.write('.sidenav a:hover {\ncolor: #0645AD;\n}\n')
        f.write('.main {\nmargin-left: 220px;\nfont-size: 16px;\npadding: 0px 10 px;}\n')
        f.write(
            '@media screen and (max-height: 450px) {\n.sidenav {padding-top: 15px;}\n.sidenav a {font-size: 18px;}\n}\n')
        f.write(
            '.button {\nfont: 12px Avenir; text-decoration: none; background-color: DodgerBlue; color: white; '
            'padding: 2px 6px 2px 6px; \n}\n')
        f.write('</style>\n')
        f.write('</head>\n')

        f.write('<body>\n')

        f.write('<div class="sidenav">\n')
        f.write(
            '<img src="favicon.png" width="80px" height="80px" alt="Logo" style="margin-left: auto; margin-right: '
            'auto; display: block; width=50%;">\n')
        f.write('<p style="text-align: center; color: #737272; font-size: 12px; ">v0.1</p>\n')
        f.write('<a href="#violin">Violin Plot</a>\n')
        f.write('<a href="#heatmap">Heatmap</a>\n')
        f.write('<a href="#regression_plots">Regression plots</a>\n')
        f.write('<a href="#regression_plots_comp">Regression plots comparison</a>\n')
        f.write('</div>\n')

        f.write('<div class="main">\n')
        f.write('<img src="nf-core-drugresponseeval_logo_light.png" width="364px" height="100px" alt="Logo">\n')
        f.write(f'<h1>Results for {id}</h1>\n')
        f.write('<p>')
        f.write(
            '<a href="evaluation_results.csv" class="button" download="evaluation_results.csv" target="_blank"><i '
            'class="fa fa-download"></i> Download Evaluation Metrics</a>\n')
        f.write(
            '<a href="true_vs_pred.csv" class="button" download="true_vs_pred.csv" target="_blank"><i class="fa '
            'fa-download"></i> Download True vs. Predicted Values</a>\n')
        f.write('</p>')
        f.write('<h2 id="violin">Violin Plots of Performance Measures over CV runs</h2>\n')
        f.write('<iframe src="boxplot.html" width="100%" height="80%" frameBorder="0"></iframe>\n')
        f.write('<h2 id="heatmap">Heatmap for Performance Measures for Every Run</h2>\n')
        f.write('<iframe src="heatmap.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h2 id="regression_plots">Regression plots</h2>\n')
        f.write('<ul>\n')
        file_list = os.listdir(f'../results/{id}/regression_plots')
        file_list.sort()
        for file in file_list:
            f.write(f'<li><a href="regression_plots/{file}" target="_blank">{file}</a></li>\n')
        f.write('</ul>\n')
        f.write('<h2 id="regression_plots_comp">Comparison of Pearson correlation between all models</h2>\n')
        f.write('<iframe src="scatter_eval_models_overall.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('<h2>Comparison of Pearson correlation between two models</h2>\n')
        f.write('<iframe src="scatter_eval_models.html" width="100%" height="100%" frameBorder="0"></iframe>\n')
        f.write('</div>\n')
        f.write('</body>\n')
        f.write('</html>\n')


if __name__ == "__main__":
    # Load the dataset
    #evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred = parse_results('my_run')
    #evaluation_results_per_drug = pd.read_csv(f'../results/my_run/evaluation_results_per_drug.csv')
    #fig, fig_overall = generate_scatter_eval_models_plot(evaluation_results_per_drug, "Pearson")
    #fig.write_html('../results/my_run/scatter_eval_models.html')
    #fig_overall.write_html('../results/my_run/scatter_eval_models_overall.html')
    #fig = create_evaluation_violin(evaluation_results)
    #fig.write_html('../results/my_run/boxplot.html')
    #fig = generate_heatmap(evaluation_results)
    #fig.write_html('../results/my_run/heatmap.html')
    # generate_regression_plots(true_vs_pred, 'my_run')
    create_index_html('my_run')
