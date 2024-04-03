import pandas as pd
import pathlib
import os

from suite.dataset import DrugResponseDataset
from suite.evaluation import evaluate, visualize_results, AVAILABLE_METRICS


def parse_results(id):
    result_dir = pathlib.Path(f'../results/{id}')
    # recursively find all the files in the result directory
    result_files = list(result_dir.rglob('*.csv'))
    # exclude the following files: evaluation_results.csv, evaluation_results_per_drug.csv, evaluation_results_per_cell_line.csv, true_vs_pred.csv
    result_files = [file for file in result_files if file.name not in ['evaluation_results.csv', 'evaluation_results_per_drug.csv', 'evaluation_results_per_cell_line.csv', 'true_vs_pred.csv']]
    evaluation_results = {}
    evaluation_results_per_drug = None
    evaluation_results_per_cell_line = None
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
        # overall evaluation
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
        result_per_drug = tmp_df.groupby('drug').apply(lambda x: evaluate(DrugResponseDataset(
            response=x['y_true'],
            cell_line_ids=x['cell_line'],
            drug_ids=x['drug'],
            predictions=x['y_pred']
        ), AVAILABLE_METRICS.keys()))
        drugs = result_per_drug.index
        result_per_drug = pd.json_normalize(result_per_drug)
        result_per_drug['drug'] = drugs
        result_per_drug['model'] = f"{algorithm}_{rand_setting}_{eval_setting}"
        if evaluation_results_per_drug is None:
            evaluation_results_per_drug = pd.DataFrame(result_per_drug)
        else:
            evaluation_results_per_drug = pd.concat([evaluation_results_per_drug, result_per_drug])
        result_per_cell_line = tmp_df.groupby('cell_line').apply(lambda x: evaluate(DrugResponseDataset(
            response=x['y_true'],
            cell_line_ids=x['cell_line'],
            drug_ids=x['drug'],
            predictions=x['y_pred']
        ), AVAILABLE_METRICS.keys()))
        cell_lines = result_per_cell_line.index
        result_per_cell_line = pd.json_normalize(result_per_cell_line)
        result_per_cell_line['cell_line'] = cell_lines
        result_per_cell_line['model'] = f"{algorithm}_{rand_setting}_{eval_setting}"
        if evaluation_results_per_cell_line is None:
            evaluation_results_per_cell_line = pd.DataFrame(result_per_cell_line)
        else:
            evaluation_results_per_cell_line = pd.concat([evaluation_results_per_cell_line, result_per_cell_line])
        true_vs_pred = pd.concat([true_vs_pred, tmp_df])
    evaluation_results = pd.DataFrame.from_dict(evaluation_results, orient='index')
    evaluation_results.to_csv(f'../results/{id}/evaluation_results.csv', index=True)
    evaluation_results_per_drug.to_csv(f'../results/{id}/evaluation_results_per_drug.csv', index=True)
    evaluation_results_per_cell_line.to_csv(f'../results/{id}/evaluation_results_per_cell_line.csv', index=True)
    true_vs_pred.to_csv(f'../results/{id}/true_vs_pred.csv', index=True)
    return evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred