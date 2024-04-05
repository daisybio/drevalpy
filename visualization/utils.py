import pandas as pd
import pathlib
import os

from dreval.dataset import DrugResponseDataset
from dreval.evaluation import evaluate, AVAILABLE_METRICS


def parse_results(id):
    print('Generating result tables ...')
    result_dir = pathlib.Path(f'../results/{id}')
    # recursively find all the files in the result directory
    result_files = list(result_dir.rglob('*.csv'))
    # exclude the following files: evaluation_results.csv, evaluation_results_per_drug.csv, evaluation_results_per_cell_line.csv, true_vs_pred.csv
    result_files = [file for file in result_files if file.name not in ['evaluation_results.csv', 'evaluation_results_per_drug.csv', 'evaluation_results_per_cell_line.csv', 'true_vs_pred.csv']]
    evaluation_results = {}
    norm_drug_eval_results = {}
    norm_cell_line_eval_results = {}
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
        if 'LPO' in eval_setting or 'LCO' in eval_setting:
            # calculate the mean of y_true per drug
            tmp_df['mean_y_true_per_drug'] = tmp_df.groupby('drug')['y_true'].transform('mean')
            norm_df = tmp_df.copy()
            norm_df['y_true'] = norm_df['y_true'] - norm_df['mean_y_true_per_drug']
            norm_df['y_pred'] = norm_df['y_pred'] - norm_df['mean_y_true_per_drug']
            norm_drug_eval_results[f"{algorithm}_{rand_setting}_{eval_setting}"] = evaluate(DrugResponseDataset(
                response=norm_df['y_true'],
                cell_line_ids=norm_df['cell_line'],
                drug_ids=norm_df['drug'],
                predictions=norm_df['y_pred']
            ), AVAILABLE_METRICS.keys() - {"MSE", "RMSE", "MAE"})
            # evaluation per drug
            evaluation_results_per_drug = compute_evaluation(tmp_df, evaluation_results_per_drug, 'drug', algorithm,
                                                             rand_setting, eval_setting)
        if 'LPO' in eval_setting or 'LDO' in eval_setting:
            # calculate the mean of y_true per cell line
            tmp_df['mean_y_true_per_cell_line'] = tmp_df.groupby('cell_line')['y_true'].transform('mean')
            norm_df = tmp_df.copy()
            norm_df['y_true'] = norm_df['y_true'] - norm_df['mean_y_true_per_cell_line']
            norm_df['y_pred'] = norm_df['y_pred'] - norm_df['mean_y_true_per_cell_line']
            norm_cell_line_eval_results[f"{algorithm}_{rand_setting}_{eval_setting}"] = evaluate(DrugResponseDataset(
                response=norm_df['y_true'],
                cell_line_ids=norm_df['cell_line'],
                drug_ids=norm_df['drug'],
                predictions=norm_df['y_pred']
            ), AVAILABLE_METRICS.keys() - {"MSE", "RMSE", "MAE"})
            # evaluation per cell line
            evaluation_results_per_cell_line = compute_evaluation(tmp_df, evaluation_results_per_cell_line, 'cell_line', algorithm, rand_setting, eval_setting)

        true_vs_pred = pd.concat([true_vs_pred, tmp_df])

    evaluation_results = pd.DataFrame.from_dict(evaluation_results, orient='index')
    if norm_drug_eval_results != {}:
        norm_drug_eval_results = pd.DataFrame.from_dict(norm_drug_eval_results, orient='index')
        # append 'drug normalized ' to the column names
        norm_drug_eval_results.columns = [f'{col}: drug normalized' for col in norm_drug_eval_results.columns]
        evaluation_results = pd.concat([evaluation_results, norm_drug_eval_results], axis=1)
        evaluation_results_per_drug.to_csv(f'../results/{id}/evaluation_results_per_drug.csv', index=True)
    if norm_cell_line_eval_results != {}:
        norm_cell_line_eval_results = pd.DataFrame.from_dict(norm_cell_line_eval_results, orient='index')
        # append 'cell line normalized ' to the column names
        norm_cell_line_eval_results.columns = [f'{col}: cell line normalized' for col in norm_cell_line_eval_results.columns]
        evaluation_results = pd.concat([evaluation_results, norm_cell_line_eval_results], axis=1)
        evaluation_results_per_cell_line.to_csv(f'../results/{id}/evaluation_results_per_cell_line.csv', index=True)

    evaluation_results.to_csv(f'../results/{id}/evaluation_results.csv', index=True)
    true_vs_pred.to_csv(f'../results/{id}/true_vs_pred.csv', index=True)

    return evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred


def compute_evaluation(df, return_df, group_by, algorithm, rand_setting, eval_setting):
    result_per_group = df.groupby(group_by).apply(lambda x: evaluate(DrugResponseDataset(
        response=x['y_true'],
        cell_line_ids=x['cell_line'],
        drug_ids=x['drug'],
        predictions=x['y_pred']
    ), AVAILABLE_METRICS.keys()))
    groups = result_per_group.index
    result_per_group = pd.json_normalize(result_per_group)
    result_per_group[group_by] = groups
    result_per_group['model'] = f"{algorithm}_{rand_setting}_{eval_setting}"
    if return_df is None:
        return_df = pd.DataFrame(result_per_group)
    else:
        return_df = pd.concat([return_df, result_per_group])
    return return_df
