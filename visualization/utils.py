import pandas as pd
import pathlib
import os

from dreval.datasets.dataset import DrugResponseDataset
from dreval.evaluation import evaluate, AVAILABLE_METRICS


def parse_results(run_id):
    print('Generating result tables ...')
    # generate list of all result files
    result_dir = pathlib.Path(f'../results/{run_id}')
    result_files = list(result_dir.rglob('*.csv'))
    result_files = [file for file in result_files if file.name not in ['evaluation_results.csv',
                                                                       'evaluation_results_per_drug.csv',
                                                                       'evaluation_results_per_cell_line.csv',
                                                                       'true_vs_pred.csv']]
    # inititalize dictionaries to store the evaluation results
    evaluation_results = {}
    norm_drug_eval_results = {}
    norm_cell_line_eval_results = {}
    evaluation_results_per_drug = None
    evaluation_results_per_cell_line = None
    true_vs_pred = pd.DataFrame({'model': [], 'y_true': [], 'y_pred': []})

    # read every result file and compute the evaluation metrics
    for file in result_files:
        print('Parsing file:', os.path.normpath(file))
        result = pd.read_csv(file)
        dataset = DrugResponseDataset(
            response=result['response'],
            cell_line_ids=result['cell_line_ids'],
            drug_ids=result['drug_ids'],
            predictions=result['predictions']
        )
        model = generate_model_names(file)

        # overall evaluation
        evaluation_results[model] = evaluate(dataset, AVAILABLE_METRICS.keys())

        tmp_df = pd.DataFrame({
            'model': [model for _ in range(len(dataset.response))],
            'drug': dataset.drug_ids,
            'cell_line': dataset.cell_line_ids,
            'y_true': dataset.response,
            'y_pred': dataset.predictions})

        if 'LPO' in model or 'LCO' in model:
            norm_drug_eval_results, evaluation_results_per_drug = evaluate_per_group(df=tmp_df,
                                                                                     group_by='drug',
                                                                                     norm_group_eval_results=norm_drug_eval_results,
                                                                                     eval_results_per_group=evaluation_results_per_drug,
                                                                                     model=model)
        if 'LPO' in model or 'LDO' in model:
            norm_cell_line_eval_results, evaluation_results_per_cell_line = evaluate_per_group(df=tmp_df,
                                                                                                 group_by='cell_line',
                                                                                                 norm_group_eval_results=norm_cell_line_eval_results,
                                                                                                 eval_results_per_group=evaluation_results_per_cell_line,
                                                                                                 model=model)

        true_vs_pred = pd.concat([true_vs_pred, tmp_df])

    evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred = write_results(eval_results=evaluation_results,
                                                                                                                    norm_d_results=norm_drug_eval_results,
                                                                                                                    eval_results_d=evaluation_results_per_drug,
                                                                                                                    norm_cl_results=norm_cell_line_eval_results,
                                                                                                                    eval_results_cl=evaluation_results_per_cell_line,
                                                                                                                    t_vs_p=true_vs_pred,
                                                                                                                    run_id=run_id)
    return evaluation_results, evaluation_results_per_drug, evaluation_results_per_cell_line, true_vs_pred


def generate_model_names(file):
    file_parts = os.path.normpath(file).split('/')
    algorithm = file_parts[4]
    randomization = file_parts[-3].split('_')[0]
    rand_setting = file_parts[-2].replace('_', '-')
    if randomization == 'randomization':
        rand_setting = 'randomize-' + file_parts[-2].replace('_', '-')
    filename = file_parts[-1]
    # overall evaluation
    eval_setting = f"{filename.split('_')[2]}_split_{filename.split('_')[4].split('.')[0]}"
    model = f"{algorithm}_{rand_setting}_{eval_setting}"
    return model


def evaluate_per_group(df, group_by, norm_group_eval_results, eval_results_per_group, model):
    # calculate the mean of y_true per drug
    print(f'Calculating {group_by}-wise evaluation measures …')
    df[f'mean_y_true_per_{group_by}'] = df.groupby(group_by)['y_true'].transform('mean')
    norm_df = df.copy()
    norm_df['y_true'] = norm_df['y_true'] - norm_df[f'mean_y_true_per_{group_by}']
    norm_df['y_pred'] = norm_df['y_pred'] - norm_df[f'mean_y_true_per_{group_by}']
    norm_group_eval_results[model] = evaluate(DrugResponseDataset(
        response=norm_df['y_true'],
        cell_line_ids=norm_df['cell_line'],
        drug_ids=norm_df['drug'],
        predictions=norm_df['y_pred']
    ), AVAILABLE_METRICS.keys() - {"MSE", "RMSE", "MAE"})
    # evaluation per group
    eval_results_per_group = compute_evaluation(df, eval_results_per_group, group_by, model)
    return norm_group_eval_results, eval_results_per_group


def compute_evaluation(df, return_df, group_by, model):
    result_per_group = df.groupby(group_by).apply(lambda x: evaluate(DrugResponseDataset(
        response=x['y_true'],
        cell_line_ids=x['cell_line'],
        drug_ids=x['drug'],
        predictions=x['y_pred']
    ), AVAILABLE_METRICS.keys()))
    groups = result_per_group.index
    result_per_group = pd.json_normalize(result_per_group)
    result_per_group[group_by] = groups
    result_per_group['model'] = model
    if return_df is None:
        return_df = pd.DataFrame(result_per_group)
    else:
        return_df = pd.concat([return_df, result_per_group])
    return return_df


def write_results(eval_results, norm_d_results, eval_results_d, norm_cl_results, eval_results_cl, t_vs_p, run_id):
    eval_results = pd.DataFrame.from_dict(eval_results, orient='index')
    if norm_d_results != {}:
        eval_results, eval_results_d = write_group_results(norm_d_results, 'drug', eval_results, eval_results_d, run_id)
    if norm_cl_results != {}:
        eval_results, eval_results_cl = write_group_results(norm_cl_results, 'cell_line', eval_results, eval_results_cl, run_id)

    eval_results.to_csv(f'../results/{run_id}/evaluation_results.csv', index=True)
    t_vs_p.to_csv(f'../results/{run_id}/true_vs_pred.csv', index=True)
    return eval_results, eval_results_d, eval_results_cl, t_vs_p


def write_group_results(norm_group_res, group_by, eval_res, eval_res_group, run_id):
    norm_group_res = pd.DataFrame.from_dict(norm_group_res, orient='index')
    # append 'group normalized ' to the column names
    norm_group_res.columns = [f'{col}: {group_by} normalized' for col in norm_group_res.columns]
    eval_res = pd.concat([eval_res, norm_group_res], axis=1)
    eval_res_group.to_csv(f'../results/{run_id}/evaluation_results_per_{group_by}.csv', index=True)
    return eval_res, eval_res_group
