params.run_id = "my_run"
params.models = "NaivePredictor"
params.test_mode = "LPO"
params.randomization_mode = "None"
params.randomization_type = "permutation"
params.n_trials_robustness = 0
params.dataset_name = "GDSC1"
params.path_out = "results/"
params.curve_curator = false
params.overwrite = false
params.optim_metric = "RMSE"
params.n_cv_splits = 5
params.response_transformation = "None"
params.multiprocessing = false

models_ch = channel.from(params.models)

process runModels {
    input:
    val model

    script:
    """
    cd ~/PyCharmProjects/drp_model_suite/
    echo "Running model $model"
    python run_suite.py --models $model --test_mode ${params.test_mode} --randomization_mode ${params.randomization_mode} --randomization_type ${params.randomization_type} --n_trials_robustness ${params.n_trials_robustness} --dataset_name ${params.dataset_name} --path_out ${params.path_out} --curve_curator ${params.curve_curator} --overwrite ${params.overwrite} --optim_metric ${params.optim_metric} --n_cv_splits ${params.n_cv_splits} --response_transformation ${params.response_transformation} --multiprocessing ${params.multiprocessing}
    """
}

workflow {
    runModels(models_ch)
}
