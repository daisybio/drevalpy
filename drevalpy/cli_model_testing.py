"""For the nf-core/drugresponseeval subworkflow model_testing."""

import argparse
import json
import pathlib
import pickle

import pandas as pd
import yaml


def _prep_data_for_final_prediction(arguments):
    """Helper function to load the data and prepare it for training and prediction.

    :param arguments: Command line arguments = model_name, split_dataset_path, split_id, hyperparameters_path,
        response_transformation
    :return: The instantiated model, the best hyperparameters, training dataset (=train + val), test dataset,
        early stopping dataset, response transformation
    """
    from drevalpy.experiment import get_datasets_from_cv_split, get_model_name_and_drug_id
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    # instantiate model
    model_name, drug_id = get_model_name_and_drug_id(arguments.model_name)
    model_class = MODEL_FACTORY[model_name]
    model = model_class()
    # load the data
    with open(arguments.split_dataset_path, "rb") as split_file:
        split = pickle.load(split_file)
    train_dataset, validation_dataset, es_dataset, test_dataset = get_datasets_from_cv_split(
        split, model_class, model_name, drug_id
    )

    if model_class.early_stopping:
        validation_dataset = split["validation_es"]
        es_dataset = split["early_stopping"]
    else:
        es_dataset = None
    # append the validation dataset to the training dataset because we now predict the test set with the
    # optimal hyperparameters
    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)
    # get optimal hyperparameters
    with open(arguments.hyperparameters_path) as f:
        best_hpam_dict = yaml.safe_load(f)
    best_hpams = best_hpam_dict[f"{arguments.model_name}_{arguments.split_id}"]["best_hpam_combi"]
    # get response transformation
    response_transform = get_response_transformation(arguments.response_transformation)
    return model, drug_id, best_hpams, train_dataset, test_dataset, es_dataset, response_transform


def train_and_predict_final():
    """CLI for predicting the CV fold test set with the best hyperparameter configuration.

    Either in full mode, randomization mode, or robustness mode.
    :raises ValueError: If mode is not full, randomization, or robustness.
    """
    from drevalpy.experiment import (
        cross_study_prediction,
        generate_data_saving_path,
        randomize_train_predict,
        robustness_train_predict,
        train_and_predict,
    )

    # create parser
    parser = argparse.ArgumentParser(
        description="Train and predict: either full mode, randomization mode, " "or robustness mode."
    )
    parser.add_argument(
        "--mode", type=str, default="full", help="Mode: full, randomization, or robustness. Default: full."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for global models, <Model name>.<Drug name> for single-drug models.",
    )
    parser.add_argument("--split_id", type=str, required=True, help="Split id.")
    parser.add_argument("--split_dataset_path", type=str, required=True, help="Path to the pickled CV split dataset.")
    parser.add_argument(
        "--hyperparameters_path",
        type=str,
        required=True,
        help="Path to yaml file containing the optimal hyperparameters.",
    )
    parser.add_argument(
        "--response_transformation", type=str, default="None", help="Response transformation. Default: None."
    )
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO.")
    parser.add_argument("--path_data", type=str, default="data", required=True, help="Path to data. Default: data")
    parser.add_argument(
        "--randomization_views_path",
        type=str,
        default=None,
        help="Path to the yaml file containing the randomization configuration (only relevant if mode=randomization).",
    )
    parser.add_argument(
        "--randomization_type",
        type=str,
        default="permutation",
        help="Randomization type (permutation, invariant). Default: permutation. Only relevant if mode=randomization.",
    )
    parser.add_argument(
        "--robustness_trial", type=int, help="Robustness trial index. Only relevant if mode=robustness."
    )
    parser.add_argument("--cross_study_datasets", nargs="+", help="(List of) path(s) to pickled cross study datasets.")
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="TEMPORARY",
        help="model checkpoint directory, if not provided: temporary directory is used",
    )
    args = parser.parse_args()

    # load all required objects
    selected_model, drug_id, hpam_combi, train_set, test_set, es_set, transformation = _prep_data_for_final_prediction(
        args
    )
    if args.mode == "full":
        predictions_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="predictions",
        )
        hpam_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="best_hpams",
        )
        hpam_path = pathlib.Path(hpam_path) / f"best_hpams_{args.split_id}.json"
        # save the best hyperparameters as json
        with open(
            hpam_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(hpam_combi, f)

        test_set = train_and_predict(
            model=selected_model,
            hpams=hpam_combi,
            path_data=args.path_data,
            train_dataset=train_set,
            prediction_dataset=test_set,
            early_stopping_dataset=es_set,
            response_transformation=transformation,
            model_checkpoint_dir=args.model_checkpoint_dir,
        )
        prediction_dataset = pathlib.Path(predictions_path) / f"predictions_{args.split_id}.csv"

        test_set.to_csv(prediction_dataset)
        # cross-study prediction
        for cs_ds in args.cross_study_datasets:
            if cs_ds == "NONE.csv":
                continue
            split_index = args.split_id.split("split_")[1]
            # load cross-study dataset
            with open(cs_ds, "rb") as cs_file:
                cross_study_dataset = pickle.load(cs_file)
            cross_study_dataset.remove_nan_responses()
            cross_study_prediction(
                dataset=cross_study_dataset,
                model=selected_model,
                test_mode=args.test_mode,
                train_dataset=train_set,
                path_data=args.path_data,
                early_stopping_dataset=(es_set if selected_model.early_stopping else None),
                response_transformation=transformation,
                path_out=str(pathlib.Path(predictions_path).parent),
                split_index=split_index,
                single_drug_id=drug_id,
            )
    elif args.mode == "randomization":
        with open(args.randomization_views_path) as f:
            rand_test_view = yaml.safe_load(f)
        rand_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="randomization",
        )
        randomization_test_file = (
            pathlib.Path(rand_path) / f'randomization_{rand_test_view["test_name"]}_{args.split_id}.csv'
        )

        randomize_train_predict(
            view=rand_test_view["view"],
            test_name=rand_test_view["test_name"],
            randomization_type=args.randomization_type,
            randomization_test_file=str(randomization_test_file),
            model=selected_model,
            hpam_set=hpam_combi,
            path_data=args.path_data,
            train_dataset=train_set,
            test_dataset=test_set,
            early_stopping_dataset=es_set,
            response_transformation=transformation,
            model_checkpoint_dir=args.model_checkpoint_dir,
        )
    elif args.mode == "robustness":
        rob_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="robustness",
        )
        robustness_test_file = pathlib.Path(rob_path) / f"robustness_{args.robustness_trial}_{args.split_id}.csv"

        robustness_train_predict(
            trial=args.robustness_trial,
            trial_file=str(robustness_test_file),
            train_dataset=train_set,
            test_dataset=test_set,
            early_stopping_dataset=es_set,
            model=selected_model,
            hpam_set=hpam_combi,
            path_data=args.path_data,
            response_transformation=transformation,
            model_checkpoint_dir=args.model_checkpoint_dir,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose full, randomization, or robustness.")


def randomization_split():
    """CLI for creating randomization test view files."""
    from drevalpy.experiment import get_randomization_test_views
    from drevalpy.models import MODEL_FACTORY

    # define parser
    parser = argparse.ArgumentParser(description="Create randomization test views, saves them as yamls.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--randomization_mode", type=str, required=True, help="Randomization mode to use.")
    args = parser.parse_args()

    model_class = MODEL_FACTORY[args.model_name]
    model = model_class()

    randomization_test_views = get_randomization_test_views(model=model, randomization_mode=[args.randomization_mode])
    for test_name, views in randomization_test_views.items():
        for view in views:
            rand_dict = {"test_name": test_name, "view": view}
            with open(f"randomization_test_view_{test_name}.yaml", "w") as f:
                yaml.dump(rand_dict, f)


def final_split():
    """CLI creating the final split pkls for a production model (no prediction, training on full dataset)."""
    from drevalpy.datasets.dataset import split_early_stopping_data
    from drevalpy.experiment import make_train_val_split
    from drevalpy.models import MODEL_FACTORY

    # define parser
    parser = argparse.ArgumentParser(
        description="Splits to train a final model on the full dataset for future predictions "
        "and saves them as pickles."
    )
    parser.add_argument(
        "--response", type=str, required=True, help="Drug response data, pickled (output of load_response)."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model class name, e.g., RandomForest, SingleDrugRandomForest."
    )
    parser.add_argument("--path_data", type=str, default="data", required=True, help="Path to data. Default: data.")
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio.")
    args = parser.parse_args()

    # load data
    with open(args.response, "rb") as response_file:
        response_data = pickle.load(response_file)
    response_data.remove_nan_responses()
    # get model features to reduce dataset
    model_class = MODEL_FACTORY[args.model_name]
    model = model_class()
    cl_features = model.load_cell_line_features(data_path=args.path_data, dataset_name=response_data.dataset_name)
    drug_features = model.load_drug_features(data_path=args.path_data, dataset_name=response_data.dataset_name)
    cell_lines_to_keep = cl_features.identifiers
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None
    response_data.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    # make the final split: only train and validation
    train_dataset, validation_dataset = make_train_val_split(
        response_data, test_mode=args.test_mode, val_ratio=args.val_ratio
    )

    if model_class.early_stopping:
        validation_dataset, early_stopping_dataset = split_early_stopping_data(validation_dataset, args.test_mode)
    else:
        early_stopping_dataset = None

    # save datasets to pkl
    with open("training_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("validation_dataset.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)
    with open("early_stopping_dataset.pkl", "wb") as f:
        pickle.dump(early_stopping_dataset, f)


def tune_final_model():
    """CLI for tuning the final model on the full dataset."""
    from drevalpy.experiment import get_model_name_and_drug_id, train_and_predict
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    # define parser
    parser = argparse.ArgumentParser(
        description="Finding the optimal hyperparameters for the final model trained "
        "on the full dataset for future predictions."
    )
    parser.add_argument("--train_data", type=str, required=True, help="Train dataset, pickled.")
    parser.add_argument("--val_data", type=str, required=True, help="Validation dataset, pickled.")
    parser.add_argument("--early_stopping_data", type=str, required=True, help="Early stopping dataset, pickled.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
    )
    parser.add_argument(
        "--hpam_combi", type=str, required=True, help="Path to hyperparameter combination file, yaml format."
    )
    parser.add_argument(
        "--response_transformation", type=str, default="None", help="Response transformation. Default: None."
    )
    parser.add_argument("--path_data", type=str, default="data", required=True, help="Path to data. Default: data.")
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="TEMPORARY",
        help="model checkpoint directory, if not provided: temporary directory is used",
    )
    args = parser.parse_args()

    # load data
    with open(args.train_data, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    with open(args.val_data, "rb") as val_file:
        validation_dataset = pickle.load(val_file)
    with open(args.early_stopping_data, "rb") as es_file:
        early_stopping_dataset = pickle.load(es_file)
    response_transform = get_response_transformation(args.response_transformation)

    # instantiate and train model
    model_name, drug_id = get_model_name_and_drug_id(args.model_name)
    model_class = MODEL_FACTORY[model_name]
    with open(args.hpam_combi) as f:
        hpams = yaml.safe_load(f)
    model = model_class()

    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        path_data=args.path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=args.model_checkpoint_dir,
    )
    # save predictions to pkl
    with open(f"final_prediction_dataset_{model_name}_" f"{str(args.hpam_combi).split('.yaml')[0]}.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)


def train_final_model():
    """CLI for training the final model on the full dataset with the optimal hyperparameters."""
    from drevalpy.experiment import generate_data_saving_path, get_model_name_and_drug_id
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    # define parser
    parser = argparse.ArgumentParser(
        description="Train a final model on the full dataset for future predictions using the best hyperparameters."
    )
    parser.add_argument("--train_data", type=str, required=True, help="Train data, pickled.")
    parser.add_argument("--val_data", type=str, required=True, help="Validation data, pickled.")
    parser.add_argument("--early_stopping_data", type=str, required=True, help="Early stopping data, pickled.")
    parser.add_argument("--response_transformation", type=str, default="None", help="Response transformation.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
    )
    parser.add_argument("--path_data", type=str, default="data", required=True, help="Path to data. Default: data.")
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default="TEMPORARY",
        help="model checkpoint directory, if not provided: temporary directory is used",
    )
    parser.add_argument(
        "--best_hpam_combi", type=str, required=True, help="Best hyperparameter combination file, yaml format."
    )
    args = parser.parse_args()

    # create relevant objects from args
    model_name, drug_id = get_model_name_and_drug_id(args.model_name)

    final_model_path = generate_data_saving_path(
        model_name=model_name, drug_id=drug_id, result_path="", suffix="final_model"
    )
    response_transform = get_response_transformation(args.response_transformation)
    with open(args.train_data, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    with open(args.val_data, "rb") as val_file:
        validation_dataset = pickle.load(val_file)
    with open(args.early_stopping_data, "rb") as es_file:
        es_dataset = pickle.load(es_file)
    # create dataset
    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)
    if response_transform:
        train_dataset.fit_transform(response_transform)
        if es_dataset is not None:
            es_dataset.transform(response_transform)
    # instantiate model
    with open(args.best_hpam_combi) as f:
        best_hpam_combi = yaml.safe_load(f)[f"{model_name}_final"]["best_hpam_combi"]
    model = MODEL_FACTORY[model_name]()
    cl_features = model.load_cell_line_features(data_path=args.path_data, dataset_name=train_dataset.dataset_name)
    drug_features = model.load_drug_features(data_path=args.path_data, dataset_name=train_dataset.dataset_name)
    model.build_model(hyperparameters=best_hpam_combi)

    # train
    model.train(
        output=train_dataset,
        output_earlystopping=es_dataset,
        cell_line_input=cl_features,
        drug_input=drug_features,
        model_checkpoint_dir=args.model_checkpoint_dir,
    )

    # save model for the future
    pathlib.Path(final_model_path).mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)


def consolidate_results():
    """CLI for consolidating the results of the single-drug models."""
    from drevalpy.experiment import consolidate_single_drug_model_predictions
    from drevalpy.models import MODEL_FACTORY

    # define parser
    parser = argparse.ArgumentParser(description="Consolidate results for SingleDrugModels")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    parser.add_argument("--test_mode", type=str, required=False, default="LPO", help="Test mode (LPO, LCO, LTO, LDO)")
    parser.add_argument("--model_name", type=str, required=True, help="All Model " "names")
    parser.add_argument("--outdir_path", type=str, required=True, help="Output directory path")
    parser.add_argument("--n_cv_splits", type=int, required=True, help="Number of CV splits")
    parser.add_argument("--cross_study_datasets", type=str, nargs="+", help="All " "cross-study " "datasets")
    parser.add_argument(
        "--randomization_modes", type=str, default="[None]", required=False, help="All " "randomizations"
    )
    parser.add_argument("--n_trials_robustness", type=int, default=0, required=False, help="Number of trials")
    args = parser.parse_args()

    # load relevant objects from args
    results_path = str(pathlib.Path(args.outdir_path) / args.run_id / args.test_mode)
    if args.randomization_modes == "[None]":
        randomizations = None
    else:
        randomizations = args.randomization_modes.split("[")[1].split("]")[0].split(", ")
    model = MODEL_FACTORY[args.model_name]
    if args.cross_study_datasets is None:
        args.cross_study_datasets = []
    # consolidate results into a single file
    consolidate_single_drug_model_predictions(
        models=[model],
        n_cv_splits=args.n_cv_splits,
        results_path=results_path,
        cross_study_datasets=args.cross_study_datasets,
        randomization_mode=randomizations,
        n_trials_robustness=args.n_trials_robustness,
        out_path="",
    )


def evaluate_test_results():
    """CLI for evaluating the results obtained on the test sets of the CV splits."""
    from drevalpy.visualization.utils import evaluate_file

    # define parser
    parser = argparse.ArgumentParser(description="Evaluate the predictions.")
    parser.add_argument("--test_mode", type=str, default="LPO", help="Test mode (LPO, LCO, LDO, LTO).")
    parser.add_argument("--model_name", type=str, required=True, help="Model name.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions.")
    args = parser.parse_args()

    # evaluate the files
    results_all, eval_res_d, eval_res_cl, t_vs_pred, mname = evaluate_file(
        test_mode=args.test_mode, model_name=args.model_name, pred_file=args.pred_file
    )
    # write the results to csvs
    results_all.to_csv(f"{mname}_evaluation_results.csv")
    if eval_res_d is not None:
        eval_res_d.to_csv(f"{mname}_evaluation_results_per_drug.csv")
    if eval_res_cl is not None:
        eval_res_cl.to_csv(f"{mname}_evaluation_results_per_cl.csv")
    t_vs_pred.to_csv(f"{mname}_true_vs_pred.csv")


def _parse_results(outfiles):
    # get all files with the pattern f'{model_name}_evaluation_results.csv' from outfiles
    result_files = [file for file in outfiles if "evaluation_results.csv" in file]
    # get all files with the pattern f'{model_name}_evaluation_results_per_drug.csv' from outfiles
    result_per_drug_files = [file for file in outfiles if "evaluation_results_per_drug.csv" in file]
    # get all files with the pattern f'{model_name}_evaluation_results_per_cl.csv' from outfiles
    result_per_cl_files = [file for file in outfiles if "evaluation_results_per_cl.csv" in file]
    # get all files with the pattern f'{model_name}_true_vs_pred.csv' from outfiles
    t_vs_pred_files = [file for file in outfiles if "true_vs_pred.csv" in file]
    return result_files, result_per_drug_files, result_per_cl_files, t_vs_pred_files


def _collapse_file(files):
    out_df = None
    for file in files:
        if out_df is None:
            out_df = pd.read_csv(file, index_col=0)
        else:
            out_df = pd.concat([out_df, pd.read_csv(file, index_col=0)])
    if out_df is not None and "drug" in out_df.columns:
        out_df["drug"] = out_df["drug"].astype(str)
    return out_df


def collect_results():
    """CLI for collecting the results from the nextflow parallelization."""
    from drevalpy.visualization.utils import prep_results, write_results

    # define parser
    parser = argparse.ArgumentParser(description="Collect results and write to single files.")
    parser.add_argument(
        "--outfiles",
        type=str,
        nargs="+",
        required=True,
        help="List of all output files containing results, i.e., evaluation_results*csv + true_vs_pred.csv files.",
    )
    parser.add_argument("--path_data", type=str, default="data", help="Data directory path. Default: data.")
    args = parser.parse_args()
    # parse the results from outfiles.outfiles
    outfiles = args.outfiles
    path_data = pathlib.Path(args.path_data)
    eval_result_files, eval_result_per_drug_files, eval_result_per_cl_files, true_vs_pred_files = _parse_results(
        outfiles
    )

    # collapse the results into single dataframes
    eval_results = _collapse_file(eval_result_files)
    eval_results_per_drug = _collapse_file(eval_result_per_drug_files)
    eval_results_per_cell_line = _collapse_file(eval_result_per_cl_files)
    t_vs_p = _collapse_file(true_vs_pred_files)

    # prepare the results through introducing new columns algorithm, rand_setting, test_mode, split, CV_split
    eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p = prep_results(
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cell_line=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
        path_data=path_data,
    )

    # save the results to csv files
    write_results(
        path_out="",
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cl=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
    )
