"""For the nf-core/drugresponseeval subworkflow model_testing."""

import json
import pathlib
import pickle
from argparse import Namespace
from typing import Any

import pandas as pd
import yaml


def _prep_data_for_final_prediction(arguments: Namespace) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load data and prepare it for final CV-fold training and prediction."""
    from drevalpy.experiment import get_datasets_from_cv_split, get_model_name_and_drug_id
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    model_name, drug_id = get_model_name_and_drug_id(arguments.model_name)
    model_class = MODEL_FACTORY[model_name]
    model = model_class()
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
    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)
    with open(arguments.hyperparameters_path) as f:
        best_hpam_dict = yaml.safe_load(f)
    best_hpams = best_hpam_dict[f"{arguments.model_name}_{arguments.split_id}"]["best_hpam_combi"]
    response_transform = get_response_transformation(arguments.response_transformation)
    return model, drug_id, best_hpams, train_dataset, test_dataset, es_dataset, response_transform


def run_train_and_predict_final(
    *,
    mode: str = "full",
    model_name: str,
    split_id: str,
    split_dataset_path: str,
    hyperparameters_path: str,
    response_transformation: str = "None",
    test_mode: str = "LPO",
    path_data: str = "data",
    randomization_views_path: str | None = None,
    randomization_type: str = "permutation",
    robustness_trial: int | None = None,
    cross_study_datasets: list[str] | None = None,
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Train and predict on the CV test set (full, randomization, or robustness mode)."""
    from drevalpy.experiment import (
        cross_study_prediction,
        generate_data_saving_path,
        randomize_train_predict,
        robustness_train_predict,
        train_and_predict,
    )

    args = Namespace(
        mode=mode,
        model_name=model_name,
        split_id=split_id,
        split_dataset_path=split_dataset_path,
        hyperparameters_path=hyperparameters_path,
        response_transformation=response_transformation,
        test_mode=test_mode,
        path_data=path_data,
        randomization_views_path=randomization_views_path,
        randomization_type=randomization_type,
        robustness_trial=robustness_trial,
        cross_study_datasets=cross_study_datasets,
        model_checkpoint_dir=model_checkpoint_dir,
    )

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
        with open(hpam_path, "w", encoding="utf-8") as f:
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
        if args.cross_study_datasets:
            for cs_ds in args.cross_study_datasets:
                if cs_ds == "NONE.csv":
                    continue
                split_index = args.split_id.split("split_")[1]
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


def run_randomization_split(*, model_name: str, randomization_mode: str) -> None:
    """Create randomization test view YAML files for a model."""
    from drevalpy.experiment import get_randomization_test_views
    from drevalpy.models import MODEL_FACTORY

    model_class = MODEL_FACTORY[model_name]
    randomization_test_views: dict[str, list[str]] = {}
    for hpam_combi in model_class.get_hyperparameter_set():
        model = model_class()
        model.build_model(hpam_combi)
        randomization_test_views.update(
            get_randomization_test_views(model=model, randomization_mode=[randomization_mode])
        )

    if not randomization_test_views:
        raise RuntimeError(
            f"No randomization test views were produced for {model_name} with mode {randomization_mode}. "
            "Check that the model's hyperparameters.yaml declares cell_line_views/drug_views."
        )

    for test_name, views in randomization_test_views.items():
        for view in views:
            rand_dict = {"test_name": test_name, "view": view}
            with open(f"randomization_test_view_{test_name}.yaml", "w") as f:
                yaml.dump(rand_dict, f)


def run_final_split(
    *,
    response: str,
    model_name: str,
    path_data: str = "data",
    test_mode: str = "LPO",
    val_ratio: float = 0.1,
) -> None:
    """Create train/validation/early-stopping pickles for a final production model."""
    from drevalpy.datasets.dataset import split_early_stopping_data
    from drevalpy.experiment import make_train_val_split
    from drevalpy.models import MODEL_FACTORY

    with open(response, "rb") as response_file:
        response_data = pickle.load(response_file)
    response_data.remove_nan_responses()
    model_class = MODEL_FACTORY[model_name]
    model = model_class()
    cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=response_data.dataset_name)
    drug_features = model.load_drug_features(data_path=path_data, dataset_name=response_data.dataset_name)
    cell_lines_to_keep = cl_features.identifiers
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None
    response_data.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    train_dataset, validation_dataset = make_train_val_split(response_data, test_mode=test_mode, val_ratio=val_ratio)

    if model_class.early_stopping:
        validation_dataset, early_stopping_dataset = split_early_stopping_data(validation_dataset, test_mode)
    else:
        early_stopping_dataset = None

    with open("training_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("validation_dataset.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)
    with open("early_stopping_dataset.pkl", "wb") as f:
        pickle.dump(early_stopping_dataset, f)


def run_tune_final_model(
    *,
    train_data: str,
    val_data: str,
    early_stopping_data: str,
    model_name: str,
    hpam_combi: str,
    response_transformation: str = "None",
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Tune hyperparameters for the final model on full data."""
    from drevalpy.experiment import get_model_name_and_drug_id, train_and_predict
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    with open(train_data, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    with open(val_data, "rb") as val_file:
        validation_dataset = pickle.load(val_file)
    with open(early_stopping_data, "rb") as es_file:
        early_stopping_dataset = pickle.load(es_file)
    response_transform = get_response_transformation(response_transformation)

    resolved_name, _drug_id = get_model_name_and_drug_id(model_name)
    model_class = MODEL_FACTORY[resolved_name]
    with open(hpam_combi) as f:
        hpams = yaml.safe_load(f)
    model = model_class()

    validation_dataset = train_and_predict(
        model=model,
        hpams=hpams,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    with open(f"final_prediction_dataset_{resolved_name}_" f"{str(hpam_combi).split('.yaml')[0]}.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)


def run_train_final_model(
    *,
    train_data: str,
    val_data: str,
    early_stopping_data: str,
    response_transformation: str = "None",
    model_name: str,
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
    best_hpam_combi: str,
) -> None:
    """Train and save the final production model."""
    from drevalpy.experiment import generate_data_saving_path, get_model_name_and_drug_id
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    resolved_name, _drug_id = get_model_name_and_drug_id(model_name)
    final_model_path = generate_data_saving_path(
        model_name=resolved_name, drug_id=_drug_id, result_path="", suffix="final_model"
    )
    response_transform = get_response_transformation(response_transformation)
    with open(train_data, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    with open(val_data, "rb") as val_file:
        validation_dataset = pickle.load(val_file)
    with open(early_stopping_data, "rb") as es_file:
        es_dataset = pickle.load(es_file)
    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)
    if response_transform:
        train_dataset.fit_transform(response_transform)
        if es_dataset is not None:
            es_dataset.transform(response_transform)
    with open(best_hpam_combi) as f:
        best_hpam = yaml.safe_load(f)[f"{resolved_name}_final"]["best_hpam_combi"]
    model = MODEL_FACTORY[resolved_name]()
    cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    drug_features = model.load_drug_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    model.build_model(hyperparameters=best_hpam)
    model.train(
        output=train_dataset,
        output_earlystopping=es_dataset,
        cell_line_input=cl_features,
        drug_input=drug_features,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    pathlib.Path(final_model_path).mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)


def run_consolidate_results(
    *,
    run_id: str,
    test_mode: str = "LPO",
    model_name: str,
    outdir_path: str,
    n_cv_splits: int,
    cross_study_datasets: list[str] | None = None,
    randomization_modes: str = "[None]",
    n_trials_robustness: int = 0,
) -> None:
    """Consolidate single-drug model prediction outputs."""
    from drevalpy.experiment import consolidate_single_drug_model_predictions
    from drevalpy.models import MODEL_FACTORY

    results_path = str(pathlib.Path(outdir_path) / run_id / test_mode)
    if randomization_modes == "[None]":
        randomizations = None
    else:
        randomizations = randomization_modes.split("[")[1].split("]")[0].split(", ")
    model = MODEL_FACTORY[model_name]
    cross_study = cross_study_datasets or []
    consolidate_single_drug_model_predictions(
        models=[model],
        n_cv_splits=n_cv_splits,
        results_path=results_path,
        cross_study_datasets=cross_study,
        randomization_mode=randomizations,
        n_trials_robustness=n_trials_robustness,
        out_path="",
    )


def run_evaluate_test_results(
    *,
    test_mode: str = "LPO",
    model_name: str,
    pred_file: str,
) -> None:
    """Evaluate test predictions and write metric CSVs."""
    from drevalpy.visualization.utils import evaluate_file

    results_all, eval_res_d, eval_res_cl, t_vs_pred, mname = evaluate_file(
        test_mode=test_mode, model_name=model_name, pred_file=pred_file
    )
    results_all.to_csv(f"{mname}_evaluation_results.csv")
    if eval_res_d is not None:
        eval_res_d.to_csv(f"{mname}_evaluation_results_per_drug.csv")
    if eval_res_cl is not None:
        eval_res_cl.to_csv(f"{mname}_evaluation_results_per_cl.csv")
    t_vs_pred.to_csv(f"{mname}_true_vs_pred.csv")


def _parse_results(outfiles: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    result_files = [file for file in outfiles if "evaluation_results.csv" in file]
    result_per_drug_files = [file for file in outfiles if "evaluation_results_per_drug.csv" in file]
    result_per_cl_files = [file for file in outfiles if "evaluation_results_per_cl.csv" in file]
    t_vs_pred_files = [file for file in outfiles if "true_vs_pred.csv" in file]
    return result_files, result_per_drug_files, result_per_cl_files, t_vs_pred_files


def _collapse_file(files: list[str]) -> pd.DataFrame | None:
    out_df = None
    for file in files:
        if out_df is None:
            out_df = pd.read_csv(file, index_col=0)
        else:
            out_df = pd.concat([out_df, pd.read_csv(file, index_col=0)])
    if out_df is not None and "drug" in out_df.columns:
        out_df["drug"] = out_df["drug"].astype(str)
    return out_df


def run_collect_results(
    *,
    outfiles: list[str],
    path_data: str = "data",
) -> None:
    """Collect parallel Nextflow evaluation outputs into merged CSVs."""
    from drevalpy.visualization.utils import prep_results, write_results

    path_data_path = pathlib.Path(path_data)
    eval_result_files, eval_result_per_drug_files, eval_result_per_cl_files, true_vs_pred_files = _parse_results(
        outfiles
    )
    eval_results = _collapse_file(eval_result_files)
    eval_results_per_drug = _collapse_file(eval_result_per_drug_files)
    eval_results_per_cell_line = _collapse_file(eval_result_per_cl_files)
    t_vs_p = _collapse_file(true_vs_pred_files)
    eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p = prep_results(
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cell_line=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
        path_data=path_data_path,
    )
    write_results(
        path_out="",
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cl=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
    )
