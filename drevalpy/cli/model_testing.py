"""For the nf-core/drugresponseeval subworkflow model_testing."""

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary
from drevalpy.utils.pickle_io import dump_trusted_pickle, load_trusted_pickle


def _prep_data_for_final_prediction(
    arguments: Namespace,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load data and prepare it for final CV-fold training and prediction.

    :param arguments: Namespace with split paths, model name, and transformation options.
    :returns: Model, drug id, hyperparameters, train/test/early-stopping sets, and response
        transformer.
    """
    from drevalpy.experiment import get_model_name_and_drug_id
    from drevalpy.experiment.fold import (
        early_stopping_for_model,
        prepare_final_fold_training_data,
    )
    from drevalpy.models._model_lookup import get_model_class
    from drevalpy.utils import get_response_transformation

    model_name, drug_id = get_model_name_and_drug_id(arguments.model_name)
    model_class = get_model_class(model_name)
    split = load_trusted_pickle(arguments.split_dataset_path)
    fold = prepare_final_fold_training_data(split, model_class, model_name, drug_id)
    with open(arguments.hyperparameters_path) as f:
        best_hpam_dict = yaml.safe_load(f)
    best_hpams = best_hpam_dict[f"{arguments.model_name}_{arguments.split_id}"]["best_hpam_combi"]
    model = model_class(best_hpams)
    response_transform = get_response_transformation(arguments.response_transformation)
    return (
        model,
        drug_id,
        best_hpams,
        fold.train,
        fold.test,
        early_stopping_for_model(model, fold.early_stopping),
        response_transform,
    )


def run_train_and_predict_final(
    *,
    mode: str = "full",
    model_name: str,
    split_id: str,
    split_dataset_path: str,
    hyperparameters_path: str,
    response_transformation: str = "None",
    test_mode: str = "LPO",
    randomization_views_path: str | None = None,
    randomization_type: str = "permutation",
    robustness_trial: int | None = None,
    cross_study_datasets: list[str] | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Train and predict on the CV test set (full, randomization, or robustness mode).

    :param mode: mode.
    :param model_name: model name.
    :param split_id: split id.
    :param split_dataset_path: split dataset path.
    :param hyperparameters_path: hyperparameters path.
    :param response_transformation: response transformation.
    :param test_mode: test mode.
    :param randomization_views_path: randomization views path.
    :param randomization_type: randomization type.
    :param robustness_trial: robustness trial.
    :param cross_study_datasets: cross study datasets.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :raises ValueError: If ``mode`` is not ``full``, ``randomization``, or ``robustness``.
    """
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
            result_path=Path("."),
            suffix="predictions",
        )
        hpam_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path=Path("."),
            suffix="best_hpams",
        )
        hpam_path = Path(hpam_path) / f"best_hpams_{args.split_id}.json"
        with open(hpam_path, "w", encoding="utf-8") as f:
            json.dump(hpam_combi, f)

        test_set = train_and_predict(
            model=selected_model,
            train_dataset=train_set,
            prediction_dataset=test_set,
            early_stopping_dataset=es_set,
            response_transformation=transformation,
            model_checkpoint_dir=args.model_checkpoint_dir,
        )
        prediction_dataset = Path(predictions_path) / f"predictions_{args.split_id}.csv"
        test_set.to_csv(prediction_dataset)
        if args.cross_study_datasets:
            for cs_ds in args.cross_study_datasets:
                if cs_ds == "NONE.csv":
                    continue
                split_index = args.split_id.split("split_")[1]
                cross_study_dataset = load_trusted_pickle(cs_ds)
                cross_study_dataset.remove_nan_responses()
                cross_study_prediction(
                    dataset=cross_study_dataset,
                    model=selected_model,
                    test_mode=args.test_mode,
                    train_dataset=train_set,
                    early_stopping_dataset=(es_set if selected_model.supports_early_stopping() else None),
                    response_transformation=transformation,
                    path_out=str(Path(predictions_path).parent),
                    split_index=split_index,
                    single_drug_id=drug_id,
                )
    elif args.mode == "randomization":
        with open(args.randomization_views_path) as f:
            rand_test_view = yaml.safe_load(f)
        rand_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path=Path("."),
            suffix="randomization",
        )
        randomization_test_file = Path(rand_path) / f"randomization_{rand_test_view['test_name']}_{args.split_id}.csv"
        views = rand_test_view.get("views")
        if views is None:
            views = [rand_test_view["view"]]
        randomize_train_predict(
            views=views,
            test_name=rand_test_view["test_name"],
            randomization_type=args.randomization_type,
            randomization_test_file=str(randomization_test_file),
            model_class=type(selected_model),
            hyperparameters=hpam_combi,
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
            result_path=Path("."),
            suffix="robustness",
        )
        robustness_test_file = Path(rob_path) / f"robustness_{args.robustness_trial}_{args.split_id}.csv"
        robustness_train_predict(
            trial=args.robustness_trial,
            trial_file=str(robustness_test_file),
            train_dataset=train_set,
            test_dataset=test_set,
            early_stopping_dataset=es_set,
            model_class=type(selected_model),
            hyperparameters=hpam_combi,
            response_transformation=transformation,
            model_checkpoint_dir=args.model_checkpoint_dir,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Choose full, randomization, or robustness.")


def run_randomization_split(*, model_name: str, randomization_mode: str) -> None:
    """Create randomization test view YAML files for a model.

    :param model_name: model name.
    :param randomization_mode: randomization mode.
    :raises RuntimeError: If no randomization test views are produced for the model.
    """
    from drevalpy.experiment import get_randomization_test_views
    from drevalpy.models._model_lookup import get_model_class

    model_class = get_model_class(model_name)
    randomization_test_views = get_randomization_test_views(
        model_class=model_class,
        randomization_mode=[randomization_mode],
        hyperparameters=model_class.get_default_hyperparameters(),
    )

    if not randomization_test_views:
        raise RuntimeError(
            f"No randomization test views were produced for {model_name} with mode {randomization_mode}. "
            "Check that the model declares cell_line_views/drug_views in its public hyperparameters."
        )

    for test_name, views in randomization_test_views.items():
        rand_dict = {
            "test_name": test_name,
            "views": views,
            "view": views[0] if views else None,
        }
        with open(f"randomization_test_view_{test_name}.yaml", "w") as f:
            yaml.dump(rand_dict, f)


def run_final_split(
    *,
    response: str | Path,
    model_name: str,
    test_mode: str = "LPO",
    val_ratio: float = 0.1,
) -> None:
    """Create train/validation/early-stopping pickles for a final production model.

    :param response: response.
    :param model_name: model name.
    :param test_mode: test mode.
    :param val_ratio: val ratio.
    """
    from drevalpy.datasets.dataset import split_early_stopping_data
    from drevalpy.experiment import make_train_val_split
    from drevalpy.models._model_lookup import get_model_class

    response_data = load_trusted_pickle(response)
    response_data.remove_nan_responses()
    model_class = get_model_class(model_name)
    model = model_class()
    cl_features = model.load_cell_line_features(dataset_name=response_data.dataset_name)
    drug_features = model.load_drug_features(dataset_name=response_data.dataset_name)
    cell_lines_to_keep = cl_features.identifiers
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None
    response_data = response_data.reduced_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)

    train_dataset, validation_dataset = make_train_val_split(response_data, test_mode=test_mode, val_ratio=val_ratio)

    if model_class.supports_early_stopping():
        validation_dataset, early_stopping_dataset = split_early_stopping_data(validation_dataset, test_mode)
    else:
        early_stopping_dataset = None

    dump_trusted_pickle(train_dataset, "training_dataset.pkl")
    dump_trusted_pickle(validation_dataset, "validation_dataset.pkl")
    dump_trusted_pickle(early_stopping_dataset, "early_stopping_dataset.pkl")


def run_tune_final_model(
    *,
    train_data: str | Path,
    val_data: str | Path,
    early_stopping_data: str | Path,
    model_name: str,
    hpam_combi: str,
    response_transformation: str = "None",
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Score a final-model candidate on the validation split (no search).

    Despite the historical name, this command evaluates one hyperparameter YAML
    via ``train_and_predict``. Ray/Optuna search lives in
    ``drevalpy.experiment.train_final_model`` / ``hpam_tune``.

    :param train_data: train data.
    :param val_data: val data.
    :param early_stopping_data: early stopping data.
    :param model_name: model name.
    :param hpam_combi: hpam combi.
    :param response_transformation: response transformation.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    """
    import warnings

    from drevalpy.experiment import get_model_name_and_drug_id, train_and_predict
    from drevalpy.models._model_lookup import get_model_class
    from drevalpy.utils import get_response_transformation

    warnings.warn(
        "tune-final-model evaluates a single hyperparameter YAML; it does not run "
        "Ray/Optuna search. Prefer drevalpy.experiment.train_final_model for tuning.",
        DeprecationWarning,
        stacklevel=2,
    )

    train_dataset = load_trusted_pickle(train_data)
    validation_dataset = load_trusted_pickle(val_data)
    early_stopping_dataset = load_trusted_pickle(early_stopping_data)
    response_transform = get_response_transformation(response_transformation)

    resolved_name, _drug_id = get_model_name_and_drug_id(model_name)
    model_class = get_model_class(resolved_name)
    with open(hpam_combi) as f:
        hpams = yaml.safe_load(f)
    model = model_class(hpams)

    validation_dataset = train_and_predict(
        model=model,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    dump_trusted_pickle(
        validation_dataset,
        f"final_prediction_dataset_{resolved_name}_{str(hpam_combi).split('.yaml')[0]}.pkl",
    )


def run_train_final_model(
    *,
    train_data: str | Path,
    val_data: str | Path,
    early_stopping_data: str | Path,
    response_transformation: str = "None",
    model_name: str,
    model_checkpoint_dir: str | Path | None = None,
    best_hpam_combi: str | Path,
) -> None:
    """Train and save the final production model.

    :param train_data: train data.
    :param val_data: val data.
    :param early_stopping_data: early stopping data.
    :param response_transformation: response transformation.
    :param model_name: model name.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :param best_hpam_combi: best hpam combi.
    """
    from drevalpy.experiment import (
        generate_final_model_checkpoint_path,
        get_model_name_and_drug_id,
    )
    from drevalpy.models._model_lookup import get_model_class
    from drevalpy.utils import get_response_transformation

    resolved_name, _drug_id = get_model_name_and_drug_id(model_name)
    final_model_path = generate_final_model_checkpoint_path(
        model_name=resolved_name, drug_id=_drug_id, result_path=Path(".")
    )
    response_transform = get_response_transformation(response_transformation)
    train_dataset = load_trusted_pickle(train_data)
    validation_dataset = load_trusted_pickle(val_data)
    es_dataset = load_trusted_pickle(early_stopping_data)
    train_dataset = train_dataset.with_rows_added(validation_dataset).shuffled(random_state=42)
    if response_transform:
        train_dataset = train_dataset.fit_transformed(response_transform)
        if es_dataset is not None:
            es_dataset = es_dataset.transformed(response_transform)
    with open(best_hpam_combi) as f:
        best_hpam = yaml.safe_load(f)[f"{resolved_name}_final"]["best_hpam_combi"]
    model = get_model_class(resolved_name)(best_hpam)
    cl_features = model.load_cell_line_features(dataset_name=train_dataset.dataset_name)
    drug_features = model.load_drug_features(dataset_name=train_dataset.dataset_name)
    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        model.train(
            output=train_dataset,
            output_earlystopping=es_dataset,
            cell_line_input=cl_features,
            drug_input=drug_features,
            model_checkpoint_dir=checkpoint_dir,
        )
    Path(final_model_path).parent.mkdir(parents=True, exist_ok=True)
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
    dataset_name: str | None = None,
) -> None:
    """Consolidate single-drug model prediction outputs.

    :param run_id: run id.
    :param test_mode: test mode.
    :param model_name: model name.
    :param outdir_path: outdir path.
    :param n_cv_splits: n cv splits.
    :param cross_study_datasets: cross study datasets.
    :param randomization_modes: randomization modes.
    :param n_trials_robustness: n trials robustness.
    :param dataset_name: dataset name.
    :raises ValueError: If ``dataset_name`` is omitted.
    """
    from drevalpy.experiment import consolidate_single_drug_model_predictions
    from drevalpy.experiment.paths import consolidate_results_path
    from drevalpy.models._model_lookup import get_model_class

    if dataset_name is None:
        raise ValueError("dataset_name is required to locate experiment results")
    results_path = str(consolidate_results_path(outdir_path, run_id, dataset_name, test_mode))
    if randomization_modes == "[None]":
        randomizations = None
    else:
        randomizations = randomization_modes.split("[")[1].split("]")[0].split(", ")
    model = get_model_class(model_name)
    cross_study = cross_study_datasets or []
    consolidate_single_drug_model_predictions(
        models=[model],
        n_cv_splits=n_cv_splits,
        results_path=results_path,
        cross_study_datasets=cross_study,
        randomization_mode=randomizations,
        n_trials_robustness=n_trials_robustness,
        out_path=Path("."),
    )


def run_evaluate_test_results(
    *,
    test_mode: str = "LPO",
    model_name: str,
    pred_file: str,
) -> None:
    """Evaluate test predictions and write metric CSVs.

    :param test_mode: test mode.
    :param model_name: model name.
    :param pred_file: pred file.
    """
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


def _parse_results(
    outfiles: list[str],
) -> tuple[list[str], list[str], list[str], list[str]]:
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
) -> None:
    """Collect parallel Nextflow evaluation outputs into merged CSVs.

    :param outfiles: outfiles.
    """
    from drevalpy.visualization.utils import prep_results, write_results

    (
        eval_result_files,
        eval_result_per_drug_files,
        eval_result_per_cl_files,
        true_vs_pred_files,
    ) = _parse_results(outfiles)
    eval_results = _collapse_file(eval_result_files)
    eval_results_per_drug = _collapse_file(eval_result_per_drug_files)
    eval_results_per_cell_line = _collapse_file(eval_result_per_cl_files)
    t_vs_p = _collapse_file(true_vs_pred_files)
    eval_results, eval_results_per_drug, eval_results_per_cell_line, t_vs_p = prep_results(
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cell_line=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
    )
    write_results(
        path_out="",
        eval_results=eval_results,
        eval_results_per_drug=eval_results_per_drug,
        eval_results_per_cl=eval_results_per_cell_line,
        t_vs_p=t_vs_p,
    )
