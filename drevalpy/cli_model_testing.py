"""For the nf-core/drugresponseeval subworkflow model_testing."""

import json
import pathlib
import pickle
from functools import partial

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import TransformerMixin

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import DRPModel


def _best_hpam_combi_for_drug(best_hpam_dict: dict, drug: str) -> dict:
    """Best combination of one drug; without hyperparameter tuning, one shared ``best_hpam_combi`` for all drugs."""
    if "best_hpam_combi_per_drug" in best_hpam_dict:
        return best_hpam_dict["best_hpam_combi_per_drug"][drug]
    return best_hpam_dict["best_hpam_combi"]


def _test_split(
    drug_id: str | None,
    split: dict,
    cross_study_datasets: list,
    hpam_combi: dict,
    *,
    mode: str,
    model_class: "type[DRPModel]",
    split_id: str,
    test_mode: str,
    path_data: str,
    response_transform: TransformerMixin | None,
    rand_test_view: dict,
    randomization_type: str,
    robustness_trial: int | None,
    model_checkpoint_dir: str,
) -> None:
    """Train on a CV split and predict its test set in the given mode, for single drug models for one drug."""
    from drevalpy.experiment import (
        cross_study_prediction,
        generate_data_saving_path,
        get_datasets_from_cv_split,
        randomize_train_predict,
        robustness_train_predict,
        train_and_predict,
    )

    selected_model = model_class()
    train_set, validation_set, es_set, test_set = get_datasets_from_cv_split(
        split, model_class, model_class.get_model_name(), drug_id
    )
    train_set.add_rows(validation_set)
    train_set.shuffle(random_state=42)

    if mode == "full":
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
        hpam_path = pathlib.Path(hpam_path) / f"best_hpams_{split_id}.json"
        with open(hpam_path, "w", encoding="utf-8") as f:
            json.dump(hpam_combi, f)

        test_set = train_and_predict(
            model=selected_model,
            hpams=hpam_combi,
            path_data=path_data,
            train_dataset=train_set,
            prediction_dataset=test_set,
            early_stopping_dataset=es_set,
            response_transformation=response_transform,
            model_checkpoint_dir=model_checkpoint_dir,
        )
        prediction_dataset = pathlib.Path(predictions_path) / f"predictions_{split_id}.csv"
        test_set.to_csv(prediction_dataset)
        for cross_study_dataset in cross_study_datasets:
            split_index = split_id.split("split_")[1]
            cross_study_prediction(
                dataset=cross_study_dataset,
                model=selected_model,
                test_mode=test_mode,
                train_dataset=train_set,
                path_data=path_data,
                early_stopping_dataset=(es_set if selected_model.early_stopping else None),
                response_transformation=response_transform,
                path_out=str(pathlib.Path(predictions_path).parent),
                split_index=split_index,
                single_drug_id=drug_id,
            )
    elif mode == "randomization":
        rand_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="randomization",
        )
        randomization_test_file = (
            pathlib.Path(rand_path) / f'randomization_{rand_test_view["test_name"]}_{split_id}.csv'
        )
        randomize_train_predict(
            view=rand_test_view["view"],
            test_name=rand_test_view["test_name"],
            randomization_type=randomization_type,
            randomization_test_file=str(randomization_test_file),
            model=selected_model,
            hpam_set=hpam_combi,
            path_data=path_data,
            train_dataset=train_set,
            test_dataset=test_set,
            early_stopping_dataset=es_set,
            response_transformation=response_transform,
            model_checkpoint_dir=model_checkpoint_dir,
        )
    elif mode == "robustness":
        rob_path = generate_data_saving_path(
            model_name=selected_model.get_model_name(),
            drug_id=drug_id,
            result_path="",
            suffix="robustness",
        )
        robustness_test_file = pathlib.Path(rob_path) / f"robustness_{robustness_trial}_{split_id}.csv"
        robustness_train_predict(
            trial=robustness_trial,
            trial_file=str(robustness_test_file),
            train_dataset=train_set,
            test_dataset=test_set,
            early_stopping_dataset=es_set,
            model=selected_model,
            hpam_set=hpam_combi,
            path_data=path_data,
            response_transformation=response_transform,
            model_checkpoint_dir=model_checkpoint_dir,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose full, randomization, or robustness.")


def _consolidate_drug_predictions(model_name: str) -> None:
    """Concatenate the per-drug CSVs in <model_name>/drugs/<drug_id>/<suffix>/ into <model_name>/<suffix>/."""
    drugs_path = pathlib.Path(model_name, "drugs")
    predictions: dict[pathlib.Path, list[pd.DataFrame]] = {}
    for csv_file in sorted(drugs_path.glob("*/*/*.csv")):
        # drop the drug id: <drug_id>/<suffix>/<file>.csv -> <suffix>/<file>.csv
        consolidated_file = pathlib.Path(model_name, *csv_file.relative_to(drugs_path).parts[1:])
        predictions.setdefault(consolidated_file, []).append(pd.read_csv(csv_file, index_col=0))
    for consolidated_file, drug_predictions in predictions.items():
        consolidated_file.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(drug_predictions, axis=0).to_csv(consolidated_file)


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
    n_jobs: int = 1,
) -> None:
    """
    Train and predict on the CV test set (full, randomization, or robustness mode).

    Single drug models are trained per drug, n_jobs drugs in parallel. Their predictions are saved per drug and
    consolidated over all drugs.
    """
    from drevalpy.experiment import run_per_drug, split_by_drug
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    model_class = MODEL_FACTORY[model_name]
    with open(split_dataset_path, "rb") as split_file:
        split = pickle.load(split_file)
    with open(hyperparameters_path) as f:
        best_hpam_dict = yaml.safe_load(f)[f"{model_name}_{split_id}"]

    loaded_cross_study_datasets = []
    for cs_ds in cross_study_datasets or []:
        if cs_ds == "NONE.csv":
            continue
        with open(cs_ds, "rb") as cs_file:
            cross_study_dataset = pickle.load(cs_file)
        cross_study_dataset.remove_nan_responses()
        loaded_cross_study_datasets.append(cross_study_dataset)

    rand_test_view = {}
    if randomization_views_path is not None:
        with open(randomization_views_path) as f:
            rand_test_view = yaml.safe_load(f)

    test_split = partial(
        _test_split,
        mode=mode,
        model_class=model_class,
        split_id=split_id,
        test_mode=test_mode,
        path_data=path_data,
        response_transform=get_response_transformation(response_transformation),
        rand_test_view=rand_test_view,
        randomization_type=randomization_type,
        robustness_trial=robustness_trial,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    if model_class.is_single_drug_model:
        drug_ids = np.unique(np.concatenate([ds.drug_ids for ds in split.values() if ds is not None]))
        splits_per_drug = split_by_drug(split, drug_ids)
        cross_study_per_drug = split_by_drug({cs.dataset_name: cs for cs in loaded_cross_study_datasets}, drug_ids)
        run_per_drug(
            test_split,
            {
                drug: (
                    splits_per_drug[drug],
                    list(cross_study_per_drug[drug].values()),
                    _best_hpam_combi_for_drug(best_hpam_dict, drug),
                )
                for drug in splits_per_drug
            },
            n_jobs=n_jobs,
        )
        _consolidate_drug_predictions(model_name)
    else:
        test_split(None, split, loaded_cross_study_datasets, best_hpam_dict["best_hpam_combi"])


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


def _make_final_split(
    response_data: DrugResponseDataset, model_class: "type[DRPModel]", test_mode: str, val_ratio: float
) -> tuple:
    """Split the response data into train, validation, and early stopping data for a final model."""
    from drevalpy.datasets.dataset import split_early_stopping_data
    from drevalpy.experiment import make_train_val_split

    train_dataset, validation_dataset = make_train_val_split(response_data, test_mode=test_mode, val_ratio=val_ratio)

    if model_class.early_stopping:
        validation_dataset, early_stopping_dataset = split_early_stopping_data(validation_dataset, test_mode)
    else:
        early_stopping_dataset = None
    return train_dataset, validation_dataset, early_stopping_dataset


def run_final_split(
    *,
    response: str,
    model_name: str,
    path_data: str = "data",
    test_mode: str = "LPO",
    val_ratio: float = 0.1,
) -> None:
    """
    Create train/validation/early-stopping pickles for a final production model.

    For single drug models, each pickle is a {drug_id: dataset} dict, split per drug.
    """
    from drevalpy.experiment import split_by_drug
    from drevalpy.models import MODEL_FACTORY

    with open(response, "rb") as response_file:
        response_data = pickle.load(response_file)
    response_data.remove_nan_responses()
    model_class = MODEL_FACTORY[model_name]
    if model_class.is_single_drug_model:
        response_per_drug = split_by_drug({"full": response_data}, np.unique(response_data.drug_ids))
        splits_per_drug = {
            drug: _make_final_split(datasets["full"], model_class, test_mode, val_ratio)
            for drug, datasets in response_per_drug.items()
        }
        train_dataset = {drug: datasets[0] for drug, datasets in splits_per_drug.items()}
        validation_dataset = {drug: datasets[1] for drug, datasets in splits_per_drug.items()}
        early_stopping_dataset = {drug: datasets[2] for drug, datasets in splits_per_drug.items()}
    else:
        train_dataset, validation_dataset, early_stopping_dataset = _make_final_split(
            response_data, model_class, test_mode, val_ratio
        )

    with open("training_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("validation_dataset.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)
    with open("early_stopping_dataset.pkl", "wb") as f:
        pickle.dump(early_stopping_dataset, f)


def _load_final_split(
    train_data: str, val_data: str, early_stopping_data: str
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset] | tuple[dict, dict, dict]:
    """Load the pickles of run_final_split."""
    with open(train_data, "rb") as train_file:
        train_dataset = pickle.load(train_file)
    with open(val_data, "rb") as val_file:
        validation_dataset = pickle.load(val_file)
    with open(early_stopping_data, "rb") as es_file:
        early_stopping_dataset = pickle.load(es_file)
    return train_dataset, validation_dataset, early_stopping_dataset


def _tune_final_model(
    drug_id: str | None,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset,
    *,
    model_class: "type[DRPModel]",
    hpams: dict,
    path_data: str,
    response_transform: TransformerMixin | None,
    model_checkpoint_dir: str,
) -> DrugResponseDataset:
    """Train a final model candidate and predict the validation set, for single drug models for one drug."""
    from drevalpy.experiment import train_and_predict

    return train_and_predict(
        model=model_class(),
        hpams=hpams,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )


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
    n_jobs: int = 1,
) -> None:
    """Tune hyperparameters for the final model on full data. Single drug models: n_jobs drugs in parallel."""
    from drevalpy.experiment import run_per_drug
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    train_dataset, validation_dataset, early_stopping_dataset = _load_final_split(
        train_data, val_data, early_stopping_data
    )
    model_class = MODEL_FACTORY[model_name]
    with open(hpam_combi) as f:
        hpams = yaml.safe_load(f)

    tune = partial(
        _tune_final_model,
        model_class=model_class,
        hpams=hpams,
        path_data=path_data,
        response_transform=get_response_transformation(response_transformation),
        model_checkpoint_dir=model_checkpoint_dir,
    )
    if model_class.is_single_drug_model:
        # the datasets are {drug_id: dataset} dicts, see run_final_split
        if (
            not isinstance(train_dataset, dict)
            or not isinstance(validation_dataset, dict)
            or not isinstance(early_stopping_dataset, dict)
        ):
            raise ValueError(
                "For single drug models, the train/validation/early_stopping datasets must be dicts of "
                "{drug_id: dataset}."
            )
        validation_dataset = run_per_drug(
            tune,
            {
                drug: (train_dataset[drug], validation_dataset[drug], early_stopping_dataset[drug])
                for drug in train_dataset
            },
            n_jobs=n_jobs,
        )
    else:
        if (
            not isinstance(train_dataset, DrugResponseDataset)
            or not isinstance(validation_dataset, DrugResponseDataset)
            or not isinstance(early_stopping_dataset, DrugResponseDataset)
        ):
            raise ValueError(
                "For multi-drug models, the train/validation/early_stopping datasets must be DrugResponseDataset "
                "instances."
            )
        validation_dataset = tune(None, train_dataset, validation_dataset, early_stopping_dataset)
    with open(f"final_prediction_dataset_{model_name}_" f"{str(hpam_combi).split('.yaml')[0]}.pkl", "wb") as f:
        pickle.dump(validation_dataset, f)


def _train_final_model(
    drug_id: str | None,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    es_dataset: DrugResponseDataset,
    best_hpam: dict,
    *,
    model_class: "type[DRPModel]",
    path_data: str,
    response_transform: TransformerMixin | None,
    model_checkpoint_dir: str,
) -> None:
    """Train and save the final production model, for single drug models for one drug."""
    from drevalpy.experiment import generate_data_saving_path

    final_model_path = generate_data_saving_path(
        model_name=model_class.get_model_name(), drug_id=drug_id, result_path="", suffix="final_model"
    )
    model = model_class()
    model.build_model(hyperparameters=best_hpam)
    cl_features = model.load_cell_line_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    drug_features = model.load_drug_features(data_path=path_data, dataset_name=train_dataset.dataset_name)
    cell_lines_to_keep = cl_features.identifiers
    drugs_to_keep = drug_features.identifiers if drug_features is not None else None

    train_dataset.add_rows(validation_dataset)
    train_dataset.shuffle(random_state=42)
    len_train_before = len(train_dataset)
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    if len(train_dataset) < len_train_before:
        print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}, due to missing features")

    if es_dataset is not None:
        len_early_stopping_before = len(es_dataset)
        es_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
        if len(es_dataset) < len_early_stopping_before:
            print(
                f"Reduced early stopping dataset from {len_early_stopping_before} to "
                f"{len(es_dataset)}, due to missing features"
            )

    if response_transform:
        train_dataset.fit_transform(response_transform)
        if es_dataset is not None:
            es_dataset.transform(response_transform)

    model.train(
        output=train_dataset,
        output_earlystopping=es_dataset,
        cell_line_input=cl_features,
        drug_input=drug_features,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    pathlib.Path(final_model_path).mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    if response_transform:
        joblib.dump(response_transform, pathlib.Path(final_model_path) / "response_transform.pkl")


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
    n_jobs: int = 1,
) -> None:
    """Train and save the final production model. Single drug models: n_jobs drugs in parallel."""
    from drevalpy.experiment import run_per_drug
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    train_dataset, validation_dataset, es_dataset = _load_final_split(train_data, val_data, early_stopping_data)
    with open(best_hpam_combi) as f:
        best_hpam_dict = yaml.safe_load(f)[f"{model_name}_final"]
    model_class = MODEL_FACTORY[model_name]

    train = partial(
        _train_final_model,
        model_class=model_class,
        path_data=path_data,
        response_transform=get_response_transformation(response_transformation),
        model_checkpoint_dir=model_checkpoint_dir,
    )
    if model_class.is_single_drug_model:
        # the datasets are {drug_id: dataset} dicts, see run_final_split
        if (
            not isinstance(train_dataset, dict)
            or not isinstance(validation_dataset, dict)
            or not isinstance(es_dataset, dict)
        ):
            raise ValueError(
                "For single drug models, the train/validation/early_stopping datasets must be dicts of "
                "{drug_id: dataset}."
            )
        run_per_drug(
            train,
            {
                drug: (
                    train_dataset[drug],
                    validation_dataset[drug],
                    es_dataset[drug],
                    _best_hpam_combi_for_drug(best_hpam_dict, drug),
                )
                for drug in train_dataset
            },
            n_jobs=n_jobs,
        )
    else:
        if (
            not isinstance(train_dataset, DrugResponseDataset)
            or not isinstance(validation_dataset, DrugResponseDataset)
            or not isinstance(es_dataset, DrugResponseDataset)
        ):
            raise ValueError(
                "For multi-drug models, the train/validation/early_stopping datasets must be DrugResponseDataset "
                "instances."
            )
        train(None, train_dataset, validation_dataset, es_dataset, best_hpam_dict["best_hpam_combi"])


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
