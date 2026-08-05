"""Orchestration for the full drug response prediction experiment."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

from sklearn.base import TransformerMixin, clone

from drevalpy.components.tuning.config import build_experiment_hpo_config

from ..datasets.dataset import DrugResponseDataset
from ..models._model_lookup import get_model_class, is_single_drug_model_name
from ..models.drp_model import DRPModel
from .consolidate import consolidate_single_drug_model_predictions_impl
from .cross_study import cross_study_prediction_impl
from .final_model import train_final_model_impl
from .fold import (
    early_stopping_for_model,
    merge_train_validation,
    prepare_fold_datasets,
)
from .hpo import (
    final_model_hpo_storage_path,
    fold_hpo_storage_path,
    select_fold_hyperparameters,
)
from .model_paths import (
    generate_data_saving_path,
    get_model_name_and_drug_id,
    make_model_list,
)
from .paths import experiment_result_path
from .randomization import build_randomization_test_views, randomization_test_impl
from .robustness import robustness_test_impl
from .seed import seed_everything
from .splits import prepare_response_splits_impl
from .training import train_and_predict_impl

wandb: Any | None
try:
    import wandb as _wandb_module

    wandb = _wandb_module
except ImportError:
    wandb = None


def _normalize_baselines(
    baselines: list[type[DRPModel]] | None,
) -> list[type[DRPModel]]:
    nme = get_model_class("NaiveMeanEffectsPredictor")
    if baselines is None:
        return [nme]
    baseline_names = {b.get_model_name() for b in baselines}
    if nme not in baselines and nme.get_model_name() not in baseline_names:
        baselines.append(nme)
    return baselines


def _fold_wandb_base_config(
    model_name: str,
    drug_id: str | None,
    split_index: int,
    test_mode: str,
    response_data: DrugResponseDataset,
    actual_n_cv_splits: int,
    hyperparameter_tuning: bool,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "drug_id": drug_id,
        "split_index": split_index,
        "test_mode": test_mode,
        "dataset": response_data.dataset_name,
        "n_cv_splits": actual_n_cv_splits,
        "hyperparameter_tuning": hyperparameter_tuning,
    }


def _init_final_wandb(
    model: DRPModel,
    model_name: str,
    drug_id: str | None,
    split_index: int,
    wandb_project: str,
    base_wandb_config: dict[str, Any],
    test_mode: str,
    response_data: DrugResponseDataset,
) -> None:
    final_run_name = f"{model_name}"
    if drug_id is not None:
        final_run_name += f"_{drug_id}"
    final_run_name += f"_split_{split_index}_final"
    final_config = {**base_wandb_config, "phase": "final_training"}
    model.init_wandb(
        project=wandb_project,
        config=final_config,
        name=final_run_name,
        tags=[model_name, test_mode, response_data.dataset_name or "unknown", "final"],
    )


def _log_wandb_test_metrics(
    model: DRPModel,
    test_dataset: DrugResponseDataset,
    wandb_project: str | None,
    hpam_optimization_metric: str,
) -> None:
    if wandb_project is None or wandb is None:
        return
    if len(test_dataset) == 0 or test_dataset.predictions is None or len(test_dataset.predictions) == 0:
        return
    if wandb.run is not None:
        model.compute_and_log_final_metrics(
            test_dataset,
            additional_metrics=[hpam_optimization_metric],
            prefix="test_",
        )


def _run_cross_study_for_fold(
    cross_study_datasets: list[DrugResponseDataset],
    model: DRPModel,
    test_mode: str,
    train_dataset: DrugResponseDataset,
    path_data: str,
    es_for_model: DrugResponseDataset | None,
    fold_transform: TransformerMixin | None,
    parent_dir: str,
    split_index: int,
    model_name: str,
    drug_id: str | None,
) -> None:
    for cross_study_dataset in cross_study_datasets:
        print(f"Cross study prediction on {cross_study_dataset.dataset_name}")
        cross_study_dataset.remove_nan_responses()
        cross_study_prediction_impl(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data=path_data,
            early_stopping_dataset=es_for_model,
            response_transformation=fold_transform,
            path_out=parent_dir,
            split_index=split_index,
            single_drug_id=(drug_id if is_single_drug_model_name(model_name) else None),
        )


def _run_fresh_cv_fold(
    *,
    model_class: type[DRPModel],
    model_name: str,
    drug_id: str | None,
    split_index: int,
    fold,
    prediction_file: str,
    hpam_save_path: str,
    path_data: str,
    model_checkpoint_dir: str,
    response_transformation: TransformerMixin | None,
    hyperparameter_tuning: bool,
    hpam_optimization_metric: str,
    hpo_num_samples: int,
    hpo_random_state: int,
    hpo_resources_per_trial: dict[str, float] | None,
    result_path: str,
    wandb_project: str | None,
    base_wandb_config: dict[str, Any],
    test_mode: str,
    response_data: DrugResponseDataset,
    cross_study_datasets: list[DrugResponseDataset],
    parent_dir: str,
) -> tuple[DrugResponseDataset, DrugResponseDataset, dict[str, Any], DRPModel | None]:
    hpo_cfg = build_experiment_hpo_config(
        hpam_optimization_metric,
        n_trials=hpo_num_samples,
        random_state=hpo_random_state,
        resources_per_trial=hpo_resources_per_trial,
        storage_path=fold_hpo_storage_path(result_path),
    )
    best_hpams = select_fold_hyperparameters(
        model_class=model_class,
        train_dataset=fold.train,
        validation_dataset=fold.validation,
        early_stopping_dataset=fold.early_stopping,
        response_transformation=response_transformation,
        metric=hpam_optimization_metric,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_config=hpo_cfg,
        wandb_project=wandb_project,
        split_index=split_index,
        wandb_base_config=base_wandb_config,
    )

    print(f"Best hyperparameters: {best_hpams}")
    print("Training model on full train and validation set to predict test set")

    with open(hpam_save_path, "w", encoding="utf-8") as f:
        json.dump(best_hpams, f)

    model = model_class(best_hpams)
    if wandb_project is not None:
        _init_final_wandb(
            model,
            model_name,
            drug_id,
            split_index,
            wandb_project,
            base_wandb_config,
            test_mode,
            response_data,
        )

    train_dataset = merge_train_validation(fold.train, fold.validation)
    es_for_model = early_stopping_for_model(model, fold.early_stopping)
    fold_transform = None if response_transformation is None else clone(response_transformation)
    test_dataset = train_and_predict_impl(
        model=model,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=fold.test,
        early_stopping_dataset=es_for_model,
        response_transformation=fold_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    _log_wandb_test_metrics(model, test_dataset, wandb_project, hpam_optimization_metric)
    _run_cross_study_for_fold(
        cross_study_datasets,
        model,
        test_mode,
        train_dataset,
        path_data,
        es_for_model,
        fold_transform,
        parent_dir,
        split_index,
        model_name,
        drug_id,
    )
    test_dataset.to_csv(prediction_file)
    return train_dataset, test_dataset, best_hpams, model


def _resume_cv_fold(
    fold,
    prediction_file: str,
    hpam_save_path: str,
    split_index: int,
) -> tuple[DrugResponseDataset, DrugResponseDataset, dict[str, Any], DRPModel | None]:
    print(f"Split {split_index} already exists. Skipping.")
    with open(hpam_save_path, encoding="utf-8") as f:
        best_hpams = json.load(f)
    train_dataset = merge_train_validation(fold.train, fold.validation)
    test_dataset = DrugResponseDataset.from_csv(
        prediction_file,
        dataset_name=fold.test.dataset_name,
    )
    return train_dataset, test_dataset, best_hpams, None


def _run_post_fold_stress_tests(
    *,
    is_baseline: bool,
    model_class: type[DRPModel],
    best_hpams: dict[str, Any],
    randomization_mode: list[str] | None,
    randomization_type: str,
    n_trials_robustness: int,
    path_data: str,
    train_dataset: DrugResponseDataset,
    test_dataset: DrugResponseDataset,
    fold,
    parent_dir: str,
    split_index: int,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str,
) -> None:
    if is_baseline:
        return
    es_for_stress = early_stopping_for_model(model_class, fold.early_stopping)
    if randomization_mode is not None:
        print(f"Randomization tests for {model_class.get_model_name()}")
        randomization_test_views = build_randomization_test_views(
            model_class=model_class,
            hyperparameters=best_hpams,
            randomization_mode=randomization_mode,
        )
        randomization_test_impl(
            randomization_test_views=randomization_test_views,
            model_class=model_class,
            hyperparameters=best_hpams,
            path_data=path_data,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            early_stopping_dataset=es_for_stress,
            path_out=parent_dir,
            split_index=split_index,
            randomization_type=randomization_type,
            response_transformation=(None if response_transformation is None else clone(response_transformation)),
            model_checkpoint_dir=model_checkpoint_dir,
        )
    if n_trials_robustness > 0:
        print(f"Robustness test for {model_class.get_model_name()}")
        robustness_test_impl(
            n_trials=n_trials_robustness,
            model_class=model_class,
            hyperparameters=best_hpams,
            path_data=path_data,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            early_stopping_dataset=es_for_stress,
            path_out=parent_dir,
            split_index=split_index,
            response_transformation=(None if response_transformation is None else clone(response_transformation)),
        )


def _run_model_final_production(
    *,
    final_model_on_full_data: bool,
    model_class: type[DRPModel],
    baselines: list[type[DRPModel]],
    model_name: str,
    drug_id: str | None,
    result_path: str,
    response_data: DrugResponseDataset,
    response_transformation: TransformerMixin | None,
    path_data: str,
    model_checkpoint_dir: str,
    hpam_optimization_metric: str,
    test_mode: str,
    hyperparameter_tuning: bool,
    hpo_num_samples: int,
    hpo_random_state: int,
    hpo_resources_per_trial: dict[str, float] | None,
) -> None:
    if not final_model_on_full_data or model_class in baselines:
        return
    final_model_path = generate_data_saving_path(
        model_name=model_name,
        drug_id=drug_id,
        result_path=result_path,
        suffix="final_model",
    )
    train_final_model_impl(
        model_class=model_class,
        full_dataset=response_data.copy(),
        response_transformation=response_transformation,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        metric=hpam_optimization_metric,
        final_model_path=final_model_path,
        test_mode=test_mode,
        val_ratio=0.1,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        hpo_resources_per_trial=hpo_resources_per_trial,
        hpo_storage_path=final_model_hpo_storage_path(result_path),
    )


def _run_one_model(
    model_key: str,
    model_name: str,
    drug_id: str | None,
    *,
    models: list[type[DRPModel]],
    baselines: list[type[DRPModel]],
    response_data: DrugResponseDataset,
    response_transformation: TransformerMixin | None,
    test_mode: str,
    hpam_optimization_metric: str,
    actual_n_cv_splits: int,
    multiprocessing: bool,
    randomization_mode: list[str] | None,
    randomization_type: str,
    cross_study_datasets: list[DrugResponseDataset],
    n_trials_robustness: int,
    result_path: str,
    path_data: str,
    model_checkpoint_dir: str,
    hyperparameter_tuning: bool,
    final_model_on_full_data: bool,
    wandb_project: str | None,
    hpo_num_samples: int,
    hpo_random_state: int,
    hpo_resources_per_trial: dict[str, float] | None,
) -> None:
    model_class = get_model_class(model_name)
    baseline_names = {baseline.get_model_name() for baseline in baselines}
    is_baseline = model_name in baseline_names
    print("- Only Baseline Tests -" if is_baseline else "- Full Test -")

    predictions_path = generate_data_saving_path(
        model_name=model_name,
        drug_id=drug_id,
        result_path=result_path,
        suffix="predictions",
    )
    hpam_path = generate_data_saving_path(
        model_name=model_name,
        drug_id=drug_id,
        result_path=result_path,
        suffix="best_hpams",
    )
    parent_dir = os.path.dirname(predictions_path)

    if multiprocessing:
        warnings.warn(
            "multiprocessing=True now routes through Ray Tune with OptunaSearch; "
            "use hyperparameter_tuning and hpo_num_samples instead.",
            stacklevel=2,
        )

    if response_data.cv_splits is None:
        raise ValueError("No cv splits found.")

    for split_index, split in enumerate(response_data.cv_splits):
        print()
        print(f"################# FOLD {split_index + 1}/{len(response_data.cv_splits)} " f"#################")
        print()

        prediction_file = os.path.join(predictions_path, f"predictions_split_{split_index}.csv")
        hpam_save_path = os.path.join(hpam_path, f"best_hpams_split_{split_index}.json")
        fold = prepare_fold_datasets(split, model_class, model_name, drug_id)
        base_wandb_config = _fold_wandb_base_config(
            model_name,
            drug_id,
            split_index,
            test_mode,
            response_data,
            actual_n_cv_splits,
            hyperparameter_tuning,
        )

        if not os.path.isfile(prediction_file):
            train_dataset, test_dataset, best_hpams, model = _run_fresh_cv_fold(
                model_class=model_class,
                model_name=model_name,
                drug_id=drug_id,
                split_index=split_index,
                fold=fold,
                prediction_file=prediction_file,
                hpam_save_path=hpam_save_path,
                path_data=path_data,
                model_checkpoint_dir=model_checkpoint_dir,
                response_transformation=response_transformation,
                hyperparameter_tuning=hyperparameter_tuning,
                hpam_optimization_metric=hpam_optimization_metric,
                hpo_num_samples=hpo_num_samples,
                hpo_random_state=hpo_random_state,
                hpo_resources_per_trial=hpo_resources_per_trial,
                result_path=result_path,
                wandb_project=wandb_project,
                base_wandb_config=base_wandb_config,
                test_mode=test_mode,
                response_data=response_data,
                cross_study_datasets=cross_study_datasets,
                parent_dir=parent_dir,
            )
        else:
            train_dataset, test_dataset, best_hpams, model = _resume_cv_fold(
                fold, prediction_file, hpam_save_path, split_index
            )

        if wandb_project is not None and model is not None:
            model.finish_wandb()

        _run_post_fold_stress_tests(
            is_baseline=is_baseline,
            model_class=model_class,
            best_hpams=best_hpams,
            randomization_mode=randomization_mode,
            randomization_type=randomization_type,
            n_trials_robustness=n_trials_robustness,
            path_data=path_data,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            fold=fold,
            parent_dir=parent_dir,
            split_index=split_index,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )

    _run_model_final_production(
        final_model_on_full_data=final_model_on_full_data,
        model_class=model_class,
        baselines=baselines,
        model_name=model_name,
        drug_id=drug_id,
        result_path=result_path,
        response_data=response_data,
        response_transformation=response_transformation,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        hpam_optimization_metric=hpam_optimization_metric,
        test_mode=test_mode,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        hpo_resources_per_trial=hpo_resources_per_trial,
    )


def drug_response_experiment_impl(
    models: list[type[DRPModel]],
    response_data: DrugResponseDataset,
    baselines: list[type[DRPModel]] | None = None,
    response_transformation: TransformerMixin | None = None,
    run_id: str = "",
    test_mode: str = "LPO",
    hpam_optimization_metric: str = "RMSE",
    n_cv_splits: int = 5,
    multiprocessing: bool = False,
    randomization_mode: list[str] | None = None,
    randomization_type: str = "permutation",
    cross_study_datasets: list[DrugResponseDataset] | None = None,
    n_trials_robustness: int = 0,
    path_out: str = "results/",
    overwrite: bool = False,
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
    hyperparameter_tuning: bool = True,
    final_model_on_full_data: bool = False,
    wandb_project: str | None = None,
    custom_splitter=None,
    custom_split_name: str | None = None,
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    hpo_resources_per_trial: dict[str, float] | None = None,
) -> None:
    """Run the drug response prediction experiment and save results to disk.

    Trains each model across CV folds (with optional HPO), writes predictions
    and hyperparameters under ``path_out``, and optionally runs randomization,
    robustness, cross-study, and final-model workflows.

    :param models: ``DRPModel`` subclasses to evaluate (from ``construct_model``).
    :param response_data: Training/validation response table with CV splits attached.
    :param baselines: Optional baseline models; ``NaiveMeanEffectsPredictor`` is added by default when omitted.
    :param response_transformation: Optional sklearn transformer applied to responses.
    :param run_id: Subfolder name under ``path_out`` for this run.
    :param test_mode: Split mode (``"LPO"``, ``"LCO"``, ``"LDO"``, or custom).
    :param hpam_optimization_metric: Metric optimized during HPO (for example ``"RMSE"``).
    :param n_cv_splits: Number of cross-validation folds.
    :param multiprocessing: Deprecated; routes through Ray Tune when ``True``.
    :param randomization_mode: Feature views to permute for randomization tests.
    :param randomization_type: Permutation strategy for randomization tests.
    :param cross_study_datasets: Additional datasets for cross-study evaluation.
    :param n_trials_robustness: Number of robustness-test resampling trials.
    :param path_out: Root directory for experiment outputs.
    :param overwrite: Recompute splits and predictions even when artifacts exist.
    :param path_data: Root directory for feature tables.
    :param model_checkpoint_dir: Directory for per-fold model checkpoints.
    :param hyperparameter_tuning: Whether to run HPO before final fold training.
    :param final_model_on_full_data: Train a production model on all data after CV.
    :param wandb_project: Optional Weights & Biases project name.
    :param custom_splitter: External split creator or path to split manifest.
    :param custom_split_name: Label for custom splits in output paths.
    :param hpo_num_samples: Number of HPO trials per fold.
    :param hpo_random_state: Random seed for HPO search.
    :param hpo_resources_per_trial: Optional Ray resource limits per HPO trial.
    """
    seed_everything(42)
    baselines = _normalize_baselines(baselines)
    cross_study_datasets = cross_study_datasets or []
    split_label = custom_split_name if custom_split_name is not None else test_mode
    result_path = str(experiment_result_path(path_out, run_id, response_data._name, split_label))
    split_path = os.path.join(result_path, "splits")
    result_folder_exists = os.path.exists(result_path)
    actual_n_cv_splits = prepare_response_splits_impl(
        response_data,
        split_path=split_path,
        result_path=result_path,
        split_label=split_label,
        test_mode=test_mode,
        n_cv_splits=n_cv_splits,
        overwrite=overwrite,
        result_folder_exists=result_folder_exists,
        custom_splitter=custom_splitter,
    )

    model_list = make_model_list(models + baselines, response_data)
    for model_key in model_list.keys():
        print(f"Running {model_key}")
        model_name, drug_id = get_model_name_and_drug_id(model_key)
        _run_one_model(
            model_key,
            model_name,
            drug_id,
            models=models,
            baselines=baselines,
            response_data=response_data,
            response_transformation=response_transformation,
            test_mode=test_mode,
            hpam_optimization_metric=hpam_optimization_metric,
            actual_n_cv_splits=actual_n_cv_splits,
            multiprocessing=multiprocessing,
            randomization_mode=randomization_mode,
            randomization_type=randomization_type,
            cross_study_datasets=cross_study_datasets,
            n_trials_robustness=n_trials_robustness,
            result_path=result_path,
            path_data=path_data,
            model_checkpoint_dir=model_checkpoint_dir,
            hyperparameter_tuning=hyperparameter_tuning,
            final_model_on_full_data=final_model_on_full_data,
            wandb_project=wandb_project,
            hpo_num_samples=hpo_num_samples,
            hpo_random_state=hpo_random_state,
            hpo_resources_per_trial=hpo_resources_per_trial,
        )

    consolidate_single_drug_model_predictions_impl(
        models=models,
        n_cv_splits=actual_n_cv_splits,
        results_path=result_path,
        cross_study_datasets=[cs.dataset_name for cs in cross_study_datasets],
        randomization_mode=randomization_mode,
        n_trials_robustness=n_trials_robustness,
        out_path=result_path,
    )
    print("Done!")
