"""Orchestration for the MuData-based drug response prediction experiment."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin, clone
from upath import UPath as Path

from drevalpy.components.tuning.config import build_experiment_hpo_config

from ..data.mudataset import MuDataset
from ..data.splitting import EntityScope
from ..models._model_lookup import get_model_class
from ..models.drp_model import DRPModel
from .fold import merge_train_val_scopes, prepare_mu_fold
from .hpo import fold_hpo_storage_path, select_fold_hyperparameters
from .model_paths import generate_data_saving_path
from .paths import experiment_result_path
from .seed import seed_everything
from .splits import prepare_splits
from .training import mu_train_and_predict


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


def mu_experiment(
    models: list[type[DRPModel]],
    mudataset: MuDataset,
    dataset_name: str,
    baselines: list[type[DRPModel]] | None = None,
    response_transformation: TransformerMixin | None = None,
    run_id: str = "",
    test_mode: str = "LPO",
    hpam_optimization_metric: str = "RMSE",
    n_cv_splits: int = 5,
    path_out: str | Path = "results/",
    overwrite: bool = False,
    model_checkpoint_dir: str | Path | None = None,
    hyperparameter_tuning: bool = True,
    wandb_project: str | None = None,
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    hpo_resources_per_trial: dict[str, float] | None = None,
) -> None:
    """Run the MuData-based drug response prediction experiment.

    This is the primary entry point for the experiment pipeline. It uses
    MuDataset + MuDataSplitter for drug response prediction.

    :param models: DRPModel subclasses to evaluate.
    :param mudataset: Loaded MuDataset with all features.
    :param dataset_name: Name label for result paths.
    :param baselines: Optional baseline models.
    :param response_transformation: Optional sklearn transformer for responses.
    :param run_id: Subfolder name under path_out.
    :param test_mode: Split mode ("LPO", "LCO", "LDO", "LTO").
    :param hpam_optimization_metric: Metric optimized during HPO.
    :param n_cv_splits: Number of cross-validation folds.
    :param path_out: Root directory for experiment outputs.
    :param overwrite: Recompute even when artifacts exist.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param hyperparameter_tuning: Whether to run HPO.
    :param wandb_project: Optional W&B project name.
    :param hpo_num_samples: Number of HPO trials per fold.
    :param hpo_random_state: Random seed for HPO.
    :param hpo_resources_per_trial: Optional Ray resource limits.
    """
    seed_everything(42)
    baselines = _normalize_baselines(baselines)

    split_label = test_mode
    result_path = experiment_result_path(path_out, run_id, dataset_name, split_label)
    split_path = result_path / "splits"
    result_folder_exists = result_path.exists()

    folds = prepare_splits(
        mudataset,
        split_path=split_path,
        result_path=result_path,
        test_mode=test_mode,
        n_cv_splits=n_cv_splits,
        overwrite=overwrite,
        result_folder_exists=result_folder_exists,
    )

    all_models = models + baselines
    for model_class in all_models:
        model_name = model_class.get_model_name()
        print(f"Running {model_name}")

        predictions_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=None,
            result_path=result_path,
            suffix="predictions",
        )
        hpam_path = generate_data_saving_path(
            model_name=model_name,
            drug_id=None,
            result_path=result_path,
            suffix="best_hpams",
        )

        for split_index, split_masks in enumerate(folds):
            print()
            print(f"################# FOLD {split_index + 1}/{len(folds)} #################")
            print()

            prediction_file = predictions_path / f"predictions_split_{split_index}.csv"
            hpam_save_path = hpam_path / f"best_hpams_split_{split_index}.json"

            if prediction_file.is_file() and not overwrite:
                print(f"Split {split_index} already exists. Skipping.")
                continue

            fold_data = prepare_mu_fold(mudataset, split_masks, model_class)
            merged_scope = merge_train_val_scopes(split_masks)

            hpo_cfg = build_experiment_hpo_config(
                hpam_optimization_metric,
                n_trials=hpo_num_samples,
                random_state=hpo_random_state,
                resources_per_trial=hpo_resources_per_trial,
                storage_path=fold_hpo_storage_path(result_path),
            )

            best_hpams = select_fold_hyperparameters(
                model_class=model_class,
                mudataset=mudataset,
                train_scope=fold_data.train_scope,
                val_scope=fold_data.val_scope,
                early_stopping_scope=fold_data.early_stopping_scope,
                response_transformation=response_transformation,
                metric=hpam_optimization_metric,
                model_checkpoint_dir=model_checkpoint_dir,
                hyperparameter_tuning=hyperparameter_tuning,
                hpo_config=hpo_cfg,
            )

            print(f"Best hyperparameters: {best_hpams}")
            with open(hpam_save_path, "w", encoding="utf-8") as f:
                json.dump(best_hpams, f)

            model = model_class(best_hpams)
            fold_transform = None if response_transformation is None else clone(response_transformation)

            predictions = mu_train_and_predict(
                model=model,
                mudataset=mudataset,
                train_scope=merged_scope,
                test_scope=fold_data.test_scope,
                early_stopping_scope=fold_data.early_stopping_scope,
                response_transformation=fold_transform,
                model_checkpoint_dir=model_checkpoint_dir,
            )

            _write_mu_predictions(
                prediction_file,
                mudataset=mudataset,
                test_scope=fold_data.test_scope,
                predictions=predictions,
            )

    print("Done!")


def _write_mu_predictions(
    prediction_file: str | Path,
    mudataset: MuDataset,
    test_scope: EntityScope,
    predictions: np.ndarray,
) -> None:
    """Write predictions to CSV in the standard drevalpy format."""
    import pandas as pd

    cell_line_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids

    cl_indices = test_scope.cell_lines
    dr_indices = test_scope.drugs

    rows: dict[str, Any] = {
        "cell_line_ids": cell_line_ids[cl_indices],
    }
    if dr_indices is not None:
        rows["drug_ids"] = drug_ids[dr_indices]
    else:
        rows["drug_ids"] = np.full(len(cl_indices), "all", dtype=object)

    rows["predictions"] = predictions

    response_matrix = mudataset.response_matrix
    if dr_indices is not None:
        rows["response"] = response_matrix[cl_indices, dr_indices]
    else:
        rows["response"] = np.nanmean(response_matrix[cl_indices, :], axis=1)

    df = pd.DataFrame(rows)
    Path(prediction_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prediction_file, index=False)
