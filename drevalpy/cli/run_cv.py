"""For the nf-core/drugresponseeval subworkflow run_cv."""

import pickle
from pathlib import Path

import pandas as pd
import yaml


def run_load_response(
    *,
    response_dataset: str,
    cross_study_dataset: bool = False,
    measure: str = "LN_IC50_curvecurator",
) -> None:
    """Load drug response CSV and pickle a ``DrugResponseDataset``."""
    from drevalpy.datasets.dataset import DrugResponseDataset
    from drevalpy.datasets.loader import get_builtin_dataset_entry
    from drevalpy.datasets.utils import (
        CELL_LINE_IDENTIFIER,
        DRUG_IDENTIFIER,
        TISSUE_IDENTIFIER,
    )

    input_file = Path(response_dataset)
    dataset_name = input_file.stem
    entry = get_builtin_dataset_entry(dataset_name)
    if entry is not None:
        response_file = pd.read_csv(input_file, dtype={"pubchem_id": str})
        if entry.tissue_override is not None:
            response_file[TISSUE_IDENTIFIER] = entry.tissue_override
        response_data = DrugResponseDataset(
            response=response_file[measure].values,
            cell_line_ids=response_file[CELL_LINE_IDENTIFIER].values,
            drug_ids=response_file[DRUG_IDENTIFIER].values,
            tissues=response_file[TISSUE_IDENTIFIER].values,
            dataset_name=dataset_name,
        )
    else:
        tissue_column: str | None = TISSUE_IDENTIFIER
        if TISSUE_IDENTIFIER not in pd.read_csv(input_file, nrows=1).columns:
            tissue_column = None

        response_data = DrugResponseDataset.from_csv(
            input_file=input_file,
            dataset_name=dataset_name,
            measure=measure,
            tissue_column=tissue_column,
        )
    outfile = f"cross_study_{dataset_name}.pkl" if cross_study_dataset else "response_dataset.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(response_data, f)


def run_cv_split(
    *,
    response: str,
    n_cv_splits: int,
    test_mode: str = "LPO",
    validation_ratio: float = 0.1,
    seed: int = 42,
    custom_splitter_path: str | None = None,
) -> None:
    """Split pickled response data into CV fold pickles."""
    from drevalpy.datasets.splits import create_and_record_splits

    with open(response, "rb") as f:
        response_data = pickle.load(f)
    create_and_record_splits(
        response_data,
        split_path=".",
        split_label=test_mode,
        external_splitter=custom_splitter_path,
        test_mode=test_mode,
        n_cv_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=seed,
        split_early_stopping=True,
    )
    for split_index, split in enumerate(response_data.cv_splits):
        with open(f"split_{split_index}.pkl", "wb") as f:
            pickle.dump(split, f)


def run_hpam_split(
    *,
    model_name: str,
    hyperparameter_tuning: bool = False,
) -> None:
    """Write ``hpam_0.yaml`` with a model's default hyperparameters.

    Ray/Optuna tuning runs at experiment time; this helper no longer emits search grids.
    """
    import warnings

    from drevalpy.models._model_lookup import (
        get_model_class,
        is_multi_drug_model_name,
        is_single_drug_model_name,
    )

    if is_multi_drug_model_name(model_name):
        resolved_name = model_name
    else:
        resolved_name = str(model_name).split(".")[0]
        if not is_single_drug_model_name(resolved_name):
            raise ValueError(f"{resolved_name} is neither a multi-drug nor a single-drug zoo model name.")
    model_class = get_model_class(resolved_name)
    if hyperparameter_tuning:
        warnings.warn(
            "hyperparameter_tuning=True no longer emits a YAML search grid; "
            "enable Ray/Optuna tuning in the experiment CLI instead. "
            "Writing default hyperparameters to hpam_0.yaml.",
            stacklevel=2,
        )
    defaults = model_class.get_default_hyperparameters()
    with open("hpam_0.yaml", "w", encoding="utf-8") as yaml_file:
        yaml.dump(defaults, yaml_file, default_flow_style=False)


def run_train_and_predict_cv(
    *,
    model_name: str,
    path_data: str = "data",
    test_mode: str = "LPO",
    hyperparameters: str,
    cv_data: str,
    response_transformation: str = "None",
    model_checkpoint_dir: str = "TEMPORARY",
) -> None:
    """Train on a CV split and pickle validation predictions."""
    from drevalpy.experiment import get_model_name_and_drug_id, train_and_predict
    from drevalpy.experiment.fold import get_datasets_from_cv_split
    from drevalpy.models._model_lookup import get_model_class
    from drevalpy.utils import get_response_transformation

    resolved_name, drug_id = get_model_name_and_drug_id(model_name)
    model_class = get_model_class(resolved_name)
    with open(cv_data, "rb") as f:
        split = pickle.load(f)

    train_dataset, validation_dataset, es_dataset, _test_dataset = get_datasets_from_cv_split(
        split, model_class, resolved_name, drug_id
    )

    response_transform = get_response_transformation(response_transformation)
    with open(hyperparameters) as f:
        hpams = yaml.safe_load(f)
    model = model_class(hpams)

    validation_dataset = train_and_predict(
        model=model,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=es_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )

    with open(
        f"prediction_dataset_{resolved_name}_{str(cv_data).split('.pkl')[0]}_"
        f"{str(hyperparameters).split('.yaml')[0]}.pkl",
        "wb",
    ) as f:
        pickle.dump(validation_dataset, f)


def _best_metric(metric, current_metric, best_metric, minimization_metrics, maximization_metrics):
    if metric in minimization_metrics:
        if current_metric < best_metric:
            return True
    elif metric in maximization_metrics:
        if current_metric > best_metric:
            return True
    else:
        raise ValueError(f"Metric {metric} not recognized.")
    return False


def run_evaluate_and_find_max(
    *,
    model_name: str,
    split_id: str,
    hpam_yamls: list[str],
    pred_datas: list[str],
    optim_metric: str = "RMSE",
) -> None:
    """Pick the best hyperparameter YAML for one CV split.

    With ``make-hpam-yamls`` emitting a single defaults file, this is usually a
    no-op selector. Prefer Ray/Optuna via ``hpam_tune`` / the root experiment.
    """
    import warnings

    from drevalpy.evaluation import MAXIMIZATION_METRICS, MINIMIZATION_METRICS, evaluate

    warnings.warn(
        "evaluate-hpams selects among YAML prediction artifacts and is not Ray/Optuna "
        "tuning. Prefer drevalpy.components.tuning.hpam_tune or the root experiment CLI.",
        DeprecationWarning,
        stacklevel=2,
    )

    best_hpam_combi = None
    best_result = None
    for i in range(0, len(pred_datas)):
        with open(pred_datas[i], "rb") as pred_file:
            pred_data = pickle.load(pred_file)
        with open(hpam_yamls[i]) as yaml_file:
            hpam_combi = yaml.safe_load(yaml_file)
        results = evaluate(pred_data, optim_metric)
        if best_result is None:
            best_result = results[optim_metric]
            best_hpam_combi = hpam_combi
        elif _best_metric(
            metric=optim_metric,
            current_metric=results[optim_metric],
            best_metric=best_result,
            minimization_metrics=MINIMIZATION_METRICS,
            maximization_metrics=MAXIMIZATION_METRICS,
        ):
            best_result = results[optim_metric]
            best_hpam_combi = hpam_combi
    final_result = {
        f"{model_name}_{split_id}": {
            "best_hpam_combi": best_hpam_combi,
            "best_result": best_result,
        }
    }
    with open(f"best_hpam_combi_{split_id}.yaml", "w") as yaml_file:
        yaml.dump(final_result, yaml_file, default_flow_style=False)
