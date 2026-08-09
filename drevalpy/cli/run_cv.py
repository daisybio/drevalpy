"""For the nf-core/drugresponseeval subworkflow run_cv."""

import pandas as pd
import yaml
from upath import UPath as Path

from drevalpy.utils.pickle_io import dump_trusted_pickle, load_trusted_pickle


def run_load_response(
    *,
    response_dataset: str,
    cross_study_dataset: bool = False,
    measure: str = "LN_IC50_curvecurator",
) -> None:
    """Load drug response CSV and pickle a response DataFrame.

    :param response_dataset: response dataset.
    :param cross_study_dataset: cross study dataset.
    :param measure: measure.
    """
    from drevalpy.datasets.utils import (
        CELL_LINE_IDENTIFIER,
        DRUG_IDENTIFIER,
        TISSUE_IDENTIFIER,
    )

    input_file = Path(response_dataset)
    dataset_name = input_file.stem

    response_file = pd.read_csv(input_file, dtype={"pubchem_id": str})

    required_cols = [CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, measure]
    if TISSUE_IDENTIFIER in response_file.columns:
        required_cols.append(TISSUE_IDENTIFIER)
    response_data = response_file[required_cols].rename(columns={measure: "response"})
    response_data.attrs["dataset_name"] = dataset_name

    outfile = f"cross_study_{dataset_name}.pkl" if cross_study_dataset else "response_dataset.pkl"
    dump_trusted_pickle(response_data, outfile)


def run_cv_split(
    *,
    response: str | Path,
    n_cv_splits: int,
    test_mode: str = "LPO",
    validation_ratio: float = 0.1,
    seed: int = 42,
    custom_splitter_path: str | None = None,
) -> None:
    """Split pickled response data into CV fold pickles.

    :param response: response.
    :param n_cv_splits: n cv splits.
    :param test_mode: test mode.
    :param validation_ratio: validation ratio.
    :param seed: seed.
    :param custom_splitter_path: custom splitter path.
    """
    from drevalpy.datasets.splitting import MuDataSplitter

    response_data = load_trusted_pickle(response)
    splitter = MuDataSplitter()
    folds = splitter.split(
        response_data,
        mode=test_mode,
        n_splits=n_cv_splits,
        validation_ratio=validation_ratio,
        random_state=seed,
        external_splitter=custom_splitter_path,
    )
    for split_index, split in enumerate(folds):
        dump_trusted_pickle(split, f"split_{split_index}.pkl")


def run_hpam_split(
    *,
    model_name: str,
    hyperparameter_tuning: bool = False,
) -> None:
    """Write ``hpam_0.yaml`` with a model's default hyperparameters.

    Ray/Optuna tuning runs at experiment time; this helper no longer emits search grids.

    :param model_name: model name.
    :param hyperparameter_tuning: hyperparameter tuning.
    :raises ValueError: If ``model_name`` is neither a multi-drug nor single-drug zoo name.
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

    :param model_name: model name.
    :param split_id: split id.
    :param hpam_yamls: hpam yamls.
    :param pred_datas: pred datas.
    :param optim_metric: optim metric.
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
        pred_data = load_trusted_pickle(pred_datas[i])
        with open(hpam_yamls[i]) as yaml_file:
            hpam_combi = yaml.safe_load(yaml_file)
        results = evaluate(predictions=pred_data.predictions, response=pred_data.response, metric=optim_metric)
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
