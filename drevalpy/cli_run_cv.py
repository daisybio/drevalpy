"""For the nf-core/drugresponseeval subworkflow run_cv."""

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import TransformerMixin

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models import DRPModel


def run_load_response(
    *,
    response_dataset: str,
    cross_study_dataset: bool = False,
    measure: str = "LN_IC50_curvecurator",
    clean_min_responders: int | None = None,
    clean_min_responder_frac: float | None = None,
) -> None:
    """
    Load drug response CSV and pickle a ``DrugResponseDataset``.

    If ``clean_min_responders`` or ``clean_min_responder_frac`` is set, whole drugs failing the responder
    criterion are removed (needs a curve-curated CSV with a ``Regulation`` column). The dataset keeps its
    base name so that downstream steps still find its feature files.
    """
    from drevalpy.datasets.dataset import DrugResponseDataset
    from drevalpy.datasets.loader import AVAILABLE_DATASETS, DrugCurveFilter
    from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER

    input_file = Path(response_dataset)
    dataset_name = input_file.stem
    if dataset_name in AVAILABLE_DATASETS:
        response_file = pd.read_csv(input_file, dtype={"pubchem_id": str})
        if dataset_name == "BeatAML2":
            response_file[TISSUE_IDENTIFIER] = "Blood"
        elif dataset_name == "PDX_Bruna":
            response_file[TISSUE_IDENTIFIER] = "Breast"
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
            input_file=input_file, dataset_name=dataset_name, measure=measure, tissue_column=tissue_column
        )
    if clean_min_responders is not None or clean_min_responder_frac is not None:
        drug_filter = DrugCurveFilter(min_responders=clean_min_responders, min_responder_frac=clean_min_responder_frac)
        kept = drug_filter.apply(pd.read_csv(input_file, dtype={"pubchem_id": str}))[DRUG_IDENTIFIER].unique()
        response_data.reduce_to(drug_ids=kept)
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
    """Write one YAML per hyperparameter combination for a model."""
    from drevalpy.models import MODEL_FACTORY, MULTI_DRUG_MODEL_FACTORY, SINGLE_DRUG_MODEL_FACTORY

    if model_name in MULTI_DRUG_MODEL_FACTORY:
        resolved_name = model_name
    else:
        resolved_name = str(model_name).split(".")[0]
        if resolved_name not in SINGLE_DRUG_MODEL_FACTORY:
            raise ValueError(f"{resolved_name} neither in SINGLE_DRUG_MODEL_FACTORY nor in MULTI_DRUG_MODEL_FACTORY.")
    model_class = MODEL_FACTORY[resolved_name]
    hyperparameters = model_class.get_hyperparameter_set()
    if not hyperparameter_tuning:
        hyperparameters = [hyperparameters[0]]
    hpam_idx = 0
    for hpam_combi in hyperparameters:
        with open(f"hpam_{hpam_idx}.yaml", "w") as yaml_file:
            hpam_idx += 1
            yaml.dump(hpam_combi, yaml_file, default_flow_style=False)


def _train_and_predict_cv_split(
    drug_id: str | None,
    split: dict,
    *,
    model_class: "type[DRPModel]",
    hpams: dict,
    path_data: str,
    response_transform: TransformerMixin | None,
    model_checkpoint_dir: str,
) -> DrugResponseDataset:
    """
    Train on a CV split and predict its validation set, for single drug models for one drug.

    :param drug_id: drug id for single drug models, None for global models
    :param split: CV split
    :param model_class: model to train
    :param hpams: hyperparameters to use
    :param path_data: path to the data directory
    :param response_transform: normalizer to use for the response data
    :param model_checkpoint_dir: directory to save model checkpoints
    :returns: validation dataset with predictions
    """
    from drevalpy.experiment import get_datasets_from_cv_split, train_and_predict

    train_dataset, validation_dataset, es_dataset, _test_dataset = get_datasets_from_cv_split(
        split, model_class, model_class.get_model_name(), drug_id
    )
    return train_and_predict(
        model=model_class(),
        hpams=hpams,
        path_data=path_data,
        train_dataset=train_dataset,
        prediction_dataset=validation_dataset,
        early_stopping_dataset=es_dataset,
        response_transformation=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )


def run_train_and_predict_cv(
    *,
    model_name: str,
    path_data: str = "data",
    test_mode: str = "LPO",
    hyperparameters: str,
    cv_data: str,
    response_transformation: str = "None",
    model_checkpoint_dir: str = "TEMPORARY",
    n_jobs: int = 1,
) -> None:
    """Train on a CV split and pickle validation predictions, for single drug models a {drug_id: predictions} dict."""
    from drevalpy.experiment import run_per_drug, split_by_drug
    from drevalpy.models import MODEL_FACTORY
    from drevalpy.utils import get_response_transformation

    model_class = MODEL_FACTORY[model_name]
    with open(cv_data, "rb") as f:
        split = pickle.load(f)

    response_transform = get_response_transformation(response_transformation)
    with open(hyperparameters) as f:
        hpams = yaml.safe_load(f)

    train_and_predict_split = partial(
        _train_and_predict_cv_split,
        model_class=model_class,
        hpams=hpams,
        path_data=path_data,
        response_transform=response_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )
    if model_class.is_single_drug_model:
        drug_ids = np.unique(np.concatenate([ds.drug_ids for ds in split.values() if ds is not None]))
        splits_per_drug = split_by_drug(split, drug_ids)
        validation_dataset = run_per_drug(
            train_and_predict_split, {drug: (s,) for drug, s in splits_per_drug.items()}, n_jobs=n_jobs
        )
    else:
        validation_dataset = train_and_predict_split(None, split)

    with open(
        f"prediction_dataset_{model_name}_{str(cv_data).split('.pkl')[0]}_"
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


def _find_best_hpam_combi(
    pred_datas: list, hpam_combis: list[dict], optim_metric: str
) -> tuple[dict | None, float | None]:
    """
    Find the hyperparameter combination with the best predictions.

    :param pred_datas: predictions per hyperparameter combination
    :param hpam_combis: hyperparameter combinations, in the order of pred_datas
    :param optim_metric: metric to optimize
    :returns: best hyperparameter combination and its result
    """
    from drevalpy.evaluation import MAXIMIZATION_METRICS, MINIMIZATION_METRICS, evaluate

    best_hpam_combi = None
    best_result = None
    for pred_data, hpam_combi in zip(pred_datas, hpam_combis, strict=True):
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
    return best_hpam_combi, best_result


def run_evaluate_and_find_max(
    *,
    model_name: str,
    split_id: str,
    hpam_yamls: list[str],
    pred_datas: list[str],
    optim_metric: str = "RMSE",
) -> None:
    """Pick the best hyperparameter YAML for one CV split, for single drug models per drug."""
    from drevalpy.models import MODEL_FACTORY

    pred_data_list = []
    hpam_combis = []
    for i in range(0, len(pred_datas)):
        with open(pred_datas[i], "rb") as pred_file:
            pred_data_list.append(pickle.load(pred_file))
        with open(hpam_yamls[i]) as yaml_file:
            hpam_combis.append(yaml.safe_load(yaml_file))

    if MODEL_FACTORY[model_name].is_single_drug_model:
        # the predictions are {drug_id: predictions} dicts, see run_train_and_predict_cv
        best_per_drug = {
            drug: _find_best_hpam_combi([pred_data[drug] for pred_data in pred_data_list], hpam_combis, optim_metric)
            for drug in pred_data_list[0]
        }
        result = {
            "best_hpam_combi_per_drug": {drug: best[0] for drug, best in best_per_drug.items()},
            "best_result_per_drug": {drug: best[1] for drug, best in best_per_drug.items()},
        }
    else:
        best_hpam_combi, best_result = _find_best_hpam_combi(pred_data_list, hpam_combis, optim_metric)
        result = {"best_hpam_combi": best_hpam_combi, "best_result": best_result}
    final_result = {f"{model_name}_{split_id}": result}
    with open(f"best_hpam_combi_{split_id}.yaml", "w") as yaml_file:
        yaml.dump(final_result, yaml_file, default_flow_style=False)
