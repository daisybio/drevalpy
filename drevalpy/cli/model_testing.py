"""For the nf-core/drugresponseeval subworkflow model_testing."""

import pandas as pd
import yaml
from upath import UPath as Path

from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary
from drevalpy.utils.pickle_io import load_trusted_pickle


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

    import numpy as np

    from drevalpy.components.feature_source import CellLineFeatureSource, DrugFeatureSource
    from drevalpy.data import load_mudataset

    mudataset = load_mudataset(train_dataset.dataset_name)
    all_cl_ids = np.array(mudataset.cell_line_ids)
    all_drug_ids = np.array(mudataset.drug_ids)
    cl_features = CellLineFeatureSource(mudataset, all_cl_ids)
    drug_features = DrugFeatureSource(mudataset, all_drug_ids)
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
