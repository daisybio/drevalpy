"""nf-core/drugresponseeval hooks for CurveCurator viability preprocessing."""

from __future__ import annotations

from pathlib import Path

from drevalpy.curation import combine, load_raw_curve_df, split, write_dataset_csv
from drevalpy.curation._curvecurator.combine import combine_from_disk
from drevalpy.curation._curvecurator.io import read_fit_results_from_manifest, read_manifest, write_split_artifacts


def run_preprocess_raw_viability(
    *,
    path_data: str = "./data",
    dataset_name: str,
    cores: int = 4,
) -> None:
    """Prepare raw viability data for CurveCurator via ``curation split``.

    :param path_data: Base directory containing ``<dataset_name>/<dataset_name>_raw.csv``.
    :param dataset_name: Dataset name used for split artifacts.
    :param cores: CPU worker threads passed to split chunk sizing.
    """
    input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}_raw.csv"
    output_dir = input_file.parent
    raw_df = load_raw_curve_df(input_file)
    split_result = split(
        raw_df,
        dataset_name=dataset_name,
        input_filename=input_file.name,
        cores=cores,
    )
    write_split_artifacts(split_result, output_dir)


def run_postprocess_viability(
    *,
    dataset_name: str,
    path_data: str = "./",
) -> None:
    """Combine CurveCurator outputs into a single dataset CSV.

    Uses the manifest workflow when ``curation_manifest.json`` exists; otherwise
    falls back to legacy on-disk ``curves.tsv`` / parquet artifacts.

    :param dataset_name: Dataset name used for the output CSV.
    :param path_data: Base directory containing fitted CurveCurator outputs.
    """
    output_folder = Path(path_data).resolve() / dataset_name
    manifest = output_folder / "curation_manifest.json"
    if manifest.is_file():
        manifest_data = read_manifest(manifest)
        fit_results = read_fit_results_from_manifest(manifest)
        dataset = combine(fit_results, dataset_name=manifest_data["dataset_name"])
        write_dataset_csv(dataset, output_folder / f"{dataset_name}.csv")
        return

    dataset = combine_from_disk(output_folder, dataset_name)
    write_dataset_csv(dataset, output_folder / f"{dataset_name}.csv")
