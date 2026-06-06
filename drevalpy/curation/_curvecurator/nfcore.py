"""nf-core/drugresponseeval hooks for CurveCurator viability preprocessing."""

from __future__ import annotations

from pathlib import Path

from drevalpy.curation._curvecurator.combine import combine_from_disk, write_dataset_csv
from drevalpy.curation._curvecurator.io import combine_manifest_to_csv, split_to_disk


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
    split_to_disk(
        input_file=input_file,
        output_dir=output_dir,
        dataset_name=dataset_name,
        cores=cores,
    )


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
        combine_manifest_to_csv(manifest)
        return

    dataset = combine_from_disk(output_folder, dataset_name)
    write_dataset_csv(dataset, output_folder / f"{dataset_name}.csv")
