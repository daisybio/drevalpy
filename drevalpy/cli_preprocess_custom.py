"""For the nf-core/drugresponseeval subworkflow preprocess_custom."""

from __future__ import annotations

from pathlib import Path

from drevalpy.curation._curvecurator.io import combine_manifest_to_csv, split_to_disk


def run_preprocess_raw_viability(
    *,
    path_data: str = "./data",
    dataset_name: str,
    cores: int = 4,
) -> None:
    """Prepare raw viability data for CurveCurator via ``curation split``."""
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
    """Combine CurveCurator outputs into a single dataset CSV."""
    output_folder = Path(path_data).resolve() / dataset_name
    manifest = output_folder / "curation_manifest.json"
    if manifest.is_file():
        combine_manifest_to_csv(manifest)
        return

    from drevalpy.datasets.curvecurator import postprocess

    postprocess(output_folder=str(output_folder), dataset_name=dataset_name)
