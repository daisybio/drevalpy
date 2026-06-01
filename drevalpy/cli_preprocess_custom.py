"""For the nf-core/drugresponseeval subworkflow preprocess_custom."""

from pathlib import Path


def run_preprocess_raw_viability(
    *,
    path_data: str = "./data",
    dataset_name: str,
    cores: int = 4,
) -> None:
    """Preprocess raw viability data with CurveCurator."""
    from drevalpy.datasets.curvecurator import preprocess

    input_file = Path(path_data).resolve() / dataset_name / f"{dataset_name}_raw.csv"
    output_dir = input_file.parent
    preprocess(
        input_file=str(input_file),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        cores=cores,
    )


def run_postprocess_viability(
    *,
    dataset_name: str,
    path_data: str = "./",
) -> None:
    """Postprocess CurveCurator output into a single dataset CSV."""
    from drevalpy.datasets.curvecurator import postprocess

    output_folder = Path(path_data).resolve() / dataset_name
    postprocess(output_folder=str(output_folder), dataset_name=dataset_name)
