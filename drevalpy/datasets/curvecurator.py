"""Compatibility wrappers for CurveCurator curation.

New code should import from ``drevalpy.curation`` instead.
"""

from __future__ import annotations

from pathlib import Path

from drevalpy.curation import combine as _combine
from drevalpy.curation._curvecurator.combine import combine_from_disk, write_dataset_csv
from drevalpy.curation._curvecurator.split import build_config, load_raw_curve_df, prepare_input_table
from drevalpy.curation._curvecurator.workflow import curate_to_csv
from drevalpy.pipeline_function import pipeline_function

# Backward-compatible aliases used by tests and legacy imports.
_load_raw_curve_df = load_raw_curve_df
_build_config = build_config


def _prepare_raw_data(curve_df, output_dir: Path, prefix: str = ""):
    """Prepare CurveCurator input on disk for legacy callers.

    :param curve_df: Raw viability rows.
    :param output_dir: Directory in which the compatibility TSV is written.
    :param prefix: Optional subdirectory name below ``output_dir``.
    :returns: Experiment count, dose list, replicate count, and curve count.
    """
    input_table, n_exp, doses, n_replicates, n_curves = prepare_input_table(curve_df)
    curvecurator_folder = Path(output_dir) / prefix
    curvecurator_folder.mkdir(exist_ok=True, parents=True)
    input_table.to_csv(curvecurator_folder / "curvecurator_input.tsv", sep="\t", index=False)
    return n_exp, doses, n_replicates, n_curves


@pipeline_function
def postprocess(output_folder: str, dataset_name: str) -> None:
    """Combine fitted CurveCurator outputs on disk into ``<dataset_name>.csv``.

    :param output_folder: Directory containing fitted CurveCurator output files.
    :param dataset_name: Dataset name used for the output CSV.
    """
    dataset = combine_from_disk(output_folder, dataset_name)
    write_dataset_csv(dataset, Path(output_folder) / f"{dataset_name}.csv")


def fit_curves(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    cores: int,
    normalize: bool = False,
    *,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
) -> None:
    """Fit curves for provided raw viability data and write ``<dataset_name>.csv``.

    :param input_file: Raw viability CSV path.
    :param output_dir: Directory where the fitted dataset CSV is written.
    :param dataset_name: Dataset name used for the output file.
    :param cores: Maximum CPU worker threads.
    :param normalize: Whether to normalize responses for CurveCurator.
    :param device: Requested PyTorch device string.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before using an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    """
    curate_to_csv(
        input_file=input_file,
        output_dir=output_dir,
        dataset_name=dataset_name,
        cores=cores,
        normalize=normalize,
        device=device,
        chunk_size=chunk_size,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
    )


def combine(*args, **kwargs):
    """Deprecated alias kept for transitional imports.

    :param args: Positional arguments forwarded to ``drevalpy.curation.combine``.
    :param kwargs: Keyword arguments forwarded to ``drevalpy.curation.combine``.
    :returns: Combined CurveCurator dataset table.
    """
    return _combine(*args, **kwargs)
