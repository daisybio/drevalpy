"""Full in-memory CurveCurator curation workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.curation._curvecurator.combine import combine, write_dataset_csv
from drevalpy.curation._curvecurator.curvecurator import curvecurator_many
from drevalpy.curation._curvecurator.split import split
from drevalpy.curation._curvecurator.types import CurationFitResult, CurationSplitResult


def curate(
    input_file: str | Path,
    dataset_name: str,
    *,
    cores: int = 1,
    normalize: bool = False,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    gpu_available: bool = False,
    curve_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run split, CurveCurator, and combine in one in-memory workflow.

    :param input_file: Path to the raw viability CSV.
    :param dataset_name: Dataset name used in metadata and combine output.
    :param cores: Maximum CPU worker threads for CurveCurator execution.
    :param normalize: Whether CurveCurator should normalize responses.
    :param device: Requested PyTorch device string.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param gpu_available: Whether GPU resources are available for accelerator chunking.
    :param curve_df: Optional preloaded raw dataframe.
    :returns: Combined curated dataset table.
    """
    split_result = split(
        input_file=input_file,
        dataset_name=dataset_name,
        cores=cores,
        normalize=normalize,
        device=device,
        chunk_size=chunk_size,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        gpu_available=gpu_available,
        curve_df=curve_df,
    )
    fit_results = curvecurator_many(
        split_result.work_items,
        cores=cores,
        device=device,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
    )
    return combine(fit_results, dataset_name=dataset_name)


def curate_to_csv(
    input_file: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    *,
    cores: int = 1,
    normalize: bool = False,
    device: str = "auto",
    chunk_size: int = 1_000,
    gpu_min_curves: int = 1_000,
    gpu_chunk_size: int = 50_000,
    gpu_available: bool = False,
) -> Path:
    """Run the in-memory curation workflow and write ``<dataset_name>.csv``.

    :param input_file: Path to the raw viability CSV.
    :param output_dir: Directory for the combined dataset CSV.
    :param dataset_name: Dataset name used in metadata and output filename.
    :param cores: Maximum CPU worker threads for CurveCurator execution.
    :param normalize: Whether CurveCurator should normalize responses.
    :param device: Requested PyTorch device string.
    :param chunk_size: Maximum curves per CPU chunk.
    :param gpu_min_curves: Minimum curves before ``auto`` may select an accelerator.
    :param gpu_chunk_size: Maximum curves per accelerator chunk.
    :param gpu_available: Whether GPU resources are available for accelerator chunking.
    :returns: Path to the written dataset CSV.
    """
    dataset = curate(
        input_file=input_file,
        dataset_name=dataset_name,
        cores=cores,
        normalize=normalize,
        device=device,
        chunk_size=chunk_size,
        gpu_min_curves=gpu_min_curves,
        gpu_chunk_size=gpu_chunk_size,
        gpu_available=gpu_available,
    )
    return write_dataset_csv(dataset, Path(output_dir) / f"{dataset_name}.csv")


__all__ = ["CurationFitResult", "CurationSplitResult", "curate", "curate_to_csv"]
