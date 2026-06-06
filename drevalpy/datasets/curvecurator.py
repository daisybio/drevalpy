"""Compatibility wrappers for CurveCurator curation.

New code should import from ``drevalpy.curation`` instead.
"""

from __future__ import annotations

from drevalpy.curation._curvecurator.workflow import curate_to_csv


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
