"""Command line entry point for fitting raw viability data with CurveCurator."""

from __future__ import annotations

import argparse
from pathlib import Path

from drevalpy.datasets.curvecurator import fit_curves


def _dataset_name_from_input(input_file: Path) -> str:
    stem = input_file.stem
    return stem.removesuffix("_raw")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``drevalpy-fit-curves`` argument parser.

    :returns: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=("Fit a raw viability CSV with CurveCurator and write the fitted " "<dataset_name>.csv file.")
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Raw viability CSV with columns dose, response, sample, drug, and optional replicate.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for intermediate CurveCurator files and fitted CSV. Defaults to input file parent.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for the output CSV. Defaults to input stem with a trailing '_raw' removed.",
    )
    parser.add_argument("--cores", type=int, default=1, help="CPU worker threads for CurveCurator chunks.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize response values to [0, 1] for CurveCurator.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='PyTorch device: "auto", "cpu", "cuda", "cuda:0", or "mps".',
    )
    parser.add_argument("--chunk_size", type=int, default=1_000, help="Maximum curves per CPU chunk.")
    parser.add_argument(
        "--gpu_min_curves",
        type=int,
        default=1_000,
        help="Minimum curves before auto device selection may use an accelerator.",
    )
    parser.add_argument(
        "--gpu_chunk_size",
        type=int,
        default=50_000,
        help="Maximum curves per accelerator chunk.",
    )
    return parser


def fit_curves_cmd() -> None:
    """Fit raw viability data and write the fitted CurveCurator CSV."""
    args = build_parser().parse_args()
    input_file = args.input_file.expanduser().resolve()
    output_dir = (args.output_dir.expanduser() if args.output_dir else input_file.parent).resolve()
    dataset_name = args.dataset_name or _dataset_name_from_input(input_file)

    fit_curves(
        input_file=str(input_file),
        output_dir=str(output_dir),
        dataset_name=dataset_name,
        cores=args.cores,
        normalize=args.normalize,
        device=args.device,
        chunk_size=args.chunk_size,
        gpu_min_curves=args.gpu_min_curves,
        gpu_chunk_size=args.gpu_chunk_size,
    )

    print(output_dir / f"{dataset_name}.csv")
