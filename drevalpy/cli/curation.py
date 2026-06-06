"""``drevalpy curation`` command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from typer import _click

from drevalpy.curation import curate_to_csv
from drevalpy.curation._curvecurator.io import (
    combine_manifest_to_csv,
    curvecurator_to_disk,
    split_to_disk,
)

CURATOR_DEVICE_HELP = (
    'PyTorch device for CurveCurator fitting: "auto" (CUDA, then MPS, then CPU when enough curves), '
    '"cpu", "cuda", "cuda:0", or "mps".'
)


def register(app: typer.Typer) -> None:
    """Register the ``curation`` Typer sub-app."""
    curation_app = typer.Typer(
        name="curation",
        help="CurveCurator curation workflow: split, curvecurator, and combine.",
        no_args_is_help=False,
    )
    app.add_typer(curation_app, name="curation")

    @curation_app.callback(invoke_without_command=True)
    def curation_root(
        ctx: _click.Context,
        input_file: Annotated[
            Path | None,
            typer.Option(
                "--input-file",
                "--input_file",
                help="Raw viability CSV with dose, response, sample, and drug columns.",
            ),
        ] = None,
        output_dir: Annotated[
            Path | None,
            typer.Option("--output-dir", "--output_dir", help="Directory for the combined dataset CSV."),
        ] = None,
        dataset_name: Annotated[
            str | None,
            typer.Option(
                "--dataset-name",
                "--dataset_name",
                help="Dataset name for output CSV. Defaults to input stem without '_raw'.",
            ),
        ] = None,
        cores: Annotated[int, typer.Option("--cores", help="CPU worker threads for CurveCurator chunks.")] = 1,
        normalize: Annotated[
            bool,
            typer.Option("--normalize", help="Normalize response values to [0, 1] for CurveCurator."),
        ] = False,
        device: Annotated[str, typer.Option("--device", help=CURATOR_DEVICE_HELP)] = "auto",
        chunk_size: Annotated[
            int, typer.Option("--chunk-size", "--chunk_size", help="Maximum curves per CPU chunk.")
        ] = 1_000,
        gpu_min_curves: Annotated[
            int,
            typer.Option(
                "--gpu-min-curves",
                "--gpu_min_curves",
                help="Minimum curves before auto device selection may use an accelerator.",
            ),
        ] = 1_000,
        gpu_chunk_size: Annotated[
            int,
            typer.Option("--gpu-chunk-size", "--gpu_chunk_size", help="Maximum curves per accelerator chunk."),
        ] = 50_000,
        gpu_available: Annotated[
            bool,
            typer.Option(
                "--gpu-available/--no-gpu-available",
                help="Whether GPU resources are available for accelerator chunking during split.",
            ),
        ] = False,
    ) -> None:
        """Run split, CurveCurator, and combine in one command."""
        if ctx.invoked_subcommand is not None:
            return
        if input_file is None or output_dir is None:
            raise typer.BadParameter("Top-level curation requires --input-file and --output-dir.")
        resolved_input = input_file.expanduser().resolve()
        resolved_output = output_dir.expanduser().resolve()
        resolved_dataset = dataset_name or _dataset_name_from_input(resolved_input)
        output_path = curate_to_csv(
            input_file=resolved_input,
            output_dir=resolved_output,
            dataset_name=resolved_dataset,
            cores=cores,
            normalize=normalize,
            device=device,
            chunk_size=chunk_size,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
            gpu_available=gpu_available,
        )
        typer.echo(output_path)

    @curation_app.command("split")
    def curation_split(
        input_file: Annotated[Path, typer.Argument(help="Raw viability CSV.")],
        output_dir: Annotated[
            Path,
            typer.Option("--output-dir", "--output_dir", help="Directory for serialized work items."),
        ],
        dataset_name: Annotated[
            str | None,
            typer.Option(
                "--dataset-name", "--dataset_name", help="Dataset name. Defaults to input stem without '_raw'."
            ),
        ] = None,
        cores: Annotated[int, typer.Option("--cores", help="CPU worker threads for chunk sizing.")] = 1,
        normalize: Annotated[bool, typer.Option("--normalize", help="Normalize responses for CurveCurator.")] = False,
        device: Annotated[str, typer.Option("--device", help=CURATOR_DEVICE_HELP)] = "auto",
        chunk_size: Annotated[
            int, typer.Option("--chunk-size", "--chunk_size", help="Maximum curves per CPU chunk.")
        ] = 1_000,
        gpu_min_curves: Annotated[
            int,
            typer.Option("--gpu-min-curves", "--gpu_min_curves", help="Minimum curves before auto GPU."),
        ] = 1_000,
        gpu_chunk_size: Annotated[
            int,
            typer.Option("--gpu-chunk-size", "--gpu_chunk_size", help="Maximum curves per GPU chunk."),
        ] = 50_000,
        gpu_available: Annotated[
            bool,
            typer.Option(
                "--gpu-available/--no-gpu-available",
                help="Whether GPU resources are available for accelerator chunking.",
            ),
        ] = False,
    ) -> None:
        """Prepare CurveCurator work items and write a curation manifest."""
        resolved_input = input_file.expanduser().resolve()
        resolved_output = output_dir.expanduser().resolve()
        resolved_dataset = dataset_name or _dataset_name_from_input(resolved_input)
        manifest_path = split_to_disk(
            input_file=resolved_input,
            output_dir=resolved_output,
            dataset_name=resolved_dataset,
            cores=cores,
            normalize=normalize,
            device=device,
            chunk_size=chunk_size,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
            gpu_available=gpu_available,
        )
        typer.echo(manifest_path)

    @curation_app.command("curvecurator")
    def curation_curvecurator(
        config_file: Annotated[
            Path,
            typer.Argument(help="Job config JSON from ``curation split`` (``<job_id>_config.json``)."),
        ],
        input_file: Annotated[
            Path,
            typer.Argument(help="Prepared input parquet from ``curation split`` (``<job_id>_input.parquet``)."),
        ],
        output_file: Annotated[
            Path,
            typer.Argument(help="Destination parquet for fitted CurveCurator curves."),
        ],
        device: Annotated[str, typer.Option("--device", help=CURATOR_DEVICE_HELP)] = "auto",
        gpu_min_curves: Annotated[
            int,
            typer.Option("--gpu-min-curves", "--gpu_min_curves", help="Minimum curves before auto GPU."),
        ] = 1_000,
        gpu_chunk_size: Annotated[
            int,
            typer.Option("--gpu-chunk-size", "--gpu_chunk_size", help="Maximum curves per GPU chunk."),
        ] = 50_000,
    ) -> None:
        """Run CurveCurator for one prepared curation job."""
        curves_path = curvecurator_to_disk(
            config_file.expanduser().resolve(),
            input_file.expanduser().resolve(),
            output_file.expanduser().resolve(),
            device=device,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
        )
        typer.echo(curves_path)

    @curation_app.command("combine")
    def curation_combine(
        manifest: Annotated[Path, typer.Argument(help="Curation manifest written by ``curation split``.")],
        output_file: Annotated[
            Path | None,
            typer.Option(
                "--output-file",
                "--output_file",
                help="Destination CSV. Defaults to manifest directory/<dataset_name>.csv.",
            ),
        ] = None,
    ) -> None:
        """Combine fitted CurveCurator results into one dataset CSV."""
        output_path = combine_manifest_to_csv(
            manifest.expanduser().resolve(),
            output_file=output_file.expanduser().resolve() if output_file is not None else None,
        )
        typer.echo(output_path)


def _dataset_name_from_input(input_file: Path) -> str:
    return input_file.stem.removesuffix("_raw")
