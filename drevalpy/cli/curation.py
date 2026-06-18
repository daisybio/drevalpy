"""``drevalpy curation`` command group."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from typer import _click

from drevalpy.curation import combine, curate, curvecurator, load_raw_curve_df, split, write_dataset_csv
from drevalpy.curation._curvecurator.io import (
    read_fit_results_from_paths,
    read_work_item,
    resolve_curve_paths,
    write_fit_curves,
    write_split_artifacts,
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
        output_file: Annotated[
            Path | None,
            typer.Option(
                "--output-file",
                "--output_file",
                help="Destination CSV for the curated dataset.",
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
        if input_file is None or output_file is None:
            raise typer.BadParameter("Top-level curation requires --input-file and --output-file.")
        resolved_input = input_file.expanduser().resolve()
        resolved_output = output_file.expanduser().resolve()
        raw_df = load_raw_curve_df(resolved_input)
        dataset = curate(
            raw_df,
            input_filename=resolved_input.name,
            cores=cores,
            normalize=normalize,
            device=device,
            chunk_size=chunk_size,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
            gpu_available=gpu_available,
        )
        output_path = write_dataset_csv(dataset, resolved_output)
        typer.echo(output_path)

    @curation_app.command("split")
    def curation_split(
        input_file: Annotated[Path, typer.Argument(help="Raw viability CSV.")],
        output_dir: Annotated[
            Path,
            typer.Option("--output-dir", "--output_dir", help="Directory for serialized work items."),
        ],
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
        """Prepare CurveCurator work items and write serialized artifacts."""
        resolved_input = input_file.expanduser().resolve()
        resolved_output = output_dir.expanduser().resolve()
        raw_df = load_raw_curve_df(resolved_input)
        split_result = split(
            raw_df,
            input_filename=resolved_input.name,
            cores=cores,
            normalize=normalize,
            device=device,
            chunk_size=chunk_size,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
            gpu_available=gpu_available,
        )
        artifact_dir = write_split_artifacts(split_result, resolved_output)
        typer.echo(artifact_dir)

    @curation_app.command("curvecurator")
    def curation_curvecurator(
        config_file: Annotated[
            Path,
            typer.Argument(help="Job config JSON from ``curation split`` (``<job_id>.json``)."),
        ],
        input_file: Annotated[
            Path,
            typer.Argument(help="Prepared input parquet from ``curation split`` (``<job_id>_input.parquet``)."),
        ],
        output_file: Annotated[
            Path,
            typer.Argument(help="Destination parquet for fitted CurveCurator curves (``<job_id>.parquet``)."),
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
        work_item = read_work_item(
            config_file.expanduser().resolve(),
            input_path=input_file.expanduser().resolve(),
        )
        fit_result = curvecurator(
            work_item,
            device=device,
            gpu_min_curves=gpu_min_curves,
            gpu_chunk_size=gpu_chunk_size,
        )
        curves_path = write_fit_curves(fit_result.curves, output_file.expanduser().resolve())
        typer.echo(curves_path)

    @curation_app.command("combine")
    def curation_combine(
        curve_files: Annotated[
            list[Path],
            typer.Argument(help="Fitted curve parquet file(s) or a directory containing them."),
        ],
        output_file: Annotated[
            Path,
            typer.Option(
                "--output-file",
                "--output_file",
                help="Destination CSV for the curated dataset.",
            ),
        ],
    ) -> None:
        """Combine fitted CurveCurator results into one dataset CSV."""
        resolved_output = output_file.expanduser().resolve()
        curve_paths = resolve_curve_paths(curve_files)
        fit_results = read_fit_results_from_paths(curve_paths)
        dataset = combine(fit_results)
        output_path = write_dataset_csv(dataset, resolved_output)
        typer.echo(output_path)
