"""``drevalpy curate`` command group: dose-response curve curation.

``curate run`` reads a well-level dose-response table and writes the AnnData a full curation
produces - see :mod:`drevalpy.curation`. The ``.h5ad`` is the only output: because
:func:`drevalpy.curation.curate` keys ``obs_names``/``var_names`` from the ``sample``/``drug``
columns it was given, a pipeline can curate on native identifiers and remap the indices in a
later, cheap stage.

``plan``, ``fit-chunk`` and ``collect`` run the same curation as separate steps, so a workflow
manager can fit the chunks as parallel jobs on a cluster (:mod:`drevalpy.curation._plan`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

import typer

from drevalpy.curation import DEFAULT_FIT_SPEED, FIT_SPEEDS

if TYPE_CHECKING:
    import pandas as pd

_TABLE_SUFFIXES = (".parquet", ".pq")

FitSpeed = StrEnum("FitSpeed", {speed.upper(): speed for speed in FIT_SPEEDS}, module=__name__)
FitSpeed.__doc__ = "Values of ``--fit-speed``, one per level in :data:`drevalpy.curation.FIT_SPEEDS`."
_DEFAULT_FIT_SPEED = FitSpeed(DEFAULT_FIT_SPEED)

curate_app = typer.Typer(
    name="curate",
    help="Fit dose-response curves, in one process or as separate steps.",
    no_args_is_help=True,
)

_Input = Annotated[str, typer.Argument(help="Path to a CSV or Parquet file with well-level dose-response data.")]
_Normalize = Annotated[bool, typer.Option("--normalize", help="Apply normalization before fitting.")]
_FitSpeedOption = Annotated[FitSpeed, typer.Option("--fit-speed", case_sensitive=False, help="Fitting thoroughness.")]
_ReplicateFits = Annotated[
    bool, typer.Option("--replicate-fits", help='Also fit every replicate on its own, into uns["replicate_fits"].')
]
_MinDoses = Annotated[
    int, typer.Option("--min-doses", min=1, help="Minimum number of distinct doses for a series to be fitted.")
]
_Metadata = Annotated[
    list[str] | None,
    typer.Option("--metadata", help="Column to keep per series in the replicate-fits table; repeatable."),
]
_MaxChunkSize = Annotated[
    int, typer.Option("--max-chunk-size", min=1, help="Maximum number of curves fitted in one chunk.")
]


def _read_frame(input_path: str) -> pd.DataFrame:
    """Read a long-form dose-response table from CSV or Parquet.

    :param input_path: Path to a ``.csv``, ``.parquet`` or ``.pq`` file.
    :returns: The parsed frame.
    :raises typer.BadParameter: If the suffix is not a supported format.
    """
    import pandas as pd
    from upath import UPath

    path = UPath(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in _TABLE_SUFFIXES:
        return pd.read_parquet(path)
    msg = f"Unsupported file format: {suffix}. Use .csv or .parquet."
    raise typer.BadParameter(msg)


@curate_app.command("run")
def run_cmd(
    input_path: _Input,
    output: Annotated[str, typer.Argument(help="Output .h5ad file path.")],
    cores: Annotated[int, typer.Option("--cores", "-c", help="Number of CPU cores.")] = 4,
    normalize: _Normalize = False,
    fit_speed: _FitSpeedOption = _DEFAULT_FIT_SPEED,
    replicate_fits: _ReplicateFits = False,
    min_doses: _MinDoses = 4,
    metadata: _Metadata = None,
    max_chunk_size: _MaxChunkSize = 2000,
) -> None:
    """Fit dose-response curves from a CSV/Parquet file and write AnnData to .h5ad."""
    from upath import UPath

    from drevalpy.curation import curate

    df = _read_frame(input_path)
    adata = curate(
        df,
        replicate_fits=replicate_fits,
        min_doses=min_doses,
        metadata=tuple(metadata or ()),
        normalize=normalize,
        fit_speed=fit_speed.value,
        max_chunk_size=max_chunk_size,
        max_workers=cores,
    )
    adata.write_h5ad(UPath(output))


@curate_app.command("plan")
def plan_cmd(
    input_path: _Input,
    plan_dir: Annotated[str, typer.Argument(help="Directory to write the plan and its chunks/ to.")],
    normalize: _Normalize = False,
    fit_speed: _FitSpeedOption = _DEFAULT_FIT_SPEED,
    replicate_fits: _ReplicateFits = False,
    min_doses: _MinDoses = 4,
    metadata: _Metadata = None,
    max_chunk_size: _MaxChunkSize = 2000,
) -> None:
    """Split a curation into chunks of curves, written as PLAN_DIR/chunks/<name>.parquet."""
    from upath import UPath

    from drevalpy.curation import _plan_curation
    from drevalpy.curation._plan import write_plan

    work = _plan_curation(
        _read_frame(input_path),
        replicate_fits=replicate_fits,
        min_doses=min_doses,
        metadata=tuple(metadata or ()),
        normalize=normalize,
        fit_speed=fit_speed.value,
        max_chunk_size=max_chunk_size,
    )
    write_plan(work, UPath(plan_dir))


@curate_app.command("fit-chunk")
def fit_chunk_cmd(
    chunk: Annotated[str, typer.Argument(help="A chunk file written by `drevalpy curate plan`.")],
    output: Annotated[str, typer.Argument(help="Output Parquet file with the chunk's fitted curves.")],
) -> None:
    """Fit the curves of one chunk, on one core."""
    from upath import UPath

    from drevalpy.curation._plan import read_chunk, write_fitted

    write_fitted(read_chunk(UPath(chunk)).run(), UPath(output))


@curate_app.command("collect")
def collect_cmd(
    plan_dir: Annotated[str, typer.Argument(help="Directory written by `drevalpy curate plan`.")],
    fitted_dir: Annotated[
        str, typer.Argument(help="Directory holding `<chunk name>.parquet` from `fit-chunk` for every chunk.")
    ],
    output: Annotated[str, typer.Argument(help="Output .h5ad file path.")],
) -> None:
    """Assemble the fitted chunks of a plan into the curve AnnData."""
    from upath import UPath

    from drevalpy.curation._plan import assemble, read_fitted, read_plan
    from drevalpy.errors import InvalidInputError

    work = read_plan(UPath(plan_dir))
    fitted_root = UPath(fitted_dir)
    missing = [chunk.name for chunk in work.chunks if not (fitted_root / f"{chunk.name}.parquet").exists()]
    if missing:
        raise InvalidInputError(f"{len(missing)} chunk(s) have no fit in {fitted_dir}: {missing[:5]}")
    fitted = {chunk.name: read_fitted(fitted_root / f"{chunk.name}.parquet") for chunk in work.chunks}
    assemble(work, fitted).write_h5ad(UPath(output))
