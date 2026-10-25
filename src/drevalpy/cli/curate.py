"""CLI command for dose-response curve curation.

``curate`` reads a long-form dose-response table and writes the AnnData a full
curation produces - see :mod:`drevalpy.curation`. The ``.h5ad`` is the only
output: because :func:`drevalpy.curation.curate` keys ``obs_names``/``var_names``
from the ``sample``/``drug`` columns it was given, a pipeline can curate on
native identifiers and remap the indices in a later, cheap stage.
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


def curate_cmd(
    input_path: Annotated[str, typer.Argument(help="Path to a CSV or Parquet file with dose-response data.")],
    output: Annotated[str, typer.Argument(help="Output .h5ad file path.")],
    cores: Annotated[int, typer.Option("--cores", "-c", help="Number of CPU cores.")] = 4,
    normalize: Annotated[bool, typer.Option("--normalize", help="Apply normalization before fitting.")] = False,
    fit_speed: Annotated[
        FitSpeed, typer.Option("--fit-speed", case_sensitive=False, help="Fitting thoroughness.")
    ] = _DEFAULT_FIT_SPEED,
) -> None:
    """Fit dose-response curves from a CSV/Parquet file and write AnnData to .h5ad."""
    from upath import UPath

    from drevalpy.curation import curate

    df = _read_frame(input_path)
    adata = curate(df, max_workers=cores, normalize=normalize, fit_speed=fit_speed.value)
    adata.write_h5ad(UPath(output))
