"""CLI command for dose-response curve curation."""

from __future__ import annotations

from typing import Annotated

import typer


def curate_cmd(
    input_path: Annotated[
        str, typer.Argument(help="Path to a CSV or Parquet file with dose-response data.")
    ],
    output: Annotated[str, typer.Argument(help="Output .h5ad file path.")],
    cores: Annotated[int, typer.Option("--cores", "-c", help="Number of CPU cores.")] = 4,
    normalize: Annotated[
        bool, typer.Option("--normalize", help="Apply normalization before fitting.")
    ] = False,
    fit_type: Annotated[str, typer.Option("--fit-type", help="OLS or MLE.")] = "OLS",
    fit_speed: Annotated[
        str, typer.Option("--fit-speed", help="fast/standard/exhaustive/basinhopping.")
    ] = "exhaustive",
) -> None:
    """Fit dose-response curves from a CSV/Parquet file and write AnnData to .h5ad."""
    import pandas as pd
    from upath import UPath

    from drevalpy.curation import curate

    path = UPath(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        msg = f"Unsupported file format: {suffix}. Use .csv or .parquet."
        raise typer.BadParameter(msg)

    adata = curate(df, cores=cores, normalize=normalize, fit_type=fit_type, fit_speed=fit_speed)
    adata.write_h5ad(UPath(output))
