"""``drevalpy data synthetic`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.data._synthetic import DATASET_NAME, MAX_SYNTHETIC_DRUGS, N_DRUGS, N_SAMPLES, N_TISSUES, SEED


def synthetic_dataset(
    output: Annotated[str, typer.Option("--output", "-o", help="Output .h5mu file path.")],
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = SEED,
    n_samples: Annotated[int, typer.Option("--n-samples", help="Number of samples.")] = N_SAMPLES,
    n_drugs: Annotated[
        int, typer.Option("--n-drugs", help=f"Number of drugs, at most {MAX_SYNTHETIC_DRUGS}.")
    ] = N_DRUGS,
    n_tissues: Annotated[int, typer.Option("--n-tissues", help="Number of distinct tissues.")] = N_TISSUES,
    name: Annotated[
        str, typer.Option("--name", help="Dataset name; give each file its own to concatenate them.")
    ] = DATASET_NAME,
) -> None:
    """Write a small simulated dataset to an .h5mu file.

    The responses mimic a real screen: drugs differ strongly in average potency,
    and hidden sample factors drive both the responses and the gene expression
    features. The file needs no download, so it suits tutorials and offline
    experiments.
    """
    from upath import UPath

    from drevalpy.data import synthetic

    try:
        dataset = synthetic(name=name, seed=seed, n_samples=n_samples, n_drugs=n_drugs, n_tissues=n_tissues)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--n-drugs") from error

    path = UPath(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(path))
    typer.echo(f"Wrote {dataset.name} to {path}")
