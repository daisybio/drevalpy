"""``drevalpy predict`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.errors import InputNotFoundError

from .train import MODEL_FILE


def _checkpoint_path(model_dir: UPath) -> UPath:
    """Locate the checkpoint of a model directory written by ``drevalpy train``.

    :param model_dir: The model directory, or a checkpoint file itself.
    :returns: Path of the checkpoint archive.
    :raises InputNotFoundError: If *model_dir* does not exist or holds no checkpoint.
    """
    if not model_dir.exists():
        raise InputNotFoundError(f"Model directory not found: {model_dir}")
    if model_dir.is_file():
        return model_dir
    checkpoint = model_dir / MODEL_FILE
    if not checkpoint.is_file():
        raise InputNotFoundError(f"No {MODEL_FILE} in {model_dir}; is it a directory written by 'drevalpy train'?")
    return checkpoint


def predict_cmd(
    model_dir: Annotated[str, typer.Argument(help="Model directory written by 'drevalpy train'.")],
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="Dataset name or .h5mu path; every observed pair is predicted.")
    ],
    output: Annotated[str, typer.Option("--output", "-o", help="Output path for the predictions .npz file.")],
) -> None:
    """Predict every observed pair of a dataset with a saved model."""
    from drevalpy._predict import predict_observed
    from drevalpy.data import load
    from drevalpy.models import DRPModel

    checkpoint = _checkpoint_path(UPath(model_dir))
    data = load(dataset)
    result = predict_observed(DRPModel.load(checkpoint), data)

    out = UPath(output if output.endswith(".npz") else f"{output}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    typer.echo(f"Predictions: {result.model_name} on {result.dataset_name}, {len(result.predictions)} pairs -> {out}")
