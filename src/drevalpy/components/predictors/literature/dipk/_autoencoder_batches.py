"""Batch-size-dependent helpers of the DIPK gene-expression autoencoder trainer.

Split from :mod:`~drevalpy.components.predictors.literature.dipk.gene_expression_encoder` to keep
that module under the size gate. :func:`avoid_singleton_batch` is for every caller of the trainer
that fits on unique samples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from drevalpy.components.predictors.literature.dipk.gene_expression_encoder import (
        GeneExpressionDecoder,
        GeneExpressionEncoder,
    )


def avoid_singleton_batch(train: np.ndarray, validation: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Move the last training row to the validation rows if it would form a mini-batch of one.

    The encoder uses ``BatchNorm1d``, which cannot normalise a single row while training, so a
    trailing mini-batch of exactly one row would abort the fit. Training on unique samples makes
    that case reachable.

    :param train: Training gene expression, one row per sample.
    :param validation: Early-stopping gene expression, one row per sample.
    :param batch_size: Mini-batch size the trainer will use.
    :returns: ``(train, validation)``, unchanged unless ``len(train) % batch_size == 1`` and more than one
        training row exists.
    """
    if len(train) > 1 and len(train) % batch_size == 1:
        return train[:-1], np.concatenate([validation, train[-1:]])
    return train, validation


def validation_loss(
    encoder: GeneExpressionEncoder,
    decoder: GeneExpressionDecoder,
    gene_expression_val_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute the reconstruction MSE of the validation matrix in chunks.

    The sum of squared errors and the element count are accumulated exactly, so the result equals
    ``nn.MSELoss()`` over the whole matrix while only one chunk lives on the device.

    :param encoder: Encoder in eval mode.
    :param decoder: Decoder in eval mode.
    :param gene_expression_val_tensor: Validation matrix, kept on the host.
    :param batch_size: Number of rows moved to the device at a time.
    :param device: Device the models live on.
    :returns: Mean squared reconstruction error, ``nan`` for an empty validation matrix.
    """
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for start in range(0, len(gene_expression_val_tensor), batch_size):
            stop = start + batch_size
            val_batch = gene_expression_val_tensor[start:stop].to(device)
            val_output = decoder(encoder(val_batch))
            squared_error += float(((val_output - val_batch) ** 2).sum().item())
            elements += val_batch.numel()
    if elements == 0:
        return float("nan")
    return squared_error / elements
