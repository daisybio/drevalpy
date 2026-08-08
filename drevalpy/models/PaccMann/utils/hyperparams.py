"""Customizable model hyperparameters."""

import torch.nn as nn

from drevalpy.models.PaccMann.utils.loss_functions import (
    correlation_coefficient_loss,
    mse_cc_loss,
)

LOSS_FN_FACTORY = {
    "mse": nn.MSELoss(),
    "l1": nn.L1Loss(),
    "mse_and_pearson": mse_cc_loss,
    "pearson": correlation_coefficient_loss,
    "binary_cross_entropy": nn.BCELoss(),
}

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
}
