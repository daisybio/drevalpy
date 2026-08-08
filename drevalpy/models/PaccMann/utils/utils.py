"""Utility functions."""

import torch
import torch.nn as nn


def get_device():
    """Return the active torch device.

    :return: torch.device("cuda") if cuda is available otherwise torch.device("cpu")
    """
    return torch.device("cuda" if cuda() else "cpu")


def cuda():
    """Check whether cuda is available.

    :return: True if cuda is available otherwise False.
    """
    return torch.cuda.is_available()


def get_log_molar(y, ic50_max=None, ic50_min=None):
    """Converts PaccMann predictions from [0,1] to log(micromolar) range.

    :param y: predicted values in the normalized range
    :param ic50_max: maximum IC50 value used for scaling
    :param ic50_min: minimum IC50 value used for scaling
    :return: predictions transformed to the log-micromolar range
    """
    return y * (ic50_max - ic50_min) + ic50_min


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        """Squeeze the last dimension of the input tensor.

        :param data: input tensor
        :return: squeezed tensor
        """
        return torch.squeeze(data, -1)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        """Initialize the temperature wrapper.

        :param temperature: Temperature value used for scaling.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, data):
        """Scale the input tensor by the temperature value.

        :param data: input tensor
        :return: scaled tensor
        """
        return data / self.temperature
