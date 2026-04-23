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


def to_np(x):
    """Convert a tensor to a NumPy array.

    :param x: Input tensor
    :return: Tensor converted to a NumPy array on the CPU
    """
    return x.data.cpu().numpy()


def attention_list_to_matrix(coding_tuple, dim=2):
    """Convert a list of attention outputs to attention matrices.

    :param coding_tuple: iterable of (outputs, att_weights) tuples coming from the attention function
    :param dim: The dimension along which expansion takes place to concatenate the attention weights.
        Defaults to 2.
    :return: Tuple (raw_coeff, coeff) where 'raw_coeff' contains all
        attention weights concatenated along 'dim' and 'coeff' contains
        the averaged attention weights.
    """
    raw_coeff = torch.cat([torch.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim)
    return raw_coeff, torch.mean(raw_coeff, dim=dim)


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


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, dim):
        """Initialize the unsqueeze wrapper.

        :param dim: dimension at which to insert the new axis
        """
        super().__init__()
        self.dim = dim

    def forward(self, data):
        """Unsqueeze the input tensor at the configured dimension.

        :param data: input tensor
        :return: tensor with added dimension
        """
        return torch.unsqueeze(data, self.dim)


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
