"""Custom layers implementation."""

from collections import OrderedDict

import torch
import torch.nn as nn

from .utils import Squeeze, Temperature


def dense_layer(
    input_size,
    hidden_size,
    act_fn=None,
    batch_norm=False,
    dropout=0.0,
):
    """Build a dense layer block.

    :param input_size: Input feature size
    :param hidden_size: Output feature size
    :param act_fn: Activation module
    :param batch_norm: whether batch normalization is applied
    :param dropout: Dropout probability
    :return: Sequential dense layer block
    """
    if act_fn is None:
        act_fn = nn.ReLU()

    return nn.Sequential(
        OrderedDict(
            [
                ("projection", nn.Linear(input_size, hidden_size)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
                ),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
            ]
        )
    )


def convolutional_layer(
    num_kernel,
    kernel_size,
    act_fn=None,
    batch_norm=False,
    dropout=0.0,
    input_channels=1,
):
    """Convolutional layer.

    :param num_kernel: number of convolution kernels
    :param kernel_size: size of the convolution kernels
    :param act_fn: activation module
    :param batch_norm: whether batch normalization is applied
    :param dropout: dropout probability
    :param input_channels: number of input channels
    :return: sequential convolutional layer block
    """
    if act_fn is None:
        act_fn = nn.ReLU()

    return nn.Sequential(
        OrderedDict(
            [
                (
                    "convolve",
                    torch.nn.Conv2d(
                        input_channels,  # channel_in
                        num_kernel,  # channel_out
                        kernel_size,  # kernel_size
                        padding=[kernel_size[0] // 2, 0],  # pad for valid conv.
                    ),
                ),
                ("squeeze", Squeeze()),
                ("act_fn", act_fn),
                ("dropout", nn.Dropout(p=dropout)),
                (
                    "batch_norm",
                    nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity(),
                ),
            ]
        )
    )


class ContextAttentionLayer(nn.Module):
    """Context attention layer used in the PaccMann architecture.

    It implements context attention as described in the PaccMann paper and
    supports an optional hidden size in the context representation.
    """

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int = 1,
        attention_size: int = 16,
        individual_nonlinearity=None,
        temperature: float = 1.0,
    ):
        """Initialize the context attention layer.

        :param reference_hidden_size: hidden size of the reference input
        :param reference_sequence_length: sequence length of the reference input
        :param context_hidden_size: hidden size or feature count of the context
        :param context_sequence_length: sequence length of the context
        :param attention_size: size of the attention space
        :param individual_nonlinearity: optional activation module applied to each projection
        :param temperature: temperature used for the softmax
        """
        super().__init__()

        if individual_nonlinearity is None:
            individual_nonlinearity = nn.Sequential()

        self.reference_sequence_length = reference_sequence_length
        self.reference_hidden_size = reference_hidden_size
        self.context_sequence_length = context_sequence_length
        self.context_hidden_size = context_hidden_size
        self.attention_size = attention_size
        self.individual_nonlinearity = individual_nonlinearity
        self.temperature = temperature

        # Project the reference into the attention space
        self.reference_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        "projection",
                        nn.Linear(reference_hidden_size, attention_size),
                    ),
                    ("act_fn", individual_nonlinearity),
                ]
            )
        )

        # Project the context into the attention space
        self.context_projection = nn.Sequential(
            OrderedDict(
                [
                    (
                        "projection",
                        nn.Linear(context_hidden_size, attention_size),
                    ),
                    ("act_fn", individual_nonlinearity),
                ]
            )
        )

        # Optionally reduce the hidden size in context
        if context_sequence_length > 1:
            self.context_hidden_projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "projection",
                            nn.Linear(
                                context_sequence_length,
                                reference_sequence_length,
                            ),
                        ),
                        ("act_fn", individual_nonlinearity),
                    ]
                )
            )
        else:
            self.context_hidden_projection = nn.Sequential()

        self.alpha_projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(attention_size, 1, bias=False)),
                    ("squeeze", Squeeze()),
                    ("temperature", Temperature(self.temperature)),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(
        self,
        reference: torch.Tensor,
        context: torch.Tensor,
        average_seq: bool = True,
    ):
        """Forward pass through a context attention layer.

        :param reference: reference tensor of shape 'bs x ref_seq_length x ref_hidden_size'
        :param context: context tensor of shape 'bs x context_seq_length x context_hidden_size'
        :param average_seq: whether to average over the sequence length
        :return: Tuple (output, attention_weights)
        :raises ValueError: If reference or context is not 3-dimensional.
        """
        if len(reference.shape) != 3:
            raise ValueError("Reference tensor needs to be 3D")

        if len(context.shape) != 3:
            raise ValueError("Context tensor needs to be 3D")

        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(self.context_projection(context).permute(0, 2, 1)).permute(
            0, 2, 1
        )
        alphas = self.alpha_projection(torch.tanh(reference_attention + context_attention))

        output = reference * torch.unsqueeze(alphas, -1)
        # Squeeze only the last dimension. A bare torch.squeeze would also drop the batch dimension
        # for a batch of size 1, which breaks the concatenation of the encodings further downstream.
        output = torch.sum(output, 1) if average_seq else torch.squeeze(output, -1)

        return output, alphas
