"""PaccMann network: context attention between SMILES convolutions and gene expression.

Reimplementation of the MCA architecture from Manica et al., Molecular Pharmaceutics 2019
(https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520), as released at
https://github.com/PaccMann/paccmann_predictor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class PaccMannConfig:
    """Resolved, typed configuration for PaccMannNetwork.

    smiles_vocabulary_size, smiles_padding_length, and number_of_genes are derived from the training data;
    every other field is a model hyperparameter, see PaccMann/hyperparameters.yaml for their defaults.
    """

    smiles_vocabulary_size: int
    smiles_embedding_size: int
    smiles_padding_length: int
    number_of_genes: int
    molecule_heads: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    gene_heads: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    filters: list[int] = field(default_factory=lambda: [64, 64, 64])
    kernel_sizes: list[tuple[int, int]] | None = None
    dropout: float = 0.5
    batch_norm: bool = False
    smiles_attention_size: int = 64
    gene_attention_size: int = 1
    molecule_temperature: float = 1.0
    gene_temperature: float = 1.0
    stacked_dense_hidden_sizes: list[int] = field(default_factory=lambda: [1024, 512])

    @property
    def resolved_kernel_sizes(self) -> list[tuple[int, int]]:
        """Kernel sizes, defaulting to token windows of 3, 5, and 11 sized to the embedding dimension.

        :return: one (token window, embedding dimension) pair per convolution
        """
        if self.kernel_sizes is not None:
            return self.kernel_sizes
        return [(window, self.smiles_embedding_size) for window in (3, 5, 11)]

    def __post_init__(self) -> None:
        """Validate that head counts, filters, and kernel sizes line up.

        :raises ValueError: if the molecule/gene head counts, filters, or kernel sizes are inconsistent
        """
        if len(self.gene_heads) != len(self.molecule_heads):
            raise ValueError("gene_heads and molecule_heads must have the same length.")
        if len(self.filters) != len(self.resolved_kernel_sizes):
            raise ValueError("filters and kernel_sizes must have the same length.")
        if len(self.filters) + 1 != len(self.molecule_heads):
            raise ValueError("molecule_heads must have exactly one more entry than filters.")

    @classmethod
    def from_hyperparameters(cls, params: dict[str, Any]) -> PaccMannConfig:
        """Resolve a config from a raw hyperparameter dictionary.

        :param params: hyperparameters, see PaccMann/hyperparameters.yaml for the available keys, plus
            smiles_vocabulary_size, smiles_padding_length, and number_of_genes, which are derived from the
            training data rather than configured
        :return: resolved config
        """
        field_names = {f for f in cls.__dataclass_fields__ if f in params}
        return cls(**{name: params[name] for name in field_names})


class ContextAttentionLayer(nn.Module):
    """Context attention layer: lets one modality attend over another (PaccMann paper, Fig. 2C)."""

    def __init__(
        self,
        reference_hidden_size: int,
        reference_sequence_length: int,
        context_hidden_size: int,
        context_sequence_length: int,
        attention_size: int,
        temperature: float,
    ) -> None:
        """Initialize the context attention layer.

        :param reference_hidden_size: hidden size of the reference input
        :param reference_sequence_length: sequence length of the reference input
        :param context_hidden_size: hidden size (or feature count) of the context input
        :param context_sequence_length: sequence length of the context input
        :param attention_size: size of the shared attention space
        :param temperature: softmax temperature; below 1 sharpens the attention, above 1 smooths it
        """
        super().__init__()
        self.reference_projection = nn.Linear(reference_hidden_size, attention_size)
        self.context_projection = nn.Linear(context_hidden_size, attention_size)
        self.context_hidden_projection = (
            nn.Linear(context_sequence_length, reference_sequence_length)
            if context_sequence_length > 1
            else nn.Identity()
        )
        self.alpha_projection = nn.Linear(attention_size, 1, bias=False)
        self.temperature = temperature

    def forward(
        self, reference: torch.Tensor, context: torch.Tensor, average_seq: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend over ``reference`` using ``context``.

        :param reference: tensor of shape (batch, reference_sequence_length, reference_hidden_size)
        :param context: tensor of shape (batch, context_sequence_length, context_hidden_size)
        :param average_seq: sum the attended reference over its sequence dimension
        :return: attended output and attention weights
        """
        reference_attention = self.reference_projection(reference)
        context_attention = self.context_hidden_projection(self.context_projection(context).permute(0, 2, 1)).permute(
            0, 2, 1
        )
        alphas = self.alpha_projection(torch.tanh(reference_attention + context_attention)).squeeze(-1)
        alphas = torch.softmax(alphas / self.temperature, dim=1)

        output = reference * alphas.unsqueeze(-1)
        # Squeeze only the trailing dimension: a batch of size 1 must keep its batch dimension.
        return (output.sum(dim=1) if average_seq else output.squeeze(-1)), alphas


class _ConvolutionBlock(nn.Module):
    """Convolution over embedded SMILES tokens, followed by activation, dropout, and batch norm."""

    def __init__(self, num_kernel: int, kernel_size: tuple[int, int], dropout: float, batch_norm: bool) -> None:
        """Initialize the convolution block.

        :param num_kernel: number of convolution kernels, i.e. output channels
        :param kernel_size: (token window, embedding dimension) size of the convolution kernel
        :param dropout: dropout probability
        :param batch_norm: whether to apply batch normalization
        """
        super().__init__()
        self.convolve = nn.Conv2d(1, num_kernel, kernel_size, padding=(kernel_size[0] // 2, 0))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(num_kernel) if batch_norm else nn.Identity()

    def forward(self, embedded_smiles: torch.Tensor) -> torch.Tensor:
        """Convolve embedded SMILES tokens.

        :param embedded_smiles: tensor of shape (batch, 1, smiles_padding_length, embedding_size)
        :return: tensor of shape (batch, num_kernel, smiles_padding_length)
        """
        activated = torch.relu(self.convolve(embedded_smiles).squeeze(-1))
        return self.batch_norm(self.dropout(activated))


def _dense_block(input_size: int, hidden_size: int, dropout: float, batch_norm: bool) -> nn.Sequential:
    """Build a linear layer followed by batch norm, ReLU, and dropout.

    :param input_size: input feature size
    :param hidden_size: output feature size
    :param dropout: dropout probability
    :param batch_norm: whether to apply batch normalization
    :return: sequential dense block
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class PaccMannNetwork(nn.Module):
    """PaccMann drug-response network.

    SMILES tokens are embedded and convolved at several kernel sizes; gene expression attends over each
    resulting representation and vice versa; the concatenated attention outputs are passed through dense
    layers down to a single drug-response prediction.
    """

    def __init__(self, config: PaccMannConfig) -> None:
        """Build the network from a resolved config.

        :param config: resolved network configuration
        """
        super().__init__()

        self.smiles_embedding = nn.Embedding(config.smiles_vocabulary_size, config.smiles_embedding_size)
        self.convolutions = nn.ModuleList(
            [
                _ConvolutionBlock(num_kernel, kernel_size, config.dropout, config.batch_norm)
                for num_kernel, kernel_size in zip(config.filters, config.resolved_kernel_sizes)
            ]
        )

        # Flat lists of attention heads, grouped back into per-layer chunks in forward() via self._molecule_heads
        # and self._gene_heads: nn.ModuleList cannot be nested and stay iterable under mypy's torch stubs.
        smiles_hidden_sizes = [config.smiles_embedding_size] + config.filters
        self._molecule_heads = config.molecule_heads
        self.molecule_attentions = nn.ModuleList(
            [
                ContextAttentionLayer(
                    reference_hidden_size=smiles_hidden_sizes[layer],
                    reference_sequence_length=config.smiles_padding_length,
                    context_hidden_size=1,
                    context_sequence_length=config.number_of_genes,
                    attention_size=config.smiles_attention_size,
                    temperature=config.molecule_temperature,
                )
                for layer, heads in enumerate(config.molecule_heads)
                for _ in range(heads)
            ]
        )
        self._gene_heads = config.gene_heads
        self.gene_attentions = nn.ModuleList(
            [
                ContextAttentionLayer(
                    reference_hidden_size=1,
                    reference_sequence_length=config.number_of_genes,
                    context_hidden_size=smiles_hidden_sizes[layer],
                    context_sequence_length=config.smiles_padding_length,
                    attention_size=config.gene_attention_size,
                    temperature=config.gene_temperature,
                )
                for layer, heads in enumerate(config.gene_heads)
                for _ in range(heads)
            ]
        )

        attention_output_size = (
            config.molecule_heads[0] * config.smiles_embedding_size
            + sum(heads * num_filters for heads, num_filters in zip(config.molecule_heads[1:], config.filters))
            + sum(config.gene_heads) * config.number_of_genes
        )
        hidden_sizes = [attention_output_size, *config.stacked_dense_hidden_sizes]

        self.batch_norm = nn.BatchNorm1d(hidden_sizes[0]) if config.batch_norm else nn.Identity()
        self.dense_layers = nn.ModuleList(
            [
                _dense_block(hidden_sizes[i], hidden_sizes[i + 1], config.dropout, config.batch_norm)
                for i in range(len(hidden_sizes) - 1)
            ]
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, smiles: torch.Tensor, gene_expression: torch.Tensor) -> torch.Tensor:
        """Predict a drug-response score from tokenized SMILES and gene expression.

        :param smiles: token ids of shape (batch, smiles_padding_length)
        :param gene_expression: gene expression of shape (batch, number_of_genes)
        :return: predicted response of shape (batch, 1)
        """
        gene_expression = gene_expression.unsqueeze(-1)
        embedded_smiles = self.smiles_embedding(smiles)
        smiles_encodings = [embedded_smiles] + [
            conv(embedded_smiles.unsqueeze(1)).permute(0, 2, 1) for conv in self.convolutions
        ]

        attended = []
        molecule_attentions = iter(self.molecule_attentions)
        for heads, encoding in zip(self._molecule_heads, smiles_encodings):
            attended += [next(molecule_attentions)(encoding, gene_expression)[0] for _ in range(heads)]
        gene_attentions = iter(self.gene_attentions)
        for heads, encoding in zip(self._gene_heads, smiles_encodings):
            attended += [next(gene_attentions)(gene_expression, encoding, average_seq=False)[0] for _ in range(heads)]

        hidden = self.batch_norm(torch.cat(attended, dim=1))
        for dense in self.dense_layers:
            hidden = dense(hidden)
        return self.output_layer(hidden)
