"""Contains a custom MultiHeadAttentionLayer for the DIPK model."""

import torch
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    """Custom multi-head attention layer for the DIPK model."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str | torch.device | int | None):
        """
        Initialize the multi-head attention layer.

        :param hid_dim: dimension of hidden layer
        :param n_heads: number of heads
        :param dropout: dropout rate
        :param device: which device to use, e.g. "cuda" or "cpu"
        :raises ValueError: if hidden dimension is not divisible by the number of heads
        """
        super().__init__()

        # Ensure head dimension divides evenly
        if hid_dim % n_heads != 0:
            raise ValueError("Hidden dimension must be divisible by the number of heads.")

        # Define dimensions
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        # Define fully connected layers for Q, K, V, and output
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # Dropout and scaling factor
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-head attention layer.

        :param query: query tensor
        :param key: key tensor
        :param value: value tensor
        :param mask: mask tensor
        :returns: output tensor and attention tensor
        """
        batch_size = query.size(0)

        # Transform inputs
        transformed_query = self.fc_q(query)
        transformed_key = self.fc_k(key)
        transformed_value = self.fc_v(value)

        # Split into heads and transpose for multi-head processing
        transformed_query = transformed_query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        transformed_key = transformed_key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        transformed_value = transformed_value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(transformed_query, transformed_key.transpose(-2, -1)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(energy, dim=-1)

        # Apply attention weights
        x = torch.matmul(self.dropout(attention), transformed_value)

        # Concatenate heads and pass through output layer
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
