r"""Neural network components for the Precily model.

Exact port of the Keras architecture from Chawla et al. (Nat Commun 2022),

    Input(input_dim)
      -> Dense(1429) -> ReLU
      -> Dense(512)  -> ReLU -> Dropout(p)
      -> Dense(140)  -> ReLU -> Dropout(p)
      -> Dense(200)  -> ReLU -> Dropout(p)
      -> Dense(1)

input_dim = n_pathways (GSVA) + n_drug_features (Morgan/SMILESVec).
With Morgan fingerprints the drug dimension differs and input_dim
is set accordingly at build time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PrecilyNetwork(nn.Module):
    """Feed-forward regressor predicting LN(IC50) from pathway + drug features."""

    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Initialize the Precily network.

        :param input_dim: total feature dimension (pathways + drug features)
        :param dropout: dropout probability between hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1429),
            nn.ReLU(),
            nn.Linear(1429, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 140),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(140, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass.

        :param x: [batch, input_dim] feature tensor
        :return: [batch] predicted LN(IC50)
        """
        return self.net(x).squeeze(-1)
