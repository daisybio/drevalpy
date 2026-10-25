"""Two-tower matrix-factorization network of the EnsembleMF model.

The sample and drug latent factors are the outputs of small residual MLPs over their
features. On top of the dot product sit per-sample, per-drug and global bias terms, an
optional free per-drug id embedding and an optional interaction head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ResidualEncoder(nn.Module):
    """Project features to a latent factor through residual blocks, with a direct input skip."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, n_layers: int, dropout: float) -> None:
        """Build the encoder.

        :param in_dim: Width of the input features.
        :param hidden_dim: Width of the residual blocks.
        :param emb_dim: Width of the latent factor.
        :param n_layers: Number of residual blocks.
        :param dropout: Dropout probability inside the blocks.
        """
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.skip_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode entity features into latent factors.

        :param x: ``(n_entities, in_dim)`` feature matrix.
        :returns: ``(n_entities, emb_dim)`` latent factors.
        """
        h0 = self.act(self.input_proj(x))
        h = self.dropout(h0)
        for block, norm in zip(self.blocks, self.norms, strict=True):
            h = self.dropout(self.act(norm(block(h)))) + h
        return self.output_proj(h) + self.skip_proj(h0)


@dataclass(frozen=True)
class MFNetConfig:
    """Widths and switches of one ensemble member."""

    sample_in_dim: int
    drug_in_dim: int
    n_drugs: int
    hidden_dim: int = 256
    emb_dim: int = 128
    n_layers: int = 1
    dropout: float = 0.2
    use_drug_id_embedding: bool = True
    use_mlp_head: bool = True
    mlp_hidden: int = 256


class MFNet(nn.Module):
    """One ensemble member: two encoders plus a bilinear head with main effects."""

    def __init__(self, config: MFNetConfig) -> None:
        """Build the member.

        :param config: Dimensions and switches of the network.
        """
        super().__init__()
        emb_dim = config.emb_dim
        self.sample_encoder = ResidualEncoder(
            config.sample_in_dim, config.hidden_dim, emb_dim, config.n_layers, config.dropout
        )
        self.drug_encoder = ResidualEncoder(
            config.drug_in_dim, config.hidden_dim, emb_dim, config.n_layers, config.dropout
        )

        self.use_drug_id_embedding = config.use_drug_id_embedding
        if config.use_drug_id_embedding:
            self.drug_id_emb = nn.Embedding(config.n_drugs, emb_dim)
            nn.init.normal_(self.drug_id_emb.weight, std=0.01)

        self.sample_bias = nn.Linear(emb_dim, 1)
        self.drug_bias = nn.Linear(emb_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dot_scale = nn.Parameter(torch.ones(1))

        self.use_mlp_head = config.use_mlp_head
        if config.use_mlp_head:
            self.mlp = nn.Sequential(
                nn.Linear(3 * emb_dim, config.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.mlp_hidden, 1),
            )

    def encode(
        self, x_sample: torch.Tensor, x_drug: torch.Tensor, drug_id_emb_rows: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent factors for every sample and every drug row.

        :param x_sample: ``(n_samples, sample_in_dim)`` sample features.
        :param x_drug: ``(n_drugs, drug_in_dim)`` drug features.
        :param drug_id_emb_rows: ``(n_drugs, emb_dim)`` id-embedding row to add per drug, aligned with the
            rows of *x_drug*; defaults to this network's own table, which is only correct when *x_drug* uses
            the training-time drug order. A caller encoding another drug set gathers the rows itself and
            passes zeros for drugs that never appeared in a training batch.
        :returns: Sample factors and drug factors.
        """
        z_sample = self.sample_encoder(x_sample)
        z_drug = self.drug_encoder(x_drug)
        if self.use_drug_id_embedding:
            z_drug = z_drug + (self.drug_id_emb.weight if drug_id_emb_rows is None else drug_id_emb_rows)
        return z_sample, z_drug

    def score_pairs(self, z_sample: torch.Tensor, z_drug: torch.Tensor) -> torch.Tensor:
        """Score a batch of sample-drug pairs from their gathered factors.

        :param z_sample: ``(batch, emb_dim)`` sample factors.
        :param z_drug: ``(batch, emb_dim)`` drug factors.
        :returns: ``(batch,)`` predicted responses.
        """
        pred = (z_sample * z_drug).sum(dim=-1, keepdim=True) * self.dot_scale
        pred = pred + self.sample_bias(z_sample) + self.drug_bias(z_drug) + self.global_bias
        if self.use_mlp_head:
            pred = pred + self.mlp(torch.cat([z_sample, z_drug, z_sample * z_drug], dim=-1))
        return pred.squeeze(-1)
