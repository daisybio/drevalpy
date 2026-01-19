"""Neural network components for PharmaFormer model."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class FeatureExtractor(nn.Module):
    """Feature extractor for gene expression and drug SMILES."""

    def __init__(self, gene_input_size: int, gene_hidden_size: int, drug_hidden_size: int):
        """
        Initialize the feature extractor.

        :param gene_input_size: Input size for gene expression features
        :param gene_hidden_size: Hidden size for gene expression MLP
        :param drug_hidden_size: Hidden size for drug SMILES MLP
        """
        super().__init__()
        self.gene_fc1 = nn.Linear(gene_input_size, gene_hidden_size)
        self.gene_fc2 = nn.Linear(gene_hidden_size, gene_hidden_size)
        self.smiles_fc = nn.Linear(128, drug_hidden_size)

    def forward(self, gene_expr: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.

        :param gene_expr: Gene expression features [batch_size, gene_input_size]
        :param smiles: BPE-encoded SMILES features [batch_size, 128]
        :return: Combined features [batch_size, gene_hidden_size + drug_hidden_size]
        """
        gene_out = functional.relu(self.gene_fc1(gene_expr))
        gene_out = functional.relu(self.gene_fc2(gene_out))
        smiles_out = functional.relu(self.smiles_fc(smiles))
        combined_features = torch.cat((gene_out, smiles_out), dim=1)
        return combined_features


class TransModel(nn.Module):
    """Transformer model for processing combined features."""

    def __init__(
        self,
        feature_dim: int,
        nhead: int,
        seq_len: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_layers: int = 3,
    ):
        """
        Initialize the transformer model.

        :param feature_dim: Dimension of each feature in the sequence
        :param nhead: Number of attention heads
        :param seq_len: Length of the input sequence
        :param dim_feedforward: Dimension of feedforward network
        :param dropout: Dropout rate
        :param num_layers: Number of transformer encoder layers
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Linear(seq_len * feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer model.

        :param x: Input tensor [batch_size, seq_len, feature_dim]
        :return: Output predictions [batch_size, 1]
        """
        x = self.transformer_encoder(x)
        x = torch.flatten(x, 1)
        return self.output(x)


class CombinedModel(nn.Module):
    """Combined model integrating feature extraction and transformer."""

    def __init__(
        self,
        gene_input_size: int,
        gene_hidden_size: int,
        drug_hidden_size: int,
        feature_dim: int,
        nhead: int,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Initialize the combined model.

        :param gene_input_size: Input size for gene expression features
        :param gene_hidden_size: Hidden size for gene expression MLP
        :param drug_hidden_size: Hidden size for drug SMILES MLP
        :param feature_dim: Dimension of each feature in the transformer sequence
        :param nhead: Number of attention heads
        :param num_layers: Number of transformer encoder layers
        :param dim_feedforward: Dimension of feedforward network
        :param dropout: Dropout rate
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(gene_input_size, gene_hidden_size, drug_hidden_size)
        self.feature_dim = feature_dim
        self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim
        self.transformer = TransModel(
            feature_dim=feature_dim,
            nhead=nhead,
            seq_len=self.seq_len,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(self, gene_expr: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the combined model.

        :param gene_expr: Gene expression features [batch_size, gene_input_size]
        :param smiles: BPE-encoded SMILES features [batch_size, 128]
        :return: Output predictions [batch_size, 1]
        """
        features = self.feature_extractor(gene_expr, smiles)
        batch_size = features.size(0)
        features = features.view(batch_size, self.seq_len, self.feature_dim)
        output = self.transformer(features)
        return output
