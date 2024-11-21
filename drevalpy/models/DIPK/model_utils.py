"""Includes custom torch.nn.Modules for the DIPK model: AttentionLayer, DenseLayer, Predictor."""

import torch
import torch.nn as nn

from .attention_utils import MultiHeadAttentionLayer

features_dim_gene = 512
features_dim_bionic = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    """Custom attention layer for the DIPK model."""

    def __init__(self, heads: int = 1):
        """
        Initialize the attention layer with a multi-head attention layer with a specified number of heads.

        :param heads: number of heads for the multi-head attention layer
        """
        super().__init__()
        self.fc_layer_0 = nn.Linear(features_dim_gene, 768)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 768)
        self.attention_0 = MultiHeadAttentionLayer(hid_dim=768, n_heads=heads, dropout=0.3, device=DEVICE)
        self.attention_1 = MultiHeadAttentionLayer(hid_dim=768, n_heads=heads, dropout=0.3, device=DEVICE)

    def forward(
        self, molgnet_features: torch.Tensor, mask: torch.Tensor, gene_expression: torch.Tensor, bionic: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention layer.

        :param molgnet_features: MolGNet features
        :param mask: mask for the MolGNet features, as molecules have varying sizes (valid atom features are True)
        :param gene_expression: gene expression features of the graph data
        :param bionic: bionic network features of the graph data
        :returns: tensor of MolGNet features after attention layer
        """
        gene_expression = nn.functional.relu(self.fc_layer_0(gene_expression))  # Shape: [batch_size, feature_dim_gene]
        bionic = nn.functional.relu(self.fc_layer_1(bionic))  # Shape: [batch_size, feature_dim_bionic]

        # Preparing query, key, value for attention layers
        query_0 = torch.unsqueeze(gene_expression, 1)  # Shape: [batch_size, 1, 768] for gene
        query_1 = torch.unsqueeze(bionic, 1)  # Shape: [batch_size, 1, 768] for bionic
        key = molgnet_features  # Shape: [batch_size, seq_len, 768] (features from MolGNet)
        value = molgnet_features  # Shape: [batch_size, seq_len, 768] (same as key)

        mask = torch.unsqueeze(mask, 1).unsqueeze(2)

        # Apply the first attention layer
        x_att = self.attention_0(query_0, key, value, mask)  # Output: [batch_size, seq_len, hid_dim]
        x = torch.squeeze(x_att[0])  # Squeeze to remove the extra dimension (1)

        # Apply the second attention layer
        x_att = self.attention_1(query_1, key, value, mask)  # Output: [batch_size, seq_len, hid_dim]
        x += torch.squeeze(x_att[0])  # Add the result of the second attention to the first

        return x


class DenseLayers(nn.Module):
    """Custom dense layers for the DIPK model."""

    def __init__(self, fc_layer_num: int, fc_layer_dim: list[int], dropout_rate: float):
        """
        Initialize the dense layers of the DIPK model which follow the attention layer.

        :param fc_layer_num: number of fully connected layers
        :param fc_layer_dim: list of dimensions for each fully connected layer
        :param dropout_rate: dropout rate for all fully connected layers
        """
        super().__init__()
        self.fc_layer_num = fc_layer_num
        self.fc_layer_0 = nn.Linear(features_dim_gene, 512)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 512)
        self.fc_input = nn.Linear(768 + 512, 768 + 512)
        self.fc_layers = torch.nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.Linear(512, fc_layer_dim[0]),
            nn.Linear(fc_layer_dim[0], fc_layer_dim[1]),
            nn.Linear(fc_layer_dim[1], fc_layer_dim[2]),
            nn.Linear(fc_layer_dim[2], fc_layer_dim[3]),
            nn.Linear(fc_layer_dim[3], fc_layer_dim[4]),
            nn.Linear(fc_layer_dim[4], fc_layer_dim[5]),
        )
        self.dropout_layers = torch.nn.ModuleList([nn.Dropout(p=dropout_rate) for _ in range(fc_layer_num)])
        self.fc_output = nn.Linear(fc_layer_dim[fc_layer_num - 2], 1)

    def forward(self, x: torch.Tensor, gene: torch.Tensor, bionic: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dense layers.

        :param x: output tensor from the attention layer
        :param gene: gene expression features (GEF) of the graph data
        :param bionic: biological network features (BNF) of the graph data
        :returns: output tensor after the dense layers
        """
        gene = torch.nn.functional.relu(self.fc_layer_0(gene))
        bionic = torch.nn.functional.relu(self.fc_layer_1(bionic))
        f = torch.cat((x, gene + bionic), 1)
        f = torch.nn.functional.relu(self.fc_input(f))
        for layer_index in range(self.fc_layer_num):
            f = torch.nn.functional.relu(self.fc_layers[layer_index](f))
            f = self.dropout_layers[layer_index](f)
        f = self.fc_output(f)
        return f


class Predictor(nn.Module):
    """Whole DIPK model."""

    def __init__(self, heads: int, fc_layer_num: int, fc_layer_dim: list[int], dropout_rate: float):
        """
        Initialize the DIPK model with the specified hyperparameters.

        :param heads: number of heads for the multi-head attention layer
        :param fc_layer_num: number of fully connected layers for the dense layers
        :param fc_layer_dim: number of neurons for each fully connected layer
        :param dropout_rate: dropout rate for all fully connected layers
        """
        super().__init__()
        self.attention_layer = AttentionLayer(heads=heads)
        self.dense_layers = DenseLayers(fc_layer_num=fc_layer_num, fc_layer_dim=fc_layer_dim, dropout_rate=dropout_rate)

    def forward(
        self,
        molgnet_drug_features: torch.Tensor,
        gene_expression: torch.Tensor,
        bionic: torch.Tensor,
        molgnet_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the DIPK model.

        :param molgnet_drug_features: tensor of MolGNet features from graph data
        :param gene_expression: gene expression features (GEF) of the graph data
        :param bionic: biological network features (BNF) of the graph data
        :param molgnet_mask: mask for the MolGNet features, as molecules have varying sizes
        :returns: output tensor of the DIPK model
        """
        molgnet_drug_features = self.attention_layer(molgnet_drug_features, molgnet_mask, gene_expression, bionic)
        f = self.dense_layers(molgnet_drug_features, gene_expression, bionic)
        return f
