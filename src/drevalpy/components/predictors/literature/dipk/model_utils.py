"""Custom torch.nn.Modules of the DIPK model: MultiHeadAttentionLayer, AttentionLayer, DenseLayer and Predictor."""

import torch
import torch.nn as nn

from drevalpy._utils.accelerator import select_device

features_dim_gene = 512
features_dim_bionic = 512


class MultiHeadAttentionLayer(nn.Module):
    """Custom multi-head attention layer for the DIPK model."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str | torch.device | int | None) -> None:
        """Initialize the multi-head attention layer.

        :param hid_dim: Width of the query, key, value and output projections; split evenly across the heads.
        :param n_heads: Number of attention heads.
        :param dropout: Dropout probability applied to the attention weights.
        :param device: Device on which the attention scaling constant is created, e.g. "cuda" or "cpu".
        :raises ValueError: If ``hid_dim`` is not divisible by ``n_heads``.
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
        # A buffer, so that moving the module (e.g. to the CPU for saving) moves it too.
        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device)), persistent=False
        )
        self.scale: torch.Tensor

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the multi-head attention layer.

        :param query: Query tensor of shape [batch, query_len, hid_dim].
        :param key: Key tensor of shape [batch, key_len, hid_dim].
        :param value: Value tensor with the same shape as ``key``.
        :param mask: Optional mask; positions equal to 0 are excluded from attention.
        :returns: The attended output of shape [batch, query_len, hid_dim] and the per-head attention weights.
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


class AttentionLayer(nn.Module):
    """Custom attention layer for the DIPK model."""

    def __init__(self, heads: int = 1) -> None:
        """Initialize the attention layer with a multi-head attention layer with a specified number of heads.

        :param heads: Number of heads of each multi-head attention layer.
        """
        super().__init__()
        self.fc_layer_0 = nn.Linear(features_dim_gene, 768)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 768)
        device = select_device()
        self.attention_0 = MultiHeadAttentionLayer(hid_dim=768, n_heads=heads, dropout=0.3, device=device)
        self.attention_1 = MultiHeadAttentionLayer(hid_dim=768, n_heads=heads, dropout=0.3, device=device)

    def forward(
        self, molgnet_features: torch.Tensor, mask: torch.Tensor, gene_expression: torch.Tensor, bionic: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the attention layer.

        :param molgnet_features: MolGNet atom features of shape [batch, n_atoms, 768]; used as key and value.
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

    def __init__(self, fc_layer_num: int, fc_layer_dim: list[int], dropout_rate: float) -> None:
        """Initialize the dense layers of the DIPK model which follow the attention layer.

        :param fc_layer_num: Number of fully connected layers (and dropout layers) that are applied.
        :param fc_layer_dim: Output widths of the stacked linear layers; at least six entries are read.
        :param dropout_rate: Dropout probability after each fully connected layer.
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
        """Forward pass of the dense layers.

        :param x: output tensor from the attention layer
        :param gene: gene expression features (GEF) of the graph data
        :param bionic: biological network features (BNF) of the graph data

        :returns: output tensor after the dense layers
        """
        if len(x.shape) == 1:  # happens if the batch size is 1
            x = torch.unsqueeze(x, 0)

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

    def __init__(self, heads: int, fc_layer_num: int, fc_layer_dim: list[int], dropout_rate: float) -> None:
        """Initialize the DIPK model with the specified hyperparameters.

        :param heads: Number of heads of each multi-head attention layer.
        :param fc_layer_num: Number of fully connected layers that are applied in the dense layers.
        :param fc_layer_dim: Output widths of the stacked linear layers in the dense layers.
        :param dropout_rate: Dropout probability after each fully connected layer.
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
        """Forward pass of the DIPK model.

        :param molgnet_drug_features: tensor of MolGNet features from graph data
        :param gene_expression: gene expression features (GEF) of the graph data
        :param bionic: biological network features (BNF) of the graph data
        :param molgnet_mask: mask for the MolGNet features, as molecules have varying sizes

        :returns: output tensor of the DIPK model
        """
        molgnet_drug_features = self.attention_layer(molgnet_drug_features, molgnet_mask, gene_expression, bionic)
        f = self.dense_layers(molgnet_drug_features, gene_expression, bionic)
        return f
