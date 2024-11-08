import torch
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        # Ensure head dimension divides evenly
        assert hid_dim % n_heads == 0, "Hidden dimension must be divisible by the number of heads."
        
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

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Transform inputs
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Split into heads and transpose for multi-head processing
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention weights
        x = torch.matmul(self.dropout(attention), V)
        
        # Concatenate heads and pass through output layer
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        
        return x, attention
