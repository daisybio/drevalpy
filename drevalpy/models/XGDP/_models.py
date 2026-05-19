"""Models for XGDP model."""

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import (
    FiLMConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    RGATConv,
    RGCNConv,
    SAGEConv,
    global_add_pool,
)
from torch_geometric.nn import global_max_pool as gmp

"""
    DeepChem feature set: 78
    ECFP4: 192
    ECFP4 + DeepChem: 270
    ECFP6: 256
    ECFP6 + DeepChem: 334
"""


class GCNNet(torch.nn.Module):
    """Standard graph convolutions to capture structural SMILES information."""

    def __init__(
        self,
        n_output=1,
        n_filters=32,
        embed_dim=128,
        num_features_xd=334,
        num_features_xt=25,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialization method for GCNNet.

        :param n_output: Number of output units (default: 1)
        :param n_filters: Number of convolution filters for cell line CNN branch
        :param embed_dim: Embedding dimension for optional embeddings
        :param num_features_xd: Number of molecular graph node features
        :param num_features_xt: Number of cell line features
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)
        # self.fc1_xt = nn.Linear(2944, output_dim)
        # self.fc1_xt = nn.Linear(4224, output_dim)
        # self.fc1_xt = nn.Linear(61824, output_dim)
        self.fc1_xt = nn.Linear(4096, output_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, edge_weight=None, return_attention_weights=False):
        """
        Forward pass of the GCNNet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features (unused for GCN)
        :param edge_weight: Optional edge weights
        :returns: Predicted drug response
        """
        # get graph input
        # edge_weight is only used for decoding

        if edge_feat is not None:
            pass

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_index = edge_index.long()

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # get protein input
        # target = data.target
        # print(x_cell_mut.shape)

        # add this line for CNV data, remove for gene expr data
        # x_cell_mut = x_cell_mut[:,None,:]

        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out


class GATNet(torch.nn.Module):
    """Uses attention to weigh importance of neighboring nodes."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the GATNet model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for cell line CNN branch
        :param embed_dim: Embedding dimension for optional embeddings
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the GATNet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features of the molecular graph
        :param return_attention_weights: Whether to return attention weights
        :returns: Predicted drug response or (prediction, attention weights)
        """
        # graph input feed-forward
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        if edge_feat is not None:
            pass

        # x = self.dropout(x)
        # x = f.dropout(x, p=0.2, training=self.training)
        x = f.elu(self.gcn1(x, edge_index))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)
        if return_attention_weights:
            x, attn_weights = self.gcn2(x, edge_index, return_attention_weights=return_attention_weights)
        else:
            x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten

        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            # return out, x
            return out


class GATv2Net(torch.nn.Module):
    """More expressive attention mechanism that supports edge features."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the GATv2Net model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for cell line CNN branch
        :param embed_dim: Embedding dimension for optional embeddings
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = GATv2Conv(
            num_features_xd, num_features_xd, heads=25, dropout=dropout, edge_dim=7, add_self_loops=False
        )
        self.gcn2 = GATv2Conv(num_features_xd * 25, output_dim, dropout=dropout, edge_dim=7, add_self_loops=False)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)
        # self.fc1_xt = nn.Linear(2944, output_dim)
        # self.fc1_xt = nn.Linear(4224, output_dim)
        # self.fc1_xt = nn.Linear(61824, output_dim)
        self.fc1_xt = nn.Linear(4096, output_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the GATv2Net model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features of the molecular graph
        :param return_attention_weights: Whether to return attention weights
        :returns: Predicted drug response or (prediction, attention weights)
        """
        # graph input feed-forward
        # x, edge_index, batch, edge_feat = data.x, data.edge_index, data.batch, data.edge_features
        # print(data.x.shape)
        # print(edge_feat.shape)
        if edge_feat is not None:
            pass

        # x = f.dropout(x, p=0.2, training=self.training)
        # x = self.dropout(x)
        x = f.elu(self.gcn1(x, edge_index, edge_attr=edge_feat))
        x = self.dropout(x)
        # x = f.dropout(x, p=0.2, training=self.training)
        if return_attention_weights:
            x, attn_weights = self.gcn2(
                x, edge_index, edge_attr=edge_feat, return_attention_weights=return_attention_weights
            )
        else:
            x = self.gcn2(x, edge_index, edge_attr=edge_feat)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class GATNetE(torch.nn.Module):
    """A GAT variant explicitly incorporating edge attributes."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the GATNetE model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for cell line CNN branch
        :param embed_dim: Embedding dimension for optional embeddings
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout, edge_dim=7)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout, edge_dim=7)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, output_dim)
        else:
            # combined layers
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the GATNet_E model.

        :param x: feature matrix of molecular graph
        :param edge_index: edges of molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: edge features of molecular graph
        :param return_attention_weights: Whether to return attention weights
        :return: out and attention weights/ out
        """
        # graph input feed-forward
        # x, edge_index, batch, edge_feat = data.x, data.edge_index, data.batch, data.edge_features
        # print(data.x.shape)
        if edge_feat is not None:
            pass

        # x = f.dropout(x, p=0.2, training=self.training)
        # x = self.dropout(x)
        x = f.elu(self.gcn1(x, edge_index, edge_attr=edge_feat))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)
        if return_attention_weights:
            x, attn_weights = self.gcn2(
                x, edge_index, edge_attr=edge_feat, return_attention_weights=return_attention_weights
            )
        else:
            x = self.gcn2(x, edge_index, edge_attr=edge_feat)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class SAGENet(torch.nn.Module):
    """Focuses on sampling and aggregating local node neighborhoods."""

    def __init__(
        self,
        n_output=1,
        n_filters=32,
        embed_dim=128,
        num_features_xd=334,
        num_features_xt=25,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the SAGENet model.

        :param n_output: Number of output units
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param num_features_xd: Number of molecular graph node features
        :param num_features_xt: Number of cell line features
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        """
        super().__init__()
        self.use_attn = use_attn
        # SMILES graph branch

        # GCNSAGE
        self.n_output = n_output
        self.conv1 = SAGEConv(num_features_xd, num_features_xd)
        self.conv2 = SAGEConv(num_features_xd, num_features_xd * 2)
        self.conv3 = SAGEConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the SAGENet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features (unused for SAGEConv)
        :returns: Predicted drug response
        """
        # get graph input
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        if edge_feat is not None:
            pass

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # GCNSAGE
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # get protein input
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out


class GINNet(torch.nn.Module):
    """Structurally powerful architecture for distinguishing complex graph patterns."""

    def __init__(
        self,
        n_output=1,
        num_features_xd=334,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the GINNet model.

        :param n_output: Number of output units
        :param num_features_xd: Number of molecular graph node features
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        """
        super().__init__()
        self.use_attn = use_attn

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the GINNet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features (unused for GINConv)
        :returns: Predicted drug response
        """
        if edge_feat is not None:
            pass

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x)
        # print(data.target)
        x = f.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = f.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = f.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = f.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = f.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = f.relu(self.fc1_xd(x))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]

        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out


class GINENet(torch.nn.Module):
    """Combines GIN's structural power with edge feature integration."""

    def __init__(
        self,
        n_output=1,
        num_features_xd=334,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the GINENet model.

        :param n_output: Number of output units
        :param num_features_xd: Number of molecular graph node features
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        """
        super().__init__()
        self.use_attn = use_attn

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINEConv(nn1, edge_dim=7)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINEConv(nn2, edge_dim=7)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINEConv(nn3, edge_dim=7)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINEConv(nn4, edge_dim=7)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINEConv(nn5, edge_dim=7)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the GINENet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge features of the molecular graph
        :returns: Predicted drug response
        """
        if edge_feat is not None:
            pass

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x)
        # print(data.target)
        x = f.relu(self.conv1(x, edge_index, edge_attr=edge_feat))
        x = self.bn1(x)
        x = f.relu(self.conv2(x, edge_index, edge_attr=edge_feat))
        x = self.bn2(x)
        x = f.relu(self.conv3(x, edge_index, edge_attr=edge_feat))
        x = self.bn3(x)
        x = f.relu(self.conv4(x, edge_index, edge_attr=edge_feat))
        x = self.bn4(x)
        x = f.relu(self.conv5(x, edge_index, edge_attr=edge_feat))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = f.relu(self.fc1_xd(x))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]

        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out


class RGCNNet(torch.nn.Module):
    """Uses relation-specific weights for multi-relational drug graphs."""

    def __init__(
        self,
        n_output=1,
        n_filters=32,
        embed_dim=128,
        num_features_xd=334,
        num_features_xt=25,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the RGCNNet model.

        :param n_output: Number of output units
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param num_features_xd: Number of molecular graph node features
        :param num_features_xt: Number of cell line features
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = RGCNConv(num_features_xd, num_features_xd, num_relations=4)
        self.conv2 = RGCNConv(num_features_xd, num_features_xd * 2, num_relations=4)
        self.conv3 = RGCNConv(num_features_xd * 2, num_features_xd * 4, num_relations=4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout, batch_first=True)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, output_dim)
        else:
            # combined layers
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, edge_weight=None, return_attention_weights=False):
        """
        Forward pass of the RGCNNet model.

        :param x: Node feature matrix of the molecular graph
        :param edge_index: Edge indices of the molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge type indices for relational graph convolution
        :param edge_weight: Optional edge weights
        :returns: Predicted drug response
        """
        # get graph input
        # edge_weight is only used for decoding

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        if edge_feat is not None:
            edge_feat = edge_feat.long().view(-1)

        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # edge_index = edge_index.long()
        # edge_feat = edge_feat.long().squeeze()

        x = self.conv1(x, edge_index, edge_type=edge_feat)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_type=edge_feat)
        x = self.relu(x)
        x = self.conv3(x, edge_index, edge_type=edge_feat)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # get protein input
        # target = data.target
        # print(x_cell_mut.shape)

        # add this line for CNV data, remove for gene expr data
        # x_cell_mut = x_cell_mut[:,None,:]
        if x_cell_mut.dim() == 2:
            x_cell_mut = x_cell_mut.unsqueeze(1)

        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out


class WIRGATNet(torch.nn.Module):
    """Relational attention focused on interactions within the same relation."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the WIRGATNet model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = RGATConv(
            num_features_xd,
            num_features_xd,
            num_relations=4,
            attention_mechanism="within-relation",
            heads=10,
            dropout=dropout,
        )
        self.gcn2 = RGATConv(
            num_features_xd * 10, output_dim, num_relations=4, attention_mechanism="within-relation", dropout=dropout
        )
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            # combined layers
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the WIRGATNet model.

        :param x: feature matrix of molecular graph
        :param edge_index: edges of molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: edge features of molecular graph
        :param return_attention_weights: Whether to return attention weights
        :returns: Predicted drug response or (prediction, attention weights)
        """
        # graph input feed-forward
        # x, edge_index, batch, edge_feat = data.x, data.edge_index, data.batch, data.edge_features
        # print(data.x.shape)
        if edge_feat is not None:
            pass
        edge_feat = edge_feat.int().squeeze()
        # print(edge_feat)

        # x = f.dropout(x, p=0.2, training=self.training)
        # x = self.dropout(x)
        x = f.elu(self.gcn1(x, edge_index, edge_type=edge_feat))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)
        if return_attention_weights:
            x, attn_weights = self.gcn2(
                x, edge_index, edge_type=edge_feat, return_attention_weights=return_attention_weights
            )
        else:
            x = self.gcn2(x, edge_index, edge_type=edge_feat)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class ARGATNet(torch.nn.Module):
    """Relational attention designed to process features across different relations."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the ARGATNet model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = RGATConv(
            num_features_xd,
            num_features_xd,
            num_relations=4,
            attention_mechanism="across-relation",
            heads=10,
            dropout=dropout,
        )
        self.gcn2 = RGATConv(
            num_features_xd * 10, output_dim, num_relations=4, attention_mechanism="across-relation", dropout=dropout
        )
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            # combined layers
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the ARGATNet model.

        :param x: feature matrix of molecular graph
        :param edge_index: edges of molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: edge features of molecular graph
        :param return_attention_weights: Whether to return attention weights
        :returns: Predicted drug response or (prediction, attention weights)
        """
        # graph input feed-forward
        # x, edge_index, batch, edge_feat = data.x, data.edge_index, data.batch, data.edge_features
        # print(data.x.shape)
        if edge_feat is not None:
            pass
        edge_feat = edge_feat.int().squeeze()

        # x = f.dropout(x, p=0.2, training=self.training)
        # x = self.dropout(x)
        x = f.elu(self.gcn1(x, edge_index, edge_type=edge_feat))
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)
        if return_attention_weights:
            x, attn_weights = self.gcn2(
                x, edge_index, edge_type=edge_feat, return_attention_weights=return_attention_weights
            )
        else:
            x = self.gcn2(x, edge_index, edge_type=edge_feat)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        if return_attention_weights:
            return out, attn_weights
        else:
            return out


class FiLMNet(torch.nn.Module):
    """Adaptively modulates features based on specific graph relations."""

    def __init__(
        self,
        num_features_xd=334,
        n_output=1,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.5,
        use_attn=False,
    ):
        """
        Initialize the FiLMNet model.

        :param num_features_xd: Number of molecular graph node features
        :param n_output: Number of output units
        :param num_features_xt: Number of cell line features
        :param n_filters: Number of convolution filters for the cell line CNN branch
        :param embed_dim: Embedding dimension (unused but kept for API consistency)
        :param output_dim: Dimensionality of the latent representation
        :param dropout: Dropout probability
        :param use_attn: Whether to use cross‑attention between drug and cell line features
        """
        super().__init__()
        self.use_attn = use_attn

        # graph layers
        self.gcn1 = FiLMConv(num_features_xd, num_features_xd, num_relations=4, act=nn.LeakyReLU(), edge_dim=7)
        self.gcn2 = FiLMConv(num_features_xd, output_dim, num_relations=4, act=nn.LeakyReLU(), edge_dim=7)

        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        if num_features_xt < 50:
            k = 3
            p = 2
        else:
            k = 8
            p = 3

        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=k)
        self.pool_xt_1 = nn.MaxPool1d(p)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=k)
        self.pool_xt_2 = nn.MaxPool1d(p)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=k)
        self.pool_xt_3 = nn.MaxPool1d(p)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_features_xt)
            conv_xt = self.pool_xt_1(self.conv_xt_1(dummy))
            conv_xt = self.pool_xt_2(self.conv_xt_2(conv_xt))
            conv_xt = self.pool_xt_3(self.conv_xt_3(conv_xt))
            flat_dim = conv_xt.shape[1] * conv_xt.shape[2]

        self.fc1_xt = nn.Linear(flat_dim, output_dim)

        if self.use_attn:
            self.cross_attn1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.cross_attn2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout)
            self.norm1 = nn.LayerNorm(output_dim)
            self.norm2 = nn.LayerNorm(output_dim)
            self.fc = nn.Linear(2 * output_dim, 128)
        else:
            # combined layers
            self.fc1 = nn.Linear(2 * output_dim, 1024)
            self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch, x_cell_mut, edge_feat, return_attention_weights=False):
        """
        Forward pass of the FiLMNet model.

        :param x: feature matrix of molecular graph
        :param edge_index: edges of molecular graph
        :param batch: Batch vector assigning nodes to graphs
        :param x_cell_mut: Cell line omics features
        :param edge_feat: Edge type indices for FiLM modulation
        :param return_attention_weights: Whether to return attention weights
        :returns: Predicted drug response
        """
        # graph input feed-forward
        # x, edge_index, batch, edge_feat = data.x, data.edge_index, data.batch, data.edge_features
        # print(data.x.shape)
        if edge_feat is not None:
            pass
        edge_feat = edge_feat.int().squeeze()

        # x = f.dropout(x, p=0.2, training=self.training)
        # x = self.dropout(x)
        self.gcn1(x, edge_index, edge_type=edge_feat)
        # x = f.dropout(x, p=0.2, training=self.training)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index, edge_type=edge_feat)
        # x = self.relu(x)
        x = gmp(x, batch)  # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        MIN_CNN_INPUT = 22
        if x_cell_mut.shape[-1] < MIN_CNN_INPUT:
            pad = MIN_CNN_INPUT - x_cell_mut.shape[-1]
            x_cell_mut = torch.nn.functional.pad(x_cell_mut, (0, pad))

        # protein input feed-forward:
        # target = data.target
        # x_cell_mut = x_cell_mut[:,None,:]
        # 1d conv layers
        conv_xt = self.conv_xt_1(x_cell_mut)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = f.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        xt = self.fc1_xt(xt)

        if self.use_attn:
            xc1, _ = self.cross_attn1(x, xt, xt)
            xc1 = xc1 + x
            xc1 = self.norm1(xc1)
            xc2, _ = self.cross_attn2(xt, x, x)
            xc2 = xc2 + xt
            xc2 = self.norm2(xc2)
            xc = torch.cat((xc1, xc2), 1)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
        else:
            # concat
            xc = torch.cat((x, xt), 1)
            # add some dense layers
            xc = self.fc1(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)
            xc = self.fc2(xc)
            xc = self.relu(xc)
            xc = self.dropout(xc)

        if self.use_attn:
            pass
        out = self.out(xc)
        out = nn.Sigmoid()(out)

        return out
