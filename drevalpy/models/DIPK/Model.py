import torch
import random
import numpy as np
from optuna.samplers import TPESampler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .Model_MHA import MultiHeadAttentionLayer

features_dim_gene = 512
features_dim_bionic = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):
    def __init__(self, heads):
        super(AttentionLayer, self).__init__()
        self.fc_layer_0 = nn.Linear(features_dim_gene, 768)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 768)
        self.attention_0 = MultiHeadAttentionLayer(hid_dim=768, n_heads=1, dropout=0.3, device=DEVICE)
        self.attention_1 = MultiHeadAttentionLayer(hid_dim=768, n_heads=1, dropout=0.3, device=DEVICE)

    def forward(self, x, g, gene, bionic):
        gene = F.relu(self.fc_layer_0(gene))
        bionic = F.relu(self.fc_layer_1(bionic))
        x = to_dense_batch(x, g.batch)
        query_0 = torch.unsqueeze(gene, 1)
        query_1 = torch.unsqueeze(bionic, 1)
        key = x[0]
        value = x[0]
        mask = torch.unsqueeze(torch.unsqueeze(x[1], 1), 1)
        x_att = self.attention_0(query_0, key, value, mask)
        x = torch.squeeze(x_att[0])
        x_att = self.attention_1(query_1, key, value, mask)
        x += torch.squeeze(x_att[0])
        return x


class DenseLayers(nn.Module):
    def __init__(self, heads, fc_layer_num, fc_layer_dim, dropout_rate):
        super(DenseLayers, self).__init__()
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

    def forward(self, x, gene, bionic):
        gene = F.relu(self.fc_layer_0(gene))
        bionic = F.relu(self.fc_layer_1(bionic))
        f = torch.cat((x, gene + bionic), 1)
        f = F.relu(self.fc_input(f))
        for layer_index in range(self.fc_layer_num):
            f = F.relu(self.fc_layers[layer_index](f))
            f = self.dropout_layers[layer_index](f)
        f = self.fc_output(f)
        return f


class Predictor(nn.Module):
    def __init__(self, embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate):
        super(Predictor, self).__init__()
        # self.graph_encoder = GraphEncoder(embedding_dim, heads)
        self.attention_layer = AttentionLayer(heads)
        self.dense_layers = DenseLayers(heads, fc_layer_num, fc_layer_dim, dropout_rate)

    def forward(self, x, g, gene, bionic):
        # x = self.graph_encoder(g)
        x = self.attention_layer(x, g, gene, bionic)
        f = self.dense_layers(x, gene, bionic)
        return f


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = TPESampler(seed=seed)
    return sampler
