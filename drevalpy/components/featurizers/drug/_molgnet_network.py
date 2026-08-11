"""MolGNet model and graph conversion utilities (adapted from DIPK).

These classes were originally in ``scripts/featurizer/create_molgnet_embeddings.py``
and are used at runtime by :class:`MolGNetDrugFeaturizer` to compute embeddings
on the fly when precomputed views are missing.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_nn_f
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, softmax

try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import Mol as RDMol
except ImportError as err:
    raise ImportError("Please install rdkit package for MolGNet featurizer: pip install rdkit") from err

allowable_features: dict[str, list[Any]] = {
    "atomic_num": list(range(1, 122)),
    "formal_charge": ["unk", -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "chirality": [
        "unk",
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "hybridization": [
        "unk",
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "numH": ["unk", 0, 1, 2, 3, 4, 5, 6, 7, 8],
    "implicit_valence": ["unk", 0, 1, 2, 3, 4, 5, 6],
    "degree": ["unk", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "isaromatic": [False, True],
    "bond_type": [
        "unk",
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "bond_dirs": [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ],
    "bond_isconjugated": [False, True],
    "bond_inring": [False, True],
    "bond_stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
}

atom_dic = [
    len(allowable_features["atomic_num"]),
    len(allowable_features["formal_charge"]),
    len(allowable_features["chirality"]),
    len(allowable_features["hybridization"]),
    len(allowable_features["numH"]),
    len(allowable_features["implicit_valence"]),
    len(allowable_features["degree"]),
    len(allowable_features["isaromatic"]),
]
bond_dic = [
    len(allowable_features["bond_type"]),
    len(allowable_features["bond_dirs"]),
    len(allowable_features["bond_isconjugated"]),
    len(allowable_features["bond_inring"]),
    len(allowable_features["bond_stereo"]),
]
atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)


def mol_to_graph_data_obj_complex(mol: RDMol) -> Data:
    """Convert an RDKit Mol into a torch_geometric Data object.

    :param mol: RDKit Mol instance.
    :return: torch_geometric.data.Data with node and edge fields.
    :raises ValueError: If mol is None.
    """
    if mol is None:
        raise ValueError("mol must not be None")
    atom_features_list: list = []
    fc_list = allowable_features["formal_charge"]
    ch_list = allowable_features["chirality"]
    hyb_list = allowable_features["hybridization"]
    numh_list = allowable_features["numH"]
    imp_list = allowable_features["implicit_valence"]
    deg_list = allowable_features["degree"]
    isa_list = allowable_features["isaromatic"]
    bt_list = allowable_features["bond_type"]
    bd_list = allowable_features["bond_dirs"]
    bic_list = allowable_features["bond_isconjugated"]
    bir_list = allowable_features["bond_inring"]
    bs_list = allowable_features["bond_stereo"]
    for atom in mol.GetAtoms():
        a_idx = allowable_features["atomic_num"].index(atom.GetAtomicNum())
        fc_idx = fc_list.index(atom.GetFormalCharge()) + atom_cumsum[0]
        ch_idx = ch_list.index(atom.GetChiralTag()) + atom_cumsum[1]
        hyb_idx = hyb_list.index(atom.GetHybridization()) + atom_cumsum[2]
        numh_idx = numh_list.index(atom.GetTotalNumHs()) + atom_cumsum[3]
        imp_idx = imp_list.index(atom.GetValence(Chem.ValenceType.IMPLICIT)) + atom_cumsum[4]
        deg_idx = deg_list.index(atom.GetDegree()) + atom_cumsum[5]
        isa_idx = isa_list.index(atom.GetIsAromatic()) + atom_cumsum[6]
        atom_feature = [a_idx, fc_idx, ch_idx, hyb_idx, numh_idx, imp_idx, deg_idx, isa_idx]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 5
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bt = bt_list.index(bond.GetBondType())
            bd = bd_list.index(bond.GetBondDir()) + bond_cumsum[0]
            bic = bic_list.index(bond.GetIsConjugated()) + bond_cumsum[1]
            bir = bir_list.index(bond.IsInRing()) + bond_cumsum[2]
            bs = bs_list.index(str(bond.GetStereo())) + bond_cumsum[3]
            edge_feature = [bt, bd, bic, bir, bs]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SelfLoop:
    """Append self-loops and matching edge attributes to a Data object."""

    def __call__(self, data: Data) -> Data:
        """Add self-loop indices and corresponding edge attributes.

        :param data: torch_geometric.data.Data to modify.
        :return: The modified Data object.
        """
        num_nodes = data.num_nodes
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
        self_loop_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(num_nodes, 1)
        data.edge_attr = torch.cat((data.edge_attr, self_loop_attr), dim=0)
        return data


class AddSegId:
    """Attach zero-valued segment id tensors to nodes and edges."""

    def __call__(self, data: Data) -> Data:
        """Attach zero-filled node_seg and edge_seg tensors.

        :param data: torch_geometric.data.Data to modify.
        :return: The modified Data object.
        """
        data.edge_seg = torch.LongTensor([0] * data.num_edges)
        data.node_seg = torch.LongTensor([0] * data.num_nodes)
        return data


class BertLayerNorm(nn.Module):
    """Layer normalization compatible with BERT-style implementations."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def _bias_gelu(bias: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class LinearActivation(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if bias:
            self.biased_act_fn = _bias_gelu
        else:
            self.act_fn = _gelu
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return self.biased_act_fn(self.bias, torch_nn_f.linear(input, self.weight, None))
        return self.act_fn(torch_nn_f.linear(input, self.weight, self.bias))


class Intermediate(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.dense_act = LinearActivation(hidden, 4 * hidden)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dense_act(hidden_states)


class AttentionOut(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class GTOut(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden * 4, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class _MessagePassing(nn.Module):
    """Minimal MessagePassing base for MolGNet layers."""

    def __init__(self, aggr: str = "add", flow: str = "source_to_target", node_dim: int = 0) -> None:
        super().__init__()
        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

    def propagate(self, edge_index: torch.Tensor, size=None, **kwargs) -> torch.Tensor:
        i = 1 if self.flow == "source_to_target" else 0
        j = 0 if i == 1 else 1
        x = kwargs.get("x")
        if x is None:
            raise ValueError("propagate requires 'x'")
        x_i = x[edge_index[i]]
        x_j = x[edge_index[j]]
        msg = self.message(edge_index_i=edge_index[i], edge_index_j=edge_index[j], x_i=x_i, x_j=x_j, **kwargs)
        dim_size = x.size(0) if hasattr(x, "size") else len(x)
        out = self.aggregate(msg, index=edge_index[i], dim_size=dim_size)
        return self.update(out)

    def message(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        x_j = kwargs.get("x_j")
        if x_j is None:
            raise ValueError("message requires 'x_j'")
        return x_j

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: int | None = None) -> torch.Tensor:
        from torch_scatter import scatter

        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=self.aggr)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class GraphAttentionConv(_MessagePassing):
    def __init__(self, hidden: int, heads: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.query = nn.Linear(hidden, heads * int(hidden / heads))
        self.key = nn.Linear(hidden, heads * int(hidden / heads))
        self.value = nn.Linear(hidden, heads * int(hidden / heads))
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, size=None) -> torch.Tensor:
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index=edge_index, x=x, pseudo=pseudo)

    def message(self, edge_index_i, x_i, x_j, pseudo, size_i=None, **kwargs) -> torch.Tensor:
        head_dim = int(self.hidden / self.heads)
        query = self.query(x_i).view(-1, self.heads, head_dim)
        key = self.key(x_j + pseudo).view(-1, self.heads, head_dim)
        value = self.value(x_j + pseudo).view(-1, self.heads, head_dim)
        alpha = (query * key).sum(dim=-1) / math.sqrt(head_dim)
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = self.attn_drop(alpha.view(-1, self.heads, 1))
        return alpha * value

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out.view(-1, self.heads * int(self.hidden / self.heads))


class GTLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float, num_message_passing: int) -> None:
        super().__init__()
        self.attention = GraphAttentionConv(hidden, heads, dropout)
        self.att_out = AttentionOut(hidden, dropout)
        self.intermediate = Intermediate(hidden)
        self.output = GTOut(hidden, dropout)
        self.gru = nn.GRU(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.time_step = num_message_passing

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        h = x.unsqueeze(0)
        for _ in range(self.time_step):
            attention_output = self.attention.forward(x, edge_index, edge_attr)
            attention_output = self.att_out.forward(attention_output, x)
            intermediate_output = self.intermediate.forward(attention_output)
            m = self.output.forward(intermediate_output, attention_output)
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.LayerNorm.forward(x.squeeze(0))
        return x


class MolGNet(torch.nn.Module):
    """MolGNet model for node embeddings."""

    def __init__(self, num_layer: int, emb_dim: int, heads: int, num_message_passing: int, drop_ratio: float = 0):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.x_embedding = torch.nn.Embedding(178, emb_dim)
        self.x_seg_embed = torch.nn.Embedding(3, emb_dim)
        self.edge_embedding = torch.nn.Embedding(18, emb_dim)
        self.edge_seg_embed = torch.nn.Embedding(3, emb_dim)
        self.reset_parameters()
        self.gnns = torch.nn.ModuleList(
            [GTLayer(emb_dim, heads, drop_ratio, num_message_passing) for _ in range(num_layer)]
        )

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.x_seg_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_seg_embed.weight.data)

    def forward(self, *argv: Any) -> torch.Tensor:
        if len(argv) == 5:
            x, edge_index, edge_attr, node_seg, edge_seg = argv
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, node_seg, edge_seg = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.node_seg,
                data.edge_seg,
            )
        else:
            raise ValueError("unmatched number of arguments.")
        x = self.x_embedding(x).sum(1) + self.x_seg_embed(node_seg)
        edge_attr = self.edge_embedding(edge_attr).sum(1) + self.edge_seg_embed(edge_seg)
        for gnn in self.gnns:
            x = gnn(x, edge_index, edge_attr)
        return x
