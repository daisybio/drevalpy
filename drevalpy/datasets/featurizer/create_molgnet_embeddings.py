#!/usr/bin/env python3
"""MolGNet feature extraction utilities (needed for DIPK and adapted from the DIPK github).

Creates MolGNet embeddings for molecules given their SMILES strings. This module needs torch_scatter.
    python create_molgnet_embeddings.py dataset_name --checkpoint meta/MolGNet.pt --data_path data
"""
import argparse
import math
import os
import pickle  # noqa: S403
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_nn_f
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, softmax
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import Mol as RDMol
except ImportError:
    raise ImportError("Please install rdkit package for MolGNet featurizer: pip install rdkit")

# building graphs
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
    """Convert an RDKit Mol into a torch_geometric ``Data`` object.

    The function encodes a fixed set of atom and bond categorical
    features and returns a ``Data`` instance with ``x``, ``edge_index``
    and ``edge_attr`` fields. It mirrors the feature layout expected by
    the MolGNet implementation used in this repository.

    :param mol: RDKit ``Mol`` instance. Must not be ``None``.
    :return: A ``torch_geometric.data.Data`` object with node and edge fields.
    :raises ValueError: If ``mol`` is ``None``.
    """
    if mol is None:
        raise ValueError("mol must not be None")
    atom_features_list: list = []
    # Shortcuts for feature lists
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
        imp_idx = imp_list.index(atom.GetImplicitValence()) + atom_cumsum[4]
        deg_idx = deg_list.index(atom.GetDegree()) + atom_cumsum[5]
        isa_idx = isa_list.index(atom.GetIsAromatic()) + atom_cumsum[6]

        atom_feature = [
            a_idx,
            fc_idx,
            ch_idx,
            hyb_idx,
            numh_idx,
            imp_idx,
            deg_idx,
            isa_idx,
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class SelfLoop:
    """Callable that appends self-loops and matching edge attributes.

    This helper mutates the provided ``Data`` object by adding self-loop
    entries to ``edge_index`` and a corresponding edge attribute row for
    every node.
    """

    def __call__(self, data: Data) -> Data:
        """Modify ``data`` in-place by adding self-loop indices and corresponding edge attributes.

        :param data: ``torch_geometric.data.Data`` to modify.
        :return: The modified ``Data`` object (same instance).
        """
        num_nodes = data.num_nodes
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
        self_loop_attr = torch.LongTensor([0, 5, 8, 10, 12]).repeat(num_nodes, 1)
        data.edge_attr = torch.cat((data.edge_attr, self_loop_attr), dim=0)
        return data


class AddSegId:
    """Attach zero-valued segment id tensors to nodes and edges.

    The created ``node_seg`` and ``edge_seg`` tensors are added to the
    provided ``Data`` instance and used by the MolGNet embedding layers.
    """

    def __init__(self) -> None:
        """Create an AddSegId callable (no parameters)."""
        pass

    def __call__(self, data: Data) -> Data:
        """Attach zero-filled ``node_seg`` and ``edge_seg`` tensors to ``data``.

        :param data: ``torch_geometric.data.Data`` to modify.
        :return: The modified ``Data`` object (same instance).
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        node_seg = [0 for _ in range(num_nodes)]
        edge_seg = [0 for _ in range(num_edges)]
        data.edge_seg = torch.LongTensor(edge_seg)
        data.node_seg = torch.LongTensor(node_seg)
        return data


# MolGNet model


class BertLayerNorm(nn.Module):
    """Layer normalization compatible with BERT-style implementations.

    :param hidden_size: Dimension of the last axis to normalize.
    :param eps: Small epsilon for numerical stability.
    """

    def __init__(self, hidden_size, eps=1e-12):
        """Create a BertLayerNorm module.

        :param hidden_size: Dimension of the last axis to normalize.
        :param eps: Small epsilon for numerical stability.
        """
        super().__init__()
        self.shape = torch.Size((hidden_size,))
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the last dimension of ``x``.

        :param x: Input tensor.
        :return: Normalized tensor with same shape as ``x``.
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit activation (approximation).

    :param x: Input tensor.
    :return: Activated tensor.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def bias_gelu(bias: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Apply GELU to ``bias + y``.

    :param bias: Bias tensor to add.
    :param y: Linear output tensor.
    :return: GELU applied to ``bias + y``.
    """
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class LinearActivation(nn.Module):
    """Linear layer with optional bias-aware GELU activation.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param bias: Whether to use a bias parameter and the biased GELU.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Create a LinearActivation module.

        :param in_features: Input feature dimension.
        :param out_features: Output feature dimension.
        :param bias: Whether to use a bias parameter and the biased GELU.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.biased_act_fn = bias_gelu
        else:
            self.act_fn = gelu
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the layer parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation and activation.

        :param input: Input tensor of shape [N, in_features].
        :return: Transformed tensor of shape [N, out_features].
        """
        if self.bias is not None:
            linear_out = torch_nn_f.linear(input, self.weight, None)
            return self.biased_act_fn(self.bias, linear_out)
        else:
            return self.act_fn(torch_nn_f.linear(input, self.weight, self.bias))


class Intermediate(nn.Module):
    """Intermediate feed-forward block used inside GT layers.

    :param hidden: Hidden dimension size.
    """

    def __init__(self, hidden: int) -> None:
        """Create the intermediate dense activation block.

        :param hidden: Hidden dimension size.
        """
        super().__init__()
        self.dense_act = LinearActivation(hidden, 4 * hidden)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the dense activation to the hidden states.

        :param hidden_states: Input tensor of shape [N, hidden].
        :return: Transformed tensor of shape [N, 4*hidden].
        """
        hidden_states = self.dense_act(hidden_states)
        return hidden_states


class AttentionOut(nn.Module):
    """Post-attention output block: projection, dropout and residual norm.

    :param hidden: Hidden dimension used for the linear projection.
    :param dropout: Dropout probability.
    """

    def __init__(self, hidden: int, dropout: float) -> None:
        """Create an AttentionOut block.

        :param hidden: Hidden dimension used for projection.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project attention outputs and apply layer norm with residual.

        :param hidden_states: Attention output tensor.
        :param input_tensor: Residual tensor to add before normalization.
        :return: Normalized tensor with the same shape as ``input_tensor``.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GTOut(nn.Module):
    """Output projection used in GT blocks.

    :param hidden: Hidden dimension.
    :param dropout: Dropout probability.
    """

    def __init__(self, hidden: int, dropout: float) -> None:
        """Create a GTOut projection block.

        :param hidden: Hidden dimension.
        :param dropout: Dropout probability.
        """
        super().__init__()
        self.dense = nn.Linear(hidden * 4, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Project intermediate states back to hidden dimension and normalize.

        :param hidden_states: Intermediate tensor of shape [N, 4*hidden].
        :param input_tensor: Residual tensor to add.
        :return: Tensor of shape [N, hidden].
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MessagePassing(nn.Module):
    """Minimal MessagePassing base class used by the MolGNet layers.

    This class provides a lightweight implementation of propagate/
    message/aggregate/update used in graph convolutions.

    :param aggr: Aggregation method (e.g., 'add', 'mean').
    :param flow: Message flow direction.
    :param node_dim: Node dimension index (unused in this minimal impl).
    """

    def __init__(self, aggr: str = "add", flow: str = "source_to_target", node_dim: int = 0) -> None:
        """Create a MessagePassing helper.

        :param aggr: Aggregation method (e.g., 'add' or 'mean').
        :param flow: Message flow direction.
        :param node_dim: Node dimension index.
        """
        super().__init__()
        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

    def propagate(self, edge_index: torch.Tensor, size: Optional[tuple[int, int]] = None, **kwargs) -> torch.Tensor:
        """Run full message-passing: message -> aggregate -> update.

        :param edge_index: Edge indices tensor of shape [2, E].
        :param size: Optional pair describing (num_nodes_source, num_nodes_target).
        :param kwargs: Additional data (e.g., node features) needed for message computation.
        :raises ValueError: If required inputs (e.g., 'x') are missing or indexing fails.
        :return: Updated node tensor after aggregation.
        """
        i = 1 if self.flow == "source_to_target" else 0
        j = 0 if i == 1 else 1
        x = kwargs.get("x")
        if x is None:
            raise ValueError("propagate requires node features passed as keyword 'x'")
        try:
            x_i = x[edge_index[i]]
            x_j = x[edge_index[j]]
        except Exception as exc:  # defensive
            raise ValueError("failed to index node features with edge_index") from exc
        msg = self.message(
            edge_index_i=edge_index[i],
            edge_index_j=edge_index[j],
            x_i=x_i,
            x_j=x_j,
            **kwargs,
        )
        # determine number of destination nodes for aggregation
        if hasattr(x, "size"):
            dim_size = x.size(0)
        else:
            dim_size = len(x)
        out = self.aggregate(msg, index=edge_index[i], dim_size=dim_size)
        out = self.update(out)
        return out

    def message(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Default message function returning neighbor features.

        Subclasses may provide richer signatures; this generic form allows
        subclass overrides while keeping the base class typed.

        :param args: Positional arguments forwarded by propagate.
        :param kwargs: Keyword arguments forwarded by propagate.
        :raises ValueError: If required node features are not present.
        :return: Message tensor.
        """
        x_j = kwargs.get("x_j") if "x_j" in kwargs else (args[1] if len(args) > 1 else None)
        if x_j is None:
            raise ValueError("message requires node features 'x_j'")
        return x_j

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
        """Aggregate messages using ``torch_scatter.scatter``.

        :param inputs: Message tensor of shape [E, hidden].
        :param index: Indices to aggregate into nodes.
        :param dim_size: Optional target size for the aggregation dimension.
        :return: Aggregated node tensor.
        """
        from torch_scatter import scatter  # local dependency

        return scatter(
            inputs,
            index,
            dim=0,
            dim_size=dim_size,
            reduce=self.aggr,
        )

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        """Identity update by default.

        Override to apply post-aggregation transformations.

        :param inputs: Aggregated node tensor.
        :return: Updated tensor.
        """
        return inputs


class GraphAttentionConv(MessagePassing):
    """Graph attention convolution used by MolGNet.

    :param hidden: Hidden feature dimension.
    :param heads: Number of attention heads.
    :param dropout: Attention dropout probability.
    """

    def __init__(self, hidden: int, heads: int = 3, dropout: float = 0.0) -> None:
        """Create a GraphAttentionConv.

        :param hidden: Hidden feature dimension.
        :param heads: Number of attention heads.
        :param dropout: Dropout probability.
        :raises ValueError: If hidden is not divisible by heads.
        """
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.query = nn.Linear(hidden, heads * int(hidden / heads))
        self.key = nn.Linear(hidden, heads * int(hidden / heads))
        self.value = nn.Linear(hidden, heads * int(hidden / heads))
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Execute the graph attention conv over the provided inputs.

        :param x: Node feature tensor.
        :param edge_index: Edge indices tensor.
        :param edge_attr: Edge attribute tensor.
        :param size: Optional size tuple.
        :return: Updated node tensor after attention.
        """
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index=edge_index, x=x, pseudo=pseudo)

    def message(
        self,
        edge_index_i: torch.Tensor,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pseudo: torch.Tensor,
        size_i: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute messages using multi-head attention between nodes.

        :param edge_index_i: Source indices for edges.
        :param x_i: Node features for source nodes.
        :param x_j: Node features for target nodes.
        :param pseudo: Edge pseudo-features (edge attributes).
        :param size_i: Optional number of destination nodes.
        :param kwargs: Additional keyword arguments (ignored).
        :return: Message tensor shaped for aggregation.
        """
        query = self.query(x_i).view(
            -1,
            self.heads,
            int(self.hidden / self.heads),
        )
        key = self.key(x_j + pseudo).view(
            -1,
            self.heads,
            int(self.hidden / self.heads),
        )
        value = self.value(x_j + pseudo).view(
            -1,
            self.heads,
            int(self.hidden / self.heads),
        )
        denom = math.sqrt(int(self.hidden / self.heads))
        alpha = (query * key).sum(dim=-1) / denom
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = self.attn_drop(alpha.view(-1, self.heads, 1))
        return alpha * value

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Reshape aggregated outputs from multi-head to flat hidden dim.

        :param aggr_out: Aggregated output tensor of shape [N*heads, head_dim].
        :return: Reshaped tensor of shape [N, hidden].
        """
        aggr_out = aggr_out.view(-1, self.heads * int(self.hidden / self.heads))
        return aggr_out


class GTLayer(nn.Module):
    """Graph Transformer layer composed from attention and feed-forward blocks.

    :param hidden: Hidden dimension size.
    :param heads: Number of attention heads.
    :param dropout: Dropout probability.
    :param num_message_passing: Number of internal message passing steps.
    """

    def __init__(self, hidden: int, heads: int, dropout: float, num_message_passing: int) -> None:
        """Create a GTLayer composed of attention and feed-forward blocks.

        :param hidden: Hidden dimension size.
        :param heads: Number of attention heads.
        :param dropout: Dropout probability.
        :param num_message_passing: Number of internal message passing steps.
        """
        super().__init__()
        self.attention = GraphAttentionConv(hidden, heads, dropout)
        self.att_out = AttentionOut(hidden, dropout)
        self.intermediate = Intermediate(hidden)
        self.output = GTOut(hidden, dropout)
        self.gru = nn.GRU(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.time_step = num_message_passing

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Run the GT layer for the configured number of message-passing steps.

        :param x: Node feature tensor of shape [N, hidden].
        :param edge_index: Edge index tensor.
        :param edge_attr: Edge attribute tensor.
        :return: Updated node tensor of shape [N, hidden].
        """
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
    """MolGNet model implementation used for node embeddings.

    This implementation is intentionally minimal and only includes the
    components required to run a checkpoint and produce per-node
    embeddings saved by the featurizer script.

    :param num_layer: Number of GT layers.
    :param emb_dim: Embedding dimensionality per node.
    :param heads: Number of attention heads.
    :param num_message_passing: Message passing steps per layer.
    :param drop_ratio: Dropout probability.
    """

    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        heads: int,
        num_message_passing: int,
        drop_ratio: float = 0,
    ) -> None:
        """Create a MolGNet instance.

        :param num_layer: Number of GT layers.
        :param emb_dim: Embedding dimensionality per node.
        :param heads: Number of attention heads.
        :param num_message_passing: Message passing steps per layer.
        :param drop_ratio: Dropout probability.
        """
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
        """Re-initialize embedding parameters with Xavier uniform.

        This mirrors common initialization used for transformer-style
        embeddings.
        """
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.x_seg_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_seg_embed.weight.data)

    def forward(self, *argv: Any) -> torch.Tensor:
        """Forward pass supporting two calling conventions.

        Accepts either explicit tensors (x, edge_index, edge_attr, node_seg,
        edge_seg) or a single ``Data`` object containing those attributes.

        :param argv: Positional arguments as described above.
        :raises ValueError: If an unsupported number of arguments is provided.
        :return: Node embeddings tensor of shape [N, emb_dim].
        """
        if len(argv) == 5:
            x, edge_index, edge_attr, node_seg, edge_seg = (argv[0], argv[1], argv[2], argv[3], argv[4])
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
        edge_attr = self.edge_embedding(edge_attr).sum(1)
        edge_attr = edge_attr + self.edge_seg_embed(edge_seg)
        for gnn in self.gnns:
            x = gnn(x, edge_index, edge_attr)
        return x


def tensor_to_csv_friendly(tensor: Any) -> np.ndarray:
    """Convert a tensor-like object into a NumPy array safe for CSV output.

    :param tensor: Input tensor or array-like object.
    :return: NumPy array on CPU.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return np.array(tensor)


def run(args: argparse.Namespace) -> None:
    """Execute the featurization pipeline for a given dataset.

    The function builds graphs from SMILES, runs the MolGNet checkpoint
    to extract node embeddings, and writes per-drug CSVs and pickles in
    the dataset folder.

    :param args: Parsed CLI arguments.
    :raises FileNotFoundError: If expected files or directories are missing.
    :raises ValueError: If expected columns are missing in the input CSV.
    :raises Exception: For various failures during graph building or inference.
    """
    # Use dataset-oriented paths: {data_path}/{dataset_name}/...
    # Expand user (~) and resolve to an absolute path.
    data_dir = Path(args.data_path).expanduser().resolve()
    dataset_dir = data_dir / args.dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_graphs = str(dataset_dir / "GRAPH_dict.pkl")
    out_molg = str(dataset_dir / "MolGNet_dict.pkl")

    # read input csv (expected at {data_path}/{dataset_name}/drug_smiles.csv)
    smiles_csv = dataset_dir / "drug_smiles.csv"
    if not smiles_csv.exists():
        raise FileNotFoundError(f"Expected SMILES CSV at: {smiles_csv}")
    df = pd.read_csv(smiles_csv)
    if args.smiles_col not in df.columns or args.id_col not in df.columns:
        msg = f"Provided columns not in CSV: {args.smiles_col}, " f"{args.id_col}"
        raise ValueError(msg)
    df = df.dropna(subset=[args.smiles_col])
    smiles_map = dict(zip(df[args.id_col], df[args.smiles_col]))

    # Build graphs
    graph_dict: dict[Any, Data] = {}
    failed_conversions = []
    for idx, smi in tqdm(smiles_map.items(), desc="building graphs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed_conversions.append((idx, smi, "MolFromSmiles returned None"))
            continue
        try:
            graph_dict[idx] = mol_to_graph_data_obj_complex(mol)
        except Exception as e:
            failed_conversions.append((idx, smi, str(e)))
    if failed_conversions:
        print(f"\n{len(failed_conversions)} molecules failed to convert to graphs.")
        for idx, smi, err in failed_conversions:
            print(f"Failed to convert {idx} (SMILES: {smi}): {err}")
    else:
        print("\nAll molecules converted to graphs successfully.")
    # save graphs to dataset folder
    with open(out_graphs, "wb") as f:
        pickle.dump(graph_dict, f)
    # load model
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layer = 5
    emb_dim = 768
    heads = 12
    msg_pass = 3
    drop = 0.0
    model = MolGNet(
        num_layer=num_layer,
        emb_dim=emb_dim,
        heads=heads,
        num_message_passing=msg_pass,
        drop_ratio=drop,
    )
    # Prefer pathlib operations when working with Path objects
    checkpoint_path = data_dir / args.checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)  # noqa S614
    try:
        model.load_state_dict(ckpt)
    except Exception:
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            raise
    model = model.to(device)
    model.eval()

    self_loop = SelfLoop()
    add_seg = AddSegId()

    molgnet_dict: dict[Any, torch.Tensor] = {}
    with torch.no_grad():
        for idx, graph in tqdm(graph_dict.items(), desc="running model"):
            try:
                g = self_loop(graph)
                g = add_seg(g)
                g = g.to(device)
                emb = model(g)
                molgnet_dict[idx] = emb.cpu()
            except Exception as e:
                print(f"Inference failed for {idx}: {e}")

    with open(out_molg, "wb") as f:
        pickle.dump(molgnet_dict, f)

    # write per-drug CSVs to {dataset_dir}/DIPK_features/Drugs
    out_drugs_dir = dataset_dir / "DIPK_features/Drugs"
    os.makedirs(out_drugs_dir, exist_ok=True)
    for idx, emb in tqdm(molgnet_dict.items(), desc="writing csvs"):
        arr = tensor_to_csv_friendly(emb)
        df_emb = pd.DataFrame(arr)
        out_path = out_drugs_dir / f"MolGNet_{idx}.csv"
        df_emb.to_csv(out_path, sep="\t", index=False)

    print("Done.")
    print("Graphs saved to:", out_graphs)
    print("Node embeddings saved to:", out_molg)
    print("Per-drug CSVs in:", out_drugs_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    :return: Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=("Standalone MolGNet extractor " "(dataset-oriented)"))
    p.add_argument(
        "dataset_name",
        help="Name of the dataset (folder under data_path)",
    )
    p.add_argument(
        "--data_path",
        default="data",
        help="Top-level data folder path",
    )
    p.add_argument(
        "--smiles-col",
        dest="smiles_col",
        default="canonical_smiles",
        help="Column name for SMILES in input CSV",
    )
    p.add_argument(
        "--id-col",
        dest="id_col",
        default="pubchem_id",
        help="Column name for unique ID in input CSV",
    )
    p.add_argument(
        "--checkpoint",
        default="MolGNet.pt",
        help="MolGNet checkpoint (state_dict), can be obtained from Zenodo: https://doi.org/10.5281/zenodo.12633909",
    )
    p.add_argument(
        "--device",
        default=None,
        help="torch device string, e.g. cpu or cuda:0",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
