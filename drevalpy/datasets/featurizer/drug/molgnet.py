"""MolGNet drug featurizer for generating graph-based embeddings.

This module provides a featurizer that uses the MolGNet model to generate
node embeddings for molecules. It requires a pre-trained MolGNet checkpoint.
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

from drevalpy.datasets.dataset import FeatureDataset

from .base import DrugFeaturizer

try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import Mol as RDMol
except ImportError:
    Chem = None
    RDMol = None


# Feature configuration for MolGNet graph building
allowable_features: dict[str, list[Any]] = {
    "atomic_num": list(range(1, 122)),
    "formal_charge": ["unk", -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "chirality": [],  # Populated after rdkit import check
    "hybridization": [],  # Populated after rdkit import check
    "numH": ["unk", 0, 1, 2, 3, 4, 5, 6, 7, 8],
    "implicit_valence": ["unk", 0, 1, 2, 3, 4, 5, 6],
    "degree": ["unk", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "isaromatic": [False, True],
    "bond_type": [],  # Populated after rdkit import check
    "bond_dirs": [],  # Populated after rdkit import check
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


def _init_rdkit_features():
    """Initialize RDKit-dependent feature configurations.

    :raises ImportError: If rdkit package is not installed
    """
    if Chem is None:
        raise ImportError("Please install rdkit package for MolGNet featurizer: pip install rdkit")

    allowable_features["chirality"] = [
        "unk",
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    allowable_features["hybridization"] = [
        "unk",
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
    allowable_features["bond_type"] = [
        "unk",
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    allowable_features["bond_dirs"] = [
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]


# Compute cumulative sums for feature indexing
atom_dic = [
    len(allowable_features["atomic_num"]),
    12,  # formal_charge
    5,  # chirality
    8,  # hybridization
    10,  # numH
    7,  # implicit_valence
    12,  # degree
    2,  # isaromatic
]
bond_dic = [
    5,  # bond_type
    3,  # bond_dirs
    2,  # bond_isconjugated
    2,  # bond_inring
    6,  # bond_stereo
]
atom_cumsum = np.cumsum(atom_dic)
bond_cumsum = np.cumsum(bond_dic)


def mol_to_graph_data_obj_complex(mol: "RDMol") -> Data:
    """Convert an RDKit Mol into a torch_geometric Data object for MolGNet.

    :param mol: RDKit Mol instance
    :returns: torch_geometric.data.Data object
    :raises ValueError: If mol is None
    """
    if mol is None:
        raise ValueError("mol must not be None")

    _init_rdkit_features()

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
        imp_idx = imp_list.index(atom.GetImplicitValence()) + atom_cumsum[4]
        deg_idx = deg_list.index(atom.GetDegree()) + atom_cumsum[5]
        isa_idx = isa_list.index(atom.GetIsAromatic()) + atom_cumsum[6]

        atom_feature = [a_idx, fc_idx, ch_idx, hyb_idx, numh_idx, imp_idx, deg_idx, isa_idx]
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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class SelfLoop:
    """Callable that appends self-loops and matching edge attributes."""

    def __call__(self, data: Data) -> Data:
        """Add self-loop indices and corresponding edge attributes.

        :param data: torch_geometric.data.Data to modify
        :returns: Modified Data object
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

        :param data: torch_geometric.data.Data to modify
        :returns: Modified Data object
        """
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        data.edge_seg = torch.LongTensor([0] * num_edges)
        data.node_seg = torch.LongTensor([0] * num_nodes)
        return data


# MolGNet model components


class BertLayerNorm(nn.Module):
    """Layer normalization compatible with BERT-style implementations."""

    def __init__(self, hidden_size: int, eps: float = 1e-12) -> None:
        """Initialize the layer normalization.

        :param hidden_size: Size of the hidden dimension
        :param eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        :param x: Input tensor
        :returns: Normalized tensor
        """
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian Error Linear Unit activation.

    :param x: Input tensor
    :returns: Activated tensor
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def bias_gelu(bias: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation to bias + y.

    :param bias: Bias tensor
    :param y: Input tensor
    :returns: Activated tensor
    """
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


class LinearActivation(nn.Module):
    """Linear layer with optional bias-aware GELU activation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize the linear activation layer.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param bias: Whether to include a bias term
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
        """Reset layer parameters using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation with GELU activation.

        :param input: Input tensor
        :returns: Transformed tensor
        """
        if self.bias is not None:
            linear_out = torch_nn_f.linear(input, self.weight, None)
            return self.biased_act_fn(self.bias, linear_out)
        else:
            return self.act_fn(torch_nn_f.linear(input, self.weight, self.bias))


class Intermediate(nn.Module):
    """Intermediate feed-forward block used inside GT layers."""

    def __init__(self, hidden: int) -> None:
        """Initialize the intermediate layer.

        :param hidden: Hidden dimension size
        """
        super().__init__()
        self.dense_act = LinearActivation(hidden, 4 * hidden)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        :param hidden_states: Input tensor
        :returns: Transformed tensor
        """
        return self.dense_act(hidden_states)


class AttentionOut(nn.Module):
    """Post-attention output block: projection, dropout and residual norm."""

    def __init__(self, hidden: int, dropout: float) -> None:
        """Initialize the attention output layer.

        :param hidden: Hidden dimension size
        :param dropout: Dropout probability
        """
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply output transformation with residual connection.

        :param hidden_states: Attention output tensor
        :param input_tensor: Original input for residual connection
        :returns: Transformed tensor
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class GTOut(nn.Module):
    """Output projection used in GT blocks."""

    def __init__(self, hidden: int, dropout: float) -> None:
        """Initialize the GT output layer.

        :param hidden: Hidden dimension size
        :param dropout: Dropout probability
        """
        super().__init__()
        self.dense = nn.Linear(hidden * 4, hidden)
        self.LayerNorm = BertLayerNorm(hidden, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply output transformation with residual connection.

        :param hidden_states: Intermediate output tensor
        :param input_tensor: Original input for residual connection
        :returns: Transformed tensor
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class MessagePassing(nn.Module):
    """Minimal MessagePassing base class used by the MolGNet layers."""

    def __init__(self, aggr: str = "add", flow: str = "source_to_target", node_dim: int = 0) -> None:
        """Initialize the message passing layer.

        :param aggr: Aggregation method ('add', 'mean', 'max')
        :param flow: Direction of message flow
        :param node_dim: Dimension along which to aggregate
        """
        super().__init__()
        self.aggr = aggr
        self.flow = flow
        self.node_dim = node_dim

    def propagate(self, edge_index: torch.Tensor, size: Optional[tuple[int, int]] = None, **kwargs) -> torch.Tensor:
        """Propagate messages along edges.

        :param edge_index: Edge connectivity tensor
        :param size: Optional size tuple for bipartite graphs
        :param kwargs: Additional arguments including node features 'x'
        :returns: Aggregated messages
        :raises ValueError: If node features 'x' are not provided
        """
        i = 1 if self.flow == "source_to_target" else 0
        j = 0 if i == 1 else 1
        x = kwargs.get("x")
        if x is None:
            raise ValueError("propagate requires node features passed as keyword 'x'")
        x_i = x[edge_index[i]]
        x_j = x[edge_index[j]]
        msg = self.message(
            edge_index_i=edge_index[i],
            edge_index_j=edge_index[j],
            x_i=x_i,
            x_j=x_j,
            **kwargs,
        )
        dim_size = x.size(0) if hasattr(x, "size") else len(x)
        out = self.aggregate(msg, index=edge_index[i], dim_size=dim_size)
        return self.update(out)

    def message(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Compute messages for each edge.

        :param args: Positional arguments
        :param kwargs: Keyword arguments including 'x_j' for source node features
        :returns: Message tensor
        :raises ValueError: If 'x_j' is not provided
        """
        x_j = kwargs.get("x_j") if "x_j" in kwargs else (args[1] if len(args) > 1 else None)
        if x_j is None:
            raise ValueError("message requires node features 'x_j'")
        return x_j

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor:
        """Aggregate messages at target nodes.

        :param inputs: Message tensor
        :param index: Target node indices
        :param dim_size: Number of target nodes
        :returns: Aggregated tensor
        """
        from torch_scatter import scatter

        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce=self.aggr)

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        """Update node representations after aggregation.

        :param inputs: Aggregated messages
        :returns: Updated node representations
        """
        return inputs


class GraphAttentionConv(MessagePassing):
    """Graph attention convolution used by MolGNet."""

    def __init__(self, hidden: int, heads: int = 3, dropout: float = 0.0) -> None:
        """Initialize the graph attention convolution.

        :param hidden: Hidden dimension size
        :param heads: Number of attention heads
        :param dropout: Dropout probability
        :raises ValueError: If hidden is not divisible by heads
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
        """Apply graph attention convolution.

        :param x: Node feature tensor
        :param edge_index: Edge connectivity tensor
        :param edge_attr: Edge attribute tensor
        :param size: Optional size tuple for bipartite graphs
        :returns: Updated node features
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
        """Compute attention-weighted messages.

        :param edge_index_i: Target node indices
        :param x_i: Target node features
        :param x_j: Source node features
        :param pseudo: Edge features
        :param size_i: Number of target nodes
        :param kwargs: Additional arguments
        :returns: Attention-weighted messages
        """
        query = self.query(x_i).view(-1, self.heads, int(self.hidden / self.heads))
        key = self.key(x_j + pseudo).view(-1, self.heads, int(self.hidden / self.heads))
        value = self.value(x_j + pseudo).view(-1, self.heads, int(self.hidden / self.heads))
        denom = math.sqrt(int(self.hidden / self.heads))
        alpha = (query * key).sum(dim=-1) / denom
        alpha = softmax(src=alpha, index=edge_index_i, num_nodes=size_i)
        alpha = self.attn_drop(alpha.view(-1, self.heads, 1))
        return alpha * value

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Reshape aggregated output.

        :param aggr_out: Aggregated attention output
        :returns: Reshaped tensor
        """
        return aggr_out.view(-1, self.heads * int(self.hidden / self.heads))


class GTLayer(nn.Module):
    """Graph Transformer layer composed from attention and feed-forward blocks."""

    def __init__(self, hidden: int, heads: int, dropout: float, num_message_passing: int) -> None:
        """Initialize the Graph Transformer layer.

        :param hidden: Hidden dimension size
        :param heads: Number of attention heads
        :param dropout: Dropout probability
        :param num_message_passing: Number of message passing iterations
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
        """Apply Graph Transformer layer.

        :param x: Node feature tensor
        :param edge_index: Edge connectivity tensor
        :param edge_attr: Edge attribute tensor
        :returns: Updated node features
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
    """MolGNet model implementation used for node embeddings."""

    def __init__(
        self,
        num_layer: int,
        emb_dim: int,
        heads: int,
        num_message_passing: int,
        drop_ratio: float = 0,
    ) -> None:
        """Initialize the MolGNet model.

        :param num_layer: Number of Graph Transformer layers
        :param emb_dim: Embedding dimension
        :param heads: Number of attention heads
        :param num_message_passing: Number of message passing iterations per layer
        :param drop_ratio: Dropout ratio
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
        """Reset model parameters using Xavier initialization."""
        torch.nn.init.xavier_uniform_(self.x_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.x_seg_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_seg_embed.weight.data)

    def forward(self, *argv: Any) -> torch.Tensor:
        """Forward pass through the MolGNet model.

        :param argv: Either 5 tensors (x, edge_index, edge_attr, node_seg, edge_seg)
                    or a single Data object
        :returns: Node embeddings
        :raises ValueError: If incorrect number of arguments provided
        """
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
        edge_attr = self.edge_embedding(edge_attr).sum(1)
        edge_attr = edge_attr + self.edge_seg_embed(edge_seg)
        for gnn in self.gnns:
            x = gnn(x, edge_index, edge_attr)
        return x


class MolGNetFeaturizer(DrugFeaturizer):
    """Featurizer that generates MolGNet node embeddings from SMILES strings.

    MolGNet is a graph neural network that produces per-node embeddings for
    molecules. This featurizer requires a pre-trained MolGNet checkpoint.

    Example usage::

        featurizer = MolGNetFeaturizer(checkpoint_path="data/MolGNet.pt", device="cuda")
        features = featurizer.load_or_generate("data", "GDSC1")
    """

    # Default model hyperparameters
    NUM_LAYER = 5
    EMB_DIM = 768
    HEADS = 12
    MSG_PASS = 3
    DROP = 0.0

    def __init__(self, checkpoint_path: str = "data/MolGNet.pt", device: str = "cpu"):
        """Initialize the MolGNet featurizer.

        :param checkpoint_path: Path to the MolGNet checkpoint file
        :param device: Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._self_loop = SelfLoop()
        self._add_seg = AddSegId()

    def _load_model(self):
        """Lazily load the MolGNet model.

        :raises Exception: If checkpoint loading fails
        """
        if self._model is None:
            _init_rdkit_features()

            self._model = MolGNet(
                num_layer=self.NUM_LAYER,
                emb_dim=self.EMB_DIM,
                heads=self.HEADS,
                num_message_passing=self.MSG_PASS,
                drop_ratio=self.DROP,
            )

            device = torch.device(self.device)
            ckpt = torch.load(self.checkpoint_path, map_location=device)  # noqa: S614
            try:
                self._model.load_state_dict(ckpt)
            except Exception:
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    self._model.load_state_dict(ckpt["state_dict"])
                else:
                    raise

            self._model = self._model.to(device)
            self._model.eval()

    def featurize(self, smiles: str) -> torch.Tensor | None:
        """Convert a SMILES string to MolGNet node embeddings.

        :param smiles: SMILES string representing the drug
        :returns: Node embeddings tensor or None if conversion fails
        :raises RuntimeError: If model is not loaded
        """
        _init_rdkit_features()
        self._load_model()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        graph = mol_to_graph_data_obj_complex(mol)
        graph = self._self_loop(graph)
        graph = self._add_seg(graph)
        graph = graph.to(self.device)

        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        with torch.no_grad():
            embeddings = self._model(graph)

        return embeddings.cpu()

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the feature view name.

        :returns: 'molgnet_embeddings'
        """
        return "molgnet_embeddings"

    @classmethod
    def get_output_filename(cls) -> str:
        """Return the output filename for cached embeddings.

        :returns: 'MolGNet_dict.pkl'
        """
        return "MolGNet_dict.pkl"

    def _save_embeddings(self, embeddings: list, drug_ids: list[str], output_path: Path) -> None:
        """Save MolGNet embeddings to disk as a pickle file.

        :param embeddings: List of embedding tensors
        :param drug_ids: List of drug identifiers
        :param output_path: Path to save the embeddings
        """
        molgnet_dict = {}
        for drug_id, emb in zip(drug_ids, embeddings, strict=True):
            if emb is not None:
                molgnet_dict[drug_id] = emb

        with open(output_path, "wb") as f:
            pickle.dump(molgnet_dict, f)

        # Also save per-drug CSVs for DIPK compatibility
        dataset_dir = output_path.parent
        out_drugs_dir = dataset_dir / "DIPK_features" / "Drugs"
        os.makedirs(out_drugs_dir, exist_ok=True)

        for drug_id, emb in molgnet_dict.items():
            arr = emb.cpu().detach().numpy() if isinstance(emb, torch.Tensor) else np.array(emb)
            df_emb = pd.DataFrame(arr)
            out_csv = out_drugs_dir / f"MolGNet_{drug_id}.csv"
            df_emb.to_csv(out_csv, sep="\t", index=False)

    def load_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load pre-generated MolGNet embeddings from disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the embeddings
        :raises FileNotFoundError: If the embeddings file is not found
        """
        embeddings_file = Path(data_path) / dataset_name / self.get_output_filename()

        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"MolGNet embeddings file not found: {embeddings_file}. "
                f"Use load_or_generate() to automatically generate embeddings."
            )

        with open(embeddings_file, "rb") as f:
            molgnet_dict = pickle.load(f)  # noqa: S301

        feature_name = self.get_feature_name()
        features = {}

        for drug_id, emb in molgnet_dict.items():
            features[str(drug_id)] = {feature_name: emb}

        return FeatureDataset(features)


class MolGNetMixin:
    """Mixin that provides MolGNet drug embeddings loading for DRP models.

    This mixin implements load_drug_features using the MolGNetFeaturizer.
    It automatically generates embeddings if they don't exist.

    Class attributes that can be overridden:
        - molgnet_checkpoint_path: Path to MolGNet checkpoint (default: 'data/MolGNet.pt')
        - molgnet_device: Device for MolGNet model ('cpu', 'cuda', or 'auto')

    Example usage::

        from drevalpy.models.drp_model import DRPModel
        from drevalpy.datasets.featurizer.drug.molgnet import MolGNetMixin

        class MyModel(MolGNetMixin, DRPModel):
            drug_views = ["molgnet_embeddings"]
            ...
    """

    molgnet_checkpoint_path: str = "data/MolGNet.pt"
    molgnet_device: str = "auto"

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load MolGNet drug embeddings.

        Uses the MolGNetFeaturizer to load pre-generated embeddings or generate
        them automatically if they don't exist.

        :param data_path: Path to the data directory, e.g., 'data/'
        :param dataset_name: Name of the dataset, e.g., 'GDSC1'
        :returns: FeatureDataset containing the MolGNet embeddings
        """
        device = self.molgnet_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        featurizer = MolGNetFeaturizer(checkpoint_path=self.molgnet_checkpoint_path, device=device)
        return featurizer.load_or_generate(data_path, dataset_name)


def main():
    """Process drug SMILES and save MolGNet embeddings from command line."""
    parser = argparse.ArgumentParser(description="Generate MolGNet embeddings for drugs.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/MolGNet.pt",
        help="Path to MolGNet checkpoint (can be obtained from Zenodo: https://doi.org/10.5281/zenodo.12633909)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    args = parser.parse_args()

    featurizer = MolGNetFeaturizer(checkpoint_path=args.checkpoint, device=args.device)
    featurizer.generate_embeddings(args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()
