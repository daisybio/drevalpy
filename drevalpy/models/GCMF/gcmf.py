"""
GCMF family: graph-convolutional matrix factorization for drug-response prediction.

Predicts the cell-line x drug response matrix as ``R = U V^T`` (as in ordinary matrix
factorization), but the latent factors ``U`` (cell lines) and ``V`` (drugs) are not free
parameters: they are learned end-to-end by graph convolutions over feature-similarity or
prior-knowledge graphs. The relational variant uses several such graphs per side.

Idea
----
Combine learning in the primal and the dual space. A high-dimensional feature space
(the primal: gene expression, fingerprints, ...) is turned into a node-similarity graph
(the dual), which the model exploits without its complexity scaling with the feature
dimension. Each graph convolution *smooths* a node's embedding with those of its graph
neighbours, so similar cell lines (and similar drugs) are pulled toward shared latent
factors. This neighbourhood smoothing is the model's main inductive bias. The smoothed
factors are read out by the dot product ``U V^T``, plus learned per-cell and per-drug bias
terms (these matter: the drug identity carries most of the raw signal) and an optional
small MLP interaction head.

* cell lines form a k-NN graph from (gene-expression / multi-omics) similarity,
* drugs form a k-NN graph from Morgan-fingerprint (Tanimoto) similarity, or from
  prior-knowledge relations such as shared pathways / bioassays.

Variants
--------
* ``GCMF`` - base single-graph model (one cell graph, one drug graph).
* ``RGCMF`` - relational / multi-graph variant: several graphs per side (currently
  multi-omics cell graphs; pathway and bioassay drug relations) are fused by a relational
  graph convolution. Works best on leave-cell-line-out in our experiments.
* ``PGCMF`` / ``PRGCMF`` - probabilistic variants of GCMF / RGCMF that add a heteroscedastic
  Gaussian-NLL head, emitting a calibrated per-prediction aleatoric uncertainty (point
  accuracy on par with their deterministic counterparts).

All variants share a per-drug id embedding, an optional within-drug auxiliary loss, and an
N-model ensemble; see ``hyperparameters.yaml`` for the per-model configurations.
"""

import hashlib
import os
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.utils import DRUG_IDENTIFIER

from ..drp_model import DRPModel
from ..utils import (
    load_and_select_gene_features,
    load_drug_fingerprint_features,
    load_multi_cell_line_view,
    load_single_cell_line_view,
)


def _select_device() -> torch.device:
    """
    Pick CUDA, then Apple MPS, then CPU.

    :returns: the selected torch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _adj_to_numpy(adj: "torch.Tensor | list[torch.Tensor]") -> "np.ndarray | list[np.ndarray]":
    """
    Move an adjacency (single tensor or list of relation tensors) to numpy for saving.

    :param adj: a single adjacency tensor or a list of per-relation adjacency tensors
    :returns: the adjacency as a numpy array or list of numpy arrays
    """
    if isinstance(adj, list):
        return [a.cpu().numpy() for a in adj]
    return adj.cpu().numpy()


def _adj_from_numpy(adj: "np.ndarray | list", device: torch.device) -> "torch.Tensor | list[torch.Tensor]":
    """
    Inverse of ``_adj_to_numpy``: restore device tensor(s) from numpy.

    :param adj: a single numpy adjacency or a list of per-relation numpy adjacencies
    :param device: target torch device for the restored tensor(s)
    :returns: the adjacency as a device tensor or list of device tensors
    """
    if isinstance(adj, list):
        return [torch.tensor(a, device=device) for a in adj]
    return torch.tensor(adj, device=device)


# Bumped whenever a similarity kernel changes, so cached matrices from an older definition are
# not silently reused. v2: kendall maps tau with the fixed (tau + 1) / 2 instead of rescaling
# against the cohort's observed minimum.
_SIM_KERNEL_VERSION = 2


def _similarity_matrix(features: np.ndarray, metric: str) -> np.ndarray:
    """
    Compute a dense node-by-node similarity matrix with the requested kernel.

    Kernels:

    * ``cosine``   - cosine similarity (continuous features),
    * ``tanimoto`` / ``jaccard`` - Tanimoto/Jaccard over binarized features (``x > 0``);
      used for fingerprints and binary mutation profiles,
    * ``pearson``  - Pearson correlation between node feature profiles (methylation,
      gene expression),
    * ``kendall``  - Kendall's tau between profiles, mapped to [0, 1] as ``(tau + 1) / 2``
      (copy number).

    :param features: (n_nodes, n_feat) feature matrix
    :param metric: kernel name (see above)
    :returns: (n_nodes, n_nodes) similarity matrix as float64
    :raises ValueError: if the metric is unknown
    """
    x = features.astype(np.float64)
    if metric in ("tanimoto", "jaccard"):
        b = (x > 0).astype(np.float64)
        inter = b @ b.T
        sums = b.sum(axis=1)
        union = sums[:, None] + sums[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, inter / union, 0.0)
    if metric == "cosine":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        xn = x / norm
        return xn @ xn.T
    if metric == "pearson":
        with np.errstate(invalid="ignore", divide="ignore"):  # zero-variance rows -> NaN, handled below
            sim = np.corrcoef(x)
        return np.asarray(np.nan_to_num(sim, nan=0.0))
    if metric == "kendall":
        from scipy.stats import kendalltau

        n = x.shape[0]
        sim = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                tau = kendalltau(x[i], x[j])[0]
                tau = 0.0 if np.isnan(tau) else tau
                sim[i, j] = sim[j, i] = tau
        # map tau from [-1, 1] to [0, 1] with a fixed transform. Rescaling against the observed
        # minimum instead would make a pair's similarity depend on which other nodes are in the
        # cohort, so the same two cell lines would score differently in different CV splits.
        return (sim + 1.0) / 2.0
    raise ValueError(f"Unknown similarity metric: {metric!r}")


def _knn_normalize(sim: np.ndarray, k: int, use_edge_weights: bool) -> np.ndarray:
    """
    Sparsify a dense similarity matrix to a symmetric-normalized k-NN adjacency.

    For every node we keep its ``k`` most similar neighbours, symmetrize the graph (union),
    add self-loops, and apply the standard GCN normalization ``D^-1/2 (A+I) D^-1/2``. Nodes
    with no similarity to any other (e.g. drugs missing from a relation) end up with only a
    self-loop, which is exactly how missing entities are handled.

    :param sim: (n_nodes, n_nodes) similarity matrix
    :param k: number of neighbours per node
    :param use_edge_weights: if True weight edges by similarity, else binary edges
    :returns: dense (n_nodes, n_nodes) normalized adjacency as float32
    """
    n = sim.shape[0]
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)  # exclude self when picking neighbours
    k = int(min(k, n - 1)) if n > 1 else 0

    adj = np.zeros((n, n), dtype=np.float64)
    if k > 0:
        # indices of the k largest similarities per row
        nbr = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k)
        cols = nbr.reshape(-1)
        weights = sim[rows, cols] if use_edge_weights else np.ones_like(rows, dtype=np.float64)
        weights = np.clip(weights, 0.0, None)  # similarities can be slightly negative / -inf
        adj[rows, cols] = weights

    adj = np.maximum(adj, adj.T)  # symmetrize (union of neighbourhoods)
    adj = adj + np.eye(n, dtype=np.float64)  # self-loops
    deg = adj.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    adj = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return adj.astype(np.float32)


def _normalized_knn_adjacency(features: np.ndarray, k: int, metric: str, use_edge_weights: bool) -> np.ndarray:
    """
    Build a symmetric-normalized k-NN adjacency from node features (similarity + sparsify).

    :param features: (n_nodes, n_feat) feature matrix
    :param k: number of neighbours per node
    :param metric: similarity kernel (see ``_similarity_matrix``)
    :param use_edge_weights: if True weight edges by similarity, else binary edges
    :returns: dense (n_nodes, n_nodes) normalized adjacency as float32
    """
    return _knn_normalize(_similarity_matrix(features, metric), k, use_edge_weights)


class _DenseGraphConv(nn.Module):
    """A single dense graph-convolution layer: ``A_hat @ (X W) + b``."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Propagate features over the graph.

        :param x: (n_nodes, in_dim) node features
        :param adj: (n_nodes, n_nodes) normalized adjacency
        :returns: (n_nodes, out_dim) propagated features
        """
        return adj @ self.linear(x)


class _GraphEncoder(nn.Module):
    """
    Encode node features into embeddings via stacked dense graph convolutions.

    Uses residual (skip) connections between graph layers and a direct skip from the
    input projection to the output, so the graph smoothing does not wash out per-node
    signal.
    """

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([_DenseGraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.skip_proj = nn.Linear(hidden_dim, emb_dim)  # skip from input projection

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Return (n_nodes, emb_dim) embeddings.

        :param x: (n_nodes, in_dim) node features
        :param adj: (n_nodes, n_nodes) normalized adjacency
        :returns: (n_nodes, emb_dim) node embeddings
        """
        h0 = self.act(self.input_proj(x))
        h = self.dropout(h0)
        for layer, norm in zip(self.layers, self.norms):
            h_new = self.act(norm(layer(h, adj)))
            h = self.dropout(h_new) + h  # residual skip
        return self.output_proj(h) + self.skip_proj(h0)


class _RelationalDenseGraphConv(nn.Module):
    """
    Relational dense graph convolution (RGCN-style).

    Holds one linear map per relation and averages ``A_r @ (X W_r)`` over the relations,
    so several similarity graphs (e.g. fingerprint and ChemBERTa for drugs) are fused with
    relation-specific weights. With a single relation it reduces exactly to
    ``_DenseGraphConv``.
    """

    def __init__(
        self, in_dim: int, out_dim: int, n_relations: int, relation_attention: bool = False, root: bool = False
    ):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_relations)])
        # learnable per-relation softmax weights (init 0 -> uniform == plain mean), so the model
        # can downweight noisy relations (e.g. sparse perturbation graph)
        self.relation_attention = relation_attention
        if relation_attention:
            self.rel_logits = nn.Parameter(torch.zeros(n_relations))
        # optional RGCN-style root (relation-independent) self-transform
        self.root = nn.Linear(in_dim, out_dim) if root else None

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        """
        Propagate features over every relation graph and combine (mean or learned weights).

        :param x: (n_nodes, in_dim) node features
        :param adjs: one (n_nodes, n_nodes) normalized adjacency per relation
        :returns: (n_nodes, out_dim) combined propagated features
        """
        terms = [adjs[r] @ self.linears[r](x) for r in range(len(self.linears))]
        if self.relation_attention:
            w = torch.softmax(self.rel_logits, dim=0)
            out = sum(w[r] * terms[r] for r in range(len(terms)))
        else:
            out = terms[0]
            for r in range(1, len(terms)):
                out = out + terms[r]
            out = out / len(terms)
        if self.root is not None:
            out = out + self.root(x)
        return out


class _RelationalGraphEncoder(nn.Module):
    """Like ``_GraphEncoder``, but convolves over a list of relation graphs per layer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        n_layers: int,
        dropout: float,
        n_relations: int,
        relation_attention: bool = False,
        root: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                _RelationalDenseGraphConv(hidden_dim, hidden_dim, n_relations, relation_attention, root)
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.skip_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        """
        Return (n_nodes, emb_dim) embeddings, smoothing over all relation graphs.

        :param x: (n_nodes, in_dim) node features
        :param adjs: one (n_nodes, n_nodes) normalized adjacency per relation
        :returns: (n_nodes, emb_dim) node embeddings
        """
        h0 = self.act(self.input_proj(x))
        h = self.dropout(h0)
        for layer, norm in zip(self.layers, self.norms):
            h_new = self.act(norm(layer(h, adjs)))
            h = self.dropout(h_new) + h  # residual skip
        return self.output_proj(h) + self.skip_proj(h0)


class _GCMFNet(nn.Module):
    """The full two-tower model: cell encoder + drug encoder + factorization head."""

    def __init__(
        self,
        cell_in_dim: int,
        drug_in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        n_layers: int,
        dropout: float,
        use_mlp_head: bool,
        mlp_hidden: int,
        n_mlp_layers: int = 1,
        head_raw_features: bool = False,
        n_drugs: int = 0,
        use_drug_id_embedding: bool = False,
        probabilistic: bool = False,
    ):
        super().__init__()
        # subclasses (``_RGCMFNet``) swap these for relational encoders, hence the union type
        self.cell_encoder: "_GraphEncoder | _RelationalGraphEncoder" = _GraphEncoder(
            cell_in_dim, hidden_dim, emb_dim, n_layers, dropout
        )
        self.drug_encoder: "_GraphEncoder | _RelationalGraphEncoder" = _GraphEncoder(
            drug_in_dim, hidden_dim, emb_dim, n_layers, dropout
        )

        # Free per-drug embedding: every drug is seen in LCO training, so an id-based
        # latent vector (added to the fingerprint-derived one) can capture drug-specific
        # response patterns that fingerprints only approximate. Cells get no such table
        # (they are unseen at test time and must stay purely feature-derived).
        self.use_drug_id_embedding = use_drug_id_embedding
        if use_drug_id_embedding:
            self.drug_id_emb = nn.Embedding(n_drugs, emb_dim)
            nn.init.normal_(self.drug_id_emb.weight, std=0.01)

        self.cell_bias = nn.Linear(emb_dim, 1)
        self.drug_bias = nn.Linear(emb_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dot_scale = nn.Parameter(torch.ones(1))

        self.use_mlp_head = use_mlp_head
        self.head_raw_features = head_raw_features
        in_dim = 3 * emb_dim + (cell_in_dim + drug_in_dim if head_raw_features else 0)
        if use_mlp_head:
            blocks: list[nn.Module] = []
            d = in_dim
            for _ in range(max(1, n_mlp_layers)):
                blocks += [nn.Linear(d, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = mlp_hidden
            blocks.append(nn.Linear(d, 1))
            self.mlp = nn.Sequential(*blocks)

        # Probabilistic (heteroscedastic) head: predicts log-variance per pair, trained with
        # Gaussian NLL. The "loss attenuation" can sharpen the mean by down-weighting noisy pairs.
        self.probabilistic = probabilistic
        if probabilistic:
            self.var_head = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
            )

    def encode(
        self,
        x_cell: torch.Tensor,
        adj_cell: "torch.Tensor | list[torch.Tensor]",
        x_drug: torch.Tensor,
        adj_drug: "torch.Tensor | list[torch.Tensor]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for every cell line and every drug.

        :param x_cell: (n_cells, cell_in_dim) cell node features
        :param adj_cell: normalized cell adjacency (single tensor, or list of relations in subclasses)
        :param x_drug: (n_drugs, drug_in_dim) drug node features
        :param adj_drug: normalized drug adjacency (single tensor, or list of relations in subclasses)
        :returns: (cell embeddings, drug embeddings)
        """
        z_cell = self.cell_encoder(x_cell, adj_cell)
        z_drug = self.drug_encoder(x_drug, adj_drug)
        if self.use_drug_id_embedding:
            z_drug = z_drug + self.drug_id_emb.weight
        return z_cell, z_drug

    def score_pairs(
        self,
        z_cell: torch.Tensor,
        z_drug: torch.Tensor,
        raw_cell: torch.Tensor | None = None,
        raw_drug: torch.Tensor | None = None,
        return_log_var: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Score a batch of (cell, drug) pairs given their gathered embeddings.

        :param z_cell: (batch, emb_dim) cell embeddings
        :param z_drug: (batch, emb_dim) drug embeddings
        :param raw_cell: (batch, cell_in_dim) raw cell features, used iff head_raw_features
        :param raw_drug: (batch, drug_in_dim) raw drug features, used iff head_raw_features
        :param return_log_var: if True (probabilistic), also return predicted log-variance
        :returns: (batch,) predicted means, or (means, log_var) if return_log_var
        """
        dot = (z_cell * z_drug).sum(dim=-1, keepdim=True) * self.dot_scale
        pred = dot + self.cell_bias(z_cell) + self.drug_bias(z_drug) + self.global_bias
        feat = torch.cat(
            [z_cell, z_drug, z_cell * z_drug]
            + (
                [raw_cell, raw_drug]
                if (self.head_raw_features and raw_cell is not None and raw_drug is not None)
                else []
            ),
            dim=-1,
        )
        if self.use_mlp_head:
            pred = pred + self.mlp(feat)
        pred = pred.squeeze(-1)
        if return_log_var and self.probabilistic:
            return pred, self.var_head(feat).squeeze(-1)
        return pred


class _RGCMFNet(_GCMFNet):
    """
    Two-tower net whose encoders convolve over several relation graphs per tower.

    Reuses the whole ``_GCMFNet`` head (dot product, biases, drug-id embedding, MLP and
    optional probabilistic head); only the two graph encoders are swapped for relational ones.
    """

    def __init__(
        self,
        *,
        n_cell_relations: int,
        n_drug_relations: int,
        relation_attention: bool = False,
        gnn_root: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cell_encoder = _RelationalGraphEncoder(
            kwargs["cell_in_dim"],
            kwargs["hidden_dim"],
            kwargs["emb_dim"],
            kwargs["n_layers"],
            kwargs["dropout"],
            n_cell_relations,
            relation_attention,
            gnn_root,
        )
        self.drug_encoder = _RelationalGraphEncoder(
            kwargs["drug_in_dim"],
            kwargs["hidden_dim"],
            kwargs["emb_dim"],
            kwargs["n_layers"],
            kwargs["dropout"],
            n_drug_relations,
            relation_attention,
            gnn_root,
        )

    def encode(
        self,
        x_cell: torch.Tensor,
        adj_cell: "torch.Tensor | list[torch.Tensor]",
        x_drug: torch.Tensor,
        adj_drug: "torch.Tensor | list[torch.Tensor]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings, smoothing each tower over its list of relation graphs.

        :param x_cell: (n_cells, cell_in_dim) cell node features
        :param adj_cell: one normalized cell adjacency per cell relation (a list at runtime)
        :param x_drug: (n_drugs, drug_in_dim) drug node features
        :param adj_drug: one normalized drug adjacency per drug relation (a list at runtime)
        :returns: (cell embeddings, drug embeddings)
        """
        z_cell = self.cell_encoder(x_cell, adj_cell)
        z_drug = self.drug_encoder(x_drug, adj_drug)
        if self.use_drug_id_embedding:
            z_drug = z_drug + self.drug_id_emb.weight
        return z_cell, z_drug


class GCMF(DRPModel):
    """Graph Convolutional Matrix Factorization drug-response model."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the model. The network is built lazily in ``train``."""
        super().__init__()
        self.nets: list[_GCMFNet] = []
        self.hyperparameters: dict[str, Any] = {}
        self.device = _select_device()

        # populated during train(), reused at predict()
        self._cell_id_to_idx: dict[str, int] = {}
        self._drug_id_to_idx: dict[str, int] = {}
        self._x_cell: torch.Tensor | None = None
        # a single tensor in GCMF, a list of per-relation tensors in RGCMF
        self._adj_cell: "torch.Tensor | list[torch.Tensor] | None" = None
        self._x_drug: torch.Tensor | None = None
        self._adj_drug: "torch.Tensor | list[torch.Tensor] | None" = None
        self._cell_in_dim: int = 0
        self._drug_in_dim: int = 0
        self._n_drugs: int = 0
        self._n_cell_relations: int = 1
        self._n_drug_relations: int = 1
        self._scalers: dict[str, StandardScaler] = {}
        self._train_cell_ids: np.ndarray = np.array([])
        self.training_mean: float = 0.0

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "GCMF"."""
        return "GCMF"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store hyperparameters and resolve the requested feature views.

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)
        cl_views = hyperparameters.get("cell_line_views", ["gene_expression"])
        dr_views = hyperparameters.get("drug_views", ["fingerprints"])
        self.cell_line_views = cl_views if isinstance(cl_views, list) else [cl_views]
        self.drug_views = dr_views if isinstance(dr_views, list) else [dr_views]
        if int(hyperparameters.get("seed", 0)) >= 0:
            torch.manual_seed(int(hyperparameters.get("seed", 0)))
            np.random.seed(int(hyperparameters.get("seed", 0)))

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the requested cell-line views (gene expression and/or other omics).

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with one or several cell-line views
        """
        if self.cell_line_views == ["gene_expression"]:
            # configurable gene list (default landmark_genes_reduced, like the sklearn baselines)
            gene_list = self.hyperparameters.get("gene_list", "landmark_genes_reduced")
            return load_and_select_gene_features(
                feature_type="gene_expression",
                gene_list=gene_list,
                data_path=data_path,
                dataset_name=dataset_name,
            )
        if len(self.cell_line_views) == 1:
            return load_single_cell_line_view(
                cell_line_views=self.cell_line_views,
                data_path=data_path,
                dataset_name=dataset_name,
                model_name=self.get_model_name(),
            )
        return load_multi_cell_line_view(
            cell_line_views=self.cell_line_views,
            data_path=data_path,
            dataset_name=dataset_name,
            model_name=self.get_model_name(),
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load drug Morgan fingerprints.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "fingerprints" view
        """
        n_bits = int(self.hyperparameters.get("n_bits", 128))
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True, n_bits=n_bits)

    def _build_cell_matrix(self, cell_line_input: FeatureDataset, cell_ids: np.ndarray, training: bool) -> np.ndarray:
        """
        Concatenate (and scale) the requested cell-line views into one matrix.

        Gene expression is transformed (``feature_transform`` hp: ``arcsinh`` default, or
        ``rank`` = per-gene rank across cells in [0, 1], which resolves cell-specific signal
        better); every continuous view is standardized with a scaler fit on training cells only.

        :param cell_line_input: cell-line FeatureDataset
        :param cell_ids: ordered cell-line ids (all cell lines)
        :param training: whether to fit the scalers (train) or reuse them (predict)
        :returns: (n_cells, total_feat) scaled feature matrix
        """
        train_ids = np.unique(self._train_cell_ids)
        train_mask = np.isin(cell_ids, train_ids)
        transform = str(self.hyperparameters.get("feature_transform", "arcsinh"))
        blocks = []
        for view in self.cell_line_views:
            mat = cell_line_input.get_feature_matrix(view=view, identifiers=cell_ids).astype(np.float64)
            if view == "gene_expression":
                if transform == "rank":  # per-gene rank across cells -> [0,1] (transductive feature transform)
                    mat = mat.argsort(axis=0).argsort(axis=0) / max(1, mat.shape[0] - 1)
                else:
                    mat = np.arcsinh(mat)
            if training:
                scaler = StandardScaler()
                scaler.fit(mat[train_mask])
                self._scalers[view] = scaler
            mat = self._scalers[view].transform(mat)
            blocks.append(mat)
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Build the feature-similarity graphs and train the network on (cell, drug) pairs.

        :param output: training responses
        :param cell_line_input: cell-line features (all cell lines)
        :param drug_input: drug features (all drugs)
        :param output_earlystopping: optional early-stopping responses
        :param model_checkpoint_dir: unused (kept for interface compatibility)
        :raises ValueError: if drug_input is None
        """
        if drug_input is None:
            raise ValueError("GCMF requires drug features (fingerprints).")

        hp = self.hyperparameters
        self.training_mean = float(np.nanmean(output.response))
        self._train_cell_ids = np.asarray(output.cell_line_ids)

        # node sets (transductive over features only, like SRMF)
        cell_ids = np.unique(cell_line_input.identifiers)
        drug_ids = np.unique(drug_input.identifiers)
        self._cell_id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
        self._drug_id_to_idx = {did: i for i, did in enumerate(drug_ids)}

        x_cell = self._build_cell_matrix(cell_line_input, cell_ids, training=True)
        x_drug = drug_input.get_feature_matrix(view="fingerprints", identifiers=drug_ids).astype(np.float32)

        self._x_cell = torch.tensor(x_cell, device=self.device)
        self._x_drug = torch.tensor(x_drug, device=self.device)
        self._cell_in_dim = x_cell.shape[1]
        self._drug_in_dim = x_drug.shape[1]
        self._n_drugs = x_drug.shape[0]

        # similarity graphs: a single relation in GCMF, several in RGCMF
        self._adj_cell = self._build_cell_adj(x_cell, cell_line_input, cell_ids, hp)
        self._adj_drug = self._build_drug_adj(x_drug, drug_input, drug_ids, hp)

        # training pairs
        ci, di, y = self._pairs_to_tensors(output)
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            val = self._pairs_to_tensors(output_earlystopping)
        else:  # carve a 10% validation set out of training
            n = len(y)
            perm = torch.randperm(n, device=self.device)
            n_val = max(1, int(0.1 * n))
            val = (ci[perm[:n_val]], di[perm[:n_val]], y[perm[:n_val]])
            ci, di, y = ci[perm[n_val:]], di[perm[n_val:]], y[perm[n_val:]]

        # train an ensemble of networks (n_ensemble=1 -> single model)
        n_ensemble = int(hp.get("n_ensemble", 1))
        self.nets = []
        for member in range(n_ensemble):
            torch.manual_seed(int(hp.get("seed", 0)) + member)
            net = self._build_net().to(self.device)
            self._train_net(net, ci, di, y, val)
            self.nets.append(net)

    def _net_kwargs(self) -> dict[str, Any]:
        """
        Collect the ``_GCMFNet`` constructor arguments from dimensions + hyperparameters.

        :returns: keyword arguments for the ``_GCMFNet`` constructor
        """
        hp = self.hyperparameters
        return dict(
            cell_in_dim=self._cell_in_dim,
            drug_in_dim=self._drug_in_dim,
            hidden_dim=int(hp.get("hidden_dim", 128)),
            emb_dim=int(hp.get("emb_dim", 64)),
            n_layers=int(hp.get("n_gnn_layers", 2)),
            dropout=float(hp.get("dropout", 0.3)),
            use_mlp_head=bool(hp.get("use_mlp_head", True)),
            mlp_hidden=int(hp.get("mlp_hidden", 64)),
            n_mlp_layers=int(hp.get("n_mlp_layers", 1)),
            head_raw_features=bool(hp.get("head_raw_features", False)),
            n_drugs=self._n_drugs,
            use_drug_id_embedding=bool(hp.get("use_drug_id_embedding", False)),
            probabilistic=bool(hp.get("probabilistic", False)),
        )

    def _build_net(self) -> _GCMFNet:
        """
        Instantiate a fresh network from the stored dimensions and hyperparameters.

        :returns: a new ``_GCMFNet`` instance
        """
        return _GCMFNet(**self._net_kwargs())

    def _build_cell_adj(
        self, x_cell: np.ndarray, cell_line_input: FeatureDataset, cell_ids: np.ndarray, hp: dict[str, Any]
    ) -> "torch.Tensor | list[torch.Tensor]":
        """
        Build the cell-line similarity graph(s).

        GCMF uses a single cosine k-NN graph over the (fused) cell features. Subclasses may
        return a list of relation graphs instead.

        :param x_cell: (n_cells, cell_in_dim) fused cell feature matrix
        :param cell_line_input: cell-line FeatureDataset (used by subclasses)
        :param cell_ids: ordered cell-line ids
        :param hp: hyperparameter dictionary
        :returns: a single normalized cell adjacency (or a list of them in subclasses)
        """
        adj = _normalized_knn_adjacency(
            x_cell, int(hp.get("k_cell", 15)), "cosine", bool(hp.get("use_edge_weights", True))
        )
        return torch.tensor(adj, device=self.device)

    def _build_drug_adj(
        self, x_drug: np.ndarray, drug_input: FeatureDataset, drug_ids: np.ndarray, hp: dict[str, Any]
    ) -> "torch.Tensor | list[torch.Tensor]":
        """
        Build the drug similarity graph(s).

        GCMF uses a single Tanimoto k-NN graph over the fingerprints. Subclasses may return a
        list of relation graphs instead.

        :param x_drug: (n_drugs, drug_in_dim) fingerprint feature matrix
        :param drug_input: drug FeatureDataset (used by subclasses)
        :param drug_ids: ordered drug ids
        :param hp: hyperparameter dictionary
        :returns: a single normalized drug adjacency (or a list of them in subclasses)
        """
        adj = _normalized_knn_adjacency(
            x_drug, int(hp.get("k_drug", 15)), "tanimoto", bool(hp.get("use_edge_weights", True))
        )
        return torch.tensor(adj, device=self.device)

    def _train_net(
        self,
        net: _GCMFNet,
        ci: torch.Tensor,
        di: torch.Tensor,
        y: torch.Tensor,
        val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Train a single network in place with early stopping on ``val``.

        :param net: the network to train (modified in place)
        :param ci: (n_train,) cell indices of the training pairs
        :param di: (n_train,) drug indices of the training pairs
        :param y: (n_train,) target responses of the training pairs
        :param val: (cell idx, drug idx, response) tensors of the validation set
        """
        # train() always populates the feature/graph tensors before calling _train_net
        x_cell = cast(torch.Tensor, self._x_cell)
        x_drug = cast(torch.Tensor, self._x_drug)
        adj_cell = cast("torch.Tensor | list[torch.Tensor]", self._adj_cell)
        adj_drug = cast("torch.Tensor | list[torch.Tensor]", self._adj_drug)
        hp = self.hyperparameters
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(hp.get("learning_rate", 1e-3)),
            weight_decay=float(hp.get("weight_decay", 1e-5)),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(hp.get("batch_size", 8192))
        max_epochs = int(hp.get("max_epochs", 300))
        patience = int(hp.get("patience", 15))
        emb_l2 = float(hp.get("emb_l2", 0.0))
        within_drug_weight = float(hp.get("within_drug_weight", 0.0))
        raw = net.head_raw_features

        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_no_improve = 0
        n_train = len(y)

        for _epoch in range(max_epochs):
            net.train()
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                idx = perm[start:end]
                bci, bdi = ci[idx], di[idx]
                optimizer.zero_grad()
                z_cell, z_drug = net.encode(x_cell, adj_cell, x_drug, adj_drug)
                rc = x_cell[bci] if raw else None
                rd = x_drug[bdi] if raw else None
                if net.probabilistic:
                    preds, log_var = net.score_pairs(z_cell[bci], z_drug[bdi], rc, rd, return_log_var=True)
                    var = torch.exp(log_var).clamp(min=1e-6)
                    loss = nn.functional.gaussian_nll_loss(preds, y[idx], var, eps=1e-6)
                else:
                    preds = net.score_pairs(z_cell[bci], z_drug[bdi], rc, rd)
                    loss = loss_fn(preds, y[idx])
                if within_drug_weight > 0:
                    # auxiliary loss on drug-mean-centered residuals: directly optimizes the
                    # within-drug ranking of cell lines (the hard signal in leave-cell-line-out)
                    bd = di[idx]
                    uniq, inv = torch.unique(bd, return_inverse=True)
                    cnt = torch.zeros(uniq.numel(), device=self.device).index_add(0, inv, torch.ones_like(preds))
                    pmean = torch.zeros(uniq.numel(), device=self.device).index_add(0, inv, preds) / cnt
                    ymean = torch.zeros(uniq.numel(), device=self.device).index_add(0, inv, y[idx]) / cnt
                    keep = cnt[inv] >= 2
                    if keep.any():
                        pres = (preds - pmean[inv])[keep]
                        yres = (y[idx] - ymean[inv])[keep]
                        loss = loss + within_drug_weight * loss_fn(pres, yres)
                if emb_l2 > 0:
                    loss = loss + emb_l2 * (z_cell.pow(2).mean() + z_drug.pow(2).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

            val_mse = self._eval_mse(net, val)
            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)

    def _pairs_to_tensors(self, data: DrugResponseDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map a response dataset to (cell_idx, drug_idx, response) tensors on device.

        :param data: response dataset to convert (pairs with unknown ids are dropped)
        :returns: (cell idx, drug idx, response) tensors on the model device
        """
        ci, di, y = [], [], []
        for cl, dr, resp in zip(data.cell_line_ids, data.drug_ids, data.response):
            if cl in self._cell_id_to_idx and dr in self._drug_id_to_idx:
                ci.append(self._cell_id_to_idx[cl])
                di.append(self._drug_id_to_idx[dr])
                y.append(resp)
        ci_t = torch.tensor(ci, dtype=torch.long, device=self.device)
        di_t = torch.tensor(di, dtype=torch.long, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        return ci_t, di_t, y_t

    @torch.no_grad()
    def _eval_mse(self, net: _GCMFNet, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Compute MSE of a single net on a validation pair set.

        :param net: the network to evaluate
        :param val: (cell idx, drug idx, response) tensors of the validation set
        :returns: mean squared error on ``val`` (inf if empty)
        """
        net.eval()
        ci, di, y = val
        if len(y) == 0:
            return float("inf")
        # the feature/graph tensors are set in train() before any evaluation
        x_cell = cast(torch.Tensor, self._x_cell)
        x_drug = cast(torch.Tensor, self._x_drug)
        adj_cell = cast("torch.Tensor | list[torch.Tensor]", self._adj_cell)
        adj_drug = cast("torch.Tensor | list[torch.Tensor]", self._adj_drug)
        raw = net.head_raw_features
        z_cell, z_drug = net.encode(x_cell, adj_cell, x_drug, adj_drug)
        # not return_log_var here, so score_pairs returns a single Tensor
        preds = cast(
            torch.Tensor,
            net.score_pairs(z_cell[ci], z_drug[di], x_cell[ci] if raw else None, x_drug[di] if raw else None),
        )
        return float(nn.functional.mse_loss(preds, y).item())

    @torch.no_grad()
    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict responses for (cell, drug) pairs.

        Cell lines / drugs unseen at training (no features) fall back to the training mean.

        :param cell_line_ids: cell-line ids to predict
        :param drug_ids: drug ids to predict
        :param cell_line_input: cell-line features (unused; graphs are cached from train)
        :param drug_input: drug features (unused; graphs are cached from train)
        :returns: (n,) predicted responses
        """
        if not self.nets:
            return np.full(len(cell_line_ids), self.training_mean, dtype=np.float32)

        preds = np.full(len(cell_line_ids), self.training_mean, dtype=np.float32)
        valid_rows, ci, di = [], [], []
        for row, (cl, dr) in enumerate(zip(cell_line_ids, drug_ids)):
            if cl in self._cell_id_to_idx and dr in self._drug_id_to_idx:
                valid_rows.append(row)
                ci.append(self._cell_id_to_idx[cl])
                di.append(self._drug_id_to_idx[dr])
        if valid_rows:
            # self.nets is non-empty here, so train() ran and the feature/graph tensors are set
            x_cell = cast(torch.Tensor, self._x_cell)
            x_drug = cast(torch.Tensor, self._x_drug)
            adj_cell = cast("torch.Tensor | list[torch.Tensor]", self._adj_cell)
            adj_drug = cast("torch.Tensor | list[torch.Tensor]", self._adj_drug)
            ci_t = torch.tensor(ci, dtype=torch.long, device=self.device)
            di_t = torch.tensor(di, dtype=torch.long, device=self.device)
            member_preds = []
            for net in self.nets:
                net.eval()
                raw = net.head_raw_features
                z_cell, z_drug = net.encode(x_cell, adj_cell, x_drug, adj_drug)
                # not return_log_var here, so score_pairs returns a single Tensor
                member_pred = cast(
                    torch.Tensor,
                    net.score_pairs(
                        z_cell[ci_t],
                        z_drug[di_t],
                        x_cell[ci_t] if raw else None,
                        x_drug[di_t] if raw else None,
                    ),
                )
                member_preds.append(member_pred.cpu().numpy())
            preds[np.asarray(valid_rows)] = np.mean(member_preds, axis=0)
        return preds

    def save(self, directory: str) -> None:
        """
        Persist the trained model.

        :param directory: target directory
        :raises RuntimeError: if there is nothing to save
        """
        if not self.nets:
            raise RuntimeError("No trained model to save.")
        # a trained model (self.nets non-empty) always has its feature/graph tensors set
        x_cell = cast(torch.Tensor, self._x_cell)
        x_drug = cast(torch.Tensor, self._x_drug)
        adj_cell = cast("torch.Tensor | list[torch.Tensor]", self._adj_cell)
        adj_drug = cast("torch.Tensor | list[torch.Tensor]", self._adj_drug)
        os.makedirs(directory, exist_ok=True)
        torch.save([net.state_dict() for net in self.nets], os.path.join(directory, "nets.pt"))  # noqa: S614
        joblib.dump(
            {
                "hyperparameters": self.hyperparameters,
                "cell_id_to_idx": self._cell_id_to_idx,
                "drug_id_to_idx": self._drug_id_to_idx,
                "x_cell": x_cell.cpu().numpy(),
                "adj_cell": _adj_to_numpy(adj_cell),
                "x_drug": x_drug.cpu().numpy(),
                "adj_drug": _adj_to_numpy(adj_drug),
                "training_mean": self.training_mean,
                "cell_line_views": self.cell_line_views,
                "drug_views": self.drug_views,
            },
            os.path.join(directory, "state.pkl"),
        )

    @classmethod
    def load(cls, directory: str) -> "GCMF":
        """
        Load a model saved with ``save``.

        :param directory: directory containing the saved files
        :returns: a restored GCMF instance
        """
        instance = cls()
        state = joblib.load(os.path.join(directory, "state.pkl"))
        instance.build_model(state["hyperparameters"])
        instance._cell_id_to_idx = state["cell_id_to_idx"]
        instance._drug_id_to_idx = state["drug_id_to_idx"]
        instance.training_mean = state["training_mean"]
        instance.cell_line_views = state["cell_line_views"]
        instance.drug_views = state["drug_views"]
        instance._x_cell = torch.tensor(state["x_cell"], device=instance.device)
        instance._adj_cell = _adj_from_numpy(state["adj_cell"], instance.device)
        instance._x_drug = torch.tensor(state["x_drug"], device=instance.device)
        instance._adj_drug = _adj_from_numpy(state["adj_drug"], instance.device)
        instance._cell_in_dim = instance._x_cell.shape[1]
        instance._drug_in_dim = instance._x_drug.shape[1]
        instance._n_drugs = instance._x_drug.shape[0]
        instance._n_cell_relations = len(instance._adj_cell) if isinstance(instance._adj_cell, list) else 1
        instance._n_drug_relations = len(instance._adj_drug) if isinstance(instance._adj_drug, list) else 1
        state_dicts = torch.load(os.path.join(directory, "nets.pt"))  # noqa: S614
        instance.nets = []
        for sd in state_dicts:
            net = instance._build_net().to(instance.device)
            net.load_state_dict(sd)
            net.eval()
            instance.nets.append(net)
        return instance


class PGCMF(GCMF):
    """
    Probabilistic GCMF.

    Identical architecture to ``GCMF``, but trained with a heteroscedastic Gaussian
    negative-log-likelihood head (``probabilistic=True``): a second head predicts a
    per-pair log-variance, so the model exposes a calibrated uncertainty for every
    prediction. Point accuracy is on par with the deterministic GCMF on CTRPv2 (within
    fold noise); choose PGCMF when you also want predictive uncertainty.
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "PGCMF"."""
        return "PGCMF"

    def build_model(self, hyperparameters: dict) -> None:
        """
        Build the model, forcing the probabilistic (Gaussian-NLL) head on.

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        super().build_model({**hyperparameters, "probabilistic": True})


class RGCMF(GCMF):
    """
    Relational GCMF: each tower convolves over several graphs per side.

    Instead of the single cell / single drug graph of ``GCMF``, each tower fuses several
    similarity or prior-knowledge graphs with a relational graph convolution (one weight per
    relation, averaged; RGCN-style). Node features are unchanged from ``GCMF`` (gene expression
    for cells, Morgan fingerprints for drugs) - the relations only add edge structure.

    **Cell-line relations** are computed on this dataset's own omics:

    * ``gene_expression`` - Pearson correlation,
    * ``methylation`` - Pearson correlation,
    * ``mutations`` - Jaccard over binary mutation profiles,
    * ``copy_number_variation_gistic`` - Kendall's tau, mapped to [0, 1] as ``(tau + 1) / 2``.

    **Drug relations** are biologically-informed graphs joined onto this dataset's drugs on
    ``pubchem_id``:

    * ``drug_pathways`` - Jaccard over the KEGG/Reactome pathways the drug's targets belong to
      (broad mechanism-of-action similarity),
    * ``drug_bioassay`` - Jaccard over PubChem BioAssays in which the drug is an active hit
      (a biological-activity fingerprint independent of 2D structure).

    Their resources are downloaded with the dataset's meta bundle into
    ``<data_path>/meta/gcmf_drug_relations/``. Further relations (``drug_targets``,
    ``string_targets``, ``drug_perturbation``) are optional: add the CSV there to enable one.

    Drugs / cell lines absent from a relation simply get a self-loop in that relation (handled
    by ``_knn_normalize``), so partial coverage is fine - they keep their fingerprint /
    gene-expression node features and the free per-drug id embedding, just no graph neighbours
    in that relation. A configured relation whose resource is missing raises an error rather than
    being silently skipped; use ``GCMF`` if you want the single-graph model.
    Relation sets are configurable via the ``cell_relation_views`` / ``drug_relation_views``
    hyperparameters; dense cell similarities are cached under ``<dataset>/gcmf_cache/``
    (the Kendall CNV kernel is slow to recompute).
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    # similarity kernel per cell-line omics view (built from this dataset's own omics)
    _CELL_KERNELS = {
        "gene_expression": "pearson",
        "methylation": "pearson",
        "mutations": "jaccard",
        "copy_number_variation_gistic": "kendall",
    }
    # gene list used to reduce each omics view before computing its similarity (None = no reduction)
    _CELL_GENE_LISTS = {
        "gene_expression": "landmark_genes",
        "methylation": "methylation_intersection",
        "mutations": "drug_target_genes_all_drugs",
        "copy_number_variation_gistic": "drug_target_genes_all_drugs",
    }
    # drug-relation resources live under <data_path>/meta/<_DRUG_SIM_DIR>/ (downloaded with the
    # dataset's meta bundle); a configured relation whose resource is absent raises an error.
    _DRUG_SIM_DIR = "gcmf_drug_relations"
    # how each drug relation is built: "matrix" = precomputed pubchem-indexed similarity CSV;
    # "targets" = per-drug (drug_name, feature) long table -> feature-set Jaccard at load time
    _DRUG_RELATION_KIND = {
        "drug_pathways": "targets",  # drugs sharing KEGG/Reactome pathways (via their targets)
        "drug_bioassay": "targets",  # drugs co-active in the same PubChem high-throughput screens
        "drug_targets": "targets",  # drug-target Jaccard
        "string_targets": "matrix",  # precomputed STRING drug-target similarity (GDSC drugs only)
        "drug_perturbation": "matrix",  # precomputed perturbation-signature similarity (GDSC drugs only)
    }

    def __init__(self) -> None:
        """Initialize the model; relation sets are resolved in ``build_model``."""
        super().__init__()
        self.cell_relation_views: list[str] = [
            "gene_expression",
            "methylation",
            "mutations",
            "copy_number_variation_gistic",
        ]
        self.drug_relation_views: list[str] = ["drug_pathways", "drug_bioassay"]
        # view -> (ids, dense similarity matrix), precomputed once in the loaders
        self._cell_sims: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._drug_sims: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "RGCMF"."""
        return "RGCMF"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Resolve the per-tower relation sets (node-feature views stay gene-expr / fingerprints).

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        super().build_model(hyperparameters)
        crv = hyperparameters.get("cell_relation_views", self.cell_relation_views)
        drv = hyperparameters.get("drug_relation_views", self.drug_relation_views)
        self.cell_relation_views = crv if isinstance(crv, list) else [crv]
        self.drug_relation_views = drv if isinstance(drv, list) else [drv]

    @staticmethod
    def _relation_key(table: pd.DataFrame, path: str) -> pd.Series:
        """
        Return the ``pubchem_id`` join key of a drug-relation table.

        Relations are joined on ``pubchem_id``, the same identifier the datasets use for drugs,
        so no drug-name matching is involved.

        :param table: the loaded relation table
        :param path: path the table was read from (for the error message)
        :returns: the ``pubchem_id`` column as string
        :raises ValueError: if the table has no ``pubchem_id`` column
        """
        if DRUG_IDENTIFIER not in table.columns:
            raise ValueError(
                f"drug-relation table {path} has no '{DRUG_IDENTIFIER}' column. Relations are joined on "
                f"{DRUG_IDENTIFIER}; re-download the meta bundle to get the current tables."
            )
        return table[DRUG_IDENTIFIER].astype(str)

    @classmethod
    def _drug_resource_path(cls, view: str, data_path: str) -> str | None:
        """
        Locate a drug-relation resource under ``<data_path>/meta/<_DRUG_SIM_DIR>/``.

        Both gzip-compressed (``<view>.csv.gz``) and plain (``<view>.csv``) files are accepted;
        pandas reads either transparently.

        :param view: relation view name (resource file is ``<view>.csv[.gz]``)
        :param data_path: data directory
        :returns: path to the resource, or None if it does not exist
        """
        directory = os.path.join(data_path, "meta", cls._DRUG_SIM_DIR)
        for ext in (".csv.gz", ".csv"):
            path = os.path.join(directory, f"{view}{ext}")
            if os.path.exists(path):
                return path
        return None

    def _relation_similarity(
        self, data_path: str, dataset_name: str, view: str, kernel: str, ids: np.ndarray, node_fd: FeatureDataset
    ) -> np.ndarray:
        """
        Return the dense cell-relation similarity, loading from cache when possible.

        The cache is keyed by (view, kernel, kernel version, gene list, cell-id set), so a hit
        skips loading the (large) omics CSV entirely - important for the 10-fold benchmark. The
        key does not cover the *contents* of the omics table or gene list, so delete
        ``<dataset>/gcmf_cache/`` to force recomputation if those change under the same name.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :param view: cell-relation omics view name
        :param kernel: similarity kernel for this view
        :param ids: ordered cell-line ids the similarity is computed over
        :param node_fd: node-feature dataset (source of the gene-expression relation)
        :returns: dense (n_cells, n_cells) similarity matrix
        """
        cache_dir = os.path.join(data_path, dataset_name, "gcmf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        # the gene_expression relation is built from the node features (hp gene_list), so its
        # cache key must reflect that list, not the hardcoded default (else gene-set sweeps collide)
        if view == "gene_expression":
            gene_list = str(self.hyperparameters.get("gene_list", "landmark_genes"))
        else:
            gene_list = str(self._CELL_GENE_LISTS.get(view, ""))
        # non-cryptographic: SHA1 only forms a stable cache key over the cell-id set
        id_sig = hashlib.sha1(",".join(sorted(str(c) for c in ids)).encode(), usedforsecurity=False).hexdigest()[:16]
        path = os.path.join(cache_dir, f"cell_{view}_{kernel}v{_SIM_KERNEL_VERSION}_{gene_list}_{id_sig}.npy")
        if os.path.exists(path):
            return np.load(path)
        if view == "gene_expression":
            feats = node_fd.get_feature_matrix(view="gene_expression", identifiers=ids).astype(np.float64)
        else:
            feats = self._load_omics_matrix(data_path, dataset_name, view, ids)
        sim = _similarity_matrix(feats, kernel)
        np.save(path, sim)
        return sim

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load gene-expression node features and precompute each cell-relation similarity graph.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the gene-expression node features
        """
        node_fd = super().load_cell_line_features(data_path, dataset_name)
        ids = node_fd.identifiers
        self._cell_sims = {}
        for view in self.cell_relation_views:
            kernel = self._CELL_KERNELS.get(view, "pearson")
            sim = self._relation_similarity(data_path, dataset_name, view, kernel, ids, node_fd)
            self._cell_sims[view] = (np.asarray(ids), sim)
        return node_fd

    def _load_omics_matrix(self, data_path: str, dataset_name: str, view: str, ids: np.ndarray) -> np.ndarray:
        """
        Load one omics view (gene-list reduced) and align it to ``ids``, zero-filling missing cells.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :param view: omics view name to load
        :param ids: ordered cell-line ids to align the matrix to
        :returns: (n_cells, n_feat) omics matrix aligned to ``ids``
        """
        gene_list = self._CELL_GENE_LISTS.get(view)
        fd = load_and_select_gene_features(
            feature_type=view, gene_list=gene_list, data_path=data_path, dataset_name=dataset_name
        )
        feats = fd.features
        dim = next(len(v[view]) for v in feats.values())
        mat = np.zeros((len(ids), dim), dtype=np.float64)
        for i, cid in enumerate(ids):
            if cid in feats:
                mat[i] = feats[cid][view]
        return mat

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load fingerprint node features and build each drug-relation graph for this dataset.

        Each relation resource lives under ``data/meta/gcmf_drug_relations/`` and is joined to
        this dataset's drugs on ``pubchem_id``. ``matrix`` relations are precomputed
        pubchem-indexed similarity matrices; ``targets`` relations are a
        (pubchem_id, drug_name, feature) table from which a Jaccard graph is computed over this
        dataset's drugs. Drugs absent from a relation are left isolated; a missing resource
        raises an error.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the fingerprint node features
        :raises FileNotFoundError: if a configured drug-relation resource is missing
        """
        node_fd = super().load_drug_features(data_path, dataset_name)
        drug_ids = node_fd.identifiers
        self._drug_sims = {}
        for view in self.drug_relation_views:
            csv_path = self._drug_resource_path(view, data_path)
            if csv_path is None:
                expected = os.path.join(data_path, "meta", self._DRUG_SIM_DIR, f"{view}.csv[.gz]")
                raise FileNotFoundError(
                    f"RGCMF drug relation '{view}' needs {expected}, which is missing. It is distributed "
                    f"with the dataset's meta bundle - download the data, or set 'drug_relation_views' to "
                    f"the relations you have."
                )
            if self._DRUG_RELATION_KIND.get(view, "matrix") == "targets":
                sim = self._build_target_jaccard(csv_path, drug_ids)
            else:
                sim = self._map_indexed_similarity(csv_path, drug_ids)
            off = sim.copy()
            np.fill_diagonal(off, 0.0)
            covered = int((off.sum(axis=1) > 0).sum())
            print(f"RGCMF drug relation '{view}': {covered}/{len(drug_ids)} drugs with edges.")
            self._drug_sims[view] = (np.asarray(drug_ids), sim)
        return node_fd

    def _build_target_jaccard(self, table_path: str, drug_ids: np.ndarray) -> np.ndarray:
        """
        Build a drug feature-set Jaccard similarity over ``drug_ids`` from a relation table.

        The table is keyed by ``pubchem_id`` and carries one set-membership feature per row
        (pathway for ``drug_pathways``, assay id for ``drug_bioassay``, target gene for
        ``drug_targets``). Drugs sharing more features are more similar; drugs with no features
        are isolated.

        :param table_path: path to the (pubchem_id, drug_name, feature) table
        :param drug_ids: ordered drug ids to build the similarity over
        :returns: dense (n_drugs, n_drugs) Jaccard similarity matrix
        :raises ValueError: if the table carries no feature column besides the join key
        """
        table = pd.read_csv(table_path)
        key = self._relation_key(table, table_path)
        # the feature is whichever column is neither the join key nor the human-readable name
        feature_cols = [c for c in table.columns if c not in (DRUG_IDENTIFIER, "drug_name")]
        if not feature_cols:
            raise ValueError(f"drug-relation table {table_path} has no feature column besides {DRUG_IDENTIFIER}")
        pid_to_targets: dict[str, set] = {}
        for pid, target in zip(key, table[feature_cols[0]]):
            pid_to_targets.setdefault(pid, set()).add(str(target))
        target_sets = [pid_to_targets.get(str(d), set()) for d in drug_ids]
        targets = sorted({t for s in target_sets for t in s})
        t_index = {t: i for i, t in enumerate(targets)}
        binary = np.zeros((len(drug_ids), len(targets)), dtype=np.float64)
        for i, s in enumerate(target_sets):
            for t in s:
                binary[i, t_index[t]] = 1.0
        return _similarity_matrix(binary, "jaccard")

    @staticmethod
    def _map_indexed_similarity(csv_path: str, drug_ids: np.ndarray) -> np.ndarray:
        """
        Reindex a pubchem-indexed similarity CSV onto ``drug_ids``, zero-filling misses.

        :param csv_path: path to the similarity CSV, indexed by ``pubchem_id`` on both axes
        :param drug_ids: ordered drug ids to reindex onto
        :returns: dense (n_drugs, n_drugs) similarity matrix in ``drug_ids`` order
        """
        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(str)
        src_vals = df.to_numpy(dtype=np.float64)
        pos = {str(d): i for i, d in enumerate(drug_ids)}
        n = len(drug_ids)
        sim = np.zeros((n, n), dtype=np.float64)
        # rows/cols of the source matrix that correspond to a drug in this dataset
        mapped = [(i, pos[pid]) for i, pid in enumerate(df.index) if pid in pos]
        if mapped:
            src_idx = np.array([s for s, _ in mapped])
            dst_idx = np.array([d for _, d in mapped])
            sim[np.ix_(dst_idx, dst_idx)] = src_vals[np.ix_(src_idx, src_idx)]
        return sim

    @staticmethod
    def _align_similarity(sim: np.ndarray, ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
        """
        Reorder a similarity matrix from ``ids`` order to ``target_ids`` order (missing -> zeros).

        :param sim: dense similarity matrix in ``ids`` order
        :param ids: ids matching the rows/cols of ``sim``
        :param target_ids: desired output id order
        :returns: similarity matrix reordered to ``target_ids`` (missing ids -> zero rows/cols)
        """
        pos = {str(c): i for i, c in enumerate(ids)}
        idx = np.array([pos.get(str(t), -1) for t in target_ids])
        n = len(target_ids)
        out = np.zeros((n, n), dtype=sim.dtype)
        valid = np.where(idx >= 0)[0]
        src = idx[valid]
        out[np.ix_(valid, valid)] = sim[np.ix_(src, src)]
        return out

    def _build_cell_adj(
        self, x_cell: np.ndarray, cell_line_input: FeatureDataset, cell_ids: np.ndarray, hp: dict[str, Any]
    ) -> list[torch.Tensor]:
        """
        Sparsify each precomputed cell-relation similarity to a k-NN adjacency.

        :param x_cell: (n_cells, cell_in_dim) fused cell feature matrix (unused; relations are precomputed)
        :param cell_line_input: cell-line FeatureDataset (unused; relations are precomputed)
        :param cell_ids: ordered cell-line ids to align each relation to
        :param hp: hyperparameter dictionary
        :returns: one normalized cell adjacency per cell relation
        :raises ValueError: if no cell relation resolves
        """
        use_weights = bool(hp.get("use_edge_weights", True))
        k = int(hp.get("k_cell", 15))
        adjs = []
        for view in self.cell_relation_views:
            ids, sim = self._cell_sims[view]
            sim = self._align_similarity(sim, ids, cell_ids)
            adjs.append(torch.tensor(_knn_normalize(sim, k, use_weights), device=self.device))
        if not adjs:
            raise ValueError("RGCMF has no cell relations; set 'cell_relation_views' or use GCMF instead.")
        self._n_cell_relations = len(adjs)
        return adjs

    def _build_drug_adj(
        self, x_drug: np.ndarray, drug_input: FeatureDataset, drug_ids: np.ndarray, hp: dict[str, Any]
    ) -> list[torch.Tensor]:
        """
        Sparsify each mapped drug-relation similarity to a k-NN adjacency.

        :param x_drug: (n_drugs, drug_in_dim) fingerprint feature matrix (unused; relations are precomputed)
        :param drug_input: drug FeatureDataset (unused; relations are precomputed)
        :param drug_ids: ordered drug ids to align each relation to
        :param hp: hyperparameter dictionary
        :returns: one normalized drug adjacency per drug relation
        :raises ValueError: if no drug relation resolves
        """
        use_weights = bool(hp.get("use_edge_weights", True))
        k = int(hp.get("k_drug", 15))
        adjs = []
        for view in self.drug_relation_views:
            ids, sim = self._drug_sims[view]
            sim = self._align_similarity(sim, ids, drug_ids)
            adjs.append(torch.tensor(_knn_normalize(sim, k, use_weights), device=self.device))
        if not adjs:
            raise ValueError("RGCMF has no drug relations; set 'drug_relation_views' or use GCMF instead.")
        self._n_drug_relations = len(adjs)
        return adjs

    def _build_net(self) -> _GCMFNet:
        """
        Instantiate a relational network with the resolved per-tower relation counts.

        :returns: a new ``_RGCMFNet`` instance
        """
        hp = self.hyperparameters
        return _RGCMFNet(
            n_cell_relations=self._n_cell_relations,
            n_drug_relations=self._n_drug_relations,
            relation_attention=bool(hp.get("relation_attention", False)),
            gnn_root=bool(hp.get("gnn_root", False)),
            **self._net_kwargs(),
        )


class PRGCMF(RGCMF):
    """
    Probabilistic Relational GCMF.

    ``RGCMF``'s multi-relation architecture combined with ``PGCMF``'s heteroscedastic
    Gaussian-NLL head (``probabilistic=True``): a second head predicts a per-pair log-variance,
    so the relational model also exposes a calibrated uncertainty for every prediction.
    """

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "PRGCMF"."""
        return "PRGCMF"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Build the relational model, forcing the probabilistic (Gaussian-NLL) head on.

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        super().build_model({**hyperparameters, "probabilistic": True})
