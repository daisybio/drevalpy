"""SparseGO model for drug response prediction.

A sparse visible neural network (VNN) structured according to the Gene Ontology (GO)
hierarchy, combined with an ANN for drug fingerprints.

Original authors: Sada Del Real & Rubio (2023, 10.1016/j.ebiom.2023.104767)
Code adapted from https://github.com/KatynaSada/SparseGO_lightning
"""

import warnings
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

from .utils import create_index


def _validate_sparse_linear_dimensions(
    in_features: int, out_features: int, sparsity: float, connectivity: torch.Tensor | None
) -> None:
    if not (in_features < 2**31 and out_features < 2**31 and sparsity < 1.0):
        raise ValueError("in_features and out_features must be < 2^31, sparsity must be < 1.0")
    if connectivity is None:
        return
    if connectivity.shape[0] != 2 or connectivity.shape[1] <= 0:
        raise ValueError("Input shape for connectivity should be (2, nnz)")
    if connectivity.shape[1] > in_features * out_features:
        raise ValueError("Nnz can't be bigger than the weight matrix")


def _sparse_connectivity_indices(
    in_features: int,
    out_features: int,
    sparsity: float,
    connectivity: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, int, float]:
    """Return COO indices, nnz count, and effective sparsity.

    :param in_features: Input feature dimension.
    :param out_features: Output feature dimension.
    :param sparsity: Target sparsity when *connectivity* is not provided.
    :param connectivity: Optional fixed sparse connectivity tensor.
    :param device: Torch device for generated indices.

    :returns: Tuple of connectivity indices, non-zero count, and effective sparsity.
    """
    if connectivity is None:
        nnz = round((1.0 - sparsity) * in_features * out_features)
        if in_features * out_features <= 10**8:
            idx = np.random.choice(in_features * out_features, nnz, replace=False)
            indices = torch.as_tensor(idx, device=device)
            row_ind = indices.floor_divide(in_features)
            col_ind = indices.fmod(in_features)
        else:
            warnings.warn(
                "Matrix too large to sample non-zero indices without replacement, sparsity will be approximate",
                RuntimeWarning,
                stacklevel=3,
            )
            row_ind = torch.randint(0, out_features, (nnz,), device=device)
            col_ind = torch.randint(0, in_features, (nnz,), device=device)
        stacked = torch.stack((row_ind, col_ind))
        return stacked, nnz, sparsity

    nnz = connectivity.shape[1]
    effective_sparsity = nnz / (out_features * in_features)
    return connectivity.to(device=device), nnz, effective_sparsity


class SparseLinearNew(nn.Module):
    """Sparse linear layer with user-defined connectivity.

    Applies a linear transformation y = xA^T + b where A is a sparse weight
    matrix. Only the connections specified in the connectivity tensor are learned;
    all other weights are permanently zero.

    :param in_features: Size of each input sample.
    :param out_features: Size of each output sample.
    :param bias: If True, adds a learnable bias. Default: True.
    :param sparsity: Sparsity of weight matrix if connectivity is None. Default: 0.9.
    :param connectivity: LongTensor of shape ``(2, nnz)`` with non-zero weight indices for
        GO-structured layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9,
        connectivity: torch.Tensor | None = None,
    ):
        """Initialize SparseLinearNew layer.

        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample.
        :param bias: If True, adds a learnable bias. Default: True.
        :param sparsity: Sparsity of weight matrix if connectivity is None. Default: 0.9.
        :param connectivity: LongTensor of shape (2, nnz) specifying non-zero weight positions.
        """
        _validate_sparse_linear_dimensions(in_features, out_features, sparsity, connectivity)

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity = connectivity

        coalesce_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        indices, nnz, effective_sparsity = _sparse_connectivity_indices(
            in_features, out_features, sparsity, connectivity, coalesce_device
        )
        self.sparsity = effective_sparsity

        values = torch.empty(nnz, device=coalesce_device)
        sparse = torch.sparse_coo_tensor(indices, values, (out_features, in_features)).coalesce()
        indices, values = sparse.indices(), sparse.values()

        self.register_buffer("indices", indices.cpu())
        self.weights = nn.Parameter(values.cpu())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias with uniform distribution."""
        bound = 1 / self.in_features**0.5
        nn.init.uniform_(self.weights, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through sparse linear layer.

        :param inputs: Input tensor of shape (batch_size, in_features).

        :returns: Output tensor of shape (batch_size, out_features).
        """
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        if len(output_shape) == 1:
            inputs = inputs.view(1, -1)
        inputs = inputs.flatten(end_dim=-2)

        indices = cast(torch.Tensor, self.indices)
        sparse_matrix = torch.sparse_coo_tensor(
            indices,
            self.weights,
            [self.out_features, self.in_features],
        )
        output = torch.sparse.mm(sparse_matrix, inputs.t()).t()

        if self.bias is not None:
            output += self.bias

        return output.view(output_shape)


class SparseGONetwork(nn.Module):
    """Sparse Visible Neural Network structured according to the Gene Ontology hierarchy.

    Two-branch architecture:

    - VNN branch: sparse layers following GO parent-child relationships, takes
      gene expression or mutation data as input.
    - ANN branch: fully connected layers processing Morgan drug fingerprints.

    Both branches are concatenated and fed into a final regression head that
    predicts the drug response.

    Adapted from sparseGO_nn in https://github.com/KatynaSada/SparseGO

    :param layer_connections: List of (parent, child) pair arrays per layer, output of pairs_in_layers().
    :param num_neurons_per_GO: Number of neurons per GO term (default 6).
    :param num_neurons_per_final_GO: Number of neurons in the final GO layer.
    :param num_neurons_drug: List of hidden layer sizes for the drug ANN branch.
    :param num_neurons_final: Number of neurons in the final combined layer.
    :param drug_dim: Dimensionality of the drug fingerprint vector.
    :param gene2id_mapping: Mapping from gene names to ontology indices matching expression columns.
    :param p_drop_final: Dropout rate for the final combined layers.
    :param p_drop_genes: Dropout rate for the gene input layer.
    :param p_drop_terms: Dropout rate for GO term layers.
    :param p_drop_drugs: Dropout rate for drug ANN layers.
    """

    def __init__(
        self,
        layer_connections: list,
        num_neurons_per_go: int,
        num_neurons_per_final_go: int,
        num_neurons_drug: list[int],
        num_neurons_final: int,
        drug_dim: int,
        gene2id_mapping: dict,
        p_drop_final: float = 0,
        p_drop_genes: float = 0.1,
        p_drop_terms: float = 0.1,
        p_drop_drugs: float = 0.1,
    ):
        """Initialize SparseGONetwork.

        :param layer_connections: List of (parent, child) pair arrays per layer.
        :param num_neurons_per_go: Number of neurons per GO term.
        :param num_neurons_per_final_go: Number of neurons in the final GO layer.
        :param num_neurons_drug: List of hidden layer sizes for the drug ANN branch.
        :param num_neurons_final: Number of neurons in the final combined layer.
        :param drug_dim: Dimensionality of the drug fingerprint vector.
        :param gene2id_mapping: Dictionary mapping gene names to integer indices.
        :param p_drop_final: Dropout rate for the final combined layers.
        :param p_drop_genes: Dropout rate for the gene input layer.
        :param p_drop_terms: Dropout rate for GO term layers.
        :param p_drop_drugs: Dropout rate for drug ANN layers.
        """
        super().__init__()

        self.num_neurons_per_GO = num_neurons_per_go
        self.num_neurons_per_final_GO = num_neurons_per_final_go
        self.num_neurons_drug = num_neurons_drug
        self.drug_dim = drug_dim
        self.layer_connections = layer_connections

        print("\nNumber of neurons per GO term: ", num_neurons_per_go)
        print("Number of neurons of final GO term: ", num_neurons_per_final_go)
        print("Number of drug neurons: ", num_neurons_drug)
        print("Number of final neurons: ", num_neurons_final)

        # (1) Layer of genes with terms
        input_id = self._genes_layer(layer_connections[0], p_drop_genes, gene2id_mapping)

        print("Number of term-term hierarchy levels:", len(layer_connections))

        # (2...) Layers of terms with terms
        for i in range(1, len(layer_connections)):
            neurons = num_neurons_per_final_go if i == len(layer_connections) - 1 else num_neurons_per_go
            input_id = self._terms_layer(input_id, layer_connections[i], str(i), neurons, p_drop_terms)

        # Drug ANN branch
        self._construct_drug_branch(p_drop_drugs)

        # Final combined layers
        final_input_size = num_neurons_per_final_go + num_neurons_drug[-1]
        self.add_module("final_batchnorm_layer", nn.BatchNorm1d(final_input_size))
        self.add_module("drop_final", nn.Dropout(p_drop_final))
        self.add_module("final_linear_layer", nn.Linear(final_input_size, num_neurons_final))
        self.add_module("final_tanh", nn.Tanh())
        self.add_module("final_aux_batchnorm_layer", nn.BatchNorm1d(num_neurons_final))
        self.add_module("drop_aux_final", nn.Dropout(p_drop_final))
        self.add_module("final_aux_linear_layer", nn.Linear(num_neurons_final, 1))
        self.add_module("final_aux_tanh", nn.Tanh())
        self.add_module("final_linear_layer_output", nn.Linear(1, 1))

    def _m(self, name: str) -> nn.Module:
        """Get a registered submodule by name.

        :param name: Module name as registered via add_module.

        :returns: The submodule.

        :raises ValueError: if the module is not found.
        """
        module = self._modules[name]
        if module is None:
            raise ValueError(f"Module '{name}' not found")
        return module

    def _genes_layer(self, genes_terms_pairs: np.ndarray, p_drop_genes: float, gene2id: dict) -> dict:
        """Build the first sparse layer connecting genes to GO terms.

        :param genes_terms_pairs: Array of (GO_term, gene) pairs.
        :param p_drop_genes: Dropout rate.
        :param gene2id: Dictionary mapping gene names to indices.

        :returns: Dictionary mapping GO term names to their indices in this layer.
        """
        term2id = create_index(genes_terms_pairs[:, 0])

        self.gene_dim = len(gene2id)
        self.term_dim = len(term2id)

        rows = [term2id[term] for term in genes_terms_pairs[:, 0]]
        columns = [gene2id[gene] for gene in genes_terms_pairs[:, 1]]
        data = np.ones(len(rows))

        genes_terms = sparse.coo_matrix((data, (rows, columns)), shape=(self.term_dim, self.gene_dim))

        # Expand to k neurons per GO term by repeating each row k times
        genes_terms_more_neurons = sparse.lil_matrix((self.term_dim * self.num_neurons_per_GO, self.gene_dim))
        genes_terms = genes_terms.tolil()
        row = 0
        for i in range(genes_terms_more_neurons.shape[0]):
            if (i != 0) and (i % self.num_neurons_per_GO) == 0:
                row += 1
            genes_terms_more_neurons[i, :] = genes_terms[row, :]

        rows_t = torch.from_numpy(sparse.find(genes_terms_more_neurons)[0]).view(1, -1).long()
        cols_t = torch.from_numpy(sparse.find(genes_terms_more_neurons)[1]).view(1, -1).long()
        connections = torch.cat((rows_t, cols_t), dim=0)

        input_terms = len(gene2id)
        output_terms = self.num_neurons_per_GO * len(term2id)

        self.genes_terms_sparse_linear_1 = SparseLinearNew(input_terms, output_terms, connectivity=connections)
        self.genes_terms_batchnorm = nn.BatchNorm1d(input_terms)
        self.genes_terms_tanh = nn.Tanh()
        self.drop_0 = nn.Dropout(p_drop_genes)

        return term2id

    def _terms_layer(
        self,
        input_id: dict,
        layer_pairs: np.ndarray,
        number: str,
        neurons_per_go: int,
        p_drop_terms: float,
    ) -> dict:
        """Build one sparse layer connecting GO terms to GO terms.

        :param input_id: Dictionary mapping child GO term names to indices.
        :param layer_pairs: Array of (parent_term, child_term) pairs for this layer.
        :param number: Layer number as string, used for module naming.
        :param neurons_per_go: Number of neurons for parent terms in this layer.
        :param p_drop_terms: Dropout rate.

        :returns: Dictionary mapping parent GO term names to their indices.
        """
        output_id = create_index(layer_pairs[:, 0])

        rows = [output_id[term] for term in layer_pairs[:, 0]]
        columns = [input_id[term] for term in layer_pairs[:, 1]]
        data = np.ones(len(rows))

        connections_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(len(output_id), len(input_id)))

        # Kronecker product to expand to k neurons per term
        ones = sparse.csr_matrix(np.ones([neurons_per_go, self.num_neurons_per_GO], dtype=int))
        connections_matrix_more_neurons = sparse.csr_matrix(sparse.kron(connections_matrix, ones))

        rows_t = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[0]).view(1, -1).long()
        cols_t = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[1]).view(1, -1).long()
        connections = torch.cat((rows_t, cols_t), dim=0)

        input_terms = self.num_neurons_per_GO * len(input_id)
        output_terms = neurons_per_go * len(output_id)

        self.add_module(
            f"GO_terms_sparse_linear_{number}",
            SparseLinearNew(input_terms, output_terms, connectivity=connections),
        )
        self.add_module(f"drop_{number}", nn.Dropout(p_drop_terms))
        self.add_module(f"GO_terms_tanh_{number}", nn.Tanh())
        self.add_module(f"GO_terms_batchnorm_{number}", nn.BatchNorm1d(input_terms))

        return output_id

    def _construct_drug_branch(self, p_drop_drugs: float) -> None:
        """Build the fully connected ANN branch for drug fingerprints.

        :param p_drop_drugs: Dropout rate for drug layers.
        """
        input_size = self.drug_dim
        for i in range(len(self.num_neurons_drug)):
            self.add_module(f"drug_linear_layer_{i + 1}", nn.Linear(input_size, self.num_neurons_drug[i]))
            self.add_module(f"drug_drop_{i + 1}", nn.Dropout(p_drop_drugs))
            self.add_module(f"drug_tanh_{i + 1}", nn.Tanh())
            self.add_module(f"drug_batchnorm_layer_{i + 1}", nn.BatchNorm1d(input_size))
            input_size = self.num_neurons_drug[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full SparseGO network.

        :param x: Input tensor of shape (batch_size, gene_dim + drug_dim).

        :returns: Predicted drug response of shape (batch_size, 1).
        """
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)

        # VNN branch
        gene_output = cast(nn.Module, self._modules["genes_terms_batchnorm"])(gene_input)
        gene_output = cast(nn.Module, self._modules["drop_0"])(gene_output)
        terms_output = cast(nn.Module, self._modules["genes_terms_tanh"])(
            cast(nn.Module, self._modules["genes_terms_sparse_linear_1"])(gene_output)
        )

        for i in range(1, len(self.layer_connections)):
            terms_output = cast(nn.Module, self._modules[f"GO_terms_batchnorm_{i}"])(terms_output)
            terms_output = cast(nn.Module, self._modules[f"drop_{i}"])(terms_output)
            terms_output = cast(nn.Module, self._modules[f"GO_terms_tanh_{i}"])(
                cast(nn.Module, self._modules[f"GO_terms_sparse_linear_{i}"])(terms_output)
            )

        # ANN branch
        drug_out = drug_input
        for i in range(1, len(self.num_neurons_drug) + 1):
            drug_out = cast(nn.Module, self._modules[f"drug_batchnorm_layer_{i}"])(drug_out)
            drug_out = cast(nn.Module, self._modules[f"drug_drop_{i}"])(drug_out)
            drug_out = cast(nn.Module, self._modules[f"drug_tanh_{i}"])(
                cast(nn.Module, self._modules[f"drug_linear_layer_{i}"])(drug_out)
            )

        # Final
        final_input = torch.cat((terms_output, drug_out), 1)
        output = cast(nn.Module, self._modules["final_batchnorm_layer"])(final_input)
        output = cast(nn.Module, self._modules["drop_final"])(output)
        output = cast(nn.Module, self._modules["final_tanh"])(
            cast(nn.Module, self._modules["final_linear_layer"])(output)
        )
        output = cast(nn.Module, self._modules["final_aux_batchnorm_layer"])(output)
        output = cast(nn.Module, self._modules["drop_aux_final"])(output)
        output = cast(nn.Module, self._modules["final_aux_tanh"])(
            cast(nn.Module, self._modules["final_aux_linear_layer"])(output)
        )
        return cast(nn.Module, self._modules["final_linear_layer_output"])(output)
