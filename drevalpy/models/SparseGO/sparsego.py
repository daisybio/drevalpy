"""SparseGO model for drug response prediction.

A sparse visible neural network (VNN) structured according to the Gene Ontology (GO)
hierarchy, combined with an ANN for drug fingerprints.

Original authors: Sada Del Real & Rubio (2023, 10.1016/j.ebiom.2023.104767)
Code adapted from https://github.com/KatynaSada/SparseGO_lightning
"""

import json
import os
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_sparse
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features

from .utils import create_index, load_mapping, load_ontology, pairs_in_layers, sort_pairs

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)


class SparseLinearNew(nn.Module):
    """Sparse linear layer with user-defined connectivity.

    Applies a linear transformation y = xA^T + b where A is a sparse weight
    matrix. Only the connections specified in the connectivity tensor are learned;
    all other weights are permanently zero.

    :param in_features: Size of each input sample.
    :param out_features: Size of each output sample.
    :param bias: If True, adds a learnable bias. Default: True.
    :param sparsity: Sparsity of weight matrix if connectivity is None. Default: 0.9.
    :param connectivity: LongTensor of shape (2, nnz) specifying the (row, col)
        indices of non-zero weights. Used for GO-structured layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.9,
        connectivity: torch.LongTensor | None = None,
    ):
        """Initialize SparseLinearNew layer.

        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample.
        :param bias: If True, adds a learnable bias. Default: True.
        :param sparsity: Sparsity of weight matrix if connectivity is None. Default: 0.9.
        :param connectivity: LongTensor of shape (2, nnz) specifying non-zero weight positions.
        :raises ValueError: if connectivity has wrong shape or nnz exceeds matrix size.
        """
        if not (in_features < 2**31 and out_features < 2**31 and sparsity < 1.0):
            raise ValueError("in_features and out_features must be < 2^31, sparsity must be < 1.0")
        if connectivity is not None:
            if not isinstance(connectivity, (torch.LongTensor, torch.cuda.LongTensor)):
                raise ValueError("Connectivity must be a Long Tensor")
            if not (connectivity.shape[0] == 2 and connectivity.shape[1] > 0):
                raise ValueError("Input shape for connectivity should be (2, nnz)")
            if connectivity.shape[1] > in_features * out_features:
                raise ValueError("Nnz can't be bigger than the weight matrix")

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.connectivity = connectivity

        coalesce_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if connectivity is None:
            self.sparsity = sparsity
            nnz = round((1.0 - sparsity) * in_features * out_features)
            if in_features * out_features <= 10**8:
                indices = np.random.choice(in_features * out_features, nnz, replace=False)
                indices = torch.as_tensor(indices, device=coalesce_device)
                row_ind = indices.floor_divide(in_features)
                col_ind = indices.fmod(in_features)
            else:
                warnings.warn(
                    "Matrix too large to sample non-zero indices without replacement, sparsity will be approximate",
                    RuntimeWarning,
                    stacklevel=2,
                )
                row_ind = torch.randint(0, out_features, (nnz,), device=coalesce_device)
                col_ind = torch.randint(0, in_features, (nnz,), device=coalesce_device)
            indices = torch.stack((row_ind, col_ind))
        else:
            nnz = connectivity.shape[1]
            self.sparsity = nnz / (out_features * in_features)
            indices = connectivity.to(device=coalesce_device)

        values = torch.empty(nnz, device=coalesce_device)
        indices, values = torch_sparse.coalesce(indices, values, out_features, in_features)

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
        :return: Output tensor of shape (batch_size, out_features).
        """
        output_shape = list(inputs.shape)
        output_shape[-1] = self.out_features

        if len(output_shape) == 1:
            inputs = inputs.view(1, -1)
        inputs = inputs.flatten(end_dim=-2)

        sparse_matrix = torch.sparse.FloatTensor(
            self.indices,
            self.weights,
            torch.Size([self.out_features, self.in_features]),
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

    :param layer_connections: List of (parent, child) pair arrays per layer,
        output of pairs_in_layers().
    :param num_neurons_per_GO: Number of neurons per GO term (default 6).
    :param num_neurons_per_final_GO: Number of neurons in the final GO layer.
    :param num_neurons_drug: List of hidden layer sizes for the drug ANN branch.
    :param num_neurons_final: Number of neurons in the final combined layer.
    :param drug_dim: Dimensionality of the drug fingerprint vector.
    :param gene2id_mapping: Dictionary mapping gene names to indices in the
        ontology (matches columns of the gene expression matrix).
    :param p_drop_final: Dropout rate for the final combined layers.
    :param p_drop_genes: Dropout rate for the gene input layer.
    :param p_drop_terms: Dropout rate for GO term layers.
    :param p_drop_drugs: Dropout rate for drug ANN layers.
    """

    def __init__(
        self,
        layer_connections: list,
        num_neurons_per_GO: int,
        num_neurons_per_final_GO: int,
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
        :param num_neurons_per_GO: Number of neurons per GO term.
        :param num_neurons_per_final_GO: Number of neurons in the final GO layer.
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

        self.num_neurons_per_GO = num_neurons_per_GO
        self.num_neurons_per_final_GO = num_neurons_per_final_GO
        self.num_neurons_drug = num_neurons_drug
        self.drug_dim = drug_dim
        self.layer_connections = layer_connections

        print("\nNumber of neurons per GO term: ", num_neurons_per_GO)
        print("Number of neurons of final GO term: ", num_neurons_per_final_GO)
        print("Number of drug neurons: ", num_neurons_drug)
        print("Number of final neurons: ", num_neurons_final)

        # (1) Layer of genes with terms
        input_id = self._genes_layer(layer_connections[0], p_drop_genes, gene2id_mapping)

        print("Number of term-term hierarchy levels:", len(layer_connections))

        # (2...) Layers of terms with terms
        for i in range(1, len(layer_connections)):
            neurons = num_neurons_per_final_GO if i == len(layer_connections) - 1 else num_neurons_per_GO
            input_id = self._terms_layer(input_id, layer_connections[i], str(i), neurons, p_drop_terms)

        # Drug ANN branch
        self._construct_drug_branch(p_drop_drugs)

        # Final combined layers
        final_input_size = num_neurons_per_final_GO + num_neurons_drug[-1]
        self.add_module("final_batchnorm_layer", nn.BatchNorm1d(final_input_size))
        self.add_module("drop_final", nn.Dropout(p_drop_final))
        self.add_module("final_linear_layer", nn.Linear(final_input_size, num_neurons_final))
        self.add_module("final_tanh", nn.Tanh())
        self.add_module("final_aux_batchnorm_layer", nn.BatchNorm1d(num_neurons_final))
        self.add_module("drop_aux_final", nn.Dropout(p_drop_final))
        self.add_module("final_aux_linear_layer", nn.Linear(num_neurons_final, 1))
        self.add_module("final_aux_tanh", nn.Tanh())
        self.add_module("final_linear_layer_output", nn.Linear(1, 1))

    def _genes_layer(self, genes_terms_pairs: np.ndarray, p_drop_genes: float, gene2id: dict) -> dict:
        """Build the first sparse layer connecting genes to GO terms.

        :param genes_terms_pairs: Array of (GO_term, gene) pairs.
        :param p_drop_genes: Dropout rate.
        :param gene2id: Dictionary mapping gene names to indices.
        :return: Dictionary mapping GO term names to their indices in this layer.
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
        neurons_per_GO: int,
        p_drop_terms: float,
    ) -> dict:
        """Build one sparse layer connecting GO terms to GO terms.

        :param input_id: Dictionary mapping child GO term names to indices.
        :param layer_pairs: Array of (parent_term, child_term) pairs for this layer.
        :param number: Layer number as string, used for module naming.
        :param neurons_per_GO: Number of neurons for parent terms in this layer.
        :param p_drop_terms: Dropout rate.
        :return: Dictionary mapping parent GO term names to their indices.
        """
        output_id = create_index(layer_pairs[:, 0])

        rows = [output_id[term] for term in layer_pairs[:, 0]]
        columns = [input_id[term] for term in layer_pairs[:, 1]]
        data = np.ones(len(rows))

        connections_matrix = sparse.coo_matrix((data, (rows, columns)), shape=(len(output_id), len(input_id)))

        # Kronecker product to expand to k neurons per term
        ones = sparse.csr_matrix(np.ones([neurons_per_GO, self.num_neurons_per_GO], dtype=int))
        connections_matrix_more_neurons = sparse.csr_matrix(sparse.kron(connections_matrix, ones))

        rows_t = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[0]).view(1, -1).long()
        cols_t = torch.from_numpy(sparse.find(connections_matrix_more_neurons)[1]).view(1, -1).long()
        connections = torch.cat((rows_t, cols_t), dim=0)

        input_terms = self.num_neurons_per_GO * len(input_id)
        output_terms = neurons_per_GO * len(output_id)

        self.add_module(
            f"GO_terms_sparse_linear_{number}", SparseLinearNew(input_terms, output_terms, connectivity=connections)
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
            Gene features occupy the first gene_dim columns, drug fingerprint
            occupies the remaining drug_dim columns.
        :return: Predicted drug response of shape (batch_size, 1).
        """
        gene_input = x.narrow(1, 0, self.gene_dim)
        drug_input = x.narrow(1, self.gene_dim, self.drug_dim)

        # VNN branch: batch -> dropout -> sparse linear -> activation
        gene_output = self._modules["genes_terms_batchnorm"](gene_input)
        gene_output = self._modules["drop_0"](gene_output)
        terms_output = self._modules["genes_terms_tanh"](self._modules["genes_terms_sparse_linear_1"](gene_output))

        for i in range(1, len(self.layer_connections)):
            terms_output = self._modules["GO_terms_batchnorm_" + str(i)](terms_output)
            terms_output = self._modules["drop_" + str(i)](terms_output)
            terms_output = self._modules["GO_terms_tanh_" + str(i)](
                self._modules["GO_terms_sparse_linear_" + str(i)](terms_output)
            )

        # ANN branch: batch -> dropout -> dense -> activation
        drug_out = drug_input
        for i in range(1, len(self.num_neurons_drug) + 1):
            drug_out = self._modules["drug_batchnorm_layer_" + str(i)](drug_out)
            drug_out = self._modules["drug_drop_" + str(i)](drug_out)
            drug_out = self._modules["drug_tanh_" + str(i)](self._modules["drug_linear_layer_" + str(i)](drug_out))

        # Concatenate and predict
        final_input = torch.cat((terms_output, drug_out), 1)
        output = self._modules["final_batchnorm_layer"](final_input)
        output = self._modules["drop_final"](output)
        output = self._modules["final_tanh"](self._modules["final_linear_layer"](output))
        output = self._modules["final_aux_batchnorm_layer"](output)
        output = self._modules["drop_aux_final"](output)
        output = self._modules["final_aux_tanh"](self._modules["final_aux_linear_layer"](output))
        return self._modules["final_linear_layer_output"](output)


class SparseGOModel(DRPModel):
    """SparseGO drug response prediction model.

    Wraps SparseGONetwork as a drevalpy DRPModel. Supports gene expression
    or mutation data as cell line features, and Morgan fingerprints as drug
    features.

    Requires two additional files in the dataset directory:

    - sparseGO_ont.txt: GO ontology file with term-term and gene-term connections.
    - gene2ind.txt: Mapping of gene names to indices in the expression matrix.

    These files can be generated using the create_sparsego_features.py featurizer.
    """

    cell_line_views = ["gene_expression", "mutations"]
    drug_views = ["fingerprints"]
    early_stopping = False

    def __init__(self) -> None:
        """Initialize SparseGOModel."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SparseGONetwork | None = None
        self.hyperparameters: dict[str, Any] = {}
        self.layer_connections: list | None = None
        self.gene2id_mapping_ont: dict | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model name.

        :return: SparseGO
        """
        return "SparseGO"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Store hyperparameters and build the network if ontology is already loaded.

        If load_cell_line_features() has not been called yet, network construction
        is deferred to the first call of train() or predict().

        :param hyperparameters: Dictionary with model hyperparameters.
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters

        if self.layer_connections is not None and self.gene2id_mapping_ont is not None:
            self._build_network()

    def _build_network(self) -> None:
        """Build SparseGONetwork once layer_connections and hyperparameters are both available."""
        self.model = SparseGONetwork(
            layer_connections=self.layer_connections,
            num_neurons_per_GO=self.hyperparameters.get("num_neurons_per_GO", 6),
            num_neurons_per_final_GO=self.hyperparameters.get("num_neurons_per_final_GO", 6),
            num_neurons_drug=self.hyperparameters.get("num_neurons_drug", [200, 100, 50]),
            num_neurons_final=self.hyperparameters.get("num_neurons_final", 12),
            drug_dim=self.hyperparameters.get("drug_dim", 2048),
            gene2id_mapping=self.gene2id_mapping_ont,
            p_drop_final=self.hyperparameters.get("p_drop_final", 0.0),
            p_drop_genes=self.hyperparameters.get("p_drop_genes", 0.1),
            p_drop_terms=self.hyperparameters.get("p_drop_terms", 0.1),
            p_drop_drugs=self.hyperparameters.get("p_drop_drugs", 0.1),
        ).to(self.DEVICE)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Train the SparseGO model.

        :param output: Training drug response dataset.
        :param cell_line_input: Cell line features (gene expression or mutations).
        :param drug_input: Drug features (Morgan fingerprints).
        :param output_earlystopping: Unused, kept for API compatibility.
        :param model_checkpoint_dir: Unused, kept for API compatibility.
        :raises ValueError: if drug_input is None or ontology not loaded.
        """
        if drug_input is None:
            raise ValueError("SparseGO requires drug features (fingerprints).")
        if self.model is None:
            if self.layer_connections is None or self.gene2id_mapping_ont is None:
                raise ValueError("Call load_cell_line_features() before train().")
            self._build_network()

        input_type = self.hyperparameters.get("input_type", "expression")
        feature_key = "gene_expression" if input_type == "expression" else "mutations"

        cell_matrix = cell_line_input.get_feature_matrix(view=feature_key, identifiers=output.cell_line_ids)
        drug_matrix = drug_input.get_feature_matrix(view="fingerprints", identifiers=output.drug_ids)
        features = np.concatenate([cell_matrix, drug_matrix], axis=1)
        labels = output.response.reshape(-1, 1)

        features_t = torch.from_numpy(features).float()
        labels_t = torch.from_numpy(labels).float()

        dataset = TensorDataset(features_t, labels_t)
        loader = DataLoader(
            dataset,
            batch_size=self.hyperparameters.get("batch_size", 10000),
            shuffle=True,
        )

        criterion = nn.MSELoss()
        lr = self.hyperparameters.get("learning_rate", 0.1)
        decay_rate = self.hyperparameters.get("decay_rate", 0.002)
        momentum = self.hyperparameters.get("momentum", 0.9)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        epochs = self.hyperparameters.get("epochs", 100)

        self.model.train()
        for epoch in range(epochs):
            current_lr = lr * (1 / (1 + decay_rate * epoch))
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            epoch_loss = 0.0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.DEVICE)
                batch_labels = batch_labels.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

            print(f"SparseGO Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss / len(loader):.4f}")

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict drug response for the given cell line and drug pairs.

        :param cell_line_ids: Array of cell line identifiers.
        :param drug_ids: Array of drug identifiers.
        :param cell_line_input: Cell line features.
        :param drug_input: Drug features.
        :return: Predicted response values as a 1D numpy array.
        :raises ValueError: if drug_input is None or ontology not loaded.
        """
        if drug_input is None:
            raise ValueError("SparseGO requires drug features (fingerprints).")
        if self.model is None:
            if self.layer_connections is None or self.gene2id_mapping_ont is None:
                raise ValueError("Call load_cell_line_features() before predict().")
            self._build_network()

        input_type = self.hyperparameters.get("input_type", "expression")
        feature_key = "gene_expression" if input_type == "expression" else "mutations"

        cell_matrix = cell_line_input.get_feature_matrix(view=feature_key, identifiers=cell_line_ids)
        drug_matrix = drug_input.get_feature_matrix(view="fingerprints", identifiers=drug_ids)
        features = np.concatenate([cell_matrix, drug_matrix], axis=1)
        features_t = torch.from_numpy(features).float()

        dataset = TensorDataset(features_t, torch.zeros(len(features_t)))
        loader = DataLoader(
            dataset,
            batch_size=self.hyperparameters.get("batch_size", 10000),
            shuffle=False,
        )

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_features, _ in loader:
                batch_features = batch_features.to(self.DEVICE)
                outputs = self.model(batch_features)
                predictions.append(outputs.squeeze().cpu().numpy())

        return np.concatenate(predictions)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load cell line features and build the GO ontology structure.

        Loads gene expression (or mutation) features via the standard drevalpy
        mechanism, and additionally reads sparseGO_ont.txt and gene2ind.txt
        to build the layer_connections required by the VNN.

        :param data_path: Path to the data directory.
        :param dataset_name: Name of the dataset (e.g. 'CTRPv2').
        :return: FeatureDataset with gene expression or mutation features.
        :raises FileNotFoundError: if sparseGO_ont.txt or gene2ind.txt are missing.
        """
        input_type = self.hyperparameters.get("input_type", "expression")
        feature_type = "gene_expression" if input_type == "expression" else "mutations"

        cell_line_features = load_and_select_gene_features(
            feature_type=feature_type,
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )

        ont_file = os.path.join(data_path, dataset_name, "sparseGO_ont.txt")
        gene2ind_file = os.path.join(data_path, dataset_name, "gene2ind.txt")

        if not os.path.exists(ont_file):
            raise FileNotFoundError(
                f"Ontology file not found: {ont_file}. " "Please generate it using create_sparsego_features.py."
            )
        if not os.path.exists(gene2ind_file):
            raise FileNotFoundError(
                f"Gene index file not found: {gene2ind_file}. " "Please generate it using create_sparsego_features.py."
            )

        self.gene2id_mapping_ont = load_mapping(gene2ind_file)
        dG, terms_pairs, genes_terms_pairs = load_ontology(ont_file, self.gene2id_mapping_ont)
        sorted_pairs, level_list, level_number = sort_pairs(
            genes_terms_pairs, terms_pairs, dG, self.gene2id_mapping_ont
        )
        self.layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)
        self.gene_dim_input = len(self.gene2id_mapping_ont)

        return cell_line_features

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load drug features (Morgan fingerprints) and auto-detect drug_dim.

        :param data_path: Path to the data directory.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset with Morgan fingerprint features.
        """
        features = load_drug_fingerprint_features(
            data_path=data_path,
            dataset_name=dataset_name,
        )
        sample_id = list(features.identifiers)[0]
        self.hyperparameters["drug_dim"] = features.get_feature_matrix(
            view="fingerprints", identifiers=np.array([sample_id])
        ).shape[1]
        return features

    def save(self, directory: str) -> None:
        """Save the model weights and hyperparameters.

        :param directory: Directory to save model files.
        :raises ValueError: if the model has not been trained.
        """
        os.makedirs(directory, exist_ok=True)

        if self.model is None:
            raise ValueError("Cannot save: model is not built.")

        torch.save(self.model.state_dict(), os.path.join(directory, "sparsego_model.pt"))  # noqa: S614

        save_hp = self.hyperparameters.copy()
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(save_hp, f)

    @classmethod
    def load(cls, directory: str) -> "SparseGOModel":
        """Load a previously saved SparseGOModel.

        Note: load_cell_line_features() must be called after loading to
        rebuild the ontology structure before making predictions.

        :param directory: Directory containing model files.
        :return: Loaded SparseGOModel instance.
        """
        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)

        instance._checkpoint_path = os.path.join(directory, "sparsego_model.pt")

        return instance

    def _load_weights_if_needed(self) -> None:
        """Load saved weights into the model if a checkpoint path is set.

        Called internally after build_model() when loading a saved model.
        """
        if hasattr(self, "_checkpoint_path") and self.model is not None:
            self.model.load_state_dict(
                torch.load(self._checkpoint_path, map_location=self.DEVICE, weights_only=True)  # noqa: S614
            )
            self.model.eval()
            del self._checkpoint_path
