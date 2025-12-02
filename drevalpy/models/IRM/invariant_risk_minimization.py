"""Invariant Risk Minimization (IRM) model for drug response prediction in DrEvalPy to counter cell line biases."""

import numpy as np
import torch
import torch.autograd
import torch.nn as nn

from drevalpy.models.utils import load_and_select_gene_features, load_drug_fingerprint_features

from ..drp_model import DRPModel


class IRMDRP(DRPModel):
    """
    IRM-based drug response prediction model for DrEvalPy.

    Uses IRMv1 penalty to enforce a shared classifier across cell line environments.
    Cell lines are treated as separate environments, drug and cell line features
    are concatenated and passed through a small feedforward network that will
    focus on generalizable cross-cell-line patterns.
    """

    is_single_drug_model = False
    early_stopping = False

    @classmethod
    def get_model_name(cls):
        """Return the model name used in DrEvalPy pipelines.

        :returns: model name string
        """
        return "IRMDRP"

    @classmethod
    def get_hyperparameter_set(cls):
        """
        Return default hyperparameters.

        hidden_dim: size of hidden layers
        lr: learning rate
        epochs: training epochs
        irm_lambda: IRM penalty strength
        batch_envs: number of cell line environments per batch
        batch_samples: number of samples per environment in a batch

        :returns: list of hyperparameter dictionaries
        """
        return [{"hidden_dim": 128, "lr": 1e-3, "epochs": 20, "irm_lambda": 1e-2, "batch_envs": 32, "batch_samples": 8}]

    @property
    def cell_line_views(self):
        """Return list of required cell line feature views.

        :returns: list of cell line feature view strings
        """
        return ["gene_expression"]

    @property
    def drug_views(self):
        """Return list of required drug feature views.

        :returns: list of drug feature view strings
        """
        return ["fingerprints"]

    def build_model(self, hyperparameters):
        """
        Store hyperparameters and initialize network parameters.

        :param hyperparameters: dictionary of model hyperparameters
        """
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.lr = hyperparameters["lr"]
        self.epochs = hyperparameters["epochs"]
        self.irm_lambda = hyperparameters.get("irm_lambda", 1e-2)
        self.batch_envs = hyperparameters.get("batch_envs", 32)
        self.batch_samples = hyperparameters.get("batch_samples", 8)
        self.model = None
        self.optimizer = None
        self.hyperparameters = hyperparameters

    def _init_network(self, input_dim):
        """Initialize feedforward network and optimizer.

        :param input_dim: dimension of input features
        """
        self.model = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, 1))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _irm_penalty(self, z, y):
        """
        Compute IRM penalty for one environment.

        :param z: input features for environment (torch tensor)
        :param y: target responses (torch tensor)
        :returns: IRM penalty (scalar)
        """
        scale = torch.tensor(1.0, requires_grad=True, device=z.device)
        pred = self.model(z * scale)
        loss = self.loss_fn(pred, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def train(
        self, output, cell_line_input, drug_input=None, output_earlystopping=None, model_checkpoint_dir="checkpoints"
    ):
        """
        Train the IRM model.

        :param output: DrugResponseDataset containing response values
        :param cell_line_input: FeatureDataset with cell line features
        :param drug_input: FeatureDataset with drug features (optional)
        :param output_earlystopping: DrugResponseDataset for early stopping (optional)
        :param model_checkpoint_dir: directory to save checkpoints (optional)
        """
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        y = output.response.astype(np.float32)
        x_dim = x.shape[1]
        self._init_network(x_dim)

        # organize data by cell line for IRM
        cell_lines = np.unique(output.cell_line_ids)
        by_cellline = {cid: [] for cid in cell_lines}
        for xi, yi, cid in zip(x, y, output.cell_line_ids):
            by_cellline[cid].append((xi, yi))

        for _ in range(self.epochs):
            batch_envs = min(self.batch_envs, len(cell_lines))
            envs = np.random.choice(cell_lines, batch_envs, replace=False)
            x_batch, y_batch = [], []
            for e in envs:
                samples = np.random.choice(
                    len(by_cellline[e]), min(self.batch_samples, len(by_cellline[e])), replace=False
                )
                x_env = np.stack([by_cellline[e][i][0] for i in samples])
                y_env = np.stack([by_cellline[e][i][1] for i in samples])
                x_batch.append(x_env)
                y_batch.append(y_env)
            x_batch = torch.tensor(np.concatenate(x_batch, axis=0).astype(np.float32))
            y_batch = torch.tensor(np.concatenate(y_batch, axis=0).astype(np.float32)).view(-1, 1)

            self.optimizer.zero_grad()
            pred = self.model(x_batch)
            loss_emp = self.loss_fn(pred, y_batch)

            penalty = 0.0
            idx = 0
            for e in envs:
                n_samples = min(self.batch_samples, len(by_cellline[e]))
                z_env = x_batch[idx : idx + n_samples]  # noqa: E203
                y_env = y_batch[idx : idx + n_samples]  # noqa: E203
                penalty += self._irm_penalty(z_env, y_env)
                idx += n_samples
            penalty /= len(envs)

            loss = loss_emp + self.irm_lambda * penalty
            loss.backward()
            self.optimizer.step()

    def predict(self, cell_line_ids, drug_ids, cell_line_input, drug_input=None):
        """
        Predict drug response for given cell lines and drugs.

        :param cell_line_ids: array of cell line identifiers
        :param drug_ids: array of drug identifiers
        :param cell_line_input: FeatureDataset with cell line features
        :param drug_input: FeatureDataset with drug features (optional)
        :returns: predicted responses as a 1D numpy array
        """
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        x_t = torch.tensor(x.astype(np.float32))
        with torch.no_grad():
            return self.model(x_t).cpu().numpy().flatten()

    def load_cell_line_features(self, data_path, dataset_name):
        """
        Load cell line features.

        :param data_path: path to dataset
        :param dataset_name: name of dataset
        :returns: FeatureDataset object with cell line features
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features.

        :param data_path: path to dataset
        :param dataset_name: name of dataset
        :returns: FeatureDataset object with drug features
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)
