"""
MultiTaskMLP: one network from landmark-gene expression to the response of every training drug.

The network has one output per drug and is trained with a squared error over the measured (cell line, drug)
pairs only, on responses standardized per drug. No drug features are used, so drugs unseen in training get the
training mean.
"""

import os
from typing import Any

import numpy as np
import torch
from torch import nn

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

from .utils import Standardizer, expression, gather, load_landmark_expression, response_matrix


def _device() -> torch.device:
    """
    Pick the device to train on.

    :returns: cuda if available, else cpu
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_network(n_in: int, n_out: int, hidden: int = 512, depth: int = 2, dropout: float = 0.2) -> nn.Module:
    """
    Build the multi-task MLP: expression in, one output per drug.

    :param n_in: number of genes
    :param n_out: number of drugs
    :param hidden: width of the hidden layers
    :param depth: number of hidden layers
    :param dropout: dropout after each hidden layer
    :returns: the network
    """
    layers: list[nn.Module] = []
    d = n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Squared error over the measured pairs only.

    :param pred: (n, n_drugs) predictions
    :param target: (n, n_drugs) targets, any value where unmeasured
    :param mask: (n, n_drugs) 1 where measured, else 0
    :returns: mean squared error over measured pairs
    """
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)


def train_network(
    net: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    x_stop: torch.Tensor,
    target_stop: torch.Tensor,
    mask_stop: torch.Tensor,
    max_epochs: int = 200,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> tuple[nn.Module, int]:
    """
    Train with Adam on the measured pairs, stopping when the error on held-out cell lines stops improving.

    :param net: network from ``build_network``, already on the device of ``x``
    :param x: (n_cells, n_genes) standardized expression of the training cell lines
    :param target: (n_cells, n_drugs) standardized responses, 0 where unmeasured
    :param mask: (n_cells, n_drugs) 1 where the pair was measured, else 0
    :param x_stop: expression of the early-stopping cell lines
    :param target_stop: their standardized responses
    :param mask_stop: their measured-pair mask
    :param max_epochs: upper limit on passes over the training cell lines
    :param patience: epochs without improvement on the early-stopping cell lines before stopping
    :param batch_size: cell lines per step
    :param lr: Adam learning rate
    :returns: the network restored to its best epoch, in eval mode, and that epoch's number
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(x), device=x.device)
        for start in range(0, len(x), batch_size):
            stop = start + batch_size
            b = perm[start:stop]
            loss = masked_mse(net(x[b]), target[b], mask[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            err = float(masked_mse(net(x_stop), target_stop, mask_stop))
        if err < best - 1e-6:
            best, best_epoch, stale = err, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best_epoch


class MultiTaskMLP(DRPModel):
    """Multi-task MLP on gene expression with one output per drug, default settings, early stopping."""

    cell_line_views = ["gene_expression"]
    drug_views: list[str] = []
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the model; everything is fitted in ``train``."""
        super().__init__()
        self.state: dict[str, Any] = {}
        self.nets: list[nn.Module] = []

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "MultiTaskMLP"."""
        return "MultiTaskMLP"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store the hyperparameters.

        :param hyperparameters: hyperparameter dictionary, set by RidgeMLPEnsemble from its prefixed settings
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load landmark-gene expression.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "gene_expression" view
        """
        return load_landmark_expression(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        No drug features are used.

        :param data_path: path to the data directory
        :param dataset_name: dataset name
        :returns: None
        """
        return None

    def _make_net(self, n_in: int, n_out: int) -> nn.Module:
        """
        Build the network.

        :param n_in: number of genes
        :param n_out: number of drugs
        :returns: the network
        """
        hp = self.hyperparameters
        return build_network(
            n_in,
            n_out,
            hidden=int(hp.get("hidden", 512)),
            depth=int(hp.get("depth", 2)),
            dropout=float(hp.get("dropout", 0.2)),
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Fit the seed ensemble, each network stopped early on the early-stopping cell lines.

        :param output: training responses
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :param output_earlystopping: responses of the early-stopping cell lines (DrEval passes these because
            ``early_stopping`` is True); if missing, 10% of the training cell lines are held out instead
        :param model_checkpoint_dir: not used
        """
        hp = self.hyperparameters
        dev = _device()
        known = set(cell_line_input.identifiers)
        cells, drugs, y = response_matrix(output, known)
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            stop_cells, stop_drugs, y_stop_raw = response_matrix(output_earlystopping, known)
            y_stop = np.full((len(stop_cells), len(drugs)), np.nan)
            col = {d: j for j, d in enumerate(drugs)}
            for j, d in enumerate(stop_drugs):
                if d in col:
                    y_stop[:, col[d]] = y_stop_raw[:, j]
        else:
            held = np.random.default_rng(int(hp.get("seed", 0))).permutation(len(cells))[: max(1, len(cells) // 10)]
            keep = np.setdiff1d(np.arange(len(cells)), held)
            stop_cells, y_stop = cells[held], y[held]
            cells, y = cells[keep], y[keep]
        drug_mean, drug_sd = np.nanmean(y, axis=0), np.nanstd(y, axis=0)
        drug_sd[~(drug_sd > 0)] = 1.0
        x_raw = expression(cell_line_input, cells)
        scaler = Standardizer(x_raw)

        def tensors(xr: np.ndarray, yr: np.ndarray):
            t = (yr - drug_mean[None, :]) / drug_sd[None, :]
            return (
                torch.as_tensor(scaler(xr), dtype=torch.float32, device=dev),
                torch.as_tensor(np.nan_to_num(t), dtype=torch.float32, device=dev),
                torch.as_tensor(~np.isnan(t), dtype=torch.float32, device=dev),
            )

        x, target, mask = tensors(x_raw, y)
        x_stop, target_stop, mask_stop = tensors(expression(cell_line_input, stop_cells), y_stop)
        self.nets, epochs = [], []
        for s in range(int(hp.get("n_ensemble", 10))):
            torch.manual_seed(int(hp.get("seed", 0)) + s)
            net, best_epoch = train_network(
                self._make_net(x.shape[1], len(drugs)).to(dev),
                x,
                target,
                mask,
                x_stop,
                target_stop,
                mask_stop,
                max_epochs=int(hp.get("max_epochs", 200)),
                patience=int(hp.get("patience", 10)),
                batch_size=int(hp.get("batch_size", 64)),
                lr=float(hp.get("lr", 1e-3)),
            )
            self.nets.append(net)
            epochs.append(best_epoch)
        self.state = {
            "drugs": drugs,
            "drug_mean": drug_mean,
            "drug_sd": drug_sd,
            "training_mean": float(np.nanmean(y)),
            "scaler": scaler,
            "best_epochs": epochs,
        }

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict responses; drugs unseen in training get the training mean.

        :param cell_line_ids: cell-line ids to predict
        :param drug_ids: drug ids to predict
        :param cell_line_input: cell-line features
        :param drug_input: not used
        :returns: (n,) predicted responses
        """
        st = self.state
        dev = _device()
        cells = np.unique(cell_line_ids)
        x_t = torch.as_tensor(st["scaler"](expression(cell_line_input, cells)), dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = sum(net.to(dev)(x_t) for net in self.nets) / len(self.nets)
        full = out.double().cpu().numpy() * st["drug_sd"][None, :] + st["drug_mean"][None, :]
        return gather(full, cells, st["drugs"], cell_line_ids, drug_ids, st["training_mean"])

    def save(self, directory: str) -> None:
        """
        Persist the model.

        :param directory: target directory
        """
        os.makedirs(directory, exist_ok=True)
        blob = {
            "hyperparameters": self.hyperparameters,
            "state": self.state,
            "nets": [n.state_dict() for n in self.nets],
        }
        torch.save(blob, os.path.join(directory, "multitask_mlp.pt"))  # noqa: S614

    @classmethod
    def load(cls, directory: str) -> "MultiTaskMLP":
        """
        Restore a model saved with ``save``.

        :param directory: directory containing the saved file
        :returns: the restored model
        """
        blob = torch.load(os.path.join(directory, "multitask_mlp.pt"), weights_only=False)  # noqa: S614
        instance = cls()
        instance.build_model(blob["hyperparameters"])
        instance.state = blob["state"]
        for sd in blob["nets"]:
            net = instance._make_net(int(instance.state["scaler"].genes.sum()), len(instance.state["drugs"]))
            net.load_state_dict(sd)
            net.eval()
            instance.nets.append(net)
        return instance
