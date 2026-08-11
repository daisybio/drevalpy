"""
EnsembleMF: an ensembled two-tower matrix factorization for drug-response prediction.

Predicts the cell-line x drug response matrix as ``R = U V^T``, where the latent factors are not
free parameters but the outputs of small residual MLPs over cell-line and drug features. That
keeps the model usable in leave-cell-line-out, where a held-out cell line has features but no
observed responses and therefore no free factor to fit.

The design is deliberately minimal. Each piece below was kept because removing it measurably hurt
leave-cell-line-out performance on CTRPv2 (7-fold, paired per fold, measured on the *within-drug*
correlation - drug main effects dominate the plain correlation and hide everything else):

* **the ensemble** - the single largest effect. Going from 5 members to 1 costs 0.028 within-drug
  correlation, and accuracy keeps improving up to at least 40 members (+0.008 from 5 to 20, on
  every fold).
* **the residual encoder** - removing the skip and the per-block residual costs 0.018. A single
  residual block was slightly better than two (+0.002 on 6 of 7 folds) and is kept as the default.
* **the free per-drug embedding** - every drug is seen during training, so an id-indexed latent
  captures drug behaviour that fingerprints only approximate. Removing it costs 0.004.
* **per-cell/per-drug/global biases** - a drug-mean predictor alone reaches most of the plain
  correlation, so the model gets those main effects for free rather than spending capacity on them.

Things that were tried and did *not* help, and so are absent: graph convolution over cell-line or
drug similarity graphs (no effect across neighbourhood sizes 0-48, weighted or binary edges,
single or multi-relational, and also under leave-tissue-out); multi-omics side information used as
graph structure; a free per-tissue embedding; and an auxiliary within-drug ranking loss.
"""

import os
from typing import Any, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features, load_drug_fingerprint_features


def _select_device() -> torch.device:
    """
    Pick CUDA if available, else CPU.

    :returns: the selected torch device
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ResidualEncoder(nn.Module):
    """Project features to a latent factor through residual blocks, with a direct input skip."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.skip_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode node features into latent factors.

        :param x: (n_nodes, in_dim) feature matrix
        :returns: (n_nodes, emb_dim) latent factors
        """
        h0 = self.act(self.input_proj(x))
        h = self.dropout(h0)
        for block, norm in zip(self.blocks, self.norms):
            h = self.dropout(self.act(norm(block(h)))) + h
        return self.output_proj(h) + self.skip_proj(h0)


class _MFNet(nn.Module):
    """One ensemble member: two encoders plus a bilinear head with main effects."""

    def __init__(
        self,
        cell_in_dim: int,
        drug_in_dim: int,
        hidden_dim: int,
        emb_dim: int,
        n_layers: int,
        dropout: float,
        n_drugs: int,
        use_drug_id_embedding: bool,
        use_mlp_head: bool,
        mlp_hidden: int,
    ):
        super().__init__()
        self.cell_encoder = _ResidualEncoder(cell_in_dim, hidden_dim, emb_dim, n_layers, dropout)
        self.drug_encoder = _ResidualEncoder(drug_in_dim, hidden_dim, emb_dim, n_layers, dropout)

        # Every drug is present in training (only cell lines are held out), so a free id-indexed
        # latent is learnable here and transfers to test pairs. Cell lines get no such table: they
        # are unseen at test time and must stay purely feature-derived.
        self.use_drug_id_embedding = use_drug_id_embedding
        if use_drug_id_embedding:
            self.drug_id_emb = nn.Embedding(n_drugs, emb_dim)
            nn.init.normal_(self.drug_id_emb.weight, std=0.01)

        self.cell_bias = nn.Linear(emb_dim, 1)
        self.drug_bias = nn.Linear(emb_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.dot_scale = nn.Parameter(torch.ones(1))

        self.use_mlp_head = use_mlp_head
        if use_mlp_head:
            self.mlp = nn.Sequential(
                nn.Linear(3 * emb_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)
            )

    def encode(self, x_cell: torch.Tensor, x_drug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute latent factors for every cell line and every drug.

        :param x_cell: (n_cells, cell_in_dim) cell-line features
        :param x_drug: (n_drugs, drug_in_dim) drug features
        :returns: (cell factors, drug factors)
        """
        z_cell = self.cell_encoder(x_cell)
        z_drug = self.drug_encoder(x_drug)
        if self.use_drug_id_embedding:
            z_drug = z_drug + self.drug_id_emb.weight
        return z_cell, z_drug

    def score_pairs(self, z_cell: torch.Tensor, z_drug: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of (cell, drug) pairs from their gathered factors.

        :param z_cell: (batch, emb_dim) cell factors
        :param z_drug: (batch, emb_dim) drug factors
        :returns: (batch,) predicted responses
        """
        pred = (z_cell * z_drug).sum(dim=-1, keepdim=True) * self.dot_scale
        pred = pred + self.cell_bias(z_cell) + self.drug_bias(z_drug) + self.global_bias
        if self.use_mlp_head:
            pred = pred + self.mlp(torch.cat([z_cell, z_drug, z_cell * z_drug], dim=-1))
        return pred.squeeze(-1)


class EnsembleMF(DRPModel):
    """Ensembled two-tower matrix factorization over gene expression and drug fingerprints."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the model; the networks are built in ``train`` once the dimensions are known."""
        super().__init__()
        self.nets: list[_MFNet] = []
        self.hyperparameters: dict[str, Any] = {}
        self.device = _select_device()

        # filled in by train(), reused by predict()
        self._cell_id_to_idx: dict[str, int] = {}
        self._drug_id_to_idx: dict[str, int] = {}
        self._x_cell: torch.Tensor | None = None
        self._x_drug: torch.Tensor | None = None
        self._scaler: StandardScaler | None = None
        self.training_mean: float = 0.0

    @classmethod
    def get_model_name(cls) -> str:
        """:returns: the model name "EnsembleMF"."""
        return "EnsembleMF"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store hyperparameters and seed the RNG.

        :param hyperparameters: hyperparameter dictionary (see hyperparameters.yaml)
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = dict(hyperparameters)
        seed = int(hyperparameters.get("seed", 0))
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load gene expression, reduced to the configured gene list.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "gene_expression" view
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list=self.hyperparameters.get("gene_list", "landmark_genes"),
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load Morgan fingerprints.

        :param data_path: path to the data directory
        :param dataset_name: dataset name, e.g. CTRPv2
        :returns: FeatureDataset with the "fingerprints" view
        """
        return load_drug_fingerprint_features(
            data_path, dataset_name, fill_na=True, n_bits=int(self.hyperparameters.get("n_bits", 256))
        )

    def _build_cell_matrix(
        self, cell_line_input: FeatureDataset, cell_ids: np.ndarray, train_ids: np.ndarray
    ) -> np.ndarray:
        """
        Transform and standardize the cell-line features.

        ``feature_transform`` picks between ``rank`` (per-gene rank across cell lines, mapped to
        [0, 1]) and ``arcsinh``. ``rank`` is worth about 0.003 within-drug correlation on CTRPv2
        leave-cell-line-out, but it ranks each gene across *every* cell line including held-out
        ones, so it is transductive - set ``arcsinh`` when the evaluation must be strictly
        inductive. The scaler is fit on training cell lines only either way.

        :param cell_line_input: cell-line FeatureDataset
        :param cell_ids: ordered cell-line ids (all cell lines with features)
        :param train_ids: cell-line ids present in the training responses; empty at predict time,
            in which case the scaler fitted during train() is reused
        :returns: (n_cells, n_genes) scaled feature matrix
        """
        mat = cell_line_input.get_feature_matrix(view="gene_expression", identifiers=cell_ids).astype(np.float64)
        if str(self.hyperparameters.get("feature_transform", "rank")) == "rank":
            mat = mat.argsort(axis=0).argsort(axis=0) / max(1, mat.shape[0] - 1)
        else:
            mat = np.arcsinh(mat)
        if len(train_ids) > 0:
            self._scaler = StandardScaler().fit(mat[np.isin(cell_ids, np.unique(train_ids))])
        # train() always fits the scaler before any predict() call reaches this point
        return cast(StandardScaler, self._scaler).transform(mat).astype(np.float32)

    def _pairs(self, data: DrugResponseDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Map a response dataset to (cell index, drug index, response) tensors, dropping unknown ids.

        :param data: response dataset to convert
        :returns: (cell idx, drug idx, response) tensors on the model device
        """
        ci, di, y = [], [], []
        for cl, dr, resp in zip(data.cell_line_ids, data.drug_ids, data.response):
            if cl in self._cell_id_to_idx and dr in self._drug_id_to_idx:
                ci.append(self._cell_id_to_idx[cl])
                di.append(self._drug_id_to_idx[dr])
                y.append(resp)
        return (
            torch.tensor(ci, dtype=torch.long, device=self.device),
            torch.tensor(di, dtype=torch.long, device=self.device),
            torch.tensor(y, dtype=torch.float32, device=self.device),
        )

    def _build_net(self) -> _MFNet:
        """
        Instantiate one ensemble member from the stored dimensions and hyperparameters.

        :returns: a new ``_MFNet``
        """
        hp = self.hyperparameters
        # the feature tensors are set in train() before any net is built
        x_cell, x_drug = cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug)
        return _MFNet(
            cell_in_dim=x_cell.shape[1],
            drug_in_dim=x_drug.shape[1],
            hidden_dim=int(hp.get("hidden_dim", 256)),
            emb_dim=int(hp.get("emb_dim", 128)),
            n_layers=int(hp.get("n_layers", 1)),
            dropout=float(hp.get("dropout", 0.2)),
            n_drugs=x_drug.shape[0],
            use_drug_id_embedding=bool(hp.get("use_drug_id_embedding", True)),
            use_mlp_head=bool(hp.get("use_mlp_head", True)),
            mlp_hidden=int(hp.get("mlp_hidden", 256)),
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
        Train the ensemble on the observed (cell, drug) pairs.

        :param output: training responses
        :param cell_line_input: cell-line features (all cell lines)
        :param drug_input: drug features (all drugs)
        :param output_earlystopping: responses used for early stopping; a 10% split of the
            training pairs is carved out when this is None
        :param model_checkpoint_dir: unused, kept for interface compatibility
        :raises ValueError: if drug_input is None
        """
        if drug_input is None:
            raise ValueError("EnsembleMF requires drug features (fingerprints).")
        hp = self.hyperparameters
        self.training_mean = float(np.nanmean(output.response))

        cell_ids = np.unique(cell_line_input.identifiers)
        drug_ids = np.unique(drug_input.identifiers)
        self._cell_id_to_idx = {c: i for i, c in enumerate(cell_ids)}
        self._drug_id_to_idx = {d: i for i, d in enumerate(drug_ids)}

        x_cell = self._build_cell_matrix(cell_line_input, cell_ids, np.asarray(output.cell_line_ids))
        x_drug = drug_input.get_feature_matrix(view="fingerprints", identifiers=drug_ids).astype(np.float32)
        self._x_cell = torch.tensor(x_cell, device=self.device)
        self._x_drug = torch.tensor(x_drug, device=self.device)

        ci, di, y = self._pairs(output)
        if output_earlystopping is not None and len(output_earlystopping) > 0:
            val = self._pairs(output_earlystopping)
        else:
            perm = torch.randperm(len(y), device=self.device)
            n_val = max(1, int(0.1 * len(y)))
            val = (ci[perm[:n_val]], di[perm[:n_val]], y[perm[:n_val]])
            ci, di, y = ci[perm[n_val:]], di[perm[n_val:]], y[perm[n_val:]]

        self.nets = []
        for member in range(int(hp.get("n_ensemble", 20))):
            torch.manual_seed(int(hp.get("seed", 0)) + member)
            net = self._build_net().to(self.device)
            self._train_net(net, ci, di, y, val)
            self.nets.append(net)

    def _train_net(
        self,
        net: _MFNet,
        ci: torch.Tensor,
        di: torch.Tensor,
        y: torch.Tensor,
        val: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Train one ensemble member in place, keeping the weights with the best validation MSE.

        :param net: the network to train (modified in place)
        :param ci: (n_train,) cell indices of the training pairs
        :param di: (n_train,) drug indices of the training pairs
        :param y: (n_train,) target responses
        :param val: (cell idx, drug idx, response) tensors for early stopping
        """
        hp = self.hyperparameters
        x_cell, x_drug = cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug)
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(hp.get("learning_rate", 3e-4)),
            weight_decay=float(hp.get("weight_decay", 1e-5)),
        )
        loss_fn = nn.MSELoss()
        batch_size = int(hp.get("batch_size", 2048))
        patience = int(hp.get("patience", 25))
        best_val, best_state, stale = float("inf"), None, 0

        for _epoch in range(int(hp.get("max_epochs", 300))):
            net.train()
            perm = torch.randperm(len(y), device=self.device)
            for start in range(0, len(y), batch_size):
                end = start + batch_size
                idx = perm[start:end]
                optimizer.zero_grad()
                z_cell, z_drug = net.encode(x_cell, x_drug)
                loss = loss_fn(net.score_pairs(z_cell[ci[idx]], z_drug[di[idx]]), y[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                optimizer.step()

            val_mse = self._eval_mse(net, val)
            if val_mse < best_val - 1e-6:
                best_val, stale = val_mse, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

    @torch.no_grad()
    def _eval_mse(self, net: _MFNet, val: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Mean squared error of one member on the early-stopping pairs.

        :param net: the network to evaluate
        :param val: (cell idx, drug idx, response) tensors
        :returns: mean squared error, or inf if there is nothing to evaluate
        """
        ci, di, y = val
        if len(y) == 0:
            return float("inf")
        net.eval()
        z_cell, z_drug = net.encode(cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug))
        return float(nn.functional.mse_loss(net.score_pairs(z_cell[ci], z_drug[di]), y).item())

    @torch.no_grad()
    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict responses for (cell, drug) pairs, averaging over the ensemble.

        The factors are those learned in ``train``, so pairs whose cell line or drug was not in the
        training dataset's feature set fall back to the training mean. This makes the model
        transductive over features, like ``SRMF``: it cannot score a cell line it has never
        encoded, which matters for cross-study prediction.

        :param cell_line_ids: cell-line ids to predict
        :param drug_ids: drug ids to predict
        :param cell_line_input: unused; factors are cached from train()
        :param drug_input: unused; factors are cached from train()
        :returns: (n,) predicted responses
        """
        preds = np.full(len(cell_line_ids), self.training_mean, dtype=np.float32)
        if not self.nets:
            return preds
        rows = [
            (i, self._cell_id_to_idx[c], self._drug_id_to_idx[d])
            for i, (c, d) in enumerate(zip(cell_line_ids, drug_ids))
            if c in self._cell_id_to_idx and d in self._drug_id_to_idx
        ]
        if not rows:
            return preds
        idx, ci, di = (np.array(v) for v in zip(*rows))
        ci_t = torch.tensor(ci, dtype=torch.long, device=self.device)
        di_t = torch.tensor(di, dtype=torch.long, device=self.device)
        x_cell, x_drug = cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug)
        member_preds = []
        for net in self.nets:
            net.eval()
            z_cell, z_drug = net.encode(x_cell, x_drug)
            member_preds.append(net.score_pairs(z_cell[ci_t], z_drug[di_t]).cpu().numpy())
        preds[idx] = np.mean(member_preds, axis=0)
        return preds

    def save(self, directory: str) -> None:
        """
        Persist the trained ensemble.

        :param directory: target directory
        :raises RuntimeError: if the model has not been trained
        """
        if not self.nets:
            raise RuntimeError("No trained model to save.")
        x_cell, x_drug = cast(torch.Tensor, self._x_cell), cast(torch.Tensor, self._x_drug)
        os.makedirs(directory, exist_ok=True)
        torch.save([net.state_dict() for net in self.nets], os.path.join(directory, "nets.pt"))  # noqa: S614
        joblib.dump(
            {
                "hyperparameters": self.hyperparameters,
                "cell_id_to_idx": self._cell_id_to_idx,
                "drug_id_to_idx": self._drug_id_to_idx,
                "x_cell": x_cell.cpu().numpy(),
                "x_drug": x_drug.cpu().numpy(),
                "scaler": self._scaler,
                "training_mean": self.training_mean,
            },
            os.path.join(directory, "state.pkl"),
        )

    @classmethod
    def load(cls, directory: str) -> "EnsembleMF":
        """
        Restore a model saved with ``save``.

        :param directory: directory containing the saved files
        :returns: the restored model
        """
        instance = cls()
        state = joblib.load(os.path.join(directory, "state.pkl"))
        instance.build_model(state["hyperparameters"])
        instance._cell_id_to_idx = state["cell_id_to_idx"]
        instance._drug_id_to_idx = state["drug_id_to_idx"]
        instance._scaler = state["scaler"]
        instance.training_mean = state["training_mean"]
        instance._x_cell = torch.tensor(state["x_cell"], device=instance.device)
        instance._x_drug = torch.tensor(state["x_drug"], device=instance.device)
        # map_location: a model trained on a GPU node must still load on a CPU-only machine
        state_dicts = torch.load(os.path.join(directory, "nets.pt"), map_location=instance.device)  # noqa: S614
        instance.nets = []
        for sd in state_dicts:
            net = instance._build_net().to(instance.device)
            net.load_state_dict(sd)
            net.eval()
            instance.nets.append(net)
        return instance
