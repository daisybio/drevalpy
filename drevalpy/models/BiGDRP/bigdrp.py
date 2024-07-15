from typing import Optional
from drevalpy.models.utils import (
    load_drug_features_from_fingerprints,
    load_and_reduce_gene_features,
)
from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from utils import (
    reset_seed,
    normalizer,
    reindex_tuples,
    create_fold_mask,
    predict_matrix,
)


class BiGDRP(DRPModel):
    """
    BiGDRP model. Adapted from Hostallero et al. https://academic.oup.com/bioinformatics/article/38/14/3609/6604271
    """

    cell_line_views = ["gene_expression"]

    drug_views = ["RDKit_descriptors"]

    early_stopping = True

    model_name = "BiGDRP"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        """
        self.model = DRPPlus(
            n_genes=hyperparameters["n_genes"],
            hyp=hyperparameters["hyp"],
        )

    def train(
        self,
        output: DrugResponseDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        gene_expression: np.ndarray = None,
        RDKit_descriptors: np.ndarray = None,
        gene_expression_earlystopping: Optional[np.ndarray] = None,
        RDKit_descriptors_earlystopping: Optional[np.ndarray] = None,
    ):
        """
        Trains the model.
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        :param gene_expression: gene expression features
        :param RDKit_descriptors: RDKit drug descriptors features

        """

    label_mask = create_fold_mask(labels, label_matrix)
    label_matrix = label_matrix.replace(np.nan, 0)

    n_genes = cell_lines.shape[1]
    metrics = np.zeros((5, 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RDKit_descriptors = pd.read_csv(encoding_path, index_col=0)
    dr_idx = RDKit_descriptors.index

    train_folds = [x for x in range(5) if (x != test_fold)]
    train_tuples = labels.loc[labels["fold"].isin(train_folds)]
    train_samples = list(train_tuples["cell_line"].unique())
    train_x = cell_lines.loc[train_samples].values
    train_y = label_matrix.loc[train_samples].values
    train_mask = (label_mask.loc[train_samples].isin(train_folds)) * 1

    test_tuples = labels.loc[labels["fold"] == test_fold]
    test_samples = list(test_tuples["cell_line"].unique())
    test_x = cell_lines.loc[test_samples].values
    test_mask = (label_mask.loc[test_samples] == test_fold) * 1

    train_x, test_x = normalizer(train_x, test_x)

    train_tuples = train_tuples[["drug", "cell_line", "response"]]
    train_tuples = reindex_tuples(train_tuples, dr_idx, train_samples)
    train_data = TupleMapDataset(
        train_tuples,
        RDKit_descriptors,
        torch.FloatTensor(train_x),
        torch.FloatTensor(train_y),
    )
    train_data = DataLoader(train_data, batch_size=128)

    print("Loading weights from: %s..." % model_path)
    model = DRPPlus(cell_lines.shape[1], hyperparams)
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)

    # === train for 1 epoch ===
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=1e-5)
    model.train()
    for x, d, y in train_data:
        x, d, y = x.to(device), d.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x, d)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        print("train MSE: %.4f" % loss.item())

    def save(self, path: str):
        """
        Saves the model.
        :param path: path to save the model
        """
        self.model.save(path)

    @staticmethod
    def load(path: str):
        # TODO
        raise NotImplementedError("load method not implemented")

    def predict(
        self, gene_expression: np.ndarray, RDKit_descriptors: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        """
        # === test on test set ===
        RDKit_descriptors = torch.Tensor(RDKit_descriptors.values).to(device)
        test_data = TensorDataset(torch.FloatTensor(test_x))
        test_data = DataLoader(
            test_data, batch_size=hyperparams["batch_size"], shuffle=False
        )

        prediction_matrix = predict_matrix(model, test_data, RDKit_descriptors, device)
        prediction_matrix = pd.DataFrame(
            prediction_matrix, index=test_samples, columns=dr_idx
        )
        test_mask = test_mask.replace(0, np.nan)
        prediction_matrix = prediction_matrix * test_mask
        prediction_matrix.to_csv(
            FLAGS.outroot
            + "results/"
            + FLAGS.folder
            + "/val_prediction_fold_%d.csv" % i
        )

        directory = FLAGS.outroot + "/results/" + FLAGS.folder
        torch.save(model.state_dict(), directory + "/model_weights_fold_%d" % test_fold)

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

        return load_drug_features_from_fingerprints(data_path, dataset_name)


class DRPPlus(nn.Module):
    def __init__(self, n_genes, hyp):
        super(DRPPlus, self).__init__()
        n_genes = 1234

        self.expr_l1 = nn.Linear(n_genes, hyp["expr_enc"])
        self.mid = nn.Linear(hyp["expr_enc"] + hyp["conv2"], hyp["mid"])
        self.out = nn.Linear(hyp["mid"], 1)

        if hyp["drop"] == 0:
            drop = [0, 0]
        else:
            drop = [0.2, 0.5]

        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])
        self.alpha = 0.5

    def forward(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))

        x = torch.cat([expr_enc, drug_enc], -1)  # (batch, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x))  # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x)  # (batch, n_drugs, 1)

        return out

    def predict_response_matrix(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1)  # (batch, 1, expr_enc_size)
        drug_enc = drug_enc.unsqueeze(0)  # (1, n_drugs, drug_enc_size)

        expr_enc = expr_enc.repeat(
            1, drug_enc.shape[1], 1
        )  # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(
            expr_enc.shape[0], 1, 1
        )  # (batch, n_drugs, drug_enc_size)

        x = torch.cat(
            [expr_enc, drug_enc], -1
        )  # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x))  # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x)  # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1])  # (batch, n_drugs)
        return out
