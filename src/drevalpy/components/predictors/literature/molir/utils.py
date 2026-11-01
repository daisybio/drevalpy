"""Utility functions for the MOLIR model.

Original authors of MOLI: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7),
https://github.com/kramerlab/Multi-Omics_analysis
"""

from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from upath import UPath as Path

from drevalpy._utils.torch_io import load_state_dict
from drevalpy.components.predictors._lightning_training import LossSteps
from drevalpy.components.predictors.literature._omics_loaders import OmicsSplit, fit_omics_model


def generate_triplets_indices(
    y: np.ndarray,
    positive_range: float,
    negative_range: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate triplets for the MOLIR model.

    The positive and negative range are determined by the standard deviation of the response values. A pair is
    considered positive if its response value is within the positive range of the label. The positive range is ±10%
    of the standard deviation of all response values. A pair is considered negative if its response value is at
    least one standard deviation away from the response value of the pair.

    :param y: response values
    :param positive_range: positive range for the triplet loss
    :param negative_range: negative range for the triplet loss
    :param random_seed: Explicit isolated seed, or None to use the global NumPy stream, which drevalpy seeds per fit.

    :returns: positive and negative pair indices for each pair
    """
    rng = np.random if random_seed is None else np.random.default_rng(random_seed)
    positive_pair_indices = []
    negative_pair_indices = []
    # Iterate over each label in the dataset
    for idx_current_label, current_label in enumerate(y):
        positive_class_indices = _get_positive_class_indices(current_label, idx_current_label, y, positive_range)
        positive_pair_idx = rng.choice(positive_class_indices, 1)[0]
        negative_class_indices = _get_negative_class_indices(current_label, y, negative_range)
        negative_pair_idx = rng.choice(negative_class_indices, 1)[0]
        positive_pair_indices.append(positive_pair_idx)
        negative_pair_indices.append(negative_pair_idx)
    return np.array(positive_pair_indices), np.array(negative_pair_indices)


def _get_positive_class_indices(label: np.float32, idx_label: int, y: np.ndarray, positive_range: float) -> np.ndarray:
    """Find the pairs that are within the positive range of the label except the label itself.

    If there is no similar pair within the positive range, the method returns the closest pair to the label.

    :param label: response of interest
    :param idx_label: index of the response of interest
    :param y: all responses
    :param positive_range: 0.1 * the standard deviation of all training responses

    :returns: Indices of pairs considered positive examples for the response of interest.
    """
    indices_similar_pairs = np.where(np.logical_and(label - positive_range <= y, y <= label + positive_range))[0]
    indices_similar_pairs = np.delete(indices_similar_pairs, np.where(indices_similar_pairs == idx_label))
    if len(indices_similar_pairs) == 0 and len(y) > 1:
        # return the closest pairs to the label except the label itself
        indices_similar_pairs = np.array([np.argsort(np.abs(y - label))[1]])
    elif len(indices_similar_pairs) == 0:
        # this happens only when there is just one pair in the validation dataset
        indices_similar_pairs = np.where(y == label)[0]
    return indices_similar_pairs


def _get_negative_class_indices(label: np.float32, y: np.ndarray, negative_range: float) -> np.ndarray:
    """Find dissimilar pairs to the label.

    If there is no dissimilar pair within the negative range, the method returns the pair that is the furthest away.

    :param label: reponse of interest
    :param y: all responses
    :param negative_range: 1 * the standard deviation of all training responses

    :returns: Indices of pairs considered negative examples for the response of interest.
    """
    dissimilar_pairs = np.where(np.logical_or(label - negative_range >= y, y >= label + negative_range))[0]
    if len(dissimilar_pairs) == 0:
        # return the pair that is the furthest away from the label
        dissimilar_pairs = np.argsort(np.abs(y - label))[-1:]
    return dissimilar_pairs


class MOLIEncoder(nn.Module):
    """Encoders of the MOLIR model, which is identical to the encoders of the original MOLI model.

    The MOLIR model has three encoders for the gene expression, mutations, and copy number variation data which are
    trained together.
    """

    def __init__(self, input_size: int, output_size: int, dropout_rate: float) -> None:
        """Initialize the encoder for the MOLIR model.

        :param input_size: input size determined by feature selection.
        :param output_size: output size of the encoder, set as hyperparameter.
        :param dropout_rate: dropout rate for regularization, set as hyperparameter.
        """
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder.

        :param x: omic input features

        :returns: encoded omic features
        """
        return self.encode(x)


class MOLIRegressor(nn.Module):
    """Regressor of the MOLIR model.

    It is identical to the regressor of the original MOLI model, except for the omission of the final sigmoid
    activation function. After the three encoders, the encoded features are concatenated and fed into the regressor.
    """

    def __init__(self, input_size: int, dropout_rate: float) -> None:
        """Initialize the regressor for the MOLIR model.

        :param input_size: determined by the output sizes of the encoders.
        :param dropout_rate: set as hyperparameter.
        """
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the regressor.

        :param x: concatenated encoded features

        :returns: predicted drug response
        """
        return self.regressor(x)


class OmicsRegressionModule(pl.LightningModule):
    """Lightning module that encodes the three omic views, concatenates them and regresses on the result.

    The prediction of MOLIR and of the SuperFELTR regressor.
    """

    regressor: nn.Module

    @abstractmethod
    def _encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """Encode the three views and concatenate the encodings.

        :param gene_expression: Gene expression features of the batch.
        :param mutations: Mutation features of the batch.
        :param copy_number: Copy number variation features of the batch.
        :returns: The input of the regression head.
        """

    def predict(self, gene_expression: np.ndarray, mutations: np.ndarray, copy_number: np.ndarray) -> np.ndarray:
        """Predict the response of each row in eval mode.

        :param gene_expression: Gene expression features, one row per pair.
        :param mutations: Mutation features, one row per pair.
        :param copy_number: Copy number variation features, one row per pair.
        :returns: Predicted response per pair.
        """
        tensors = [torch.from_numpy(data).float().to(self.device) for data in (gene_expression, mutations, copy_number)]
        self.eval()
        with torch.no_grad():
            preds = self.regressor(self._encode_and_concatenate(*tensors))
        return preds.squeeze().cpu().detach().numpy()


class MOLIModel(LossSteps, OmicsRegressionModule):
    """PyTorch Lightning module for the MOLIR model.

    The architecture of the MOLIR model is identical to the MOLI model, except for the omission of the final sigmoid
    layer and the usage of a regression MSE loss instead of a binary cross-entropy loss. Additionally, early stopping is
    added instead of tuning the number of epochs as hyperparameter.
    """

    def __init__(
        self, hpams: dict[str, int | float], input_dim_expr: int, input_dim_mut: int, input_dim_cnv: int
    ) -> None:
        """Initialize the MOLIR model.

        The MOLIR model uses a combined loss function of a triplet margin loss for the concatenated representation
        and an MSE loss for the regression loss.

        :param hpams: Hyperparameters such as ``mini_batch``, layer sizes, learning rate, and
            margin.
        :param input_dim_expr: determined by the feature selection of the gene expression data.
        :param input_dim_mut: determined by dataset size
        :param input_dim_cnv: determined by dataset size
        """
        super().__init__()
        self.save_hyperparameters()

        self.mini_batch = int(hpams["mini_batch"])
        self.h_dim1 = int(hpams["h_dim1"])
        self.h_dim2 = int(hpams["h_dim2"])
        self.h_dim3 = int(hpams["h_dim3"])
        self.lr = hpams["learning_rate"]
        self.dropout_rate = hpams["dropout_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = int(hpams["epochs"])
        self.triplet_loss = nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.regression_loss = nn.MSELoss()
        # Positive and Negative range for triplet loss, determined by the standard deviation of the training responses,
        # set in fit method
        self.positive_range = 1.0
        self.negative_range = 1.0
        # Checkpoint callback for early stopping, set in fit method
        self.checkpoint_callback: pl.callbacks.ModelCheckpoint | None = None

        self.expression_encoder = MOLIEncoder(input_dim_expr, self.h_dim1, self.dropout_rate)
        self.mutation_encoder = MOLIEncoder(input_dim_mut, self.h_dim2, self.dropout_rate)
        self.cna_encoder = MOLIEncoder(input_dim_cnv, self.h_dim3, self.dropout_rate)
        self.regressor = MOLIRegressor(self.h_dim1 + self.h_dim2 + self.h_dim3, self.dropout_rate)

    def fit(
        self,
        train: OmicsSplit,
        val: OmicsSplit | None = None,
        patience: int = 5,
        *,
        model_checkpoint_dir: str | Path,
    ) -> None:
        """Train the MOLIR model.

        First, the ranges for the triplet loss are determined using the standard deviation of the training responses.
        Then, the training and validation data loaders are created. The model is trained using the Lightning Trainer
        with an early stopping callback.

        :param train: training split of the three omic views
        :param val: validation split for early stopping, or None to monitor the training loss
        :param patience: for early stopping
        :param model_checkpoint_dir: directory to save the model checkpoints
        """
        std = float(np.std(train.response))
        self.positive_range = std * 0.1
        self.negative_range = std

        self.checkpoint_callback = fit_omics_model(
            self,
            train,
            val,
            batch_size=self.mini_batch,
            epochs=self.epochs,
            patience=patience,
            checkpoint_dir=model_checkpoint_dir,
            save_weights_only=True,
            enable_model_summary=False,
        )
        self.load_state_dict(load_state_dict(self.checkpoint_callback.best_model_path)["state_dict"])

    def _encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """Encode the input modalities, concatenates, and normalizes the resulting embeddings.

        :param gene_expression: gene expression data
        :param mutations: mutation data
        :param copy_number: copy number variation data

        :returns: concatenated, normalized embeddings
        """
        z_ex = self.expression_encoder(gene_expression)
        z_mu = self.mutation_encoder(mutations)
        z_cn = self.cna_encoder(copy_number)

        z = torch.cat((z_ex, z_mu, z_cn), dim=1)
        z = nn.functional.normalize(z, p=2, dim=0)
        return z

    def forward(self, x_gene: torch.Tensor, x_mutation: torch.Tensor, x_cna: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MOLIR model.

        :param x_gene: gene expression input
        :param x_mutation: mutation input
        :param x_cna: copy number variation input

        :returns: predicted drug response
        """
        z = self._encode_and_concatenate(x_gene, x_mutation, x_cna)
        preds = self.regressor(z)
        return preds

    def _compute_loss(self, z: torch.Tensor, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the combined triplet loss and regression loss.

        :param z: concatenated, normalized embeddings on which the triplet loss is calculated
        :param preds: predicted drug response on which the regression loss is calculated
        :param y: true drug response

        :returns: combined loss
        """
        positive_indices, negative_indices = generate_triplets_indices(
            y.cpu().detach().numpy(), self.positive_range, self.negative_range
        )

        triplet_loss = self.triplet_loss(z, z[positive_indices], z[negative_indices])
        regression_loss = self.regression_loss(preds.squeeze(), y)
        return self.gamma * triplet_loss + regression_loss

    def _loss_and_log(self, batch: list[torch.Tensor], key: str) -> torch.Tensor:
        """Compute and log the combined loss of one batch.

        :param batch: batch of gene expression, mutations, copy number variation, and response
        :param key: name the loss is logged under

        :returns: combined loss
        """
        gene_expression, mutations, copy_number, response = batch
        response = response.squeeze(-1)
        z = self._encode_and_concatenate(gene_expression, mutations, copy_number)
        preds = self.regressor(z)
        loss = self._compute_loss(z, preds, response)
        self.log(key, loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the optimizer for Lightning.

        :returns: optimizers for the MOLIR expression, mutation, copy number variation encoders, and regressor
        """
        optimizer = torch.optim.Adagrad(
            [
                {"params": self.expression_encoder.parameters(), "lr": self.lr},
                {"params": self.mutation_encoder.parameters(), "lr": self.lr},
                {"params": self.cna_encoder.parameters(), "lr": self.lr},
                {"params": self.regressor.parameters(), "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer
