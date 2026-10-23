"""Gene expression Autoencoder for DIPK model."""

import logging
from abc import ABC
from copy import deepcopy

import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from drevalpy.experiment._tracking import log_metrics

logger = logging.getLogger(__name__)


class GeneExpressionEncoder(nn.Module):
    """Gene expression encoder.

    Code adapted from the
    DIPK model https://github.com/user15632/DIPK.
    """

    def __init__(
        self, input_dim: int, latent_dim: int = 512, h_dims: list[int] | None = None, drop_out_rate: float = 0.3
    ) -> None:
        """Initialize the gene expression encoder.

        :param input_dim: Number of input genes.
        :param latent_dim: Size of the latent embedding returned by the bottleneck.
        :param h_dims: Widths of the hidden layers between input and bottleneck; defaults to [2048, 1024].
        :param drop_out_rate: Dropout probability after each hidden layer.
        """
        super().__init__()
        if h_dims is None:
            h_dims = [2048, 1024]
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(drop_out_rate),
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Linear(hidden_dims[-1], latent_dim)
        self.latent_dim = latent_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Encode gene expression into the latent space.

        :param input: Gene expression tensor of shape [batch, input_dim].
        :returns: Latent embedding of shape [batch, latent_dim].
        """
        result = self.encoder(input)
        embedding = functional.relu(self.bottleneck(result))
        return embedding


class GeneExpressionDecoder(nn.Module):
    """Gene expression decoder."""

    def __init__(
        self, input_dim: int, latent_dim: int = 512, h_dims: list[int] | None = None, drop_out_rate: float = 0.3
    ) -> None:
        """Initialize the gene expression decoder.

        :param input_dim: Number of genes to reconstruct.
        :param latent_dim: Size of the latent embedding that is decoded.
        :param h_dims: Widths of the hidden layers of the matching encoder; the decoder mirrors them. Defaults to
            [2048, 1024].
        :param drop_out_rate: Dropout probability after each hidden layer.
        """
        super().__init__()
        if h_dims is None:
            h_dims = [2048, 1024]
        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(drop_out_rate),
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.decoder_output = nn.Linear(hidden_dims[-2], hidden_dims[-1])

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Reconstruct gene expression from a latent embedding.

        :param embedding: Latent embedding of shape [batch, latent_dim].
        :returns: Reconstructed gene expression of shape [batch, input_dim].
        """
        result = self.decoder_input(embedding)
        result = self.decoder(result)
        output = self.decoder_output(result)
        return output


class CollateFn:
    """Collate function for the DataLoader, either for training or testing."""

    def __call__(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Stack the gene expression vectors of a batch.

        :param batch: Gene expression vectors, one per sample.
        :returns: Tensor of shape [len(batch), n_genes].
        """
        batch_data = torch.stack(batch)
        return batch_data


class DataSet(Dataset, ABC):
    """Dataset class for gene expression data."""

    def __init__(self, data: torch.Tensor) -> None:
        """Initialize the dataset.

        :param data: Gene expression tensor with one row per sample.
        """
        self._data = data

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the data at the given index.

        :param idx: Row index.
        :returns: Gene expression vector of that row.
        """
        data = self._data[idx]
        return data

    def __len__(self) -> int:
        """Return the length of the dataset.

        :returns: Number of rows.
        """
        return len(self._data)


def train_gene_expession_autoencoder(
    gene_expression_input: np.ndarray, gene_expression_input_early_stopping: np.ndarray, epochs_autoencoder: int = 100
) -> GeneExpressionEncoder:
    """Train the autoencoder model for gene expression data with early stopping.

    :param gene_expression_input: Training gene expression, one row per sample.
    :param gene_expression_input_early_stopping: Held-out gene expression used to monitor the reconstruction loss.
    :param epochs_autoencoder: Maximum number of epochs; training stops earlier after 3 epochs without improvement.
    :returns: The trained encoder in eval mode.
    """
    lr = 1e-4
    batch_size = 1024
    noising = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    encoder = GeneExpressionEncoder(len(gene_expression_input[0])).to(device)
    decoder = GeneExpressionDecoder(len(gene_expression_input[0])).to(device)
    loss_func = nn.MSELoss()
    params = [{"params": encoder.parameters()}, {"params": decoder.parameters()}]
    optimizer = optim.Adam(params, lr=lr)

    # load data
    my_collate = CollateFn()
    gene_expression_tensor = torch.tensor(gene_expression_input, dtype=torch.float32).to(device)
    train_loader = DataLoader(
        DataSet(gene_expression_tensor), batch_size=batch_size, shuffle=True, collate_fn=my_collate
    )

    # prepare early stopping validation data
    gene_expression_val_tensor = torch.tensor(gene_expression_input_early_stopping, dtype=torch.float32).to(device)

    # early stopping parameters
    patience = 3
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    logger.info("Training DIPK autoencoder for gene expression data")
    for epoch_index in range(epochs_autoencoder):
        # training
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        batch_count = 0
        for gene_expression_batch in train_loader:
            gene_expression_batch = gene_expression_batch.to(device)
            if noising:
                z = gene_expression_batch.clone()
                y = np.random.binomial(1, 0.2, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype=bool)] = 0
                gene_expression_batch.requires_grad_(True)
                output = decoder(encoder(z))
            else:
                output = decoder(encoder(gene_expression_batch))
            loss = loss_func(output, gene_expression_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            batch_count += 1
        epoch_loss /= batch_count

        # validation
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            val_output = decoder(encoder(gene_expression_val_tensor))
            val_loss = loss_func(val_output, gene_expression_val_tensor).item()

        logger.info("DIPK Autoenc. Epoch: %d, Train Loss: %s, Val Loss: %s", epoch_index, epoch_loss, val_loss)
        log_metrics(
            {
                "autoencoder_epoch": epoch_index + 1,
                "autoencoder_train_loss": epoch_loss,
                "autoencoder_val_loss": val_loss,
            }
        )

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("DIPK Autoenc. Early stopping triggered at epoch %d", epoch_index)
                break

    encoder.eval()
    return encoder


def encode_gene_expression(gene_expression_input: np.ndarray, encoder: GeneExpressionEncoder) -> np.ndarray:
    """Encode gene expression data.

    :param gene_expression_input: Gene expression of shape [n_samples, n_genes], or a single vector of shape [n_genes].
    :param encoder: Trained encoder.
    :returns: Latent embeddings with the leading dimension of the input.
    """
    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)

    # Check the original input shape, because we have to unsqueeze the input if it is a single vector
    original_shape = gene_expression_input.shape

    gene_expression_tensor = torch.tensor(gene_expression_input, dtype=torch.float32).to(device)

    # Add batch dimension if input is a single vector
    if gene_expression_tensor.ndim == 1:
        gene_expression_tensor = gene_expression_tensor.unsqueeze(0)

    with torch.no_grad():
        encoded_data = encoder(gene_expression_tensor).cpu().numpy()

    # Match the output shape to the input shape
    if len(original_shape) == 1:
        encoded_data = encoded_data.squeeze(0)

    return encoded_data
