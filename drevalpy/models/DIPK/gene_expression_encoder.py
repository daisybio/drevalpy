"""Gene expression Autoencoder for DIPK model."""

from abc import ABC
from copy import deepcopy

import numpy as np
import torch
import torch.nn
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset


class GeneExpressionEncoder(nn.Module):
    """Gene expression encoder.

    Code adapted from the
    DIPK model https://github.com/user15632/DIPK.
    """

    def __init__(self, input_dim, latent_dim=512, h_dims=None, drop_out_rate=0.3):
        """Initialize the gene expression encoder.

        :param input_dim: input dimension
        :param latent_dim: latent dimension
        :param h_dims: hidden dimensions
        :param drop_out_rate: dropout rate
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

    def forward(self, input):
        """Forward pass of the gene expression encoder.

        :param input: input data
        :return: encoded data
        """
        result = self.encoder(input)
        embedding = functional.relu(self.bottleneck(result))
        return embedding


class GeneExpressionDecoder(nn.Module):
    """Gene expression decoder."""

    def __init__(self, input_dim, latent_dim=512, h_dims=None, drop_out_rate=0.3):
        """Initialize the gene expression decoder.

        :param input_dim: input dimension
        :param latent_dim: latent dimension
        :param h_dims: hidden dimensions
        :param drop_out_rate: dropout rate
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

    def forward(self, embedding):
        """
        Forward pass of the gene expression decoder.

        :param embedding: input data
        :return: decoded data
        """
        result = self.decoder_input(embedding)
        result = self.decoder(result)
        output = self.decoder_output(result)
        return output


class CollateFn:
    """Collate function for the DataLoader, either for training or testing."""

    def __call__(self, batch):
        """Collate the batch.

        :param batch: batch of PyG Data objects
        :returns: PyG Batch, gene features, and bionic features
        """
        batch_data = torch.stack(batch)
        return batch_data


class DataSet(Dataset, ABC):
    """Dataset class for gene expression data."""

    def __init__(self, data):
        """Initialize the dataset.

        :param data: data
        """
        self._data = data

    def __getitem__(self, idx):
        """Return the data at the given index.

        :param idx: index
        :return: data
        """
        data = self._data[idx]
        return data

    def __len__(self):
        """Return the length of the dataset.

        :return: length of the dataset
        """
        return len(self._data)


def _validation_loss(
    encoder: GeneExpressionEncoder,
    decoder: GeneExpressionDecoder,
    gene_expression_val_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    """Compute the reconstruction MSE of the validation matrix in chunks.

    Sum of squared errors and element count are accumulated exactly, so the result is identical
    to nn.MSELoss() over the whole matrix, but only one batch has to live on the device.

    :param encoder: encoder in eval mode
    :param decoder: decoder in eval mode
    :param gene_expression_val_tensor: validation matrix, kept on the host
    :param batch_size: number of rows moved to the device at a time
    :param device: device the models live on
    :return: mean squared reconstruction error, nan for an empty validation matrix
    """
    squared_error = 0.0
    elements = 0
    with torch.no_grad():
        for start in range(0, len(gene_expression_val_tensor), batch_size):
            stop = start + batch_size
            val_batch = gene_expression_val_tensor[start:stop].to(device)
            val_output = decoder(encoder(val_batch))
            squared_error += float(((val_output - val_batch) ** 2).sum().item())
            elements += val_batch.numel()
    if elements == 0:
        return float("nan")
    return squared_error / elements


def train_gene_expession_autoencoder(
    gene_expression_input: np.ndarray, gene_expression_input_early_stopping: np.ndarray, epochs_autoencoder: int = 100
) -> GeneExpressionEncoder:
    """Train the autoencoder model for gene expression data with early stopping.

    :param gene_expression_input: gene expression data
    :param gene_expression_input_early_stopping: validation data for early stopping
    :param epochs_autoencoder: number of epochs for training the autoencoder
    :return: trained encoder model
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
    # The full training matrix stays in host memory, only the mini batches are moved to the device
    # (the training loop below already calls .to(device) per batch). The matrix holds one row per
    # response, not per cell line, so it is large: a GDSC1 training fold has ~270k rows, which is
    # ~2.5 GB of float32 with the 2.3k genes of the default gene_expression_intersection list and
    # ~13 GB with a 12k gene list. A batch of 1024 rows is ~50 MB. Semantics are unchanged.
    my_collate = CollateFn()
    gene_expression_tensor = torch.from_numpy(np.asarray(gene_expression_input, dtype=np.float32))
    train_loader = DataLoader(
        DataSet(gene_expression_tensor), batch_size=batch_size, shuffle=True, collate_fn=my_collate
    )

    # prepare early stopping validation data (kept on the host as well, evaluated in chunks below)
    gene_expression_val_tensor = torch.from_numpy(np.asarray(gene_expression_input_early_stopping, dtype=np.float32))

    # early stopping parameters
    patience = 3
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Training DIPK autoencoder for gene expression data")
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
        val_loss = _validation_loss(encoder, decoder, gene_expression_val_tensor, batch_size, device)

        print(f"DIPK Autoenc. Epoch: {epoch_index}, Train Loss: {epoch_loss}, Val Loss: {val_loss}")

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"DIPK Autoenc. Early stopping triggered at epoch {epoch_index}")
                break

    encoder.eval()
    return encoder


def encode_gene_expression(gene_expression_input: np.ndarray, encoder: GeneExpressionEncoder) -> np.ndarray:
    """Encode gene expression data.

    :param gene_expression_input: gene expression data
    :param encoder: trained encoder model
    :return: encoded gene expression data
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
