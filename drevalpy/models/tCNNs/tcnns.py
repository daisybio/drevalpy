import warnings
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel


class tCNNs(DRPModel):
    """
    tCNNs model adapted from https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2910-6
    Liu et al.
    hyperparameters:

    """

    cell_line_views = ["gene_mutation_sequence"]
    drug_views = ["smiles_sequence"]
    early_stopping = True
    model_name = "tCNNs"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        """
        self.model = DrugCellModel()

    def train(
        self,
        output: DrugResponseDataset,
        gene_mutation_sequence: np.ndarray,
        smiles_sequence: np.ndarray,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        gene_mutation_sequence_earlystopping: Optional[np.ndarray] = None,
        smiles_sequence_earlystopping: Optional[np.ndarray] = None,
    ):
        """
        Trains the model.
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        :param gene_mutation_sequence: mutation sequence data
        :param smiles_sequence: smiles sequence data
        :param gene_mutation_sequence_earlystopping: mutation sequence data for early stopping
        :param smiles_sequence_earlystopping: smiles sequence data for early stopping
        """

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            dataset = DatasettCNNs(
                gene_mutation_sequence, smiles_sequence, output.response
            )
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                persistent_workers=True,
            )
            if all(
                [
                    ar is not None
                    for ar in [
                        output_earlystopping,
                        gene_mutation_sequence_earlystopping,
                        smiles_sequence_earlystopping,
                    ]
                ]
            ):
                dataset_earlystopping = DatasettCNNs(
                    gene_mutation_sequence_earlystopping,
                    smiles_sequence_earlystopping,
                    output_earlystopping.response,
                )
                early_stopping_loader = DataLoader(
                    dataset=dataset_earlystopping,
                    batch_size=batch_size,
                    shuffle=False,
                )

            trainer = pl.Trainer(
                max_epochs=100,
                progress_bar_refresh_rate=0,
                gpus=1 if torch.cuda.is_available() else 0,
            )
            trainer.fit(self.model, train_loader, val_dataloaders=early_stopping_loader)

            # TODO define trainer properlz and early stopinng via callback

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
        self, gene_mutation_sequence: np.ndarray, smiles_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        """
        return self.model.predict(gene_mutation_sequence, smiles_sequence)

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the data
        :return: FeatureDataset
        """

        pass

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        pass


class DatasettCNNs(Dataset):
    def __init__(
        self,
        gene_mutation_sequence: np.ndarray,
        smiles_sequence: np.ndarray,
        response: np.ndarray,
    ):
        self.gene_mutation_sequence = gene_mutation_sequence
        self.smiles_sequence = smiles_sequence
        self.response = response

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.gene_mutation_sequence[idx], dtype=torch.float32),
            torch.tensor(self.smiles_sequence[idx], dtype=torch.float32),
            torch.tensor(self.response[idx], dtype=torch.float32),
        )


# Custom DataModule class
class DrugCellDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        drug_smile_dict,
        drug_cell_dict,
        cell_mut_dict,
        label_list,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.drug_smile_dict = drug_smile_dict
        self.drug_cell_dict = drug_cell_dict
        self.cell_mut_dict = cell_mut_dict
        self.label_list = label_list
        self.positions = drug_cell_dict["positions"]
        np.random.shuffle(self.positions)

    def setup(self, stage=None):
        size = len(self.positions)
        len1 = int(size * 0.8)
        len2 = int(size * 0.9)
        train_positions, valid_positions, test_positions = (
            self.positions[:len1],
            self.positions[len1:len2],
            self.positions[len2:],
        )

        value_shape = self.drug_cell_dict["IC50"].shape
        value = np.zeros((value_shape[0], value_shape[1], len(self.label_list)))
        for i in range(len(self.label_list)):
            value[:, :, i] = self.drug_cell_dict[self.label_list[i]]
        drug_smile = self.drug_smile_dict["canonical"]
        cell_mut = self.cell_mut_dict["cell_mut"]

        self.train_dataset = DrugCellDataset(
            drug_smile, cell_mut, value, train_positions
        )
        self.valid_dataset = DrugCellDataset(
            drug_smile, cell_mut, value, valid_positions
        )
        self.test_dataset = DrugCellDataset(drug_smile, cell_mut, value, test_positions)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class DrugCellModel(pl.LightningModule):
    def __init__(self, hidden_dim, conv1_out, conv2_out, conv3_out):
        super(DrugCellModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.drug_conv1 = nn.Conv1d(
            29, 40, 7, padding="same"
        )  # 29 is the input channels of the drug (alphabet size of the SMILES)
        self.drug_conv2 = nn.Conv1d(conv1_out, conv2_out, 7, padding="same")
        self.drug_conv3 = nn.Conv1d(conv2_out, conv3_out, 7, padding="same")

        self.cell_conv1 = nn.Conv1d(1, 40, 7, padding="same")
        self.cell_conv2 = nn.Conv1d(conv1_out, conv2_out, 7, padding="same")
        self.cell_conv3 = nn.Conv1d(conv2_out, conv3_out, 7, padding="same")

        self.fc1 = nn.Linear(1980, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, drug, cell):
        # Drug branch
        drug = F.relu(self.drug_conv1(drug))
        drug = F.max_pool1d(drug, kernel_size=3, stride=3)
        drug = F.relu(self.drug_conv2(drug))
        drug = F.max_pool1d(drug, kernel_size=3, stride=3)
        drug = F.relu(self.drug_conv3(drug))
        drug = F.max_pool1d(drug, kernel_size=3, stride=3)

        # Cell branch
        cell = cell.unsqueeze(
            1
        )  # Adding channel dimension (from (batch_size, seq_len) to (batch_size, 1, seq_len))
        cell = F.relu(self.cell_conv1(cell))
        cell = F.max_pool1d(cell, kernel_size=3, stride=3)
        cell = F.relu(self.cell_conv2(cell))
        cell = F.max_pool1d(cell, kernel_size=3, stride=3)
        cell = F.relu(self.cell_conv3(cell))
        cell = F.max_pool1d(cell, kernel_size=3, stride=3)

        # Merge branches
        merged = torch.cat((drug, cell), dim=2)
        merged = merged.view(merged.size(0), -1)
        # Fully connected layers
        merged = F.relu(self.fc1(merged))
        merged = self.dropout(merged)
        merged = F.relu(self.fc2(merged))
        merged = self.dropout(merged)
        output = torch.sigmoid(self.fc3(merged))

        return output.squeeze()

    def compute_loss(self, batch, stage):
        value, drug, cell = batch
        output = self.forward(drug, cell)
        loss = F.mse_loss(output, value.squeeze())
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.compute_loss(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


# Load data
batch_size = 32
label_list = ["IC50"]

drug_smile_dict = np.load(
    "data/drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True
).item()
drug_cell_dict = np.load(
    "data/drug_cell_interaction.npy", encoding="latin1", allow_pickle=True
).item()
cell_mut_dict = np.load(
    "data/cell_mut_matrix.npy", encoding="latin1", allow_pickle=True
).item()

# Initialize DataModule
data_module = DrugCellDataModule(
    batch_size, drug_smile_dict, drug_cell_dict, cell_mut_dict, label_list
)

hidden_dim = 128  # Define hidden dimension for the model
conv1_out = 40
conv2_out = 80
conv3_out = 60

# Initialize and train the model
model = DrugCellModel(hidden_dim, conv1_out, conv2_out, conv3_out)
trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, datamodule=data_module)
mse = trainer.test(model, datamodule=data_module)
