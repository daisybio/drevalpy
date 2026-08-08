"""PaccMann model."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_select_gene_features

from .paccmann_network_v2 import PaccMannV2

# Atom-level SMILES tokenizer, copied verbatim from pytoda.smiles.processing.SMILES_TOKENIZER, which is what
# the original implementation tokenizes with. Splitting SMILES by character instead would break multi-character
# atoms: "Cl" and "Br" would collide with chlorine/bromine-free molecules that contain carbon or boron, and
# bracket atoms such as "[C@@H]" or "[Pt+2]" would fall apart into their individual characters.
SMILES_TOKENIZER = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|" r"-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def _tokenize_smiles(smiles: str) -> list[str]:
    """Split a SMILES string into atom-level tokens.

    :param smiles: SMILES string
    :return: list of tokens
    """
    return [token for token in SMILES_TOKENIZER.split(smiles) if token]


class PaccMann(DRPModel):
    """PaccMann model for drug response prediction.

    This DrEval wrapper combines cell line gene expression features and tokenized SMILES representations of drugs
    and uses the PaccMannV2 neural network to predict drug response values.

    This wrapper:
    - loads gene expression features for cell lines
    - loads SMILES strings for drugs
    - tokenizes SMILES into padded integer sequences
    - scales gene expression on training data only
    - trains a PaccMannV2 PyTorch model
    - keeps the weights of the epoch with the lowest loss on the early stopping set

    The original implementation (https://github.com/PaccMann/paccmann_predictor) trains for a fixed number of
    epochs and checkpoints the model whenever the loss on a held-out set improves, using that checkpoint as the
    final model. This wrapper follows the same procedure with the early stopping set: training always runs for
    the full epoch budget and the weights of the best epoch are restored at the end. There is no patience-based
    termination, since the original implementation does not stop early either.
    """

    early_stopping = True
    is_single_drug_model = False

    cell_line_views = ["gene_expression"]
    drug_views = ["smiles"]

    def __init__(self) -> None:
        """Initialize the PaccMann model wrapper.

        Initialized attributes:
            model: stores the PaccMann neural network
            hyperparameters: stores the passed hyperparameters
            device: CPU or GPU device
            gene_expression_scaler: scaler fitted on training gene expression
            smiles_to_idx: SMILES vocabulary
            padding_idx: index used for padding tokens
            unk_idx: index for unknown tokens
            smiles_padding_length: sequence length used for padding
            number_of_genes: number of gene features
        """
        super().__init__()
        self.model: PaccMannV2 | None = None
        self.hyperparameters = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gene_expression_scaler = StandardScaler()

        self.smiles_to_idx: dict[str, int] = {
            "<pad>": 0,
            "<unk>": 1,
        }
        self.padding_idx = 0
        self.unk_idx = 1

        self.smiles_padding_length: int | None = None
        self.number_of_genes: int | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """Return the model name.

        :return: model name
        """
        return "PaccMann"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load gene expression features.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing gene expression features
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            data_path=data_path,
            dataset_name=dataset_name,
            gene_list=self.hyperparameters.get("gene_list", "gene_list_paccmann_network_prop"),
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load raw SMILES features.

        Only the id and SMILES columns are read from the csv. The fingerprint columns
        (cactvs_fingerprint, fingerprint) are huge decimal bit-strings that are not needed
        here, and are skipped rather than loaded and dropped, since pandas' numeric type
        inference on them can raise an OverflowError.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing SMILES features
        """
        id_column = "pubchem_id"
        smiles_column = "canonical_smiles"

        data = pd.read_csv(
            f"{data_path}/{dataset_name}/drug_smiles.csv",
            usecols=[id_column, smiles_column],
            dtype={id_column: str, smiles_column: str},
        )
        data = data.drop_duplicates(subset=id_column, keep="first")

        features = {
            str(pubchem_id): {"smiles": np.array([smiles], dtype=object)}
            for pubchem_id, smiles in zip(data[id_column], data[smiles_column], strict=True)
        }

        return FeatureDataset(features=features, meta_info={"smiles": [smiles_column]})

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Store hyperparameters for later model initialization.

        The actual PaccMannV2 network is initialized in train(),
        because the number of genes depends on the loaded training data.

        :param hyperparameters: dictionary containing model hyperparameters
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters

    def _normalize_smiles_array(self, smiles_raw: np.ndarray) -> list[str]:
        """Convert SMILES output from FeatureDataset into a list of strings.

        :param smiles_raw: raw SMILES array
        :return: list of SMILES strings
        """
        smiles_raw = np.asarray(smiles_raw, dtype=object)

        if smiles_raw.ndim == 2 and smiles_raw.shape[1] == 1:
            smiles_raw = smiles_raw[:, 0]

        smiles_list = []
        for smile in smiles_raw:
            if smile is None:
                smiles_list.append("")
            else:
                smiles_list.append(str(smile))
        return smiles_list

    def _build_smiles_vocab(self, smiles_list: list[str]) -> None:
        """Build a token vocabulary from training SMILES strings.

        :param smiles_list: list of SMILES strings
        """
        for smile in smiles_list:  # Build vocabulary: "Cl", "C", "=" ... -> {"Cl": 2, "C": 3, "=": 4}
            for token in _tokenize_smiles(smile):
                if token not in self.smiles_to_idx:
                    self.smiles_to_idx[token] = len(self.smiles_to_idx)

    def _encode_smiles(self, smiles_list: list[str]) -> np.ndarray:
        """Encode SMILES strings as padded integer sequences.

        :param smiles_list: list of SMILES strings
        :return: encoded SMILES array
        :raises ValueError: if smiles_padding_length is not set
        """
        if self.smiles_padding_length is None:
            raise ValueError("smiles_padding_length is not set.")

        encoded = np.full(
            (len(smiles_list), self.smiles_padding_length),
            fill_value=self.padding_idx,
            dtype=np.int64,
        )

        for i, smile in enumerate(smiles_list):
            tokens = _tokenize_smiles(smile)
            token_ids = [self.smiles_to_idx.get(token, self.unk_idx) for token in tokens]  # "CCO" -> [2, 2, 3]
            token_ids = token_ids[: self.smiles_padding_length]
            encoded[i, : len(token_ids)] = token_ids  # Padding: [2,2,2] -> [2,2,3,0,0,0,...]

        return encoded

    def _encode_inputs(
        self,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn cell line and drug features into model input tensors.

        Gene expression is scaled with the scaler fitted on the training data and SMILES are encoded with the
        vocabulary built from the training data, so this may only be called after train() has fitted both.

        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :param cell_line_ids: array of cell line identifiers
        :param drug_ids: array of drug identifiers
        :return: tuple of the encoded SMILES tensor and the scaled gene expression tensor
        """
        gex = cell_line_input.get_feature_matrix("gene_expression", cell_line_ids)
        gex = np.asarray(gex, dtype=np.float32)
        gex = self.gene_expression_scaler.transform(gex).astype(np.float32)

        smiles = self._normalize_smiles_array(drug_input.get_feature_matrix("smiles", drug_ids))
        smiles_encoded = self._encode_smiles(smiles)

        return (
            torch.tensor(smiles_encoded, dtype=torch.long),
            torch.tensor(gex, dtype=torch.float32),
        )

    def _build_validation_loader(
        self,
        output_earlystopping: DrugResponseDataset | None,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset,
        batch_size: int,
    ) -> DataLoader | None:
        """Build a loader over the early stopping set, used to pick the best epoch.

        :param output_earlystopping: early stopping dataset, may be None
        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :param batch_size: batch size to use
        :return: DataLoader over the early stopping set, or None if there is nothing to evaluate
        """
        if output_earlystopping is None or len(output_earlystopping) == 0:
            return None

        smiles_tensor, gex_tensor = self._encode_inputs(
            cell_line_input,
            drug_input,
            output_earlystopping.cell_line_ids,
            output_earlystopping.drug_ids,
        )
        y_tensor = torch.tensor(np.asarray(output_earlystopping.response, dtype=np.float32)).view(-1, 1)

        return DataLoader(
            TensorDataset(smiles_tensor, gex_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=False,
        )

    def _validation_loss(self, validation_loader: DataLoader) -> float:
        """Compute the mean loss over the early stopping set.

        :param validation_loader: DataLoader over the early stopping set
        :return: mean loss per batch
        :raises ValueError: if the model has not been built yet
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_smiles, batch_gex, batch_y in validation_loader:
                batch_smiles = batch_smiles.to(self.device)
                batch_gex = batch_gex.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions, _ = self.model(batch_smiles, batch_gex)
                total_loss += self.model.loss(predictions, batch_y).item()

        return total_loss / len(validation_loader)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str | None = None,
    ) -> None:
        """Train the PaccMann model on gene DrEval data.

        Procedure:
        - get gene expression data for all the cell lines
        - get raw SMILES for the drugs
        - scale gene expression features
        - build a SMILES vocabulary
        - encode and pad the SMILES strings
        - initialize the PaccMann network
        - convert both inputs to tensors
        - train the network for the full epoch budget, evaluating the early stopping set after every epoch
        - restore the weights of the epoch with the lowest loss on the early stopping set

        :param output: training dataset containing response values, cell line ids, and drug ids
        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :param output_earlystopping: dataset used to select the best epoch. If None, the weights of the last
            epoch are kept.
        :param model_checkpoint_dir: optional directory to save a model checkpoint
        :raises ValueError: if drug_input is None
        :raises ValueError: if the model has not been built yet
        """
        if drug_input is None:
            raise ValueError("drug_input (SMILES) is required for PaccMann.")

        if self.hyperparameters is None:
            raise ValueError("Model has not been built yet. Call build_model first.")

        # Retrieve gene expression features for the training cell lines
        gex = cell_line_input.get_feature_matrix("gene_expression", output.cell_line_ids)

        # Retrieve raw SMILES features for the corresponding drugs
        smiles_raw = drug_input.get_feature_matrix("smiles", output.drug_ids)

        # Target vector containing the drug response values
        y = output.response

        # Convert to numpy arrays with explicit dtypes
        gex = np.asarray(gex, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Convert SMILES to a list of strings
        smiles = self._normalize_smiles_array(smiles_raw)

        # Scale gene expression on training data only
        gex = self.gene_expression_scaler.fit_transform(gex).astype(np.float32)

        # Build SMILES vocabulary from training data only
        self.smiles_to_idx = {
            "<pad>": 0,
            "<unk>": 1,
        }
        self._build_smiles_vocab(smiles)

        # Determine SMILES padding length
        if "smiles_padding_length" in self.hyperparameters:
            self.smiles_padding_length = int(self.hyperparameters["smiles_padding_length"])
        else:
            self.smiles_padding_length = max(len(_tokenize_smiles(smile)) for smile in smiles)

        # Encode and pad SMILES strings
        smiles_encoded = self._encode_smiles(smiles)

        # Copy hyperparameters and adapt the number of genes to the training data
        model_params = dict(self.hyperparameters)
        model_params["number_of_genes"] = gex.shape[1]
        model_params["smiles_padding_length"] = self.smiles_padding_length
        model_params["smiles_vocabulary_size"] = len(self.smiles_to_idx)
        self.number_of_genes = gex.shape[1]

        # Build the PaccMann neural network
        self.model = PaccMannV2(model_params).to(self.device)

        # Convert all inputs to tensors
        smiles_tensor = torch.tensor(smiles_encoded, dtype=torch.long)
        gex_tensor = torch.tensor(gex, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Create PyTorch dataset and dataloader
        dataset = TensorDataset(smiles_tensor, gex_tensor, y_tensor)
        batch_size = model_params.get("batch_size", 64)

        # The batch norm layers cannot process a batch that holds a single sample, so a trailing batch of size 1
        # has to be dropped. The original implementation always drops the last batch; dropping it only when it
        # would contain a single sample keeps training sets smaller than one batch usable.
        drop_last = len(dataset) > batch_size and len(dataset) % batch_size == 1

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
        )

        # Initialize optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=model_params.get("learning_rate", 1e-3),
            weight_decay=model_params.get("weight_decay", 0.0),
        )

        epochs = model_params.get("epochs", 20)

        # The early stopping set is evaluated after every epoch to pick the best epoch
        validation_loader = self._build_validation_loader(
            output_earlystopping,
            cell_line_input,
            drug_input,
            batch_size,
        )

        best_validation_loss = float("inf")
        best_state_dict: dict[str, Any] | None = None

        # Train the model
        for epoch in range(epochs):
            self.model.train()
            for batch_smiles, batch_gex, batch_y in train_loader:
                batch_smiles = batch_smiles.to(self.device)
                batch_gex = batch_gex.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                predictions, _ = self.model(batch_smiles, batch_gex)
                loss = self.model.loss(predictions, batch_y)

                loss.backward()
                optimizer.step()

            if validation_loader is None:
                continue

            # Keep the weights of the epoch with the lowest loss on the early stopping set, which is what the
            # original implementation checkpoints. Training itself is never cut short.
            validation_loss = self._validation_loss(validation_loader)
            self.log_metrics({"validation_loss": validation_loss}, step=epoch)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())

        # Restore the weights of the best epoch
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        # Optional: save trained model checkpoint
        if model_checkpoint_dir is not None:
            self.model.save(f"{model_checkpoint_dir}/paccmann.pt")

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict drug response values.

        Procedure:
        - load appropriate cell lines features and drug features
        - scale gene expression features
        - encode and pad SMILES strings
        - convert inputs to tensors
        - run the trained PaccMann model in evaluation mode
        - return predicted drug response values

        :param cell_line_ids: array of cell line identifiers
        :param drug_ids: array of drug identifiers
        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :return: predicted drug response values
        :raises ValueError: if drug_input is None
        :raises ValueError: if the model has not been trained yet
        """
        if drug_input is None:
            raise ValueError("drug_input (SMILES) is required for PaccMann.")

        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Scale gene expression and encode SMILES with the training scaler and vocabulary.
        # These are CPU tensors; batches are moved to the device one at a time below.
        smiles_tensor, gex_tensor = self._encode_inputs(cell_line_input, drug_input, cell_line_ids, drug_ids)

        dataset = TensorDataset(smiles_tensor, gex_tensor)
        predict_loader = DataLoader(
            dataset,
            batch_size=self.hyperparameters.get("batch_size", 64),
            shuffle=False,
        )

        # Predict drug response values batch-wise
        self.model.eval()
        predictions_list = []
        with torch.no_grad():
            for batch_smiles, batch_gex in predict_loader:
                batch_smiles = batch_smiles.to(self.device)
                batch_gex = batch_gex.to(self.device)
                batch_predictions, _ = self.model(batch_smiles, batch_gex)
                predictions_list.append(batch_predictions.cpu())

        predictions = torch.cat(predictions_list, dim=0)
        return predictions.numpy().reshape(-1)

    def save(self, path: str) -> None:
        """Save the trained PaccMann wrapper.

        Saved files:
            - model.pt: trained model weights
            - config.json: model hyperparameters
            - scaler.pkl: fitted gene expression scaler
            - vocab.json: SMILES vocabulary
            - meta.json: additional metadata needed for loading

        :param path: directory where the model should be saved
        :raises ValueError: if no model is available
        """
        os.makedirs(path, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save.")

        torch.save(self.model.state_dict(), f"{path}/model.pt")

        with open(f"{path}/config.json", "w") as f:
            json.dump(self.hyperparameters, f)

        joblib.dump(self.gene_expression_scaler, f"{path}/scaler.pkl")

        with open(f"{path}/vocab.json", "w") as f:
            json.dump(self.smiles_to_idx, f)

        with open(f"{path}/meta.json", "w") as f:
            json.dump(
                {
                    "padding_length": self.smiles_padding_length,
                    "num_genes": self.number_of_genes,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> PaccMann:
        """Load a trained PaccMann wrapper.

        :param path: directory containing the saved model files
        :return: loaded PaccMann instance
        """
        instance = cls()

        with open(f"{path}/config.json") as f:
            instance.hyperparameters = json.load(f)

        instance.gene_expression_scaler = joblib.load(f"{path}/scaler.pkl")

        with open(f"{path}/vocab.json") as f:
            instance.smiles_to_idx = json.load(f)

        with open(f"{path}/meta.json") as f:
            meta = json.load(f)
            instance.smiles_padding_length = meta["padding_length"]
            instance.number_of_genes = meta["num_genes"]

        params = dict(instance.hyperparameters)
        params["smiles_padding_length"] = instance.smiles_padding_length
        params["smiles_vocabulary_size"] = len(instance.smiles_to_idx)
        params["number_of_genes"] = instance.number_of_genes

        instance.model = PaccMannV2(params).to(instance.device)
        instance.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=instance.device))  # noqa: S614
        instance.model.eval()

        return instance
