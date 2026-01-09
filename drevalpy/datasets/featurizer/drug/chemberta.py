"""ChemBERTa drug featurizer for generating embeddings from SMILES strings."""

import argparse

import numpy as np
import torch

from drevalpy.datasets.dataset import FeatureDataset

from .base import DrugFeaturizer


class ChemBERTaFeaturizer(DrugFeaturizer):
    """Featurizer that generates ChemBERTa embeddings from SMILES strings.

    ChemBERTa is a transformer model pre-trained on chemical SMILES strings.
    This featurizer uses the model to generate fixed-size embeddings for drugs.

    Example usage::

        featurizer = ChemBERTaFeaturizer(device="cuda")
        features = featurizer.load_or_generate("data", "GDSC1")
    """

    def __init__(self, device: str = "cpu"):
        """Initialize the ChemBERTa featurizer.

        :param device: Device to use for computation ('cpu' or 'cuda')
        """
        super().__init__(device=device)
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazily load the ChemBERTa model and tokenizer.

        :raises ImportError: If transformers or torch packages are not installed
        """
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "Please install transformers package for ChemBERTa featurizer: pip install transformers torch"
                )

            self._tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            self._model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
            self._model.to(self.device)
            self._model.eval()

    def featurize(self, smiles: str) -> np.ndarray:
        """Convert a SMILES string to a ChemBERTa embedding.

        :param smiles: SMILES string representing the drug
        :returns: ChemBERTa embedding as numpy array
        :raises RuntimeError: If model is not loaded
        """
        self._load_model()

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        inputs = self._tokenizer(smiles, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden_states = outputs.last_hidden_state

        # Mean pooling over sequence length
        embedding = hidden_states.mean(dim=1).squeeze(0)
        return embedding.cpu().numpy()

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the feature view name.

        :returns: 'chemberta_embeddings'
        """
        return "chemberta_embeddings"

    @classmethod
    def get_output_filename(cls) -> str:
        """Return the output filename for cached embeddings.

        :returns: 'drug_chemberta_embeddings.csv'
        """
        return "drug_chemberta_embeddings.csv"


class ChemBERTaMixin:
    """Mixin that provides ChemBERTa drug embeddings loading for DRP models.

    This mixin implements load_drug_features using the ChemBERTaFeaturizer.
    It automatically generates embeddings if they don't exist.

    Class attributes that can be overridden:
        - chemberta_device: Device for ChemBERTa model ('cpu', 'cuda', or 'auto')

    Example usage::

        from drevalpy.models.drp_model import DRPModel
        from drevalpy.datasets.featurizer.drug.chemberta import ChemBERTaMixin

        class MyModel(ChemBERTaMixin, DRPModel):
            drug_views = ["chemberta_embeddings"]
            ...
    """

    chemberta_device: str = "auto"

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load ChemBERTa drug embeddings.

        Uses the ChemBERTaFeaturizer to load pre-generated embeddings or generate
        them automatically if they don't exist.

        :param data_path: Path to the data directory, e.g., 'data/'
        :param dataset_name: Name of the dataset, e.g., 'GDSC1'
        :returns: FeatureDataset containing the ChemBERTa embeddings
        """
        device = self.chemberta_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        featurizer = ChemBERTaFeaturizer(device=device)
        return featurizer.load_or_generate(data_path, dataset_name)


def main():
    """Process drug SMILES and save ChemBERTa embeddings from command line."""
    parser = argparse.ArgumentParser(description="Generate ChemBERTa embeddings for drugs.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda)")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    args = parser.parse_args()

    featurizer = ChemBERTaFeaturizer(device=args.device)
    featurizer.generate_embeddings(args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()
