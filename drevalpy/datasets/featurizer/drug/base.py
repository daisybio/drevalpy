"""Abstract base class for drug featurizers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import DRUG_IDENTIFIER


class DrugFeaturizer(ABC):
    """Abstract base class for drug featurizers.

    Drug featurizers convert drug representations (e.g., SMILES strings) into
    numerical embeddings that can be used as input features for machine learning models.

    Subclasses must implement:
        - featurize(): Convert a single drug to its embedding
        - get_feature_name(): Return the name of the feature view
        - get_output_filename(): Return the filename for cached embeddings

    The base class provides:
        - load_or_generate(): Load cached embeddings or generate and cache them
        - generate_embeddings(): Generate embeddings for all drugs in a dataset
        - load_embeddings(): Load pre-generated embeddings from disk
    """

    def __init__(self, device: str = "cpu"):
        """Initialize the featurizer.

        :param device: Device to use for computation (e.g., 'cpu', 'cuda')
        """
        self.device = device

    @abstractmethod
    def featurize(self, smiles: str) -> np.ndarray | Any:
        """Convert a SMILES string to a feature representation.

        :param smiles: SMILES string representing the drug
        :returns: Feature representation (numpy array or other format like torch_geometric.Data)
        """

    @classmethod
    @abstractmethod
    def get_feature_name(cls) -> str:
        """Return the name of the feature view.

        This name is used as the key in the FeatureDataset.

        :returns: Feature view name (e.g., 'chemberta_embeddings')
        """

    @classmethod
    @abstractmethod
    def get_output_filename(cls) -> str:
        """Return the filename for cached embeddings.

        :returns: Filename (e.g., 'drug_chemberta_embeddings.csv')
        """

    def load_or_generate(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load cached embeddings or generate and cache them if not available.

        This is the main entry point for using a featurizer. It checks if
        pre-generated embeddings exist and loads them, otherwise generates
        new embeddings and saves them for future use.

        :param data_path: Path to the data directory (e.g., 'data/')
        :param dataset_name: Name of the dataset (e.g., 'GDSC1')
        :returns: FeatureDataset containing the drug embeddings
        """
        output_path = Path(data_path) / dataset_name / self.get_output_filename()

        if output_path.exists():
            return self.load_embeddings(data_path, dataset_name)
        else:
            print(f"Embeddings not found at {output_path}. Generating...")
            return self.generate_embeddings(data_path, dataset_name)

    def generate_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Generate embeddings for all drugs in a dataset and save to disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the generated embeddings
        :raises FileNotFoundError: If the drug_smiles.csv file is not found
        """
        data_dir = Path(data_path).resolve()
        smiles_file = data_dir / dataset_name / "drug_smiles.csv"
        output_file = data_dir / dataset_name / self.get_output_filename()

        if not smiles_file.exists():
            raise FileNotFoundError(f"SMILES file not found: {smiles_file}")

        smiles_df = pd.read_csv(smiles_file, dtype={"canonical_smiles": str, DRUG_IDENTIFIER: str})

        embeddings_list = []
        drug_ids = []

        print(f"Processing {len(smiles_df)} drugs for dataset {dataset_name}...")

        for row in smiles_df.itertuples(index=False):
            drug_id = getattr(row, DRUG_IDENTIFIER)
            smiles = row.canonical_smiles

            try:
                embedding = self.featurize(smiles)
                embeddings_list.append(embedding)
                drug_ids.append(drug_id)
            except Exception as e:
                print(f"Failed to process drug {drug_id} (SMILES: {smiles}): {e}")
                continue

        # Save embeddings
        self._save_embeddings(embeddings_list, drug_ids, output_file)

        print(f"Embeddings saved to {output_file}")

        # Return as FeatureDataset
        return self._create_feature_dataset(embeddings_list, drug_ids)

    def _save_embeddings(self, embeddings: list, drug_ids: list[str], output_path: Path) -> None:
        """Save embeddings to disk.

        Default implementation saves as CSV. Subclasses can override for other formats.

        :param embeddings: List of embedding arrays
        :param drug_ids: List of drug identifiers
        :param output_path: Path to save the embeddings
        """
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(0, DRUG_IDENTIFIER, drug_ids)
        embeddings_df.to_csv(output_path, index=False)

    def _create_feature_dataset(self, embeddings: list, drug_ids: list[str]) -> FeatureDataset:
        """Create a FeatureDataset from embeddings.

        :param embeddings: List of embedding arrays
        :param drug_ids: List of drug identifiers
        :returns: FeatureDataset containing the embeddings
        """
        feature_name = self.get_feature_name()
        features = {}
        for drug_id, embedding in zip(drug_ids, embeddings, strict=True):
            if isinstance(embedding, np.ndarray):
                features[drug_id] = {feature_name: embedding.astype(np.float32)}
            else:
                features[drug_id] = {feature_name: embedding}
        return FeatureDataset(features)

    def load_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load pre-generated embeddings from disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the embeddings
        :raises FileNotFoundError: If the embeddings file is not found
        """
        embeddings_file = Path(data_path) / dataset_name / self.get_output_filename()

        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}. "
                f"Use load_or_generate() to automatically generate embeddings."
            )

        embeddings_df = pd.read_csv(embeddings_file, dtype={DRUG_IDENTIFIER: str})
        feature_name = self.get_feature_name()
        features = {}

        for _, row in embeddings_df.iterrows():
            drug_id = row[DRUG_IDENTIFIER]
            embedding = row.drop(DRUG_IDENTIFIER).to_numpy(dtype=np.float32)
            features[drug_id] = {feature_name: embedding}

        return FeatureDataset(features)


def main():
    """Entry point for running featurizer from command line.

    This function should be overridden by subclasses that support CLI usage.

    :raises NotImplementedError: Always, as subclasses should implement their own main()
    """
    raise NotImplementedError("Subclasses should implement their own main() function")


if __name__ == "__main__":
    main()
