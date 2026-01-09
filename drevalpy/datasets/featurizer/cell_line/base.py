"""Abstract base class for cell line featurizers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


class CellLineFeaturizer(ABC):
    """Abstract base class for cell line featurizers.

    Cell line featurizers convert omics data (e.g., gene expression, methylation)
    into numerical embeddings that can be used as input features for machine learning models.

    Supports both single-omics and multi-omics featurization through the `omics_types`
    parameter.

    Subclasses must implement:
        - featurize(): Convert omics data for a single cell line to its embedding
        - get_feature_name(): Return the name of the feature view
        - get_output_filename(): Return the filename for cached embeddings

    The base class provides:
        - load_or_generate(): Load cached embeddings or generate and cache them
        - generate_embeddings(): Generate embeddings for all cell lines in a dataset
        - load_embeddings(): Load pre-generated embeddings from disk
    """

    # Supported omics types and their corresponding file names
    OMICS_FILE_MAPPING = {
        "gene_expression": "gene_expression.csv",
        "methylation": "methylation.csv",
        "mutations": "mutations.csv",
        "copy_number_variation": "copy_number_variation.csv",
    }

    def __init__(self, omics_types: list[str] | str = "gene_expression"):
        """Initialize the featurizer.

        :param omics_types: Single omics type or list of omics types to use.
                           Supported types: 'gene_expression', 'methylation',
                           'mutations', 'copy_number_variation'
        :raises ValueError: If an unsupported omics type is provided
        """
        if isinstance(omics_types, str):
            omics_types = [omics_types]

        for omics_type in omics_types:
            if omics_type not in self.OMICS_FILE_MAPPING:
                raise ValueError(
                    f"Unsupported omics type: {omics_type}. " f"Supported types: {list(self.OMICS_FILE_MAPPING.keys())}"
                )

        self.omics_types = omics_types

    @abstractmethod
    def featurize(self, omics_data: dict[str, np.ndarray]) -> np.ndarray | Any:
        """Convert omics data to a feature representation.

        :param omics_data: Dictionary mapping omics type to data array for a single cell line
        :returns: Feature representation (numpy array or other format)
        """

    @classmethod
    @abstractmethod
    def get_feature_name(cls) -> str:
        """Return the name of the feature view.

        This name is used as the key in the FeatureDataset.

        :returns: Feature view name (e.g., 'gene_expression_pca')
        """

    @abstractmethod
    def get_output_filename(self) -> str:
        """Return the filename for cached embeddings.

        Note: This is an instance method (not classmethod) because the filename
        may depend on featurizer parameters (e.g., n_components for PCA).

        :returns: Filename (e.g., 'cell_line_gene_expression_pca_100.csv')
        """

    def load_or_generate(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load cached embeddings or generate and cache them if not available.

        This is the main entry point for using a featurizer. It checks if
        pre-generated embeddings exist and loads them, otherwise generates
        new embeddings and saves them for future use.

        :param data_path: Path to the data directory (e.g., 'data/')
        :param dataset_name: Name of the dataset (e.g., 'GDSC1')
        :returns: FeatureDataset containing the cell line embeddings
        """
        output_path = Path(data_path) / dataset_name / self.get_output_filename()

        if output_path.exists():
            return self.load_embeddings(data_path, dataset_name)
        else:
            print(f"Embeddings not found at {output_path}. Generating...")
            return self.generate_embeddings(data_path, dataset_name)

    def _load_omics_data(self, data_path: str, dataset_name: str) -> dict[str, pd.DataFrame]:
        """Load omics data files for the specified omics types.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: Dictionary mapping omics type to DataFrame
        :raises FileNotFoundError: If any required omics file is not found
        """
        data_dir = Path(data_path) / dataset_name
        omics_data = {}

        for omics_type in self.omics_types:
            filename = self.OMICS_FILE_MAPPING[omics_type]
            filepath = data_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Omics data file not found: {filepath}. " f"Please ensure the {omics_type} data is available."
                )

            df = pd.read_csv(filepath, dtype={CELL_LINE_IDENTIFIER: str})
            df = df.set_index(CELL_LINE_IDENTIFIER)
            omics_data[omics_type] = df

        return omics_data

    def generate_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Generate embeddings for all cell lines in a dataset and save to disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the generated embeddings
        """
        data_dir = Path(data_path).resolve()
        output_file = data_dir / dataset_name / self.get_output_filename()

        # Load omics data
        omics_data = self._load_omics_data(data_path, dataset_name)

        # Get common cell line IDs across all omics types
        cell_line_ids = None
        for _omics_type, df in omics_data.items():
            if cell_line_ids is None:
                cell_line_ids = set(df.index)
            else:
                cell_line_ids = cell_line_ids.intersection(set(df.index))

        cell_line_ids = sorted(list(cell_line_ids))
        print(f"Processing {len(cell_line_ids)} cell lines for dataset {dataset_name}...")

        # Generate embeddings
        embeddings_list = []
        valid_cell_line_ids = []

        for cell_line_id in cell_line_ids:
            try:
                # Prepare omics data for this cell line
                cell_omics = {}
                for omics_type, df in omics_data.items():
                    cell_omics[omics_type] = df.loc[cell_line_id].to_numpy(dtype=np.float32)

                embedding = self.featurize(cell_omics)
                embeddings_list.append(embedding)
                valid_cell_line_ids.append(cell_line_id)
            except Exception as e:
                print(f"Failed to process cell line {cell_line_id}: {e}")
                continue

        # Save embeddings
        self._save_embeddings(embeddings_list, valid_cell_line_ids, output_file, omics_data)

        print(f"Embeddings saved to {output_file}")

        # Return as FeatureDataset
        return self._create_feature_dataset(embeddings_list, valid_cell_line_ids)

    def _save_embeddings(
        self,
        embeddings: list,
        cell_line_ids: list[str],
        output_path: Path,
        omics_data: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        """Save embeddings to disk.

        Default implementation saves as CSV. Subclasses can override for other formats.

        :param embeddings: List of embedding arrays
        :param cell_line_ids: List of cell line identifiers
        :param output_path: Path to save the embeddings
        :param omics_data: Optional omics data (may be used by subclasses for saving fitted models)
        """
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(0, CELL_LINE_IDENTIFIER, cell_line_ids)
        embeddings_df.to_csv(output_path, index=False)

    def _create_feature_dataset(self, embeddings: list, cell_line_ids: list[str]) -> FeatureDataset:
        """Create a FeatureDataset from embeddings.

        :param embeddings: List of embedding arrays
        :param cell_line_ids: List of cell line identifiers
        :returns: FeatureDataset containing the embeddings
        """
        feature_name = self.get_feature_name()
        features = {}
        for cell_line_id, embedding in zip(cell_line_ids, embeddings, strict=True):
            if isinstance(embedding, np.ndarray):
                features[cell_line_id] = {feature_name: embedding.astype(np.float32)}
            else:
                features[cell_line_id] = {feature_name: embedding}
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

        embeddings_df = pd.read_csv(embeddings_file, dtype={CELL_LINE_IDENTIFIER: str})
        feature_name = self.get_feature_name()
        features = {}

        for _, row in embeddings_df.iterrows():
            cell_line_id = row[CELL_LINE_IDENTIFIER]
            embedding = row.drop(CELL_LINE_IDENTIFIER).to_numpy(dtype=np.float32)
            features[cell_line_id] = {feature_name: embedding}

        return FeatureDataset(features)


def main():
    """Entry point for running featurizer from command line.

    This function should be overridden by subclasses that support CLI usage.

    :raises NotImplementedError: Always, as subclasses should implement their own main()
    """
    raise NotImplementedError("Subclasses should implement their own main() function")


if __name__ == "__main__":
    main()
