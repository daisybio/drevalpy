"""PCA featurizer for cell line gene expression data."""

import argparse
import pickle  # noqa: S403
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER

from .base import CellLineFeaturizer


class PCAFeaturizer(CellLineFeaturizer):
    """Featurizer that applies PCA to gene expression data.

    This featurizer standardizes gene expression data and applies PCA
    to reduce dimensionality. It is designed specifically for transcriptomics
    (gene expression) data.

    Example usage::

        featurizer = PCAFeaturizer(n_components=100)
        features = featurizer.load_or_generate("data", "GDSC1")
    """

    def __init__(self, n_components: int = 100):
        """Initialize the PCA featurizer.

        :param n_components: Number of principal components to keep
        """
        super().__init__(omics_types="gene_expression")
        self.n_components = n_components
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._fitted = False

    def featurize(self, omics_data: dict[str, np.ndarray]) -> np.ndarray:
        """Apply PCA transformation to gene expression data.

        :param omics_data: Dictionary with 'gene_expression' key containing the data
        :returns: PCA-transformed features
        :raises RuntimeError: If the PCA model is not fitted
        :raises ValueError: If gene_expression data is not provided
        """
        if not self._fitted:
            raise RuntimeError("PCA model is not fitted. Call generate_embeddings() or fit() first.")

        if "gene_expression" not in omics_data:
            raise ValueError("gene_expression data is required for PCA featurizer")

        data = omics_data["gene_expression"].reshape(1, -1)
        scaled = self._scaler.transform(data)
        return self._pca.transform(scaled).flatten()

    def fit(self, gene_expression_df: pd.DataFrame) -> None:
        """Fit the scaler and PCA model on gene expression data.

        :param gene_expression_df: DataFrame with cell lines as rows and genes as columns
        """
        data = gene_expression_df.values

        self._scaler = StandardScaler()
        scaled_data = self._scaler.fit_transform(data)

        n_components = min(self.n_components, min(scaled_data.shape))
        self._pca = PCA(n_components=n_components)
        self._pca.fit(scaled_data)

        self._fitted = True

    @classmethod
    def get_feature_name(cls) -> str:
        """Return the feature view name.

        :returns: 'gene_expression_pca'
        """
        return "gene_expression_pca"

    def get_output_filename(self) -> str:
        """Return the output filename for cached embeddings.

        :returns: Filename like 'cell_line_gene_expression_pca_100.csv'
        """
        return f"cell_line_gene_expression_pca_{self.n_components}.csv"

    def _get_model_filename(self) -> str:
        """Return the filename for the fitted model.

        :returns: Filename like 'cell_line_gene_expression_pca_100_models.pkl'
        """
        return f"cell_line_gene_expression_pca_{self.n_components}_models.pkl"

    def generate_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Generate PCA embeddings for all cell lines and save to disk.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the PCA embeddings
        :raises FileNotFoundError: If the gene expression file is not found
        """
        data_dir = Path(data_path).resolve()
        output_file = data_dir / dataset_name / self.get_output_filename()
        model_file = data_dir / dataset_name / self._get_model_filename()

        # Load gene expression data
        ge_file = data_dir / dataset_name / "gene_expression.csv"
        if not ge_file.exists():
            raise FileNotFoundError(f"Gene expression file not found: {ge_file}")

        ge_df = pd.read_csv(ge_file, dtype={CELL_LINE_IDENTIFIER: str})
        ge_df = ge_df.set_index(CELL_LINE_IDENTIFIER)

        cell_line_ids = list(ge_df.index)
        print(f"Processing {len(cell_line_ids)} cell lines for dataset {dataset_name}...")

        # Fit the model
        self.fit(ge_df)

        # Transform all cell lines
        scaled_data = self._scaler.transform(ge_df.values)
        embeddings = self._pca.transform(scaled_data)

        # Save embeddings
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.insert(0, CELL_LINE_IDENTIFIER, cell_line_ids)
        embeddings_df.to_csv(output_file, index=False)

        # Save fitted models
        with open(model_file, "wb") as f:
            pickle.dump({"scaler": self._scaler, "pca": self._pca}, f)

        print(f"Embeddings saved to {output_file}")
        print(f"Fitted models saved to {model_file}")

        # Return as FeatureDataset
        return self._create_feature_dataset(list(embeddings), cell_line_ids)

    def load_embeddings(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """Load pre-generated PCA embeddings from disk.

        Also loads the fitted scaler and PCA model for future transformations.

        :param data_path: Path to the data directory
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the embeddings
        :raises FileNotFoundError: If the embeddings file is not found
        """
        embeddings_file = Path(data_path) / dataset_name / self.get_output_filename()
        model_file = Path(data_path) / dataset_name / self._get_model_filename()

        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}. "
                f"Use load_or_generate() to automatically generate embeddings."
            )

        # Load fitted models if available (optional - only needed for transforming new data)
        if model_file.exists():
            with open(model_file, "rb") as f:
                models = pickle.load(f)  # noqa: S301
                self._scaler = models["scaler"]
                self._pca = models["pca"]
                self._fitted = True

        # Load embeddings
        embeddings_df = pd.read_csv(embeddings_file, dtype={CELL_LINE_IDENTIFIER: str})
        feature_name = self.get_feature_name()
        features = {}

        for _, row in embeddings_df.iterrows():
            cell_line_id = row[CELL_LINE_IDENTIFIER]
            embedding = row.drop(CELL_LINE_IDENTIFIER).to_numpy(dtype=np.float32)
            features[cell_line_id] = {feature_name: embedding}

        return FeatureDataset(features)


def main():
    """Generate PCA embeddings for cell line gene expression from command line."""
    parser = argparse.ArgumentParser(description="Generate PCA embeddings for cell line gene expression.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data folder")
    parser.add_argument("--n_components", type=int, default=100, help="Number of PCA components")
    args = parser.parse_args()

    featurizer = PCAFeaturizer(n_components=args.n_components)
    featurizer.generate_embeddings(args.data_path, args.dataset_name)


if __name__ == "__main__":
    main()
