"""Preprocesses transcriptome (gene expression) data using PCA dimensionality reduction."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


def main():
    """Process transcriptome data and save PCA-transformed features.

    :raises FileNotFoundError: If the gene expression file is not found.
    """
    parser = argparse.ArgumentParser(description="Preprocess transcriptome (gene expression) data using PCA.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument(
        "--n_components",
        type=int,
        default=100,
        help="Number of principal components to keep (default: 100)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to the data folder (default: data)",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="gene_expression",
        help="Type of transcriptome feature to use (default: gene_expression)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    n_components = args.n_components
    data_dir = Path(args.data_path).resolve()
    feature_type = args.feature_type

    # Input file: gene expression CSV
    input_file = data_dir / dataset_name / f"{feature_type}.csv"
    # Output files: PCA features CSV and fitted PCA/scaler objects
    output_file = data_dir / dataset_name / f"cell_line_{feature_type}_pca_{n_components}.csv"
    pca_file = data_dir / dataset_name / f"cell_line_{feature_type}_pca_{n_components}_pca.pkl"
    scaler_file = data_dir / dataset_name / f"cell_line_{feature_type}_pca_{n_components}_scaler.pkl"

    if not input_file.exists():
        raise FileNotFoundError(f"Error: {input_file} not found.")

    print(f"Loading transcriptome data from {input_file}...")
    # Load gene expression data
    # Format: rows are cell lines (indexed by cell_line_name), columns are genes
    ge_df = pd.read_csv(input_file, index_col=CELL_LINE_IDENTIFIER)
    ge_df.index = ge_df.index.astype(str)

    # Drop cellosaurus_id if present
    if "cellosaurus_id" in ge_df.columns:
        ge_df = ge_df.drop(columns=["cellosaurus_id"])

    print(f"Loaded {len(ge_df)} cell lines with {len(ge_df.columns)} genes")
    print(f"Performing PCA with {n_components} components...")

    # Extract cell line IDs and gene expression matrix
    cell_line_ids = ge_df.index.values
    gene_expression_matrix = ge_df.values.astype(np.float32)

    # Handle missing values: fill with 0 or mean (using 0 as default)
    if np.isnan(gene_expression_matrix).any():
        print("Warning: Found NaN values. Filling with 0.")
        gene_expression_matrix = np.nan_to_num(gene_expression_matrix, nan=0.0)

    # Standardize the data before PCA
    scaler = StandardScaler()
    gene_expression_scaled = scaler.fit_transform(gene_expression_matrix)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(gene_expression_scaled)

    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA explained variance (first 10 components): {pca.explained_variance_ratio_[:10]}")

    # Create output DataFrame
    pca_df = pd.DataFrame(
        pca_features,
        index=cell_line_ids,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    pca_df.index.name = CELL_LINE_IDENTIFIER
    pca_df = pca_df.reset_index()

    # Save PCA-transformed features
    pca_df.to_csv(output_file, index=False)
    print(f"PCA features saved to {output_file}")

    # Save fitted PCA and scaler for potential future use (e.g., transforming new data)
    joblib.dump(pca, pca_file)
    print(f"Fitted PCA model saved to {pca_file}")

    joblib.dump(scaler, scaler_file)
    print(f"Fitted scaler saved to {scaler_file}")

    print("Finished processing transcriptome PCA featurization.")


if __name__ == "__main__":
    main()
