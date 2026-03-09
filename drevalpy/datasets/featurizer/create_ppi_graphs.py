"""
Preprocesses PPI network CSV files into graph representations for PPIGraphGNN.

This script takes a dataset name as input, reads the corresponding
PPI network CSV file, and converts it into a torch_geometric.data.Data object.
The PPI CSV should have columns: gene_id_1, gene_id_2, and optionally interaction_score.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data


def _load_ppi_network(ppi_file: Path, gene_list_file: Path) -> Data:
    """
    Load PPI network from CSV and create a PyTorch Geometric Data object.

    The gene order in the PPI graph will match the order in the gene list file.
    This ensures consistency when gene expression features are set as node features.

    :param ppi_file: Path to the PPI network CSV file with columns [gene_id_1, gene_id_2, (optional) interaction_score]
    :param gene_list_file: Path to the gene list CSV (e.g., landmark_genes_reduced.csv) that defines gene order
    :raises ValueError: If the PPI CSV does not contain the required columns or if the gene list file is not found
    :return: A Data object representing the PPI network graph
    """
    # Load the gene list to get the ordered list of genes (same as will be used for gene expression)
    gene_list_df = pd.read_csv(gene_list_file)
    if "Symbol" in gene_list_df.columns:
        genes = gene_list_df["Symbol"].tolist()
    elif "gene" in gene_list_df.columns:
        genes = gene_list_df["gene"].tolist()
    else:
        # Gene expression file -> columns are genes
        genes = list(gene_list_df.columns)
        genes = [g for g in genes if g not in ["cellosaurus_id", "cell_line_name"]]

    # Create a mapping from gene name to index
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # Load PPI network
    ppi_df = pd.read_csv(ppi_file)

    # Validate columns
    required_cols = {"gene_id_1", "gene_id_2"}
    if not required_cols.issubset(ppi_df.columns):
        raise ValueError(f"PPI CSV must contain columns 'gene_id_1' and 'gene_id_2'. Found: {ppi_df.columns.tolist()}")

    # Build edge list (only include genes that exist in gene expression)
    edge_list = []
    edge_weights = []

    has_weights = "interaction_score" in ppi_df.columns

    for _, row in ppi_df.iterrows():
        gene1 = str(row["gene_id_1"])
        gene2 = str(row["gene_id_2"])

        # Only add edge if both genes exist in gene expression data
        if gene1 in gene_to_idx and gene2 in gene_to_idx:
            idx1 = gene_to_idx[gene1]
            idx2 = gene_to_idx[gene2]

            # Add both directions for undirected graph
            edge_list.append([idx1, idx2])
            edge_list.append([idx2, idx1])

            if has_weights:
                weight = float(row["interaction_score"])
                edge_weights.extend([weight, weight])

    if not edge_list:
        raise ValueError("No valid edges found in PPI network (genes don't match gene expression)")

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create node feature placeholder (will be filled with gene expression at runtime)
    num_nodes = len(genes)
    x = torch.zeros((num_nodes, 1), dtype=torch.float)

    # Edge attributes
    if has_weights:
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    else:
        edge_attr = None

    # Store gene names as metadata
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.gene_names = genes  # Store for reference

    return graph


def main():
    """Main function to run the PPI preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess PPI network to graph.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset to process.")
    parser.add_argument("--path_data", type=str, default="data", help="Path to the data folder")
    parser.add_argument(
        "--ppi_file",
        type=str,
        default=None,
        help="Path to PPI CSV file (default: {path_data}/{dataset_name}/ppi_network.csv)",
    )
    parser.add_argument(
        "--gene_list",
        type=str,
        default="gene_expression.csv",
        help="Gene list name to use (default: gene_expression.csv; will take the columns)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    data_dir = Path(args.path_data).resolve()

    # Determine PPI file path
    if args.ppi_file:
        ppi_file = Path(args.ppi_file)
    else:
        ppi_file = data_dir / dataset_name / "ppi_network.csv"

    # Gene list file
    gene_list_file = data_dir / dataset_name / f"{args.gene_list}"
    output_file = data_dir / dataset_name / "ppi_graph.pt"

    if not ppi_file.exists():
        print(f"Error: {ppi_file} not found.")
        return

    if not gene_list_file.exists():
        print(f"Error: {gene_list_file} not found.")
        print(f"Available gene lists should be in {data_dir / 'meta' / 'gene_lists'}/")
        return

    print(f"Processing PPI network for dataset {dataset_name}...")
    print(f"Using gene list: {args.gene_list}")

    try:
        graph = _load_ppi_network(ppi_file, gene_list_file)
        torch.save(graph, output_file)
        print(f"PPI graph saved to {output_file}")
        print(f"  Nodes (genes): {graph.num_nodes}")
        print(f"  Edges (interactions): {graph.num_edges}")
        if graph.edge_attr is not None:
            print("  Edge attributes: Yes")
        print(f"\nGene order matches: {args.gene_list}")
        print(f"First 5 genes: {graph.gene_names[:5]}")
    except Exception as e:
        print(f"Error processing PPI network: {e}")


if __name__ == "__main__":
    main()
