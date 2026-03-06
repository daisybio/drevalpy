# PPIGraphGNN Model

A Graph Neural Network (GNN) model for drug response prediction that uses protein-protein interaction (PPI) networks, gene expression data, and drug fingerprints. The model includes GNNExplainer for extracting drug-cell-line-specific subnetworks for interpretability.

## Overview

**PPIGraphGNN** combines:

- Gene expression vectors (cell line features)
- Drug fingerprints (drug features)
- PPI network structure (protein interactions)
- Graph Convolutional Networks (GCN) for learning from network topology
- GNNExplainer for interpretable predictions

## Architecture

1. **Input**:

   - Gene expression vector for each cell line
   - Drug fingerprints for each drug
   - PPI network as a graph (edges represent protein-protein interactions)

2. **Model**:

   - **PPI Graph Encoder**: Node features are initialized with gene expression values, then multiple GCN layers propagate information through the PPI network
   - **Drug Encoder**: MLP processes drug fingerprints
   - **Combiner**: Concatenates PPI graph embedding and drug embedding
   - **Predictor**: Fully connected layers predict drug response from combined features

3. **Explainability**:
   - GNNExplainer identifies important PPI subnetworks for each drug-cell line prediction
   - Returns top-k edges and their importance scores specific to each drug-cell line pair

## Usage

### 1. Prepare PPI Network Data

Create a CSV file with PPI network at `data/{dataset_name}/ppi_network.csv`:

```csv
gene_id_1,gene_id_2,interaction_score
BRCA1,BRCA2,0.95
TP53,MDM2,0.88
EGFR,PIK3CA,0.72
...
```

**Required columns:**

- `gene_id_1`: First gene/protein identifier
- `gene_id_2`: Second gene/protein identifier
- `interaction_score` (optional): Confidence score for the interaction (0-1)

**Important:** Gene IDs must match those in your gene expression data.

### 2. Generate PPI Graph

Run the preprocessing script to convert the PPI CSV to a PyTorch Geometric graph:

```bash
python -m drevalpy.datasets.featurizer.create_ppi_graphs GDSC1 --data_path data --gene_list landmark_genes_reduced
```

This creates `data/GDSC1/ppi_graph.pt` containing the graph structure.

**Important:** The `--gene_list` parameter must match the gene list used by the model (default: `landmark_genes_reduced`). This ensures the gene order in the PPI graph matches the gene expression feature order.

### 3. Train the Model

```python
from drevalpy.models import PPIGraphGNN
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

# Initialize model
model = PPIGraphGNN()

# Build model with hyperparameters
model.build_model({
    "hidden_dim": 64,
    "num_gnn_layers": 3,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
})

# Load features (also loads PPI graph automatically in load_drug_features)
cell_line_features = model.load_cell_line_features("data", "GDSC1")
drug_features = model.load_drug_features("data", "GDSC1")  # Loads PPI graph + drug fingerprints

# Train
model.train(
    output=train_dataset,
    cell_line_input=cell_line_features,
    drug_input=drug_features,  # Required for drug fingerprints
    output_earlystopping=val_dataset
)

# Predict
predictions = model.predict(
    cell_line_ids=test_cell_line_ids,
    drug_ids=test_drug_ids,
    cell_line_input=cell_line_features,
    drug_input=drug_features  # Required for drug fingerprints
)
```

### 4. Extract Explanations

Use GNNExplainer to get important subnetworks for specific drug-cell line pairs:

```python
# Get explanation for a specific drug-cell line pair
explanation = model.explain(
    cell_line_id="ACH-000001",
    drug_id="123456",
    cell_line_input=cell_line_features,
    drug_input=drug_features,
    top_k_edges=20  # Number of top edges to return
)

# Access results
print(f"Cell line: {explanation['cell_line_id']}")
print(f"Drug: {explanation['drug_id']}")
print(f"Important PPI interactions for this drug-cell line pair:")
for gene1, gene2, score in explanation['important_edges']:
    print(f"  {gene1} <-> {gene2}: {score:.4f}")
```

## Hyperparameters

Configurable in `hyperparameters.yaml`:

- `learning_rate`: Learning rate for optimizer (default: 0.001)
- `epochs`: Number of training epochs (default: 100)
- `hidden_dim`: Hidden dimension size (default: 64)
- `num_gnn_layers`: Number of GCN layers (default: 3)
- `dropout`: Dropout probability (default: 0.2)
- `batch_size`: Batch size for training (default: 32)

## Requirements

The model requires:

- `torch_geometric` with GNNExplainer
- Gene expression data with landmark genes
- Drug fingerprints
- PPI network in CSV format

## Model Properties

- **cell_line_views**: `["gene_expression"]`
- **drug_views**: `["fingerprints"]`
- **is_single_drug_model**: `False`
- **early_stopping**: Supported via validation dataset

## Output

The `explain()` method returns a dictionary with:

- `cell_line_id`: The cell line being explained
- `drug_id`: The drug being explained
- `important_edges`: List of tuples `(gene1, gene2, score)` for top-k edges
- `edge_mask`: Full edge importance scores for all edges
- `explanation`: Raw GNNExplainer output

## Example PPI Network Sources

Common sources for PPI networks:

- **STRING**: https://string-db.org/ (comprehensive, includes scores)
- **BioGRID**: https://thebiogrid.org/ (curated interactions)
- **IntAct**: https://www.ebi.ac.uk/intact/ (molecular interaction database)
- **HIPPIE**: http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/ (human integrated protein-protein interaction)

## Notes

1. **Gene Order Consistency**: The order of nodes in the PPI graph MUST match the order of genes in the gene expression features. The preprocessing script ensures this by using the same gene list file (e.g., `landmark_genes_reduced.csv`). The model validates this at training time and will raise an error if there's a mismatch.

2. The PPI graph structure is shared across all samples; only node features (gene expression) vary per cell line

3. Drug features (fingerprints) are used to distinguish between different drugs

4. The model uses landmark genes by default - ensure your PPI network includes these genes

5. GNNExplainer provides drug-cell-line-specific explanations by considering both the PPI network context and drug features

6. GNNExplainer is computationally intensive; use `top_k_edges` parameter to limit output

## How Gene Order is Maintained

When you run:

```python
graph.x = gene_expr.unsqueeze(1)
```

The gene expression vector is assigned to graph nodes. The model ensures correct mapping by:

1. **PPI Graph Creation**: Uses the same gene list file (e.g., `landmark_genes_reduced.csv`) to define node order
2. **Gene Expression Loading**: `load_and_select_gene_features()` uses the same gene list to order features
3. **Runtime Validation**: The model validates that both orders match before training

Example:

- Gene list: `["TP53", "EGFR", "BRCA1", ...]`
- PPI graph nodes: `[0: TP53, 1: EGFR, 2: BRCA1, ...]`
- Gene expression: `[expr_TP53, expr_EGFR, expr_BRCA1, ...]`
- Assignment: Node 0 gets expr_TP53, Node 1 gets expr_EGFR, etc.
