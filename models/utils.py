import pandas as pd
from dreval.dataset import FeatureDataset


def load_ge_features_from_landmark_genes(dataset_path: str) -> FeatureDataset:
    ge = pd.read_csv(f"{dataset_path}/gene_expression.csv", index_col=0)
    landmark_genes = pd.read_csv(f"{dataset_path}/gene_lists/landmark_genes.csv", sep="\t")
    genes_to_use = set(landmark_genes["Symbol"]) & set(ge.columns)
    ge = ge[list(genes_to_use)]

    return FeatureDataset(
        {cl: {"gene_expression": ge.loc[cl].values} for cl in ge.index}
    )


def load_drug_features_from_fingerprints(dataset_path: str) -> FeatureDataset:
    fingerprints = pd.read_csv(f"{dataset_path}/drug_fingerprints/drug_name_to_demorgan_128_map.csv", index_col=0).T
    return FeatureDataset(
        {
            drug: {"fingerprints": fingerprints.loc[drug].values}
            for drug in fingerprints.index
        }
    )