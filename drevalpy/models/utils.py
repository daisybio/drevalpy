import pandas as pd
import numpy as np
from typing import Optional
from drevalpy.datasets.dataset import FeatureDataset


def load_cl_ids_from_csv(path: str, dataset_name: str) -> FeatureDataset:
    cl_names = pd.read_csv(f"{path}/{dataset_name}/cell_line_names.csv", index_col=0)
    return FeatureDataset({cl: {"cell_line_id": cl} for cl in cl_names.index})


def load_and_reduce_gene_features(
    feature_type: str, gene_list: Optional[str], data_path: str, dataset_name: str
) -> FeatureDataset:
    ge = pd.read_csv(f"{data_path}/{dataset_name}/{feature_type}.csv", index_col=0)
    if gene_list is None:
        return FeatureDataset(
            {cl: {feature_type: ge.loc[cl].values} for cl in ge.index}
        )
    else:
        gene_info = pd.read_csv(
            f"{data_path}/{dataset_name}/gene_lists/{gene_list}.csv",
            sep=(
                "\t" if gene_list == "landmark_genes" else ","
            ),  # TODO harmonize gene lists
        )
        genes_to_use = set(gene_info["Symbol"]) & set(ge.columns)
        ge = ge[list(genes_to_use)]

        return FeatureDataset(
            {cl: {feature_type: ge.loc[cl].values} for cl in ge.index}
        )


def load_drug_ids_from_csv(data_path: str, dataset_name: str) -> FeatureDataset:
    drug_names = pd.read_csv(f"{data_path}/{dataset_name}/drug_names.csv", index_col=0)
    return FeatureDataset({drug: {"drug_id": drug} for drug in drug_names.index})


def load_drug_features_from_fingerprints(
    data_path: str, dataset_name: str
) -> FeatureDataset:
    fingerprints = pd.read_csv(
        f"{data_path}/{dataset_name}/drug_fingerprints/drug_name_to_demorgan_128_map.csv",
        index_col=0,
    ).T
    return FeatureDataset(
        {
            drug: {"fingerprints": fingerprints.loc[drug].values}
            for drug in fingerprints.index
        }
    )


def unique(array):
    # ordered by first occurence
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
