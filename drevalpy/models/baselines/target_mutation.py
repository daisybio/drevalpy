"""Target-gene mutation status as an additional feature block for the sklearn models."""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#: Functional mutation status categories, in the fixed column order of the feature block.
STATUS_CATEGORIES = ["activating", "LoF", "other", "wildtype", "missing"]
#: Names of the columns appended to the feature matrix.
FEATURE_NAMES = [f"target_status_{category}" for category in STATUS_CATEGORIES] + ["target_sensitizing"]
#: File names of the two tables, relative to the target mutation directory.
CURATED_FILE = "curated_drug_target_variant.csv"
STATUS_FILE = "functional_status_matrix.csv"


class TargetMutationFeatures:
    """
    Looks up the mutation status of the gene a drug acts on, for every (cell line, drug) pair.

    The feature depends on the pair: the same cell line gets the status of BRAF for dabrafenib but the status of
    PIK3CA for alpelisib. It is therefore built in ``get_concatenated_features``, where the cell line and the drug of
    every response row are known, and not as a cell line or drug view. Both inputs are plain per-entity tables:

    * ``curated_drug_target_variant.csv``: ``pubchem_id``, ``biomarker_gene``, ``sensitive_status`` per drug.
    * ``functional_status_matrix.csv``: ``cellosaurus_id``, ``gene``, ``status`` per cell line and gene.

    The block has 6 columns: a one-hot encoding of the status (activating, LoF, other, wildtype, missing) and
    ``target_sensitizing``, which is 1 if the status equals the sensitive status of the drug. Drugs without a curated
    entry get only zeros, so the block cannot change anything for them. A cell line that is not in the status matrix
    gets the status ``missing``.
    """

    def __init__(
        self,
        drug_to_gene: dict[str, str],
        drug_to_sensitive_status: dict[str, str],
        status: dict[tuple[str, str], str],
        cell_line_to_cellosaurus: dict[str, str],
        standardize: bool = True,
    ):
        """
        Initializes the lookups.

        :param drug_to_gene: pubchem_id -> biomarker gene
        :param drug_to_sensitive_status: pubchem_id -> status that makes cells sensitive to the drug
        :param status: (cellosaurus_id, gene) -> functional status
        :param cell_line_to_cellosaurus: cell line name -> cellosaurus_id
        :param standardize: standardize the block with the mean and std of the training rows. Linear models with a
            penalty need it, because the sparse 0/1 columns are otherwise shrunk to zero. Trees do not care.
        """
        self.drug_to_gene = drug_to_gene
        self.drug_to_sensitive_status = drug_to_sensitive_status
        self.status = status
        self.cell_line_to_cellosaurus = cell_line_to_cellosaurus
        self.standardize = standardize
        self.scaler: StandardScaler | None = None
        self.feature_names = list(FEATURE_NAMES)

    @classmethod
    def from_directory(
        cls, target_mutation_path: str, cell_line_names_path: str, standardize: bool = True
    ) -> "TargetMutationFeatures":
        """
        Reads the two tables and the mapping of cell line names to Cellosaurus IDs.

        :param target_mutation_path: directory with the two tables, e.g., <data_path>/meta/target_mutation
        :param cell_line_names_path: cell_line_names.csv of the dataset (columns cellosaurus_id, cell_line_name)
        :param standardize: see __init__
        :returns: the feature builder
        :raises FileNotFoundError: if one of the files is missing
        """
        curated_path = os.path.join(target_mutation_path, CURATED_FILE)
        status_path = os.path.join(target_mutation_path, STATUS_FILE)
        for path in (curated_path, status_path, cell_line_names_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Target mutation feature needs {path}, but it does not exist.")
        curated = pd.read_csv(curated_path, dtype=str).dropna(subset=["pubchem_id", "biomarker_gene"])
        status = pd.read_csv(status_path, dtype=str)
        names = pd.read_csv(cell_line_names_path, dtype=str)
        return cls(
            drug_to_gene=dict(zip(curated["pubchem_id"], curated["biomarker_gene"])),
            drug_to_sensitive_status=dict(zip(curated["pubchem_id"], curated["sensitive_status"])),
            status={(c, g): s for c, g, s in zip(status["cellosaurus_id"], status["gene"], status["status"])},
            cell_line_to_cellosaurus=dict(zip(names["cell_line_name"], names["cellosaurus_id"])),
            standardize=standardize,
        )

    def _raw_block(self, cell_line_ids: np.ndarray, drug_ids: np.ndarray) -> np.ndarray:
        """
        Builds the unstandardized block.

        :param cell_line_ids: cell line name of every row
        :param drug_ids: pubchem_id of every row
        :returns: matrix of shape (rows, 6)
        """
        block = np.zeros((len(drug_ids), len(self.feature_names)), dtype=np.float32)
        category_to_column = {category: j for j, category in enumerate(STATUS_CATEGORIES)}
        for i, (cell_line, drug) in enumerate(zip(cell_line_ids, drug_ids)):
            gene = self.drug_to_gene.get(str(drug))
            if gene is None:
                continue
            cellosaurus_id = self.cell_line_to_cellosaurus.get(str(cell_line), "")
            status = self.status.get((cellosaurus_id, gene), "missing")
            block[i, category_to_column[status]] = 1.0
            block[i, -1] = float(status == self.drug_to_sensitive_status[str(drug)])
        return block

    def fit(self, cell_line_ids: np.ndarray, drug_ids: np.ndarray) -> None:
        """
        Fits the standardization on the training rows. Does nothing if standardize is False.

        :param cell_line_ids: cell line name of every training row
        :param drug_ids: pubchem_id of every training row
        """
        if self.standardize:
            self.scaler = StandardScaler().fit(self._raw_block(cell_line_ids, drug_ids))

    def transform(self, cell_line_ids: np.ndarray, drug_ids: np.ndarray) -> np.ndarray:
        """
        Builds the feature block for the given rows.

        :param cell_line_ids: cell line name of every row
        :param drug_ids: pubchem_id of every row
        :returns: matrix of shape (rows, 6)
        """
        block = self._raw_block(cell_line_ids, drug_ids)
        if self.scaler is not None:
            block = self.scaler.transform(block).astype(np.float32)
        return block
