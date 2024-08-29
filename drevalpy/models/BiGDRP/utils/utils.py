import random
import os
import numpy as np
import pandas as pd
import torch

_SEED = 0


def reset_seed(seed=None):
    if seed is not None:
        global _SEED
        _SEED = seed

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)


def create_fold_mask(tuples, label_matrix):
    x = pd.DataFrame(index=label_matrix.index, columns=label_matrix.columns)
    for index, row in tuples.iterrows():
        x.loc[row["cell_line"], row["drug"]] = row["fold"]
    return x


def save_flags(FLAGS):
    with open(FLAGS.outroot + "/results/" + FLAGS.folder + "/flags.cfg", "w") as f:
        for arg in vars(FLAGS):
            f.write("--%s=%s\n" % (arg, getattr(FLAGS, arg)))


def mkdir(directory):
    directories = directory.split("/")

    folder = ""
    for d in directories:
        folder += d + "/"
        if not os.path.exists(folder):
            print("creating folder: %s" % folder)
            os.mkdir(folder)


def reindex_tuples(tuples, drugs, cells, start_index=0):
    """
    Transforms strings in the drug and cell line columns to numerical indices

    tuples: dataframe with columns: cell_line_name, drug_col, drug_row
    drugs: list of drugs
    cells: list of cell line names
    """
    tuples = tuples.copy()
    for i, drug in enumerate(drugs):
        tuples = tuples.replace(drug, i)
    for i, cell in enumerate(cells):
        tuples = tuples.replace(cell, i + start_index)

    return tuples


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def predict_matrix(self, data_loader, drug_encoding=None):
    """
    returns a prediction matrix of (N, n_drugs)
    """

    self.model.eval()

    preds = []
    if drug_encoding is None:
        drug_encoding = (
            self.get_drug_encoding()
        )  # get the encoding first so that we don't have top run the conv every time
    else:
        drug_encoding = drug_encoding.to(self.device)

    with torch.no_grad():
        for (x,) in data_loader:
            x = x.to(self.device)
            pred = self.model.predict_response_matrix(x, drug_encoding)
            preds.append(pred)

    preds = torch.cat(preds, axis=0).cpu().detach().numpy()
    return preds
