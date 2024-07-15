from utils.utils import mkdir, reset_seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import utils.constants as c


def standardize(train_x, test_x):
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.transform(test_x)
    return train_x, test_x


def initialize(FLAGS, binary=False, multitask=False):

    reset_seed(FLAGS.seed)
    mkdir(FLAGS.outroot + "/results/" + FLAGS.folder)

    LABEL_FILE = FLAGS.dataroot + c._LABEL_FILE
    GENE_EXPRESSION_FILE = FLAGS.dataroot + c._GENE_EXPRESSION_FILE
    LABEL_MATRIX_FILE = FLAGS.dataroot + c._LABEL_MATRIX_FILE

    if FLAGS.drug_feat == "desc" or FLAGS.drug_feat == "mixed":
        DRUG_FEATURE_FILE = FLAGS.dataroot + c._DRUG_DESCRIPTOR_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

        df = StandardScaler().fit_transform(drug_feats.values)  # normalize
        drug_feats = pd.DataFrame(
            df, index=drug_feats.index, columns=drug_feats.columns
        )

        if FLAGS.drug_feat == "mixed":
            DRUG_MFP_FEATURE_FILE = FLAGS.dataroot + c._MORGAN_FP_FILE
            drug_mfp = pd.read_csv(DRUG_MFP_FEATURE_FILE, index_col=0)
            drug_feats[drug_mfp.columns] = drug_mfp

        valid_cols = drug_feats.columns[
            ~drug_feats.isna().any()
        ]  # remove columns with missing data
        drug_feats = drug_feats[valid_cols]

    else:
        DRUG_FEATURE_FILE = FLAGS.dataroot + c._MORGAN_FP_FILE
        drug_feats = pd.read_csv(DRUG_FEATURE_FILE, index_col=0)

    cell_lines = pd.read_csv(GENE_EXPRESSION_FILE, index_col=0).T  # need to normalize
    labels = pd.read_csv(LABEL_FILE)
    labels["cell_line"] = labels["cell_line"].astype(str)
    labels["response"] = labels["ln_ic50"]

    labels = labels.loc[
        labels["drug"].isin(drug_feats.index)
    ]  # use only cell lines with data
    labels = labels.loc[
        labels["cell_line"].isin(cell_lines.index)
    ]  # use only drugs with data
    cell_lines = cell_lines.loc[
        cell_lines.index.isin(labels["cell_line"].unique())
    ]  # use only cell lines with labels
    drug_feats = drug_feats.loc[
        drug_feats.index.isin(labels["drug"].unique())
    ]  # use only drugs in labels

    label_matrix = pd.read_csv(LABEL_MATRIX_FILE, index_col=0).T
    label_matrix = label_matrix.loc[cell_lines.index][
        drug_feats.index
    ]  # align the matrix

    if FLAGS.normalize_response:
        ss = StandardScaler()  # normalize IC50
        temp = ss.fit_transform(label_matrix.values)
        label_matrix = pd.DataFrame(
            temp, index=label_matrix.index, columns=label_matrix.columns
        )
    else:
        label_matrix = label_matrix.astype(float)

    if FLAGS.split == "lpo":  # leave pairs out
        labels["fold"] = labels["pair_fold"]
    else:  # default: leave cell lines out
        labels["fold"] = labels["cl_fold"]

    print("tuples per fold:")
    print(labels.groupby("fold").size())

    return drug_feats, cell_lines, labels, label_matrix, standardize
