import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split, GroupShuffleSplit
from pydeseq2.preprocessing import deseq2_norm
from pydeseq2.dds import DeseqDataSet
from itertools import cycle


def get_train_test_set(label_matrix, mode, train_size, metric):
    if mode == "LCO":
        # LCO setting: split data into training (80% of cell lines) and test (20% of cell lines) sets
        label_matrix_T = label_matrix.set_index("Compound").transpose()
        train_T, test_T = train_test_split(label_matrix_T, train_size=train_size, random_state=0)
        train = train_T.transpose()
        test = test_T.transpose()

    elif mode == "LDO":
        # LDO setting: split data into training (80% of drugs) and test (20% of drugs) sets
        label_matrix = label_matrix.set_index("Compound")
        train, test = train_test_split(label_matrix, train_size=train_size, random_state=0)

    elif mode == "LPO":
        label_matrix_LPO = label_matrix.drop("Compound", axis=1)
        label_matrix_LPO_np = label_matrix_LPO.to_numpy()
        label_matrix_LPO_np_flat = label_matrix_LPO_np.reshape(-1)  # flatten the matrix into 1D array

        n_points = np.prod(label_matrix_LPO.shape)
        fold_indxs = list(range(0, 5))  # we want to split data into 5 batches
        pair_batch_lengths = split(n_points, 5)  # size of each batch stored in this list

        pair_fold = np.repeat(np.array(fold_indxs), np.array(pair_batch_lengths))  # np.array repeating batch indx
        np.random.seed(0)
        np.random.shuffle(pair_fold)  # shuffle the indexes

        # split data via their group (specified in pair_fold)
        gss = GroupShuffleSplit(n_splits=1, random_state=0, train_size=0.8)
        train_index, test_index = next(gss.split(label_matrix_LPO_np_flat, groups=pair_fold))

        train = label_matrix_LPO_np_flat[train_index]
        test = label_matrix_LPO_np_flat[test_index]

        # get drug and cl info for each indx

        train_indx_drug_ls, train_indx_cl_ls = cl_drug_info_df(train_index, label_matrix, label_matrix_LPO)

        test_indx_drug_ls, test_indx_cl_ls = cl_drug_info_df(test_index, label_matrix, label_matrix_LPO)

        train = pd.DataFrame(
            {"Primary Cell Line Name": train_indx_cl_ls, "Compound": train_indx_drug_ls, metric: train})
        test = pd.DataFrame({"Primary Cell Line Name": test_indx_cl_ls, "Compound": test_indx_drug_ls, metric: test})

    return train, test


def get_cell_viab_data(path, train_2, test_2):
    cell_viab = pd.read_csv(path, index_col="Primary Cell Line Name")

    # mask for cell lines in train set (EC50 data has already been split for training, find corresp. in cell viab data)
    mask_train = [True if x in train_2.columns else False for x in cell_viab.index]
    cell_viab_train = cell_viab[mask_train]

    # mask for cell lines in test set
    mask_test = [True if x in test_2.columns else False for x in cell_viab.index]
    cell_viab_test = cell_viab[mask_test]

    # generating single drug models

    # prepare viab data
    cell_viab_train_drug = cell_viab_train[cell_viab_train["Compound"] == "17-AAG"].sort_values(
        by="Primary Cell Line Name")
    cell_viab_train_drug.drop("Compound", axis=1, inplace=True)

    cell_viab_test_drug = cell_viab_test[cell_viab_test["Compound"] == "17-AAG"].sort_values(
        by="Primary Cell Line Name")
    cell_viab_test_drug.drop("Compound", axis=1, inplace=True)

    # remove cell lines from train and test set which are not in cell viab data (probably because not tested for drug)
    missing_cl_train = [x for x in cell_viab_train.index.unique() if x not in cell_viab_train_drug.index]
    missing_cl_test = [x for x in cell_viab_test.index.unique() if x not in cell_viab_test_drug.index]

    # train: fit the lin model using training data
    X_train = cell_viab_train_drug.to_numpy()
    y_train = train_2.loc["17-AAG"].drop(missing_cl_train).to_numpy()

    X_test = cell_viab_test_drug.to_numpy()
    y_test = test_2.loc["17-AAG"].drop(missing_cl_test).to_numpy()

    # drop nans
    # nans in predicted ec50
    mask_train_2 = ~np.isnan(y_train)
    X_train_2 = X_train[mask_train_2]
    y_train_2 = y_train[mask_train_2]

    mask_test_2 = ~np.isnan(y_test)
    X_test_2 = X_test[mask_test_2]
    y_test_2 = y_test[mask_test_2]

    # nans in cell viab meassurements (probably due to technical error during experiment)
    nan_idx_row = np.unique(np.asarray([i[0] for i in np.argwhere(np.isnan(X_train_2))]))
    X_train_2 = np.delete(X_train_2, nan_idx_row, 0)
    y_train_2 = np.delete(y_train_2, nan_idx_row, 0)

    nan_idx_row_test = np.unique(np.asarray([i[0] for i in np.argwhere(np.isnan(X_test_2))]))
    if len(nan_idx_row_test) > 0:
        X_test_2 = np.delete(X_test_2, nan_idx_row_test, 0)
        y_test_2 = np.delete(y_test_2, nan_idx_row_test, 0)

    return X_train_2, y_train_2, X_test_2, y_test_2


def select_genexp_features(X_train, ntop=500, mode="NormTransform"):
    # convert the data frame to a numpy array and round the floats to integers
    gene_counts_train_np = np.round(X_train.values).astype(int)
    gene_counts_train_df = pd.DataFrame(gene_counts_train_np, index=X_train.index,
                                        columns=X_train.columns)

    if mode == "NormTransform":
        # perform normalisation using the pydeseq2 package
        deseq2_counts, size_factors = deseq2_norm(gene_counts_train_df)
        deseq2_counts = np.log2(deseq2_counts + 1)

    elif mode == "VST":
        # for whatever reason it needs a condition :(
        condition = cycle(['Bananas', 'Oranges', 'Strawberries'])
        metadata = pd.DataFrame({'cell_line': gene_counts_train_df.index})
        metadata['condition'] = [next(condition) for cond in range(len(metadata))]
        metadata.index = gene_counts_train_df.index

        dds = DeseqDataSet(counts=gene_counts_train_df, metadata=metadata)
        dds.vst(use_design=False)

        deseq2_counts = pd.DataFrame(dds.layers["vst_counts"],
                                     index=gene_counts_train_df.index, columns=gene_counts_train_df.columns)

    # calculate the variance of each gene
    rv = deseq2_counts.var(axis=0).sort_values(ascending=False)

    # select the top 500 genes with the highest variance
    ntop = 500
    gene_ids_top_var_pydeseq2 = rv.index[:ntop]
    gene_ids_top_var_pydeseq2
    deseq2_counts = deseq2_counts[gene_ids_top_var_pydeseq2]
    deseq2_counts_np = deseq2_counts.to_numpy()

    return deseq2_counts_np


def get_gene_expression_data(path, train_drp, test_drp, feature_selection=True, selection_method="NormTransform"):
    gene_counts = pd.read_csv(path, sep="\t", index_col=0).T

    # mask for cell lines in train set (EC50 data has already been split, find corresp. in cell viab data)
    mask_train = [True if x in train_drp.columns else False for x in gene_counts.index]
    gene_counts_train = gene_counts[mask_train].sort_index()  # sort is important for cl ids to be same order as in drp

    if feature_selection:
        gene_counts_train_np = select_genexp_features(gene_counts_train, ntop=500, mode=selection_method)
    else:
        gene_counts_train_np = gene_counts_train.to_numpy()

    # mask for cell lines in test set
    mask_test = [True if x in test_drp.columns else False for x in gene_counts.index]
    gene_counts_test = gene_counts[mask_test]  # might have to sort this as well in the LCO setting (for line 344)

    if feature_selection:
        gene_counts_test_np = select_genexp_features(gene_counts_test, ntop=500, mode=selection_method)
    else:
        gene_counts_test_np = gene_counts_test.to_numpy()

    # generating single drug models
    drug_dict = {}
    for drug in train_drp.index:
        # select drp data of drug
        train_drp_drug = train_drp[train_drp.index == drug].T.sort_index()
        train_drp_drug_np = train_drp_drug.to_numpy()

        test_drp_drug = test_drp[test_drp.index == drug].T.sort_index()
        test_drp_drug_np = test_drp_drug.to_numpy()

        # drop nans in drug response data
        mask_train_2 = ~np.isnan(train_drp_drug_np)
        mask_train_2 = mask_train_2.reshape(-1)
        gene_counts_train_np_nona = gene_counts_train_np[mask_train_2]
        train_drp_drug_np_nona = train_drp_drug_np[mask_train_2]

        mask_test_2 = ~np.isnan(test_drp_drug_np)
        mask_test_2 = mask_test_2.reshape(-1)
        gene_counts_test_np_nona = gene_counts_test_np[mask_test_2]
        test_drp_drug_np_nona = test_drp_drug_np[mask_test_2]

        # add data to dict of format {"drug1": {X_train:[..], y_train:[..], X_test:[..], y_test:[..]}, "drug2": ...}
        data_dict = {"X_train": gene_counts_train_np_nona, "y_train": train_drp_drug_np_nona,
                     "X_test": gene_counts_test_np_nona, "y_test": test_drp_drug_np_nona}
        drug_dict[drug] = data_dict  # nested dict

    return gene_counts, drug_dict