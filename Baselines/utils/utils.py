# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:09:13 2022

@author: jessi
"""
import random
import os
import numpy as np
import pandas as pd
import torch
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, train_test_split, GroupShuffleSplit


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, hyp, metric_matrix,
               param_save_path, hyp_save_path, metric_save_path, description):
    print('Finished training the model. Saving the model to the path: {}'.format(param_save_path))
    torch.save(model.state_dict(), param_save_path + '/model_weights_' + description + '.pt ')

    print('Finished training the model. Saving the model hyp to the path: {}'.format(hyp_save_path))
    x = json.dumps(hyp)
    f = open(hyp_save_path + "/model_hyp_" + description + ".txt", "w")
    f.write(x)
    f.close()

    print('Finished training the model. Saving metric matrix to the path: {}'.format(metric_save_path))
    np.save(metric_save_path + '/model_train_metrics_' + description, metric_matrix)


def load_pretrained_model(model, model_path):
    print('Loading pre-trained model from: {}'.format(model_path))
    model.load_state_dict(torch.load(model_path))


# normalize() takes in training data and feature matrix to be normalized
# It fits the StandardScaler() using training data and use the mean and std
# of the training data to normalize the features
def normalize(train_x, features):
    ss = StandardScaler()
    ss = ss.fit(train_x)
    norm_features = ss.transform(features)
    return norm_features


# normalize features based on training data
def norm_cl_features(cl_features, indices,
                     fold_type, train_fold):
    indices_train = indices.loc[indices[fold_type].isin(train_fold)]  # df containing all train indices

    # --------------------- cell line feature normalization ------------------
    # normalize cl features (fit_transform train data, then use the metrics(mean, std)
    # calculated from train data to fit test data)
    train_cls = indices_train['cl_idx']
    train_x_cl = cl_features[train_cls, :]  # cl features for all training data
    norm_cl_features = normalize(train_x_cl, cl_features)  # normalized dict of cl features

    return norm_cl_features


def norm_drug_features(drug_features, indices,
                       fold_type, train_fold):
    indices_train = indices.loc[indices[fold_type].isin(train_fold)]  # df containing all train indices

    train_drugs = indices_train['drug_idx']
    train_x_drug = drug_features[train_drugs, :]
    norm_drug_features = normalize(train_x_drug, drug_features)
    return norm_drug_features


def mkdir(directory):
    directories = directory.split("/")

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s' % folder)
            os.mkdir(folder)


# --------------------------------------------my own utils start here -------------------------------------------------#


def split(x, n):
    batch_lengths = []
    # If we cannot split the number into exactly 'N' parts
    if (x < n):
        print(-1)

    # If x % n == 0 then the minimum difference is 0 and all numbers are x / n
    elif (x % n == 0):
        for i in range(n):
            batch_lengths.append(x // n)
    else:
        # upto n-(x % n) the values will be x / n after that the values will be x / n + 1
        zp = n - (x % n)
        pp = x // n
        for i in range(n):
            if (i >= zp):
                batch_lengths.append(pp + 1)
            else:
                batch_lengths.append(pp)

    return batch_lengths


def cl_drug_info(label_matrix, label_matrix_LPO, indx):
    # use linear index to access original pandas df and get drug/cl info
    nr_cols = label_matrix_LPO.shape[1]
    row_nr = indx // nr_cols  # row number
    if row_nr == 0:
        col_nr = indx
    else:
        col_nr = indx - nr_cols * row_nr

    drug = label_matrix["Compound"][row_nr]  # drug of linear indx
    cl = label_matrix_LPO.columns[col_nr]  # cl of linear indx

    return drug, cl


def cl_drug_info_df(split_index_arr, label_matrix, label_matrix_LPO):
    indx_drug_ls = []
    indx_cl_ls = []

    for indx in split_index_arr:
        indx_drug, indx_cl = cl_drug_info(label_matrix, label_matrix_LPO, indx)
        indx_drug_ls.append(indx_drug)
        indx_cl_ls.append(indx_cl)

    return indx_drug_ls, indx_cl_ls


def remove_outliers(drp_con, metric, mode, data_set_str):
    Q1 = drp_con[metric].quantile(0.25)
    Q3 = drp_con[metric].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(drp_con[metric] >= upper)[0]
    lower_array = np.where(drp_con[metric] <= lower)[0]

    print(f"{metric} - {data_set_str}:\n"
          f"total upper array outliers {upper_array.size}\n"
          f"total lower array outliers {lower_array.size}\n")

    if mode == "replace":
        # Replace the outliers with nans
        drp_con.loc[drp_con.index[upper_array.tolist()], metric] = np.nan
        drp_con.loc[drp_con.index[lower_array.tolist()], metric] = np.nan

    elif mode == "remove":
        # Remove the rows containing outliers
        drp_con.drop(index=upper_array, inplace=True)
        drp_con.drop(index=lower_array, inplace=True)

    else:
        warnings.warn('Warning Message: mode has to be either "replace" or "remove"')

    drp_con.reset_index(inplace=True, drop=True)

    return drp_con


def normalize_data(drp_con, metric):
    '''
    normalize desired metrics
    '''

    drp_con[metric] = (drp_con[metric] - drp_con[metric].min()) / (
            drp_con[metric].max() - drp_con[metric].min())

    return drp_con


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

        train = pd.DataFrame({"Primary Cell Line Name": train_indx_cl_ls, "Compound": train_indx_drug_ls, metric: train})
        test = pd.DataFrame({"Primary Cell Line Name": test_indx_cl_ls, "Compound": test_indx_drug_ls, metric: test})

    return train, test


if __name__ == "__main__":
    label_matrix = pd.read_csv("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
                               "µM)_matrix.csv", header=0, index_col=0)
    label_matrix.reset_index(inplace=True)
    metric = "EC50 (µM)"
    task = "LPO"

    train, test = get_train_test_set(label_matrix, task, 0.8, metric)
