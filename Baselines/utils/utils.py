# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mkdir(directory):
    directories = directory.split("/")

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s' % folder)
            os.mkdir(folder)


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

    logger.info(f"\n\n{metric} - {data_set_str}:\n"
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
    """
    normalize desired metrics
    """

    drp_con[metric] = (drp_con[metric] - drp_con[metric].min()) / (
            drp_con[metric].max() - drp_con[metric].min())

    return drp_con


def preprocessing(train_prepro,
                  test_prepro,
                  task, metric,
                  remove_out=False, norm_data=False, log_transform=False):
    dataset_postpro_ls = []
    set_string = ["train", "test"]

    for i, data_set in enumerate([train_prepro, test_prepro]):
        # convert metric matrix into a format easier to process (by melting df by compound)

        if task == "LCO" or task == "LDO":
            dataset_lin = data_set.reset_index().melt(id_vars="Compound").rename(
                columns={"variable": "Primary Cell Line Name", "value": metric})

        elif task == "LPO":
            dataset_lin = data_set  # in LPO case data is already linear

        if remove_out:
            logger.info(f"Removing outliers from {set_string[i]} set")
            dataset_lin = remove_outliers(dataset_lin, metric, "replace", set_string[i])

        if norm_data:
            logger.info(f"Normalizing {set_string[i]} set")
            dataset_lin = normalize_data(dataset_lin, metric)

        if log_transform:
            # dataset_lin[metric] = np.log(dataset_lin[metric] + 1) # in M umrechnen (also mal 10^-6 und dann -log10
            # davon)
            logger.info(f"Log transforming {set_string[i]} set")

            if "µM" in metric:
                dataset_lin[metric] = -np.log10(dataset_lin[metric] * 10 ** -6)
            elif "Amax" in metric and norm_data:
                dataset_lin[metric] = np.log(dataset_lin[metric] + 1)
            elif metric != "Amax":
                dataset_lin[metric] = np.log(dataset_lin[metric] + 1)

        if task == "LCO" or task == "LDO":
            dataset_postpro = dataset_lin.pivot(index="Compound", columns="Primary Cell Line Name", values=metric)
            # dataset_postpro.reset_index(inplace=True)

        elif task == "LPO":
            dataset_postpro = dataset_lin  # .pivot(index="Compound", columns="Primary Cell Line Name", values=metric)

        dataset_postpro_ls.append(dataset_postpro)

    return dataset_postpro_ls[0], dataset_postpro_ls[1]


if __name__ == "__main__":
    label_matrix = pd.read_csv("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
                               "µM)_matrix.csv", header=0, index_col=0)
    label_matrix.reset_index(inplace=True)
    metric = "EC50 (µM)"
    task = "LPO"

    train, test = get_train_test_set(label_matrix, task, 0.8, metric)
