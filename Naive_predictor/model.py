# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import mean_squared_error

sys.path.append('/nfs/home/students/m.lorenz/Baseline models/utils')
from utils import split, cl_drug_info, remove_outliers, normalize_data, get_train_test_set


def preprocessing(train_prepro, test_prepro, task, metric, remove_out=False, norm_data=False, log_transform=False):
    dataset_postpro_ls = []

    for data_set in [train_prepro, test_prepro]:
        # convert metric matrix into a format easier to process (by melting df by compound)

        if task == "LCO" or task == "LDO":
            dataset_lin = data_set.reset_index().melt(id_vars="Compound").rename(
                columns={"variable": "Primary Cell Line Name", "value": metric})

        elif task == "LPO":
            dataset_lin = data_set  # in LPO case data is already linear

        if remove_out:
            dataset_lin = remove_outliers(dataset_lin, metric, "replace")

        if norm_data:
            dataset_lin = normalize_data(dataset_lin, metric)

        if log_transform:
            dataset_lin[metric] = np.log(dataset_lin[metric])

        if task == "LCO" or task == "LDO":
            dataset_postpro = dataset_lin.pivot(index="Compound", columns="Primary Cell Line Name", values=metric)
            # dataset_postpro.reset_index(inplace=True)

        elif task == "LPO":
            dataset_postpro = dataset_lin

        dataset_postpro_ls.append(dataset_postpro)

    return dataset_postpro_ls[0], dataset_postpro_ls[1]


def calc_metric(train, test, mode, avg_by, metric):
    pcc_ls = []
    scc_ls = []
    mse_ls = []
    rmse_ls = []

    # ---------------------------------------------------- LCO ---------------------------------------------------------
    if mode == "LCO":

        drug_avg = train.mean(axis=1)
        pred = pd.DataFrame(drug_avg.rename("avg_" + metric))

        if avg_by == "drug":
            # avg by drug: take all the avg dr of the drugs (from train) and per drug calc. ssc, mse and rmse
            # over all cell lines in test set. Avg ssc, mse and rmse over all drugs in the end
            valid_drugs = []
            # calculate the SCC, mse and rmse using the test set: averaging per drug
            # SCC doesn't work as we cant calc. using a constant (the mean)

            for drug in drug_avg.index:
                y_true = test.loc[drug].dropna()  # drop nas from drug vector -> dr vlaues of a drug over all test cl

                if y_true.shape[0] > 1:
                    valid_drugs.append(drug)
                    y_pred = np.repeat(drug_avg[drug], len(y_true))  # array len = y_true repeating predic mean of drug

                    mse_drug = mean_squared_error(y_true, y_pred)
                    rmse_drug = mean_squared_error(y_true, y_pred, squared=False)

                    mse_ls.append(mse_drug)
                    rmse_ls.append(rmse_drug)

            print('----------------- LCO, average by drug ------------')
            print('rmse std by drug:', np.std(rmse_ls))
            print('\nmse by drug:', np.mean(mse_ls), '\nrmse by drug:', np.mean(rmse_ls))

            metric_df = pd.DataFrame(data={'drug': valid_drugs, 'mse': mse_ls, 'rmse': rmse_ls})

        elif avg_by == "cl":
            # avg by cl: take all the avg dr of the drugs (from train) and per cell line calc. ssc, mse and rmse
            # avg ssc, mse and rmse over all cell lines in the end
            valid_cls = []

            for cl in test.columns:
                y_true_with_nan = test[cl]
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)  # remove nans in cell line  (drugs not meass. in cl)

                if y_true.shape[0] > 1:
                    valid_cls.append(cl)

                    y_pred_original = drug_avg
                    y_pred = np.delete(y_pred_original, nan_idx)

                    pcc_cl = stats.pearsonr(y_true, y_pred)[0]
                    scc_cl = stats.spearmanr(y_true, y_pred)[0]
                    mse_cl = mean_squared_error(y_true, y_pred)
                    rmse_cl = mean_squared_error(y_true, y_pred, squared=False)

                    pcc_ls.append(pcc_cl)
                    scc_ls.append(scc_cl)
                    mse_ls.append(mse_cl)
                    rmse_ls.append(rmse_cl)

            print('----------------- LCO, average by cl ------------')
            print('scc std by cl:', np.std(scc_ls), '\nrmse std by cl:', np.std(rmse_ls))
            print('\npcc by cl:', np.mean(pcc_ls), '\nscc by cl:', np.mean(scc_ls),
                  '\nmse by cl:', np.mean(mse_ls), '\nrmse by cl:', np.mean(rmse_ls))

            metric_df = pd.DataFrame(data={'cl': valid_cls, 'pcc': pcc_ls, 'scc': scc_ls,
                                           'mse': mse_ls, 'rmse': rmse_ls})

    # ---------------------------------------------------- LDO ---------------------------------------------------------
    elif mode == "LDO":

        cl_avg = train.mean()
        pred = pd.DataFrame(cl_avg.rename("avg_" + metric))

        if avg_by == "drug":
            valid_drugs = []

            for drug in test.index:
                y_true_with_nan = test.loc[drug]
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)

                if y_true.shape[0] > 1:
                    valid_drugs.append(drug)

                    y_pred_original = cl_avg
                    y_pred = np.delete(y_pred_original, nan_idx)

                    pcc_cl = stats.pearsonr(y_true, y_pred)[0]
                    scc_cl = stats.spearmanr(y_true, y_pred)[0]
                    mse_cl = mean_squared_error(y_true, y_pred)
                    rmse_cl = mean_squared_error(y_true, y_pred, squared=False)

                    pcc_ls.append(pcc_cl)
                    scc_ls.append(scc_cl)
                    mse_ls.append(mse_cl)
                    rmse_ls.append(rmse_cl)

            print('----------------- LDO, average by drug ------------')
            print('scc std by drug:', np.std(scc_ls), '\nrmse std by drug:', np.std(rmse_ls))
            print('\npcc by drug:', np.mean(pcc_ls), '\nscc by drug:', np.mean(scc_ls),
                  '\nmse by drug:', np.mean(mse_ls), '\nrmse by drug:', np.mean(rmse_ls))

            metric_df = pd.DataFrame(data={'drug': valid_drugs, 'pcc': pcc_ls, 'scc': scc_ls,
                                           'mse': mse_ls, 'rmse': rmse_ls})

        elif avg_by == "cl":
            valid_cls = []

            for cl in test.columns:
                y_true = test[cl].dropna()  # drop nas from drug vector -> dr values of a cl over all test drugs

                if y_true.shape[0] > 1:
                    valid_cls.append(cl)

                    y_pred = np.repeat(cl_avg[cl], len(y_true))  # array as long as y_true repeating predic mean of cl

                    mse_cl = mean_squared_error(y_true, y_pred)
                    rmse_cl = mean_squared_error(y_true, y_pred, squared=False)

                    mse_ls.append(mse_cl)
                    rmse_ls.append(rmse_cl)

            print('----------------- LDO, average by cl ------------')
            print('rmse std by cl:', np.std(rmse_ls))
            print('\nmse by cl:', np.mean(mse_ls), '\nrmse by cl:', np.mean(rmse_ls))

            metric_df = pd.DataFrame(data={'drug': valid_cls, 'mse': mse_ls, 'rmse': rmse_ls})

    # ---------------------------------------------------- LPO ---------------------------------------------------------
    elif mode == "LPO":

        if avg_by == "drug":
            drug_avg = train.groupby('Compound')[metric].mean()
            pred = pd.DataFrame(drug_avg.rename("avg_" + metric))
            test_groups = test.groupby('Compound')

            # drop drugs which have an avg value of nan
            # (by chance, none of the pair values in the train set came from these drugs)
            drug_avg.dropna(inplace=True)

            y_true = np.array([])
            y_pred = np.array([])

            for drug in drug_avg.index:
                if drug in test_groups.groups.keys():
                    y_true_group_with_nan = test_groups.get_group(drug)[metric]
                    nan_idx = np.argwhere(np.isnan(y_true_group_with_nan))
                    y_true_group = np.delete(y_true_group_with_nan, nan_idx)

                    y_pred_group = np.repeat(drug_avg[drug], len(y_true_group))

                    y_true = np.concatenate((y_true, y_true_group))
                    y_pred = np.concatenate((y_pred, y_pred_group))

            pcc = stats.pearsonr(y_true, y_pred)[0]
            scc = stats.spearmanr(y_true, y_pred)[0]
            mse = mean_squared_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            print('----------------- LPO, average by drug ------------')
            print('pcc by drug:', pcc,
                  '\nscc by drug:', scc,
                  '\nmse by drug:', mse,
                  '\nrmse by drug:', rmse)

            metric_df = pd.Series(data={'pcc': pcc, 'scc': scc, 'mse': mse, 'rmse': rmse})

        elif avg_by == "cl":
            cl_avg = train.groupby('Primary Cell Line Name')[metric].mean()
            pred = pd.DataFrame(cl_avg.rename("avg_" + metric))
            test_groups = test.groupby('Primary Cell Line Name')

            # drop cls which have an avg value of nan
            # (by chance, none of the pair values in the train set came from these cell lines)
            cl_avg.dropna(inplace=True)

            y_true = np.array([])
            y_pred = np.array([])

            for cl in cl_avg.index:
                if cl in test_groups.groups.keys():
                    y_true_group_with_nan = test_groups.get_group(cl)[metric]
                    nan_idx = np.argwhere(np.isnan(y_true_group_with_nan))
                    y_true_group = np.delete(y_true_group_with_nan, nan_idx)

                    y_pred_group = np.repeat(cl_avg[cl], len(y_true_group))

                    y_true = np.concatenate((y_true, y_true_group))
                    y_pred = np.concatenate((y_pred, y_pred_group))

            pcc = stats.pearsonr(y_true, y_pred)[0]
            scc = stats.spearmanr(y_true, y_pred)[0]
            mse = mean_squared_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            print('----------------- LPO, average by cl ------------')
            print('pcc by cl:', pcc,
                  '\nscc by cl:', scc,
                  '\nmse by cl:', mse,
                  '\nrmse by cl:', rmse)

            metric_df = pd.Series(data={'pcc': pcc, 'scc': scc, 'mse': mse, 'rmse': rmse})

    return pred, metric_df


if __name__ == "__main__":
    # indices = pd.read_csv("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/cl_drug_indices.csv", header=0)
    label_matrix = pd.read_csv("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/EC50 ("
                               "µM)_matrix.csv", header=0, index_col=0)
    label_matrix.reset_index(inplace=True)
    metric = "EC50 (µM)"
    task = "LPO"

    train, test = get_train_test_set(label_matrix, task, 0.8, metric)

    train_2, test_2 = preprocessing(train, test, task, metric, remove_out=True, log_transform=True)

    prediction, accuracy_meas = calc_metric(train_2, test_2, task, "drug", metric)

    # foldtype = "cl" + '_fold'
    # train_ic50, test_ic50, train_cls, test_cls = get_train_test_ic50(foldtype, [0,1,2], [4], indices, label_matrix)
