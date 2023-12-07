# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
from os.path import dirname, join, abspath
from scipy import stats
from sklearn.metrics import mean_squared_error

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.utils import mkdir, preprocessing, get_train_test_set


class NaivePredictor:

    def __init__(self, dataroot, metric, task, avg_by):
        self.dataroot = dataroot # path to the matrix file
        self.metric = metric # Amax, IC50, EC50, ...
        self.task = task # LCO, LDO, LPO
        self.avg_by = avg_by # avg by drug or cl

        self.train_set = None # train set with the same shape as the test set
        self.test_set = None # test set with the same shape as the train set
        self.metric_df = None # dataframe with the performance metrics
        self.predicted_means = None    # predicted means

        self.labelmatrix = pd.read_csv(self.dataroot, header=0, index_col=0)
        self.labelmatrix.reset_index(inplace=True)

    def train(self, train):
        if self.task == "LCO":
            train_avg = train.mean(axis=1)

        elif self.task == "LDO":
            train_avg = train.mean()

        elif self.task == "LPO":
            if self.avg_by == "drug":
                train_avg = train.groupby('Compound')[self.metric].mean()

            elif self.avg_by == "cl":
                train_avg = train.groupby('Primary Cell Line Name')[self.metric].mean()

        self.predicted_means = pd.DataFrame(train_avg.rename("avg_" + self.metric))

        return self.predicted_means


    def evaluate(self, test):
        pcc_ls = []
        scc_ls = []
        mse_ls = []
        rmse_ls = []

        # ---------------------------------------------------- LCO ---------------------------------------------------------
        if self.task == "LCO":

            if self.avg_by == "drug":
                # avg by drug: take all the avg dr of the drugs (from train) and per drug calc. ssc, mse and rmse
                # over all cell lines in test set. Avg ssc, mse and rmse over all drugs in the end
                valid_drugs = []
                # calculate the SCC, mse and rmse using the test set: averaging per drug
                # SCC doesn't work as we cant calc. using a constant (the mean)

                for drug in self.predicted_means.index:
                    y_true = test.loc[drug].dropna()  # drop nas from drug vector -> dr vlaues of a drug over all test cl

                    if y_true.shape[0] > 1 and not np.isnan(self.predicted_means.loc[drug]).values[0]: # only if cl is not nan in pred and test contin
                        valid_drugs.append(drug)
                        y_pred = np.repeat(self.predicted_means.loc[drug], len(y_true))  # array len = y_true repeating predic mean of drug

                        mse_drug = mean_squared_error(y_true, y_pred)
                        rmse_drug = mean_squared_error(y_true, y_pred, squared=False)

                        mse_ls.append(mse_drug)
                        rmse_ls.append(rmse_drug)

                print('----------------- LCO, average by drug ------------')
                print(f'rmse std by drug: {np.std(rmse_ls)}')
                print('\nmse by drug:', np.mean(mse_ls), '\nrmse by drug:', np.mean(rmse_ls), end="")

                self.metric_df = pd.DataFrame(data={'drug': valid_drugs, 'mse': mse_ls, 'rmse': rmse_ls})

            elif self.avg_by == "cl":
                # avg by cl: take all the avg dr of the drugs (from train) and per cell line calc. ssc, mse and rmse
                # avg ssc, mse and rmse over all cell lines in the end
                valid_cls = []

                for cl in test.columns:
                    y_true_with_nan = test[cl]
                    y_pred_original = self.predicted_means.squeeze() # squeeze to remove the extra dimension
                    mask = ~np.isnan(y_true_with_nan) & ~np.isnan(y_pred_original)

                    y_true = y_true_with_nan[mask]
                    y_pred = y_pred_original[mask]

                    if y_true.shape[0] > 1:
                        valid_cls.append(cl)

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
                      '\nmse by cl:', np.mean(mse_ls), '\nrmse by cl:', np.mean(rmse_ls), end="")

                self.metric_df = pd.DataFrame(data={'cl': valid_cls, 'pcc': pcc_ls, 'scc': scc_ls,
                                               'mse': mse_ls, 'rmse': rmse_ls})

        # ---------------------------------------------------- LDO ---------------------------------------------------------
        elif self.task == "LDO":

            if self.avg_by == "drug":
                valid_drugs = []

                for drug in test.index:
                    y_true_with_nan = test.loc[drug]
                    y_pred_original = self.predicted_means.squeeze() # squeeze to remove the extra dimension
                    mask = ~np.isnan(y_true_with_nan) & ~np.isnan(y_pred_original)

                    y_true = y_true_with_nan[mask]
                    y_pred = y_pred_original[mask]

                    if y_true.shape[0] > 1:
                        valid_drugs.append(drug)

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
                      '\nmse by drug:', np.mean(mse_ls), '\nrmse by drug:', np.mean(rmse_ls), end="")

                self.metric_df = pd.DataFrame(data={'drug': valid_drugs, 'pcc': pcc_ls, 'scc': scc_ls,
                                               'mse': mse_ls, 'rmse': rmse_ls})

            elif self.avg_by == "cl":
                valid_cls = []

                for cl in test.columns:
                    y_true = test[cl].dropna()  # drop nas from drug vector -> dr values of a cl over all test drugs

                    if y_true.shape[0] > 1 and not np.isnan(self.predicted_means.loc[cl]).values[0]:
                        valid_cls.append(cl)
                        y_pred = np.repeat(self.predicted_means.loc[cl], len(y_true))  # array as long as y_true repeating predic mean of cl

                        mse_cl = mean_squared_error(y_true, y_pred)
                        rmse_cl = mean_squared_error(y_true, y_pred, squared=False)

                        mse_ls.append(mse_cl)
                        rmse_ls.append(rmse_cl)

                print('----------------- LDO, average by cl ------------')
                print('rmse std by cl:', np.std(rmse_ls))
                print('\nmse by cl:', np.mean(mse_ls), '\nrmse by cl:', np.mean(rmse_ls), end="")

                self.metric_df = pd.DataFrame(data={'drug': valid_cls, 'mse': mse_ls, 'rmse': rmse_ls})

        # ---------------------------------------------------- LPO ---------------------------------------------------------
        elif self.task == "LPO":

            if self.avg_by == "drug":
                test_groups = test.groupby('Compound')

                # drop drugs which have an avg value of nan
                # (by chance, none of the pair values in the train set came from these drugs)
                self.predicted_means.dropna(inplace=True)

                y_true = np.array([])
                y_pred = np.array([])

                for drug in self.predicted_means.index:
                    if drug in test_groups.groups.keys():
                        y_true_group_with_nan = test_groups.get_group(drug)[self.metric]
                        nan_idx = np.argwhere(np.isnan(y_true_group_with_nan))
                        y_true_group = np.delete(y_true_group_with_nan, nan_idx)

                        y_pred_group = np.repeat(self.predicted_means.loc[drug], len(y_true_group))

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
                      '\nrmse by drug:', rmse, end="")

                self.metric_df = pd.Series(data={'pcc': pcc, 'scc': scc, 'mse': mse, 'rmse': rmse})

            elif self.avg_by == "cl":
                test_groups = test.groupby('Primary Cell Line Name')

                # drop cls which have an avg value of nan
                # (by chance, none of the pair values in the train set came from these cell lines)
                self.predicted_means.dropna(inplace=True)

                y_true = np.array([])
                y_pred = np.array([])

                for cl in self.predicted_means.index:
                    if cl in test_groups.groups.keys():
                        y_true_group_with_nan = test_groups.get_group(cl)[self.metric]
                        nan_idx = np.argwhere(np.isnan(y_true_group_with_nan))
                        y_true_group = np.delete(y_true_group_with_nan, nan_idx)

                        y_pred_group = np.repeat(self.predicted_means.loc[cl], len(y_true_group))

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
                      '\nrmse by cl:', rmse, end="")

                self.metric_df = pd.Series(data={'pcc': pcc, 'scc': scc, 'mse': mse, 'rmse': rmse})

        return self.metric_df

    def get_features(self):

        self.train_set, self.test_set = get_train_test_set(naive_predictor.labelmatrix,
                                                           naive_predictor.task,
                                                           0.8,
                                                           naive_predictor.metric)

    def save_results(self, result_path):
        self.predicted_means.to_csv(result_path + self.task + self.avg_by + '_results.csv', index=True)
        self.metric_df.to_csv(result_path + self.task + '_avg_by_' + self.avg_by + '_metrics.csv', index=True)


if __name__ == "__main__":

    naive_predictor = NaivePredictor("/nfs/home/students/m.lorenz/datasets/cell_viability/CCLE/matrixes_raw/Amax_matrix.csv",
                                     "Amax", "LPO", "drug")

    naive_predictor.get_features()

    train_2, test_2 = preprocessing(naive_predictor.train_set,
                                    naive_predictor.test_set,
                                    naive_predictor.task,
                                    naive_predictor.metric, remove_out=True, log_transform=True)

    pred = naive_predictor.train(train_2) # train the model
    accuracy_meas = naive_predictor.evaluate(test_2) # evaluate the model

    mkdir("results_2/")

    naive_predictor.save_results("results_2/")
