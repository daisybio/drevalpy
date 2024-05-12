# -*- coding: utf-8 -*-
import argparse
import logging
import numpy as np
import os
import pandas as pd
import warnings
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, ADASYN
from itertools import cycle
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.preprocessing import deseq2_norm
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def mkdir(directory):
    """
    Create a directory if it does not exist
    """
    if not os.path.exists(directory):
        logger.info('creating folder: %s' % directory)
        os.makedirs(directory)


def parse_data(argv):
    """
    Parse the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--toml", help="Path to the toml file containing the metadata", type=str)
    parser.add_argument("-d", "--dir", help="Path to the directory where the results will be saved", type=str)
    args = parser.parse_args(argv)

    return args.dir, args.toml

def split(x, n):
    """
    Split a number into 'n' parts as evenly as possible and return the lengths of the parts as a list of integers
    """
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
    """
    Get drug and cl info for a given index in the label matrix
    """
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
    """
    Get drug and cl info for each index in split_index_arr and return as lists of drugs and cls respectively
    """
    indx_drug_ls = []
    indx_cl_ls = []

    for indx in split_index_arr:
        indx_drug, indx_cl = cl_drug_info(label_matrix, label_matrix_LPO, indx)
        indx_drug_ls.append(indx_drug)
        indx_cl_ls.append(indx_cl)

    return indx_drug_ls, indx_cl_ls


def remove_outliers(drp_con, metric, mode, data_set_str):
    """
    remove or replace outliers in the dataset using the IQR method (1.5 * IQR) as a threshold for outliers
    """
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


def normalize_gene_expression(X_train, X_test, mode="NormTransform", n_cpus=1):
    """
    Normalizes the gene expression data using the specified normalization method. The normalization method is specified in
    the metadata file and can be one of the following: "NormTransform", "VST". The number of cpus to use for the
    normalization is also specified in the metadata file. The gene expression data is normalized using the pydeseq2
    package.
    """
    # convert the data frame to a numpy array and round the floats to integers
    gene_counts_train_np = np.round(X_train.values).astype(int)
    gene_counts_train_df = pd.DataFrame(gene_counts_train_np, index=X_train.index,
                                        columns=X_train.columns)
    gene_counts_test_np = np.round(X_test.values).astype(int)
    gene_counts_test_df = pd.DataFrame(gene_counts_test_np, index=X_test.index,
                                       columns=X_test.columns)

    if mode == "NormTransform":
        # perform normalisation using the pydeseq2 package
        logger.info("Using Pydeseq2 norm function to normalize counts")
        deseq2_counts_train, size_factors = deseq2_norm(gene_counts_train_df)
        X_train = np.log2(deseq2_counts_train + 1)
        deseq2_counts_test, size_factors = deseq2_norm(gene_counts_test_df)
        X_test = np.log2(deseq2_counts_test + 1)

    elif mode == "VST":
        # for whatever reason it needs a condition :(
        logger.info(f"Using Pydeseq2 vst function to normalize counts, using {n_cpus} cpus")
        condition = cycle(['Bananas', 'Oranges', 'Strawberries'])
        metadata_train = pd.DataFrame({'cell_line': gene_counts_train_df.index})
        metadata_test = pd.DataFrame({'cell_line': gene_counts_test_df.index})
        metadata_train['condition'] = [next(condition) for cond in range(len(metadata_train))]
        metadata_test['condition'] = [next(condition) for cond in range(len(metadata_test))]
        metadata_train.index = gene_counts_train_df.index
        metadata_test.index = gene_counts_test_df.index

        infer = DefaultInference(n_cpus=n_cpus)  # dds class n_cpu argument not working, so I had to change it here
        dds_train = DeseqDataSet(counts=gene_counts_train_df, metadata=metadata_train, inference=infer)
        dds_test = DeseqDataSet(counts=gene_counts_test_df, metadata=metadata_test, inference=infer)
        dds_train.vst(use_design=False)
        dds_test.vst(use_design=False)
        X_train = pd.DataFrame(dds_train.layers["vst_counts"],
                               index=gene_counts_train_df.index, columns=gene_counts_train_df.columns)
        X_test = pd.DataFrame(dds_test.layers["vst_counts"],
                              index=gene_counts_test_df.index, columns=gene_counts_test_df.columns)

    return X_train, X_test


def preprocessing(train_prepro, test_prepro, task, metric, remove_out=False, norm_data=False, log_transform=False):
    """
    Preprocesses the datasets for the linear models. The preprocessing steps are: outlier removal, normalization and
    log transformation. The preprocessing steps are specified in the metadata file.
    """
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


def cross_validation_fit(X_train, y_train, predictor, nCV_folds, hyperparameters):
    """
    Fits the model to the training set using cross validation. The number of cross validation folds is specified in the
    metadata file. The hyperparameters are tuned using grid search. If the number of samples in the training set is less
    than the number of cross validation folds, the number of cross validation folds is set to the number of samples. If
    the number of samples in the training set is 1, no cross validation or grid search is performed. In this case the
    model is fitted to the single sample and the output is constant.
    """
    # check if CV fold is legal
    if len(X_train) == 1:
        warnings.warn("Only one sample for target. No Cross validation or Grid search performed.")

        model = predictor  # fit the model
        model.fit(X_train, y_train)  # fit model to single sample (no CV or grid search, output is constant)

    elif len(X_train) < nCV_folds:
        nCV_folds = len(X_train)
        warnings.warn("Number of CV folds is larger than the number of samples. CV folds set to {}".format(nCV_folds))

        # perform grid search to find the best hyperparameters
        model = GridSearchCV(predictor, hyperparameters, cv=nCV_folds)
        model.fit(X_train, y_train)  # fit the model
    else:
        # perform grid search to find the best hyperparameters
        model = GridSearchCV(predictor, hyperparameters, cv=nCV_folds)
        model.fit(X_train, y_train)  # fit the model

    return model

def oversampling(X_train, y_train, oversampling_method, ncpus, k_neighbours = 5):
    """
    Oversamples the minority class of the training set using the specified oversampling method. The oversampling method
    is specified in the metadata file and can be one of the following: "RandomOverSampler", "SMOTE", "SMOTEN",
    "BorderlineSMOTE", "KMeansSMOTE", "SVMSMOTE", "ADASYN". The number of cpus to use for the oversampling is also
    specified in the metadata file. Accounts for class imbalance.
    If at least one unique class has less than or equal to  5 samples (i.e. <= to number of k neighbours set in oversampling method)
    oversampling is performed with the minimum number of samples in the class. If at least one unique class has only 1 sample,
    random oversampling is performed instead. Note that k_neighbors = number of samples in the class - 1, meaning its
    the actual number of neighbours around a sample.
    """
    sample_count = np.unique(y_train, return_counts=True)[1]
    if 1 < sample_count.min() <= k_neighbours:
        warnings.warn("At least one unique class has less than or equal to 5 samples. Oversampling performed with {} k_neighbours instead".format(sample_count.min() - 1))
        k_neighbours = sample_count.min() - 1
    elif sample_count.min() == 1:
        warnings.warn("At least one unique class has only 1 sample. Random oversampling performed instead")
        oversampling_method = "RandomOverSampler"

    if oversampling_method == "RandomOverSampler":
        ovs = RandomOverSampler(random_state=0)
    if oversampling_method == "SMOTE":
        ovs = SMOTE(random_state=0, n_jobs=ncpus, k_neighbors=k_neighbours)
    elif oversampling_method == "SMOTEN":
        ovs = SMOTEN(random_state=0, n_jobs=ncpus, k_neighbors=k_neighbours)
    elif oversampling_method == "BorderlineSMOTE":
        ovs = BorderlineSMOTE(random_state=0, n_jobs=ncpus, k_neighbors=k_neighbours)
    elif oversampling_method == "KMeansSMOTE":
        ovs = KMeansSMOTE(random_state=0, n_jobs=ncpus, k_neighbors=k_neighbours)
    elif oversampling_method == "SVMSMOTE":
        ovs = SVMSMOTE(random_state=0, n_jobs=ncpus, k_neighbors=k_neighbours)
    elif oversampling_method == "ADASYN":
        ovs = ADASYN(random_state=0, n_jobs=ncpus, n_neighbors=k_neighbours)

    X_train, y_train = ovs.fit_resample(X_train, y_train)

    return X_train, y_train
