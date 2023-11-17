# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 14:46:01 2022

@author: jessi
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#%%
def get_train_test_ic50(fold_type, train_fold, val_fold, indices, label_matrix):
    '''
    fold_type: 'cl_fold', 'drug_fold', 'pair_fold'
    
    i: the test fold number
    
    indices: DataFrame that contains cl and drug indices and CV fold info
    
    label_matrix: np.array that contains the normalized IC50 values (drugs by cell lines)
    '''
    label_matrix = label_matrix.copy().to_numpy()
    indices_train = indices.loc[indices[fold_type].isin(train_fold)] # df containing all train indices
    indices_test = indices.loc[indices[fold_type].isin(val_fold)]  # df containing all test indices
    
    # LCO
    if fold_type == 'cl_fold':
        train_cls = list(np.unique(indices_train['cl_idx'].values))    # only normalize based on unique cls
        test_cls = list(np.unique(indices_test['cl_idx'].values))
        
        train_ic50 = label_matrix[:, train_cls]
        test_ic50 = label_matrix[:, test_cls]
        return train_ic50, test_ic50, train_cls, test_cls

    #LDO
    elif fold_type == 'drug_fold':
        train_drugs = list(np.unique(indices_train['drug_idx'].values))
        test_drugs = list(np.unique(indices_test['drug_idx'].values))
        
        train_ic50 = label_matrix[train_drugs, :]
        test_ic50 = label_matrix[test_drugs, :]
        return train_ic50, test_ic50, train_drugs, test_drugs
        
    #LPO
    elif fold_type == 'pair_fold':
        train_drug_idx = indices_train['drug_idx'].values
        train_cl_idx = indices_train['cl_idx'].values
        train_ic_50 = label_matrix[train_drug_idx, train_cl_idx] # 1d output
        
        test_drug_idx = indices_test['drug_idx'].values
        test_cl_idx = indices_test['cl_idx'].values
        test_ic_50 = label_matrix[test_drug_idx, test_cl_idx] # 1d output
        return train_drug_idx, train_cl_idx, test_drug_idx, test_cl_idx, train_ic_50, test_ic_50
        
    

def calc_metrics_by_fold(train_ic50, test_ic50, fold_type, label_matrix, test_idx, by): #label_matrix is a df
    label_matrix_df = label_matrix.copy()
    label_matrix = label_matrix_df.to_numpy()
    pcc_ls = []
    spc_ls = []
    mse_ls = []
    rmse_ls = []
    r2_ls = []
    
    # ----------------------------- LDO ------------------------------
    if fold_type == 'drug_fold':
        valid_drugs = []
        
        # check rows and columns of train and test_ic50 to make sure no rows/cols are entirely NaN
        nan_col = []
        for col in range(train_ic50.shape[1]):
            if np.all(np.isnan(train_ic50[:, col])):
                nan_col.append(col)
        train_ic50 = np.delete(train_ic50, nan_col, axis=1)
        test_ic50 = np.delete(test_ic50, nan_col, axis=1)
        
        nan_train_row = []
        for train_row in range(train_ic50.shape[0]):
            if np.all(np.isnan(train_ic50[train_row])):
                nan_train_row.append(train_row)
        train_ic50 = np.delete(train_ic50, nan_train_row, axis=0)
        
        nan_test_row = []
        for test_row in range(test_ic50.shape[0]):
            if np.all(np.isnan(test_ic50[test_row])):
                nan_test_row.append(test_row)
        test_ic50 = np.delete(test_ic50, nan_test_row, axis=0)
        
        valid_cls = np.delete(label_matrix_df.columns.values, nan_col)
        LDO_mean_train_ic50 = np.nanmean(train_ic50, axis=0)
        
        pred_df = pd.DataFrame(data={'CCL': valid_cls, 'predictions': LDO_mean_train_ic50})
        
        if by == 'drug': # means calculating baseline metrics by drug
            for i in range(test_ic50.shape[0]):
                y_true_with_nan = test_ic50[i, :]

                # delete nan values from each test drug's IC50 (not all cls are treated with the same drug)
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx, axis=None)
                
                if y_true.shape[0] > 1:
                    valid_drugs.append(i)
                    
                    y_pred_original = LDO_mean_train_ic50
                    y_pred = np.delete(y_pred_original, nan_idx, axis=None)

                    pcc = stats.pearsonr(y_true, y_pred)[0]
                    spc = stats.spearmanr(y_true, y_pred)[0]
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    r2 = r2_score(y_true, y_pred)
                    pcc_ls.append(pcc)
                    spc_ls.append(spc)
                    mse_ls.append(mse)
                    rmse_ls.append(rmse)
                    r2_ls.append(r2)
            print('----------------- LDO, average by drug ------------')
            print('scc std by drug:', np.std(spc_ls), 'rmse std by drug:', np.std(rmse_ls))
            print('pcc by drug:', np.mean(pcc_ls), 'scc by drug:', np.mean(spc_ls), 
                  'mse by drug:', np.mean(mse_ls), 'rmse by drug:', np.mean(rmse_ls), 
                  'r2 by drug:', np.mean(r2_ls))
            
            #need to remove invalid drug names from test set (some drugs only have one sample,
            #therefore correlation metrics cannot be calculated and are thus not included)
            metric_df = pd.DataFrame(data={'drug': label_matrix_df.index.values[np.asarray(test_idx)[valid_drugs]], 
                                       'pcc': pcc_ls, 'scc': spc_ls, 
                                       'mse': mse_ls, 'rmse': rmse_ls,
                                       'r2': r2_ls})
            return pred_df, metric_df
        
        elif by == 'cl': # calculating baseline metrics by cl
            # For LDO, the "predictions" are constant for each cell line, so PCC and SCC cannot be computed
            valid_cl_idx = []
            for i in range(test_ic50.shape[1]):
                y_true_with_nan = test_ic50[:, i]
                
                # delete nan values from each CL's IC50 values
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)
               
                if y_true.shape[0] > 1:
                    valid_cl_idx.append(i)
                    y_pred = np.repeat(LDO_mean_train_ic50[i], y_true.shape[0])
               
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    mse_ls.append(mse)
                    rmse_ls.append(rmse)

            print('----------------- LDO, average by cl ------------')
            print('rmse std by cl:', np.std(rmse_ls))
            print('mse by cl:', np.mean(mse_ls), 'rmse by cl:', np.mean(rmse_ls))
            metric_df = pd.DataFrame(data={'CCL': valid_cls[valid_cl_idx], 
                                           'mse': mse_ls, 'rmse': rmse_ls})
            return pred_df, metric_df
        
        
        elif by == 'vector': # calculating baseline metrics as a vector
            y_true_ls = []
            y_pred_ls = []
            
            # for each drug in testset, delete nan values and concat pred and 
            # ground_truth vectors later for vector evalution
            for i in range(test_ic50.shape[0]):
                y_true_with_nan = test_ic50[i, :]
                
                # delete nan values from each test drug's IC50 (not all cls are treated with the same drug)
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx, axis=None)
                
                if y_true.shape[0] >= 1:
                    y_pred_original = LDO_mean_train_ic50
                    y_pred = np.delete(y_pred_original, nan_idx, axis=None)
                    y_true_ls.append(y_true)
                    y_pred_ls.append(y_pred)
            
            y_true_vec = np.concatenate(y_true_ls).ravel()
            y_pred_vec = np.concatenate(y_pred_ls).ravel()
            pcc = stats.pearsonr(y_true_vec, y_pred_vec)[0]
            spc = stats.spearmanr(y_true_vec, y_pred_vec)[0]
            mse = mean_squared_error(y_true_vec, y_pred_vec)
            rmse = mean_squared_error(y_true_vec, y_pred_vec, squared=False)
            r2 = r2_score(y_true_vec, y_pred_vec)
            
            metric_df = pd.DataFrame(data={'pcc': pcc, 'scc': spc, 
                                       'mse': mse, 'rmse': rmse,
                                       'r2': r2})
            return pred_df, metric_df
        
        
        
    # --------------------------- LCO --------------------------------
    elif fold_type == 'cl_fold':
        # check rows and columns of train and test_ic50 to make sure no rows/cols are entirely NaN
        nan_row = []
        valid_cls = []
        
        for row in range(train_ic50.shape[0]):
            if np.all(np.isnan(train_ic50[row])):
                nan_row.append(row)
        train_ic50 = np.delete(train_ic50, nan_row, axis=0)
        test_ic50 = np.delete(test_ic50, nan_row, axis=0)
        
        nan_train_col = []
        for train_col in range(train_ic50.shape[1]):
            if np.all(np.isnan(train_ic50[:,train_col])):
                nan_train_col.append(train_col)
        train_ic50 = np.delete(train_ic50, nan_train_col, axis=1)
        
        nan_test_col = []
        for test_col in range(test_ic50.shape[1]):
            if np.all(np.isnan(test_ic50[:,test_col])):
                nan_test_col.append(test_col)
        test_ic50 = np.delete(test_ic50, nan_test_col, axis=1)
        
        valid_drugs = np.delete(label_matrix_df.index.values, nan_row)
        LCO_mean_train_ic50 = np.nanmean(train_ic50, axis=1)
        
        pred_df = pd.DataFrame(data={'drug': valid_drugs, 'predictions': LCO_mean_train_ic50})
        
        if by == 'cl': # means calculating baseline metrics by cell line  
            for i in range(test_ic50.shape[1]):
                y_true_with_nan = test_ic50[:, i]

                # delete nan values from each test CL's IC50 (not all cls are treated with the same drug)
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)
                
                if y_true.shape[0] > 1:
                    valid_cls.append(i)
                    
                    y_pred_original = LCO_mean_train_ic50
                    y_pred = np.delete(y_pred_original, nan_idx)

                    pcc = stats.pearsonr(y_true, y_pred)[0]
                    spc = stats.spearmanr(y_true, y_pred)[0]
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
                    r2 = r2_score(y_true, y_pred)
                    pcc_ls.append(pcc)
                    spc_ls.append(spc)
                    mse_ls.append(mse)
                    rmse_ls.append(rmse)
                    r2_ls.append(r2)
            print('----------------- LCO, average by cl ------------')
            print('scc std by cl:', np.std(spc_ls), 'rmse std by cl:', np.std(rmse_ls))
            print('pcc by cl:', np.mean(pcc_ls), 'scc by cl:', np.mean(spc_ls), 
                  'mse by cl:', np.mean(mse_ls), 'rmse by cl:', np.mean(rmse_ls), 
                  'r2 by cl:', np.mean(r2_ls))
            
            metric_df = pd.DataFrame(data={'CCL': label_matrix_df.columns.values[np.asarray(test_idx)[valid_cls]], 
                                           'pcc': pcc_ls, 'scc': spc_ls, 
                                           'mse': mse_ls, 'rmse': rmse_ls,
                                           'r2': r2_ls})
            return pred_df, metric_df
        
        elif by == 'drug': # calculating baseline metrics by drugs
            valid_drug_idx = []
            # For LCO, the "predictions" are constant for each drug, so PCC and SCC cannot be computed
            for i in range(test_ic50.shape[0]):
                y_true_with_nan = test_ic50[i, :]
                
                # delete nan values from each drug's IC50 values
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)
                if y_true.shape[0] > 1:
                    valid_drug_idx.append(i)
                    
                    y_pred = np.repeat(LCO_mean_train_ic50[i], y_true.shape[0])
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = mean_squared_error(y_true, y_pred, squared=False)
#                     r2 = r2_score(y_true, y_pred)
                    mse_ls.append(mse)
                    rmse_ls.append(rmse)
#                     r2_ls.append(r2)
            print('----------------- LCO, average by drug ------------')
            print('rmse std by drug:', np.std(rmse_ls))
            print('mse by drug:', np.mean(mse_ls), 'rmse by drug:', np.mean(rmse_ls))
            
            metric_df = pd.DataFrame(data={'drug': valid_drugs[valid_drug_idx], 
                                           'mse': mse_ls, 'rmse': rmse_ls})
            
            return pred_df, metric_df

        elif by == 'vector':
            y_true_ls = []
            y_pred_ls = []
            
            for i in range(test_ic50.shape[1]):
                y_true_with_nan = test_ic50[:, i]

                # delete nan values from each test CL's IC50 (not all cls are treated with the same drug)
                nan_idx = np.argwhere(np.isnan(y_true_with_nan))
                y_true = np.delete(y_true_with_nan, nan_idx)
                
                if y_true.shape[0] >= 1:
                    y_pred_original = LCO_mean_train_ic50
                    y_pred = np.delete(y_pred_original, nan_idx)
                    y_true_ls.append(y_true)
                    y_pred_ls.append(y_pred)
                    
            y_true_vec = np.concatenate(y_true_ls).ravel()
            y_pred_vec = np.concatenate(y_pred_ls).ravel()
            pcc = stats.pearsonr(y_true_vec, y_pred_vec)[0]
            spc = stats.spearmanr(y_true_vec, y_pred_vec)[0]
            mse = mean_squared_error(y_true_vec, y_pred_vec)
            rmse = mean_squared_error(y_true_vec, y_pred_vec, squared=False)
            r2 = r2_score(y_true_vec, y_pred_vec)
            
            metric_df = pd.DataFrame(data={'pcc': pcc, 'scc': spc, 
                                       'mse': mse, 'rmse': rmse,
                                       'r2': r2})
            
            return pred_df, metric_df
        
        
        
def calc_metrics_by_fold_LPO(train_drug_idx, train_cl_idx, test_drug_idx, test_cl_idx, test_ic50, label_matrix, by=None):
    label_matrix_df = label_matrix.copy()
    label_matrix = label_matrix_df.to_numpy()
    # mask label matrix to only contain train pairs
    mask = np.full((label_matrix.shape[0], label_matrix.shape[1]), np.nan)
    mask[train_drug_idx, train_cl_idx] = 1
    label_matrix = label_matrix*mask
    print(mask)
    print(label_matrix)
    
    ground_truth_ls = []
    naive_pred_ls = []
    for i in range(test_drug_idx.shape[0]):
        ground_truth = label_matrix_df.iloc[test_drug_idx[i],test_cl_idx[i]]
        naive_pred = (np.nanmean(label_matrix[test_drug_idx[i], :]) +  np.nanmean(label_matrix[:, test_cl_idx[i]]))/2
        naive_pred_ls.append(naive_pred)
        ground_truth_ls.append(ground_truth)

    naive_pred_arr = np.asarray(naive_pred_ls)
    ground_truth_arr = np.asarray(ground_truth_ls)
    pred_df = pd.DataFrame(data={'CCL': label_matrix_df.columns[test_cl_idx], 
                                 'drug': label_matrix_df.index[test_drug_idx],
                                 'ground_truth': ground_truth_arr,
                                 'prediction': naive_pred_arr})
    print(pred_df)
    print(pred_df.isna().sum())
    pcc_ls = []
    spc_ls = []
    mse_ls = []
    rmse_ls = []
    r2_ls = []
    
    if by == "None":
        return pred_df, None
    elif by == 'drug':
        valid_drugs = []
        unique_drugs = np.unique(test_drug_idx)
        for drug in unique_drugs:
            drug_indices = np.where(test_drug_idx == drug)[0]
            y_true = test_ic50[drug_indices]
            y_pred = naive_pred_arr[drug_indices]
            
            if y_true.shape[0] > 1:
                valid_drugs.append(drug)
                pcc = stats.pearsonr(y_true, y_pred)[0]
                spc = stats.spearmanr(y_true, y_pred)[0]
                mse = mean_squared_error(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                r2 = r2_score(y_true, y_pred)
                pcc_ls.append(pcc)
                spc_ls.append(spc)
                mse_ls.append(mse)
                rmse_ls.append(rmse)
                r2_ls.append(r2)
        print('----------------- LPO, average by drug ------------')
        print('scc std by drug:', np.std(spc_ls), 'rmse std by drug:', np.std(rmse_ls))
        print('pcc by drug:', np.mean(pcc_ls), 'scc by drug:', np.mean(spc_ls), 
              'mse by drug:', np.mean(mse_ls), 'rmse by drug:', np.mean(rmse_ls), 
              'r2 by drug:', np.mean(r2_ls))
        
        metric_df = pd.DataFrame(data={'drug': label_matrix_df.index[valid_drugs], 
                                       'pcc': pcc_ls, 'scc': spc_ls, 
                                       'mse': mse_ls, 'rmse': rmse_ls,
                                       'r2': r2_ls})
        return pred_df, metric_df
    
    elif by == 'cl':
        unique_cls = np.unique(test_cl_idx)
        valid_cls = []
        for cl in unique_cls:
            cl_indices = np.where(test_cl_idx == cl)[0]
            y_true = test_ic50[cl_indices]
            y_pred = naive_pred_arr[cl_indices]
            
            if y_true.shape[0] > 1:
                valid_cls.append(cl)
                pcc = stats.pearsonr(y_true, y_pred)[0]
                spc = stats.spearmanr(y_true, y_pred)[0]
                mse = mean_squared_error(y_true, y_pred)
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                r2 = r2_score(y_true, y_pred)
                pcc_ls.append(pcc)
                spc_ls.append(spc)
                mse_ls.append(mse)
                rmse_ls.append(rmse)
                r2_ls.append(r2)
        print('----------------- LPO, average by cl ------------')
        print('std by cl:', np.std(spc_ls), 'rmse by cl:', np.std(rmse_ls))
        print('pcc by cl:', np.mean(pcc_ls), 'scc by cl:', np.mean(spc_ls), 
              'mse by cl:', np.mean(mse_ls), 'rmse by cl:', np.mean(rmse_ls), 
              'r2 by cl:', np.mean(r2_ls))
        metric_df = pd.DataFrame(data={'CCL': label_matrix_df.columns[valid_cls], 
                                       'pcc': pcc_ls, 'scc': spc_ls, 
                                       'mse': mse_ls, 'rmse': rmse_ls,
                                       'r2': r2_ls})
        return pred_df, metric_df

    elif by == 'vector': 
        y_true = test_ic50
        y_pred = np.asarray(naive_pred_ls)
        pcc = stats.pearsonr(y_true, y_pred)[0]
        spc = stats.spearmanr(y_true, y_pred)[0]
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        print('pcc by vector:', pcc, 'scc by vector:', spc, 
              'mse by vector:', mse, 'rmse by vector:', rmse, 
              'r2 by vector:', r2)
        metric_df = pd.DataFrame(data={'pcc': pcc, 'scc': spc, 
                                       'mse': mse, 'rmse': rmse,
                                       'r2': r2})
        return pred_df, metric_df






