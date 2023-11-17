# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:51:14 2022

@author: jessi, Michael
"""

import pandas as pd
import os, sys, argparse
from os.path import dirname, join, abspath

# sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.append('/nfs/home/students/m.lorenz/Baseline models/')
from Naive_predictor.model import get_train_test_ic50, calc_metrics_by_fold, calc_metrics_by_fold_LPO
from utils.utils import mkdir


# %% parse_parameters
def parse_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot",
                        required=True,
                        help = "path for input data")
    parser.add_argument("--outroot",
                        required=True,
                        help = "path to save results")
    parser.add_argument("--metric",
                        required=True,
                        help = "name of the dile containing the \
                               drug response metric used (e.g. ic50,ec50,AUC,ActArea etc.")
    parser.add_argument("--pathway",
                        required=True,
                        help = "name of pathway collection (KEGG, PID, Reactome)")
    parser.add_argument("--foldtype",
                        required=True,
                        help = "type of validation scheme (pair, drug, cl)")
    parser.add_argument("--avg_by",
                        required=False,
                        help = "how to calculate performance metrics (vector, drug, cl), \
                        if not specified, then only the naive predictions are outputted")

    # emmulating cl input above so that I can work from the IDE
    sys.argv = ["main.py", "--dataroot", "datasets/cell_viability/CCLE",
                "--outroot", "Baseline models/Naive_predictor/results/",
                "--metric", "ActArea_matrix.csv", "--pathway", "KEGG", "--foldtype", "cl"]

    return parser.parse_args()


# %%
if __name__ == '__main__':
    args = parse_parameters()
    mkdir(args.outroot)
    if args.avg_by is None:
        sys.stdout = open(args.outroot + args.foldtype + '_' + args.pathway + '_' + "None" + '_log.txt', 'w')
    else:
        sys.stdout = open(args.outroot + args.foldtype + '_' + args.pathway + '_' + args.avg_by + '_log.txt', 'w')

    # import data
    indices = pd.read_csv(args.dataroot + '/cl_drug_indices.csv', header=0)
    label_matrix = pd.read_csv(args.dataroot  + '/matrixes/' + args.metric, header=0, index_col=0)
    foldtype = args.foldtype + '_fold'

    
    if args.foldtype == 'pair':
        train_drug_idx, train_cl_idx, test_drug_idx, test_cl_idx, train_ic50, test_ic50 = get_train_test_ic50(foldtype, [0,1,2], [4], 
                                                                                                              indices, label_matrix)
        pred_df, metric_df = calc_metrics_by_fold_LPO(train_drug_idx, train_cl_idx, 
                                                   test_drug_idx, test_cl_idx, 
                                                   test_ic50, 
                                                   label_matrix, by=args.avg_by)
    
    elif args.foldtype == 'drug':
        train_ic50, test_ic50, train_drugs, test_drugs = get_train_test_ic50(foldtype, [0,1,2], [4], indices, label_matrix)
        pred_df, metric_df = calc_metrics_by_fold(train_ic50, test_ic50, foldtype, label_matrix, test_drugs, by=args.avg_by)
    
    elif args.foldtype == 'cl':
        train_ic50, test_ic50, train_cls, test_cls = get_train_test_ic50(foldtype, [0,1,2], [4], indices, label_matrix)
        pred_df, metric_df = calc_metrics_by_fold(train_ic50, test_ic50, foldtype, label_matrix, test_cls, by=args.avg_by)
        
    result_path = args.outroot + args.pathway + '_' + foldtype 
    pred_df.to_csv(result_path + '_result.csv', index=False)
    metric_df.to_csv(result_path + '_avg_by_' + args.avg_by + '_metrics.csv', index=False)

