# -*- coding: utf-8 -*-

import pandas as pd
import os, sys, argparse
from os.path import dirname, join, abspath

# sys.path.insert(0, abspath(join(dirname(__file__), '..')))
sys.path.append('/nfs/home/students/m.lorenz/Baseline models/')
from Naive_predictor.model import calc_metric, preprocessing
from utils.utils import mkdir, get_train_test_set


# %%
# parse_parameters
def parse_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot",
                        required=True,
                        help="path for input data, contains files with matrix info (IC50,EC50..)")
    parser.add_argument("--outroot",
                        required=True,
                        help="path to save results")
    parser.add_argument("--foldtype",
                        required=True,
                        help="type of validation scheme (LPO, LDO, LCO or all of them)")
    parser.add_argument("--avg_by",
                        required=False,
                        help="how to calculate performance metrics (drug, cl or both), \
                        if not specified, then only the naive predictions are outputted")

    # emmulating cl input above so that I can work from the IDE
    sys.argv = ["main.py", "--dataroot", "../../datasets/cell_viability/CCLE/matrixes_raw/",
                "--outroot", "results_all_tasks/", "--foldtype", "all", "--avg_by", "both"]

    return parser.parse_args()


# %%
if __name__ == '__main__':
    args = parse_parameters()
    # mkdir(args.outroot)
    os.mkdir(args.outroot)

    if args.foldtype == "all":
        tasks = ["LDO", "LCO", "LPO"]
    else:
        tasks = [args.foldtype]

    if args.avg_by == "both":
        averaging = ["drug", "cl"]
    else:
        averaging = [args.avg_by]

    for matrix in os.listdir(args.dataroot):    # loop through all matrix files in the folder

        matrix_folder_path = args.outroot + matrix.split("_")[0] + "/"
        os.mkdir(matrix_folder_path)
        metric = matrix.split("_")[0]

        for task in tasks:
            result_path = matrix_folder_path + task

            # import data
            label_matrix = pd.read_csv(args.dataroot + matrix, header=0, index_col=0)
            label_matrix.reset_index(inplace=True)

            # get training and testing data
            train, test = get_train_test_set(label_matrix, task, 0.8, metric)

            for avg_by in averaging:

                if avg_by is None:
                    sys.stdout = open(result_path + '_avg_by_' + "None" + '_log.txt', 'w')
                else:
                    sys.stdout = open(result_path + '_avg_by_' + avg_by + '_log.txt', 'w')

                # optional preprocessing (has to be in this loop for processing output to be in sys.stdout file)
                train_2, test_2 = preprocessing(train, test, task, metric, remove_out=True, log_transform=True)

                # perform prediction
                pred_df, metric_df = calc_metric(train_2, test_2, task, avg_by, metric)

                # save results
                pred_df.to_csv(result_path + '_avg_by_' + avg_by + '_result.csv', index=False)
                metric_df.to_csv(result_path + '_avg_by_' + avg_by + '_metrics.csv', index=False)

                # sys.stdout.close()  # closes the file were writing to
