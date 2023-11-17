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

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, hyp, metric_matrix,
               param_save_path, hyp_save_path, metric_save_path, description ):
    print('Finished training the model. Saving the model to the path: {}'.format(param_save_path))
    torch.save(model.state_dict(), param_save_path+'/model_weights_' + description + '.pt ')
    
    print('Finished training the model. Saving the model hyp to the path: {}'.format(hyp_save_path))
    x = json.dumps(hyp)
    f = open(hyp_save_path + "/model_hyp_" + description + ".txt","w")
    f.write(x)
    f.close()
    
    print('Finished training the model. Saving metric matrix to the path: {}'.format(metric_save_path))
    np.save(metric_save_path + '/model_train_metrics_' + description, metric_matrix)
    
def load_pretrained_model(model, model_path):
    print('Loading pre-trained model from: {}'.format(model_path))
    model.load_state_dict(torch.load(model_path))

def mkdir(directory):
    directories = directory.split("/")   

    folder = ""
    for d in directories:
        folder += d + '/'
        if not os.path.exists(folder):
            print('creating folder: %s'%folder)
            os.mkdir(folder)

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
    indices_train = indices.loc[indices[fold_type].isin(train_fold)] # df containing all train indices
    
    # --------------------- cell line feature normalization ------------------
    # normalize cl features (fit_transform train data, then use the metrics(mean, std)
    # calculated from train data to fit test data)
    train_cls = indices_train['cl_idx']
    train_x_cl = cl_features[train_cls,:]                   # cl features for all training data
    norm_cl_features = normalize(train_x_cl, cl_features)   # normalized dict of cl features

    
    return norm_cl_features
    
def norm_drug_features(drug_features, indices,
                       fold_type, train_fold):
    indices_train = indices.loc[indices[fold_type].isin(train_fold)] # df containing all train indices
    
    train_drugs = indices_train['drug_idx']
    train_x_drug = drug_features[train_drugs, :]
    norm_drug_features = normalize(train_x_drug, drug_features)
    return norm_drug_features
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
