# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:59:49 2022

@author: jessi
"""

#%% Import Libraries
import torch
from torch.utils.data import Dataset
import numpy as np
    
#%% Five_Layer_MLP
class FiveLayerMLPDataset(Dataset):
    def __init__(self, cl_features, drug_features, indices, label_matrix):
        """
        cl_features: 
            DataFrame with size (n_cls, n_cl_features) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 512-bits of Morgan Fingerprints OR 446 drug targets
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_features = torch.Tensor(cl_features)
        self.drug_features = torch.Tensor(drug_features)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_cl_features) corresponding to cell line
            2. FloatTensor with size (n, n_drug_features) corresponding to drug
            3. FloatTensor with size (n, 1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
            
        sample = (self.cl_features[cl], self.drug_features[drug], self.label_matrix[drug, cl])

        return sample

#%% PathDNN
class PathDNNDataset(Dataset):
    def __init__(self, cl_features, drug_features, indices, label_matrix):
        """
        cl_features: 
            DataFrame with size (n_cls, n_cl_features) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 512-bits of Morgan Fingerprints OR 446 drug targets
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_features = torch.Tensor(cl_features)
        self.drug_features = torch.Tensor(drug_features)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_cl_features) corresponding to cell line
            2. FloatTensor with size (n, n_drug_features) corresponding to drug
            3. FloatTensor with size (n, 1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
            
        sample = (self.cl_features[cl], self.drug_features[drug], self.label_matrix[drug, cl])

        return sample
    
    
#%% ConsDeepSignaling
class ConsDeepSignalingDataset(Dataset):
    def __init__(self, cl_exp, cl_cnv, drug_features, indices, label_matrix):
        """
        cl_exp: 
            DataFrame with size (n_cls, n_cl_exp_genes) ---> 959 cls, 5511 genes
        
        cl_cnv:
            DataFrame with size (n_cls, n_cl_cnv_genes) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 446 drug targets, size:(162, 5511)
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_exp = torch.Tensor(cl_exp)
        self.cl_cnv = torch.Tensor(cl_cnv)
        self.drug_features = torch.Tensor(drug_features)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]
        self.num_genes = self.cl_exp.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 2 elements:
            1. FloatTensor with size (n_genes * 3) corresponding to cl_exp, cl_cnv, drug_target features
            2. FloatTensor with size (1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
        
        cl_exp_features = self.cl_exp[cl]
        cl_cnv_features = self.cl_cnv[cl]
        drug_target_features = self.drug_features[drug]
        
        features = torch.zeros([self.num_genes * 3])
        
        features[0::3] = cl_exp_features
        features[1::3] = cl_cnv_features
        features[2::3] = drug_target_features
        
        sample = (features, self.label_matrix[drug,cl])

        return sample

#%% ConsDeepSignaling_exp_target (no cnv input)
class ConsDeepSignalingNoCNVDataset(Dataset):
    def __init__(self, cl_exp, drug_features, indices, label_matrix):
        """
        cl_exp: 
            DataFrame with size (n_cls, n_cl_exp_genes) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 446 drug targets, size:(162, 5511)
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_exp = torch.Tensor(cl_exp)
        self.drug_features = torch.Tensor(drug_features)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]
        self.num_genes = self.cl_exp.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 2 elements:
            1. FloatTensor with size (n_genes * 2) corresponding to cl_exp, drug_target features
            2. FloatTensor with size (1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
        
        cl_exp_features = self.cl_exp[cl]
        drug_target_features = self.drug_features[drug]
        
        features = torch.zeros([self.num_genes * 2])
        
        features[0::2] = cl_exp_features
        features[1::2] = drug_target_features
        
        sample = (features, self.label_matrix[drug,cl])

        return sample
    
#%% HiDRA
class HiDRADataset(Dataset):
    def __init__(self, cl_features, drug_features, indices, label_matrix):
        """
        cl_features: 
            DataFrame with size (n_cls, n_cl_features) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 512-bits of Morgan Fingerprints OR 446 drug targets
        
        pathway_indices:
            DataFrame with size (n_pathways, n_cl_features) ---> 332 pathways, 5511 genes
            Used to determine the member genes of each pathway
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_features = torch.Tensor(cl_features)
        self.drug_features = torch.Tensor(drug_features)
        # self.pathway_indices = torch.Tensor(pathway_indices)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_cl_features) corresponding to cell line
            2. FloatTensor with size (n, n_drug_features) corresponding to drug
            3. FloatTensor with size (n, 1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
        
        # cl_gene_exp = self.cl_features[cl]
        # member_genes = []
        # for i in range(self.pathway_indices.shape[0]):
        #     self.pathway_indices[i] * cl_gene_exp

        
        sample = (self.cl_features[cl], self.drug_features[drug], self.label_matrix[drug, cl])

        return sample
    
    
#%% PathDSP
class PathDSPDataset(Dataset):
    def __init__(self, cl_exp, cl_mut, cl_cnv, drug_fp, drug_target, indices, label_matrix):
        """
        cl_features: 
            DataFrame with size (n_cls, n_cl_features) ---> 959 cls, 5511 genes
                     
        drug_features: 
            DataFrame with size (n_drugs, n_drug_features) ---> 162 drugs, 512-bits of Morgan Fingerprints OR 446 drug targets
            
        indices: 
            DataFrame the contains cell line and drug indices (indices are used to extract corresponding ln_ic50 values from label_matrix)
            
        label_matrix: 
            NumPy array with size (n_drugs, n_cls), contains ln_ic50 values normalized drug-wise
        """
        order = ['cl_idx', 'drug_idx']
        self.tuples = indices[order].values
        self.cl_exp = torch.Tensor(cl_exp)
        self.cl_mut = torch.Tensor(cl_mut)
        self.cl_cnv = torch.Tensor(cl_cnv)
        self.drug_fp = torch.Tensor(drug_fp)
        self.drug_target = torch.Tensor(drug_target)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = self.label_matrix.shape[0]
        self.num_cls = self.label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

    def __len__(self):
        """
        returns the number of labeled combinations
        """
        return len(self.tuples)

    def __getitem__(self, idx):
        """
        returns a tuple with 3 elements:
            1. FloatTensor with size (n, n_cl_features) corresponding to cell line
            2. FloatTensor with size (n, n_drug_features) corresponding to drug
            3. FloatTensor with size (n, 1) corresponding to the ln_ic50 
        """
        if torch.is_tensor(idx):
            idx.tolist()

        tuples = self.tuples[idx]
    
        if len(tuples.shape) == 1:
            cl = tuples[0]
            drug = tuples[1]
        else:
            cl = tuples[:,0]
            drug = tuples[:,1]
        
        sample = (self.cl_exp[cl], self.cl_mut[cl], self.cl_cnv[cl], # cl features
                  self.drug_fp[drug], self.drug_target[drug],        # drug features
                  self.label_matrix[drug, cl])

        return sample   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    