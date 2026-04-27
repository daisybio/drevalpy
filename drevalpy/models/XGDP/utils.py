import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
import time
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd

def rmse(y, f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f)**2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def coeffi_determ(y, f):
    r2 = r2_score(y, f)
    return r2

def predicting(model, device, loader, return_attention_weights = False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # output, _ = model(data)
            x, x_cell_mut, edge_index, batch_drug, edge_feat = data.x, data.target, data.edge_index.long(), data.batch, data.edge_features
            if return_attention_weights:
                # output, _, attn_weights = model(x, edge_index, x_cell_mut, batch_drug, edge_feat, return_attention_weights)
                output, attn_weights = model(x, edge_index, batch_drug, x_cell_mut, edge_feat, return_attention_weights)
                attn_weights = [attn_weight.cpu().numpy() for attn_weight in attn_weights]
                # print(attn_weights)
                attn_weights = np.array(attn_weights)
                # print(attn_weights.shape)
            else: 
                # output, _ = model(x, edge_index, x_cell_mut, batch_drug, edge_feat)
                output = model(x, edge_index, batch_drug, x_cell_mut, edge_feat)
        
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    torch.cuda.empty_cache()  ## no grad
    if return_attention_weights:
        return total_labels.numpy().flatten(), total_preds.numpy().flatten(), attn_weights
    else:
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()
    
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval, return_attention_weights=False):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        x, x_cell_mut, edge_index, batch_drug, edge_feat = data.x, data.target, data.edge_index.long(), data.batch, data.edge_features
        
        output = model(x, edge_index, batch_drug, x_cell_mut, edge_feat)
        
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
    return sum(avg_loss)/len(avg_loss)
