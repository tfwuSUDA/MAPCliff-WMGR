import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import glob
import time
import numpy as np
import random
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
import argparse
from unit import  save_smiles_dicts, get_smiles_array, mGraphSNN
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
from sklearn.metrics import roc_auc_score, mean_squared_error
sns.set()
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

fold = 2

def train(model, dataset, optimizer, loss_function, args, feature_dicts):
    model.train()
    valList = np.arange(0, dataset.shape[0])
    np.random.shuffle(valList) 
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i+args.batch_size]
        batch_list.append(batch)
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch, :]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df["lipo"].values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_full_atom_neighbors, x_full_bond_neighbors, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        x_atom = torch.Tensor(x_atom).to(args.device)
        x_bonds = torch.Tensor(x_bonds).to(args.device)
        x_atom_index = torch.LongTensor(x_atom_index).to(args.device)
        x_bond_index = torch.LongTensor(x_bond_index).to(args.device)
        x_mask = torch.Tensor(x_mask).to(args.device)
        x_full_atom_neighbors = torch.Tensor(x_full_atom_neighbors).to(args.device)
        x_full_bond_neighbors = torch.Tensor(x_full_bond_neighbors).to(args.device)
        y_val_tensor = torch.Tensor(y_val).view(-1,1).to(args.device)
        mol_prediction= model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_full_atom_neighbors, x_full_bond_neighbors)
        model.zero_grad()
        loss = loss_function(mol_prediction, y_val_tensor)
        loss.backward()
        optimizer.step()

def eval(model, dataset, args, feature_dicts):
    model.eval()
    eval_predictions = []
    eval_labels = []
    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], args.batch_size):
        batch = valList[i:i+args.batch_size]
        batch_list.append(batch) 
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch, :]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df["lipo"].values
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_full_atom_neighbors, x_full_bond_neighbors, smiles_to_rdkit_list = get_smiles_array(smiles_list, feature_dicts)
        x_atom = torch.Tensor(x_atom).to(args.device)
        x_bonds = torch.Tensor(x_bonds).to(args.device)
        x_atom_index = torch.LongTensor(x_atom_index).to(args.device)
        x_bond_index = torch.LongTensor(x_bond_index).to(args.device)
        x_mask = torch.Tensor(x_mask).to(args.device)
        x_full_atom_neighbors = torch.Tensor(x_full_atom_neighbors).to(args.device)
        x_full_bond_neighbors = torch.Tensor(x_full_bond_neighbors).to(args.device)
        y_val = torch.Tensor(y_val).to(args.device) 
        mol_prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_full_atom_neighbors, x_full_bond_neighbors)
        mol_prediction = mol_prediction.squeeze().detach().cpu().numpy()
        y_val = y_val.squeeze().cpu().numpy()
        eval_predictions.extend(mol_prediction)
        eval_labels.extend(y_val)
    try:
        rmse_score = np.sqrt(mean_squared_error(eval_labels, eval_predictions))
    except ValueError as e:
        print(f"计算RMSE时出错: {e}")
        rmse_score = None
    return rmse_score

def main_process(task_name, seed = 1):
    tasks = ["lipo"]
    raw_filename = f"../dataset/MoleculeNet/{task_name}/{task_name}.csv"
    feature_filename = f"../dataset/MoleculeNet/pickle_file/{task_name}.pickle"
    split_fold_path = f"../dataset/MoleculeNet/{task_name}/splits/scaffold-{fold}.npy"
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    split_fold_npy = np.load(split_fold_path, allow_pickle=True)
    split_test = split_fold_npy[2]
    split_val = split_fold_npy[1]
    split_train = split_fold_npy[0]
    smiles_tasks_df[split_test,'split'] = 'test'
    smiles_tasks_df.loc[split_train, 'split'] = 'train'
    smiles_tasks_df.loc[split_val, 'split'] = 'val'
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
            )
        except:
            print(smiles)
            pass
    print("number of successfully processed smiles: ", len(remained_smiles))
    
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    smiles_tasks_df["cano_smiles"] = canonical_smiles_list
    torch.manual_seed(seed=seed)
    np.random.seed(seed=seed)
    parser = argparse.ArgumentParser(description="PyTorch implementation")
    args = parser.parse_args()
    snn_args = parser.parse_args()
    args.device = (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    args.batch_size = 256
    args.epochs = 800
    args.p_dropout = 0.2
    args.fingerprint_dim = 250
    args.weight_decay = 4.5  # also known as l2_regularization_lambda
    args.learning_rate = 2.5
    args.num_layers = 5
    args.loss = "mse"
    per_task_output_units_num = 1  # for regression model
    args.output_units_num = len(tasks) * per_task_output_units_num
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
    else:
        feature_dicts = save_smiles_dicts(smilesList, feature_filename)
    remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())]
    uncovered_df = smiles_tasks_df.drop(remained_df.index)
    train_df = remained_df[remained_df['split'] == 'train'].reset_index(drop=True)
    val_df = remained_df[remained_df['split'] == 'val'].reset_index(drop=True)
    test_df = remained_df[remained_df['split'] == 'test'].reset_index(drop=True)
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_full_atom_neighbors, x_full_bond_neighbors, smiles_to_rdkit_list = get_smiles_array([canonical_smiles_list[0]],feature_dicts)
    args.atom_feature_dim = x_atom.shape[-1]
    args.bond_feature_dim = x_bonds.shape[-1]
    args.input_feature_dim = x_atom.shape[-1]
    args.input_bond_dim = x_bonds.shape[-1]
    snn_args.need_initializer = True
    snn_args.use_bias = True
    snn_args.attention_combine = 'concat'
    snn_args.num_attention_heads = 1
    snn_args.attention_dropout = 0.1
    if args.loss == "mse":
        loss_function = nn.MSELoss()
    elif args.loss == "mul_class":
        loss_function = nn.BCEWithLogitsLoss()
    model = mGraphSNN(args, snn_args)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), 10**-args.learning_rate, weight_decay=10**-args.weight_decay)
    best_param ={}
    best_param["train_epoch"] = 0
    best_param["valid_epoch"] = 0
    best_param["train_rmse"] = 999
    best_param["valid_rmse"] = 999
    model_save_path = f'./saved_models/MoleculeNet/{task_name}/{args.num_layers}_{fold}'
    os.makedirs(model_save_path, exist_ok=True)
    for epoch in range(8000):
        train_rmse = eval(model, train_df, args, feature_dicts)
        valid_rmse = eval(model, val_df, args, feature_dicts)
        if train_rmse < best_param["train_rmse"]:
            best_param["train_epoch"] = epoch
            best_param["train_rmse"] = train_rmse
        if valid_rmse < best_param["valid_rmse"]:
            best_param["valid_epoch"] = epoch
            best_param["valid_rmse"] = valid_rmse
            if valid_rmse < 2:
                torch.save(model, f'./saved_models/MoleculeNet/{task_name}/{args.num_layers}_{fold}/{str(seed)}.pt')
        if (epoch - best_param["train_epoch"] > 20) and (epoch - best_param["valid_epoch"] > 40):        
            break
        print(f"epoch: {epoch}, RMSE:{valid_rmse}")
        train(model, train_df, optimizer, loss_function, args, feature_dicts)
    test_rmse = eval(model, test_df, args, feature_dicts)
    print(f"seed: {seed}, test_RMSE:{test_rmse}")
    return test_rmse

if __name__ == "__main__":
    task_name = "lipo"
    main_process(task_name=task_name)