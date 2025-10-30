import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import pickle
import copy
import pandas as pd
import argparse
import gc
import warnings
from torch.utils.data import Dataset, DataLoader
from hyperopt import fmin, tpe, hp, Trials

# Import graph modules
from units.GraphSNN_GAT import GraphSNN_GAT
from units.getFeatures import save_smiles_dicts, get_smiles_array
from rdkit import Chem

# Import IFM modules
from units.ednn_utils import IFM_DNN_4, EarlyStopping, Meter

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CombinedDataset(Dataset):
    """Dataset that combines graph and fingerprint features"""
    def __init__(self, df, feature_dicts, finger_features):
        self.df = df.reset_index(drop=True)
        self.feature_dicts = feature_dicts
        self.finger_features = finger_features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['cano_smiles']
        y = row['y']
        finger_feat = self.finger_features[idx]
        
        return smiles, finger_feat, y


def collate_combined_fn(batch):
    """Collate function for combined dataset"""
    smiles_list = [item[0] for item in batch]
    finger_feats = np.array([item[1] for item in batch])
    y_vals = np.array([item[2] for item in batch])
    
    return smiles_list, finger_feats, y_vals


class End2EndModel(nn.Module):
    """End-to-end model combining GraphSNN and IFM"""
    def __init__(self, graph_model, ifm_model, graph_output_dim):
        super(End2EndModel, self).__init__()
        self.graph_model = graph_model
        self.ifm_model = ifm_model
        self.graph_output_dim = graph_output_dim
        
    def forward(self, x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, 
                x_full_atom_neighbors, x_full_bond_neighbors, finger_feats):
        """
        Forward pass combining graph and fingerprint features
        """
        # Get graph representation (not prediction)
        _, graph_repr = self.graph_model(x_atom, x_bonds, x_atom_index, x_bond_index, 
                                         x_mask, x_full_atom_neighbors, x_full_bond_neighbors)
        
        # Concatenate graph representation with fingerprint features
        combined_features = torch.cat([graph_repr, finger_feats], dim=1)
        
        # Pass through IFM model
        output = self.ifm_model(combined_features)
        
        return output
    
    def freeze_graph(self):
        """Freeze graph model parameters"""
        for param in self.graph_model.parameters():
            param.requires_grad = False
    
    def unfreeze_graph(self):
        """Unfreeze graph model parameters"""
        for param in self.graph_model.parameters():
            param.requires_grad = True


def process_batch(batch_data, args, feature_dicts):
    """Process batch data and move to device"""
    smiles_list, finger_feats, y_vals = batch_data
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, \
    x_full_atom_neighbors, x_full_bond_neighbors, _ = get_smiles_array(smiles_list, feature_dicts)
    
    return (
        torch.Tensor(x_atom).to(args.device),
        torch.Tensor(x_bonds).to(args.device),
        torch.LongTensor(x_atom_index).to(args.device),
        torch.LongTensor(x_bond_index).to(args.device),
        torch.Tensor(x_mask).to(args.device),
        torch.Tensor(x_full_atom_neighbors).to(args.device),
        torch.Tensor(x_full_bond_neighbors).to(args.device),
        torch.Tensor(finger_feats).to(args.device),
        torch.Tensor(y_vals).view(-1, 1).to(args.device)
    )


def compute_metrics(metric):
    """Compute evaluation metrics"""
    return {k: np.mean(metric.compute_metric(k)) for k in ["rmse", "mae", "r2"]}


def train_epoch(model, data_loader, optimizer, loss_func, args, feature_dicts, train_graph_only=False):
    """Train for one epoch"""
    model.train()
    metric = Meter()
    
    for batch_data in data_loader:
        *graph_inputs, finger_feats, y_vals = process_batch(batch_data, args, feature_dicts)
        
        if train_graph_only:
            # Only train graph model, get prediction directly from graph
            pred, _ = model.graph_model(*graph_inputs[:-1])  # Exclude finger_feats
            outputs = pred
        else:
            outputs = model(*graph_inputs, finger_feats)
        
        loss = loss_func(outputs, y_vals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric.update(outputs.detach().cpu(), y_vals.detach().cpu(), torch.ones_like(y_vals).cpu())
    
    return compute_metrics(metric)


def eval_epoch(model, data_loader, args, feature_dicts, eval_graph_only=False):
    """Evaluate for one epoch"""
    model.eval()
    metric = Meter()
    
    with torch.no_grad():
        for batch_data in data_loader:
            *graph_inputs, finger_feats, y_vals = process_batch(batch_data, args, feature_dicts)
            
            if eval_graph_only:
                # Only evaluate graph model
                pred, _ = model.graph_model(*graph_inputs[:-1])
                outputs = pred
            else:
                outputs = model(*graph_inputs, finger_feats)
            
            metric.update(outputs.cpu(), y_vals.cpu(), torch.ones_like(y_vals).cpu())
    
    return compute_metrics(metric)


def canonicalize_smiles(smiles_list):
    """Canonicalize SMILES and filter valid ones"""
    canonical_list = []
    remained_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                remained_list.append(smiles)
                canonical_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        except:
            pass
    return remained_list, canonical_list


def create_model_components(args, finger_input_dim, hyper_paras):
    """Create graph and IFM models"""
    # Graph model args
    graph_args = argparse.Namespace(
        device=args.device, p_dropout=args.p_dropout,
        fingerprint_dim=args.fingerprint_dim, num_layers=args.num_layers,
        output_units_num=1, atom_feature_dim=args.atom_feature_dim,
        bond_feature_dim=args.bond_feature_dim,
        input_feature_dim=args.input_feature_dim,
        input_bond_dim=args.input_bond_dim
    )
    
    snn_args = argparse.Namespace(
        need_initializer=True, use_bias=True,
        attention_combine='concat', num_attention_heads=1,
        attention_dropout=0
    )
    
    graph_model = GraphSNN_GAT(graph_args, snn_args)
    
    # IFM model
    ifm_model = IFM_DNN_4(
        inputs=args.fingerprint_dim + finger_input_dim,
        hidden_units=[hyper_paras[f"hidden_unit{i}"] for i in range(1, 5)],
        d_out=hyper_paras["d_out"] + 1,
        sigma=hyper_paras["sigma"],
        dp_ratio=hyper_paras["dropout"],
        first_omega_0=hyper_paras["omega0"],
        hidden_omega_0=hyper_paras["omega1"],
        outputs=1, reg=True
    )
    
    return End2EndModel(graph_model, ifm_model, args.fingerprint_dim)


def main():
    parser = argparse.ArgumentParser(description="Two-stage Graph+IFM training")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--p_dropout", type=float, default=0.2)
    parser.add_argument("--fingerprint_dim", type=int, default=250)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--data_label", type=str, default="CHEMBL218_EC50")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--graph_epochs", type=int, default=1000, help="Epochs for stage 1: graph training")
    parser.add_argument("--ifm_epochs", type=int, default=2000, help="Epochs for stage 2: IFM training")
    parser.add_argument("--runseed", type=int, default=66)
    parser.add_argument("--patience", type=int, default=64)
    parser.add_argument("--opt_iters", type=int, default=50)
    parser.add_argument("--use_hyperopt", action='store_true', default=False)
    parser.add_argument("--two_stage", action='store_true', default=True, help="Use two-stage training")
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    
    args.task = args.data_label
    args.metric = "rmse"
    
    # Load data
    print(f"Loading data for task: {args.task}")
    raw_filename = f"./{args.task}.csv"
    feature_filename = f"./{args.task}.pickle"
    finger_feature_path = f"./{args.task}.npy"
    
    total_df = pd.read_csv(raw_filename)
    
    # Canonicalize SMILES
    print("Canonicalizing SMILES...")
    remained_smiles, canonical_smiles_list = canonicalize_smiles(total_df.smiles.values)
    print(f"Successfully processed {len(remained_smiles)}/{len(total_df)} SMILES")
    
    total_df = total_df[total_df["smiles"].isin(remained_smiles)]
    total_df["cano_smiles"] = canonical_smiles_list
    
    # Load or create feature dictionaries
    feature_dicts = (pickle.load(open(feature_filename, "rb")) if os.path.isfile(feature_filename)
                     else save_smiles_dicts(remained_smiles, feature_filename))
    
    # Filter dataframe
    remained_df = total_df[total_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())]
    finger_feature = np.load(finger_feature_path)
    
    # Split data
    train_df_full = remained_df[remained_df["split"] == "train"].reset_index(drop=True)
    test_df = remained_df[remained_df["split"] == "test"].reset_index(drop=True)
    cliff_df = test_df[test_df['cliff_mol'] == 1].reset_index(drop=True)
    
    # Split train into train and validation (90% train, 10% val)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(train_df_full, test_size=0.1, random_state=args.runseed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Get indices for fingerprint features
    train_idx = train_df.index.values
    val_idx = val_df.index.values
    test_idx = remained_df[remained_df["split"] == "test"].index.values
    cliff_idx = remained_df[(remained_df.split == "test") & (remained_df.cliff_mol == 1)].index.values
    
    # Get original indices from train_df_full
    train_original_idx = remained_df[remained_df["split"] == "train"].index.values
    train_finger_full = finger_feature[train_original_idx]
    
    # Map back to get correct fingerprint features
    train_finger = train_finger_full[train_df.index.values]
    val_finger = train_finger_full[val_df.index.values]
    test_finger = finger_feature[test_idx]
    cliff_finger = finger_feature[cliff_idx]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}, Cliff: {len(cliff_df)}")
    
    # Get input dimensions
    x_atom, x_bonds, *_ = get_smiles_array([canonical_smiles_list[0]], feature_dicts)
    args.atom_feature_dim = args.bond_feature_dim = args.input_feature_dim = args.input_bond_dim = x_atom.shape[-1]
    args.bond_feature_dim = args.input_bond_dim = x_bonds.shape[-1]
    finger_input_dim = train_finger.shape[1]
    
    print(f"Graph feature dim: {args.fingerprint_dim}, Finger feature dim: {finger_input_dim}")
    
    # Create datasets and loaders
    datasets = {
        'train': CombinedDataset(train_df, feature_dicts, train_finger),
        'val': CombinedDataset(val_df, feature_dicts, val_finger),
        'test': CombinedDataset(test_df, feature_dicts, test_finger),
        'cliff': CombinedDataset(cliff_df, feature_dicts, cliff_finger)
    }
    
    loaders = {
        k: DataLoader(v, batch_size=args.batch_size, 
                     shuffle=(k=='train'), collate_fn=collate_combined_fn)
        for k, v in datasets.items()
    }
    
    # Define hyperparameter optimization function
    def hyper_opt(hyper_paras):
        model = create_model_components(args, finger_input_dim, hyper_paras).to(args.device)
        loss_func = nn.MSELoss()
        
        file_name = f"../save_model/{args.task}_twostage_{hyper_paras['dropout']:.4f}_" \
                   f"{hyper_paras['hidden_unit1']}_{hyper_paras['hidden_unit2']}_" \
                   f"{hyper_paras['hidden_unit3']}_{hyper_paras['hidden_unit4']}_{hyper_paras['l2']:.4f}.pth"
        
        if args.two_stage:
            # Stage 1: Train graph model only
            print("Stage 1: Training graph model...")
            graph_optimizer = torch.optim.Adam(model.graph_model.parameters(), lr=0.001, weight_decay=hyper_paras["l2"])
            graph_stopper = EarlyStopping(mode="lower", patience=args.patience//2, 
                                         filename=file_name.replace('.pth', '_graph.pth'))
            
            for epoch in range(args.graph_epochs):
                train_scores = train_epoch(model, loaders['train'], graph_optimizer, loss_func, 
                                          args, feature_dicts, train_graph_only=True)
                val_scores = eval_epoch(model, loaders['val'], args, feature_dicts, eval_graph_only=True)
                
                if epoch % 10 == 0:
                    print(f"Graph Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, Val RMSE={val_scores['rmse']:.4f}")
                
                if graph_stopper.step(val_scores[args.metric], model):
                    break
            
            graph_stopper.load_checkpoint(model)
            print(f"Graph training finished at epoch {epoch}")
            
            # Stage 2: Freeze graph and train IFM
            print("Stage 2: Training IFM model (graph frozen)...")
            model.freeze_graph()
            ifm_optimizer = torch.optim.Adam(model.ifm_model.parameters(), lr=0.001, weight_decay=hyper_paras["l2"])
            ifm_stopper = EarlyStopping(mode="lower", patience=args.patience, filename=file_name)
            
            for epoch in range(args.ifm_epochs):
                train_scores = train_epoch(model, loaders['train'], ifm_optimizer, loss_func, 
                                          args, feature_dicts, train_graph_only=False)
                val_scores = eval_epoch(model, loaders['val'], args, feature_dicts, eval_graph_only=False)
                
                if epoch % 10 == 0:
                    print(f"IFM Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, Val RMSE={val_scores['rmse']:.4f}")
                
                if ifm_stopper.step(val_scores[args.metric], model):
                    break
            
            ifm_stopper.load_checkpoint(model)
            print(f"IFM training finished at epoch {epoch}")
            val_scores = eval_epoch(model, loaders['val'], args, feature_dicts)
            
        else:
            # End-to-end training (original approach)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=hyper_paras["l2"])
            stopper = EarlyStopping(mode="lower", patience=args.patience, filename=file_name)
            
            for epoch in range(args.epochs):
                train_scores = train_epoch(model, loaders['train'], optimizer, loss_func, args, feature_dicts)
                val_scores = eval_epoch(model, loaders['val'], args, feature_dicts)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, Val RMSE={val_scores['rmse']:.4f}")
                
                if stopper.step(val_scores[args.metric], model):
                    break
            
            stopper.load_checkpoint(model)
            val_scores = eval_epoch(model, loaders['val'], args, feature_dicts)
        
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        
        return val_scores[args.metric]
    
    # Hyperparameter space
    hyper_space = {
        "l2": hp.uniform("l2", 0, 0.01),
        "dropout": hp.uniform("dropout", 0, 0.5),
        "d_out": hp.randint("d_out", 127),
        "omega1": hp.uniform("omega1", 0.001, 1),
        "omega0": hp.uniform("omega0", 0.001, 1),
        "sigma": hp.loguniform("sigma", np.log(0.01), np.log(100)),
        **{f"hidden_unit{i}": hp.choice(f"hidden_unit{i}", [64, 128, 256, 512, 1024]) 
           for i in range(1, 5)}
    }
    
    # Default hyperparameters
    default_paras = {
        "l2": 0.0001, "dropout": 0.2, "d_out": 64,
        "omega1": 0.5, "omega0": 0.5, "sigma": 1.0,
        **{f"hidden_unit{i}": size for i, size in enumerate([256, 256, 128, 128], 1)}
    }
    
    # Hyperparameter optimization or direct training
    if args.use_hyperopt:
        print("Starting hyperparameter optimization...")
        trials = Trials()
        opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, 
                      max_evals=args.opt_iters, trials=trials)
        
        print(f"Best hyperparameters: {opt_res}")
        
        hidden_unit_ls = [64, 128, 256, 512, 1024]
        best_paras = {
            **{k: v for k, v in opt_res.items() if not k.startswith('hidden_unit')},
            **{f"hidden_unit{i}": hidden_unit_ls[opt_res[f"hidden_unit{i}"]] for i in range(1, 5)}
        }
    else:
        print("Using default hyperparameters...")
        best_paras = default_paras
    
    # Train final model
    print("\n" + "="*70)
    print("Training final model with best hyperparameters...")
    print("="*70)
    final_model = create_model_components(args, finger_input_dim, best_paras).to(args.device)
    loss_func = nn.MSELoss()
    
    if args.two_stage:
        # Two-stage training
        print("\n### STAGE 1: Training Graph Model ###")
        best_graph_file = f"../save_model/{args.task}_twostage_graph_best.pth"
        best_file = f"../save_model/{args.task}_twostage_best.pth"
        
        # Stage 1: Train graph model
        graph_optimizer = torch.optim.Adam(final_model.graph_model.parameters(), 
                                          lr=0.001, weight_decay=best_paras["l2"])
        graph_stopper = EarlyStopping(mode="lower", patience=args.patience//2, filename=best_graph_file)
        
        for epoch in range(args.graph_epochs):
            train_scores = train_epoch(final_model, loaders['train'], graph_optimizer, loss_func, 
                                      args, feature_dicts, train_graph_only=True)
            val_scores = eval_epoch(final_model, loaders['val'], args, feature_dicts, eval_graph_only=True)
            
            if epoch % 10 == 0:
                print(f"Graph Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, MAE={train_scores['mae']:.4f}, "
                      f"Val RMSE={val_scores['rmse']:.4f}, MAE={val_scores['mae']:.4f}")
            
            if graph_stopper.step(val_scores[args.metric], final_model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        graph_stopper.load_checkpoint(final_model)
        graph_val_scores = eval_epoch(final_model, loaders['val'], args, feature_dicts, eval_graph_only=True)
        print(f"\nGraph model best Val RMSE: {graph_val_scores['rmse']:.4f}")
        
        # Stage 2: Freeze graph and train IFM
        print("\n### STAGE 2: Training IFM Model (Graph Frozen) ###")
        final_model.freeze_graph()
        ifm_optimizer = torch.optim.Adam(final_model.ifm_model.parameters(), 
                                        lr=0.001, weight_decay=best_paras["l2"])
        ifm_stopper = EarlyStopping(mode="lower", patience=args.patience, filename=best_file)
        
        for epoch in range(args.ifm_epochs):
            train_scores = train_epoch(final_model, loaders['train'], ifm_optimizer, loss_func, 
                                      args, feature_dicts, train_graph_only=False)
            val_scores = eval_epoch(final_model, loaders['val'], args, feature_dicts, eval_graph_only=False)
            
            if epoch % 10 == 0:
                print(f"IFM Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, MAE={train_scores['mae']:.4f}, "
                      f"Val RMSE={val_scores['rmse']:.4f}, MAE={val_scores['mae']:.4f}")
            
            if ifm_stopper.step(val_scores[args.metric], final_model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        ifm_stopper.load_checkpoint(final_model)
        
    else:
        # End-to-end training (original approach)
        print("\n### End-to-End Training ###")
        best_file = f"../save_model/{args.task}_end2end_best.pth"
        optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=best_paras["l2"])
        stopper = EarlyStopping(mode="lower", patience=args.patience, filename=best_file)
        
        for epoch in range(args.epochs):
            train_scores = train_epoch(final_model, loaders['train'], optimizer, loss_func, args, feature_dicts)
            val_scores = eval_epoch(final_model, loaders['val'], args, feature_dicts)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train RMSE={train_scores['rmse']:.4f}, MAE={train_scores['mae']:.4f}, "
                      f"Val RMSE={val_scores['rmse']:.4f}, MAE={val_scores['mae']:.4f}")
            
            if stopper.step(val_scores[args.metric], final_model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        stopper.load_checkpoint(final_model)
    
    # Final evaluation on test and cliff sets
    stopper.load_checkpoint(final_model)
    test_scores = eval_epoch(final_model, loaders['test'], args, feature_dicts)
    cliff_scores = eval_epoch(final_model, loaders['cliff'], args, feature_dicts)
    
    print("\n" + "="*50)
    print("Final Results:")
    print(f"Test Set - RMSE: {test_scores['rmse']:.4f}, MAE: {test_scores['mae']:.4f}, R2: {test_scores['r2']:.4f}")
    print(f"Cliff Set - RMSE: {cliff_scores['rmse']:.4f}, MAE: {cliff_scores['mae']:.4f}, R2: {cliff_scores['r2']:.4f}")
    print("="*50)
    
    return best_file, test_scores, cliff_scores


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
