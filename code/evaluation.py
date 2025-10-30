import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import argparse
import warnings
from torch.utils.data import Dataset, DataLoader

# Import graph modules
from units.GraphSNN_GAT import GraphSNN_GAT
from units.getFeatures import save_smiles_dicts, get_smiles_array
from rdkit import Chem

# Import IFM modules
from units.ednn_utils import IFM_DNN_4, Meter

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


def eval_epoch(model, data_loader, args, feature_dicts):
    """Evaluate for one epoch and return predictions"""
    model.eval()
    metric = Meter()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            *graph_inputs, finger_feats, y_vals = process_batch(batch_data, args, feature_dicts)
            outputs = model(*graph_inputs, finger_feats)
            
            metric.update(outputs.cpu(), y_vals.cpu(), torch.ones_like(y_vals).cpu())
            all_predictions.extend(outputs.cpu().numpy().flatten().tolist())
            all_targets.extend(y_vals.cpu().numpy().flatten().tolist())
    
    return compute_metrics(metric), all_predictions, all_targets


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
    parser = argparse.ArgumentParser(description="Evaluation for Graph+IFM model")
    parser.add_argument("--device", type=int, default=1, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--p_dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--fingerprint_dim", type=int, default=250, help="Graph fingerprint dimension")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of graph layers")
    parser.add_argument("--data_label", type=str, default="CHEMBL218_EC50", help="Dataset label")
    parser.add_argument("--graph_model_path", type=str, required=True, help="Path to trained GraphSNN model file")
    parser.add_argument("--ifm_model_path", type=str, required=True, help="Path to trained IFM model file")
    parser.add_argument("--output_dir", type=str, default="../evaluation_results", help="Directory to save results")
    parser.add_argument("--eval_sets", type=str, nargs='+', default=['test', 'cliff'], 
                       help="Which sets to evaluate: test, cliff, or both")
    args = parser.parse_args()
    
    # Setup
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.task = args.data_label
    args.metric = "rmse"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data for task: {args.task}")
    raw_filename = f"./{args.task}.csv"
    feature_filename = f"./{args.task}.pickle"
    finger_feature_path = f"./{args.task}.npy"
    
    # Check if files exist
    if not os.path.exists(raw_filename):
        raise FileNotFoundError(f"Data file not found: {raw_filename}")
    if not os.path.exists(feature_filename):
        raise FileNotFoundError(f"Feature file not found: {feature_filename}")
    if not os.path.exists(finger_feature_path):
        raise FileNotFoundError(f"Fingerprint file not found: {finger_feature_path}")
    if not os.path.exists(args.graph_model_path):
        raise FileNotFoundError(f"Graph model file not found: {args.graph_model_path}")
    if not os.path.exists(args.ifm_model_path):
        raise FileNotFoundError(f"IFM model file not found: {args.ifm_model_path}")
    
    total_df = pd.read_csv(raw_filename)
    
    # Canonicalize SMILES
    print("Canonicalizing SMILES...")
    remained_smiles, canonical_smiles_list = canonicalize_smiles(total_df.smiles.values)
    print(f"Successfully processed {len(remained_smiles)}/{len(total_df)} SMILES")
    
    total_df = total_df[total_df["smiles"].isin(remained_smiles)]
    total_df["cano_smiles"] = canonical_smiles_list
    
    # Load feature dictionaries
    feature_dicts = pickle.load(open(feature_filename, "rb"))
    
    # Filter dataframe
    remained_df = total_df[total_df["cano_smiles"].isin(feature_dicts["smiles_to_atom_mask"].keys())]
    finger_feature = np.load(finger_feature_path)
    
    # Get test and cliff datasets
    test_df = remained_df[remained_df["split"] == "test"].reset_index(drop=True)
    cliff_df = test_df[test_df['cliff_mol'] == 1].reset_index(drop=True)
    
    # Get indices for fingerprint features
    test_idx = remained_df[remained_df["split"] == "test"].index.values
    cliff_idx = remained_df[(remained_df.split == "test") & (remained_df.cliff_mol == 1)].index.values
    
    test_finger = finger_feature[test_idx]
    cliff_finger = finger_feature[cliff_idx]
    
    print(f"Test set size: {len(test_df)}")
    print(f"Cliff set size: {len(cliff_df)}")
    
    # Get input dimensions
    x_atom, x_bonds, *_ = get_smiles_array([canonical_smiles_list[0]], feature_dicts)
    args.atom_feature_dim = args.bond_feature_dim = args.input_feature_dim = args.input_bond_dim = x_atom.shape[-1]
    args.bond_feature_dim = args.input_bond_dim = x_bonds.shape[-1]
    finger_input_dim = test_finger.shape[1]
    
    print(f"Graph feature dim: {args.fingerprint_dim}, Finger feature dim: {finger_input_dim}")
    
    # Default hyperparameters (should match training)
    default_paras = {
        "l2": 0.0001, "dropout": 0.2, "d_out": 64,
        "omega1": 0.5, "omega0": 0.5, "sigma": 1.0,
        **{f"hidden_unit{i}": size for i, size in enumerate([256, 256, 128, 128], 1)}
    }
    
    # Create model
    print("\nCreating model architecture...")
    model = create_model_components(args, finger_input_dim, default_paras).to(args.device)
    
    # Load trained models separately
    print(f"Loading GraphSNN model from: {args.graph_model_path}")
    graph_checkpoint = torch.load(args.graph_model_path, map_location=args.device)
    model.graph_model.load_state_dict(graph_checkpoint)
    print("GraphSNN model loaded successfully!")
    
    print(f"Loading IFM model from: {args.ifm_model_path}")
    ifm_checkpoint = torch.load(args.ifm_model_path, map_location=args.device)
    model.ifm_model.load_state_dict(ifm_checkpoint)
    print("IFM model loaded successfully!")
    
    model.eval()
    print("Both models loaded and set to evaluation mode!")
    
    # Create datasets and loaders
    datasets = {}
    loaders = {}
    
    if 'test' in args.eval_sets:
        datasets['test'] = CombinedDataset(test_df, feature_dicts, test_finger)
        loaders['test'] = DataLoader(datasets['test'], batch_size=args.batch_size, 
                                     shuffle=False, collate_fn=collate_combined_fn)
    
    if 'cliff' in args.eval_sets:
        datasets['cliff'] = CombinedDataset(cliff_df, feature_dicts, cliff_finger)
        loaders['cliff'] = DataLoader(datasets['cliff'], batch_size=args.batch_size, 
                                      shuffle=False, collate_fn=collate_combined_fn)
    
    # Evaluate on each requested set
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    results_summary = {}
    
    for eval_set in args.eval_sets:
        print(f"\nEvaluating on {eval_set.upper()} set...")
        scores, predictions, targets = eval_epoch(model, loaders[eval_set], args, feature_dicts)
        
        print(f"{eval_set.upper()} Set Results:")
        print(f"  RMSE: {scores['rmse']:.4f}")
        print(f"  MAE:  {scores['mae']:.4f}")
        print(f"  R2:   {scores['r2']:.4f}")
        
        # Store results
        results_summary[eval_set] = scores
        
        # Save predictions to CSV
        if eval_set == 'test':
            result_df = test_df.copy()
        else:
            result_df = cliff_df.copy()
        
        result_df['predictions'] = predictions
        result_df['targets'] = targets
        result_df['absolute_error'] = np.abs(np.array(predictions) - np.array(targets))
        
        output_file = os.path.join(args.output_dir, f"{args.task}_{eval_set}_predictions.csv")
        result_df.to_csv(output_file, index=False)
        print(f"  Predictions saved to: {output_file}")
    
    # Save summary results
    summary_df = pd.DataFrame(results_summary).T
    summary_file = os.path.join(args.output_dir, f"{args.task}_evaluation_summary.csv")
    summary_df.to_csv(summary_file)
    print(f"\nSummary results saved to: {summary_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)
    
    return results_summary


if __name__ == "__main__":
    main()
