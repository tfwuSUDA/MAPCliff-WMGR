import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import argparse
import warnings

# Import graph modules
from units.GraphSNN_GAT import GraphSNN_GAT
from units.getFeatures import save_smiles_dicts, get_smiles_array
from rdkit import Chem

# Import IFM modules
from units.ednn_utils import IFM_DNN_4

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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


def canonicalize_smiles(smiles):
    """Canonicalize a single SMILES string"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    except Exception as e:
        raise ValueError(f"Error processing SMILES {smiles}: {str(e)}")


def load_fingerprints_from_csv(fingerprint_csv):
    """Load PaDEL fingerprints from CSV file"""
    fingerprints_df = pd.read_csv(fingerprint_csv)
    if 'Name' in fingerprints_df.columns:
        fingerprints_df = fingerprints_df.drop('Name', axis=1)
    fingerprints = fingerprints_df.select_dtypes(include=[np.number]).values
    return fingerprints


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
    
    return graph_model, ifm_model


def predict_from_smiles(smiles_list, model, feature_dicts, fingerprints, args):
    """
    Predict activity values from SMILES strings
    
    Args:
        smiles_list: List of canonical SMILES strings
        model: Trained End2EndModel
        feature_dicts: Dictionary for graph feature extraction
        fingerprints: Numpy array of PaDEL fingerprints
        args: Arguments containing device info
    
    Returns:
        List of predicted values
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for idx, smiles in enumerate(smiles_list):
            # Get graph features
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, \
            x_full_atom_neighbors, x_full_bond_neighbors, _ = get_smiles_array([smiles], feature_dicts)
            
            # Prepare inputs
            x_atom = torch.Tensor(x_atom).to(args.device)
            x_bonds = torch.Tensor(x_bonds).to(args.device)
            x_atom_index = torch.LongTensor(x_atom_index).to(args.device)
            x_bond_index = torch.LongTensor(x_bond_index).to(args.device)
            x_mask = torch.Tensor(x_mask).to(args.device)
            x_full_atom_neighbors = torch.Tensor(x_full_atom_neighbors).to(args.device)
            x_full_bond_neighbors = torch.Tensor(x_full_bond_neighbors).to(args.device)
            finger_feats = torch.Tensor(fingerprints[idx:idx+1]).to(args.device)
            
            # Predict
            output = model(x_atom, x_bonds, x_atom_index, x_bond_index, 
                          x_mask, x_full_atom_neighbors, x_full_bond_neighbors, finger_feats)
            
            predictions.append(output.cpu().item())
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference pipeline for MAPCliff-WMGR")
    
    # Required arguments
    parser.add_argument("--smiles_file", type=str, required=True,
                       help="CSV file containing SMILES strings")
    parser.add_argument("--fingerprint_file", type=str, required=True,
                       help="CSV file with PaDEL fingerprints (use: java -jar PaDEL-Descriptor.jar -2d -fingerprints -dir <dir> -file <output>)")
    parser.add_argument("--graph_model_path", type=str, required=True,
                       help="Path to trained graph model (.pth)")
    parser.add_argument("--ifm_model_path", type=str, required=True,
                       help="Path to trained IFM model (.pth)")
    
    # Optional arguments
    parser.add_argument("--smiles_column", type=str, default="smiles",
                       help="Column name for SMILES (default: smiles)")
    parser.add_argument("--output_file", type=str, default="predictions.csv",
                       help="Output CSV file (default: predictions.csv)")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID, -1 for CPU (default: 0)")
    parser.add_argument("--p_dropout", type=float, default=0.2)
    parser.add_argument("--fingerprint_dim", type=int, default=250)
    parser.add_argument("--num_layers", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup device
    if args.device >= 0 and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.device}")
    else:
        args.device = torch.device("cpu")
    
    print("MAPCliff-WMGR Inference")
    print(f"Device: {args.device}")
    
    # Load SMILES from CSV
    print("Loading SMILES...")
    # Load SMILES from CSV
    print("Loading SMILES...")
    
    smiles_df = pd.read_csv(args.smiles_file)
    if args.smiles_column not in smiles_df.columns:
        raise ValueError(f"Column '{args.smiles_column}' not found")
    
    smiles_list = smiles_df[args.smiles_column].tolist()
    
    # Canonicalize SMILES
    canonical_smiles = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            canonical = canonicalize_smiles(smiles)
            canonical_smiles.append(canonical)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Warning: Invalid SMILES at row {idx}: {smiles}")
    
    print(f"Valid SMILES: {len(canonical_smiles)}/{len(smiles_list)}")
    
    # Load PaDEL fingerprints
    print("Loading fingerprints...")
    fingerprints = load_fingerprints_from_csv(args.fingerprint_file)
    fingerprints = fingerprints[valid_indices]
    
    # Build graph features
    print("Building graph features...")
    feature_dicts = save_smiles_dicts(canonical_smiles, "temp_features.pickle")
    
    # Build graph features
    print("Building graph features...")
    feature_dicts = save_smiles_dicts(canonical_smiles, "temp_features.pickle")
    
    # Get dimensions
    x_atom, x_bonds, *_ = get_smiles_array([canonical_smiles[0]], feature_dicts)
    args.atom_feature_dim = args.bond_feature_dim = args.input_feature_dim = x_atom.shape[-1]
    args.bond_feature_dim = args.input_bond_dim = x_bonds.shape[-1]
    finger_input_dim = fingerprints.shape[1]
    
    # Load models
    print("Loading models...")
    default_paras = {
        "l2": 0.0001, "dropout": 0.2, "d_out": 64,
        "omega1": 0.5, "omega0": 0.5, "sigma": 1.0,
        **{f"hidden_unit{i}": size for i, size in enumerate([256, 256, 128, 128], 1)}
    }
    
    graph_model, ifm_model = create_model_components(args, finger_input_dim, default_paras)
    
    # Load graph model
    graph_checkpoint = torch.load(args.graph_model_path, map_location=args.device)
    if isinstance(graph_checkpoint, dict) and 'graph_model' in graph_checkpoint:
        graph_model.load_state_dict(graph_checkpoint['graph_model'])
    else:
        graph_state_dict = {}
        for key, value in graph_checkpoint.items():
            if key.startswith('graph_model.'):
                graph_state_dict[key.replace('graph_model.', '')] = value
        if graph_state_dict:
            graph_model.load_state_dict(graph_state_dict)
        else:
            graph_model.load_state_dict(graph_checkpoint)
    graph_model = graph_model.to(args.device).eval()
    
    # Load IFM model
    ifm_checkpoint = torch.load(args.ifm_model_path, map_location=args.device)
    if isinstance(ifm_checkpoint, dict) and 'ifm_model' in ifm_checkpoint:
        ifm_model.load_state_dict(ifm_checkpoint['ifm_model'])
    else:
        ifm_state_dict = {}
        for key, value in ifm_checkpoint.items():
            if key.startswith('ifm_model.'):
                ifm_state_dict[key.replace('ifm_model.', '')] = value
        if ifm_state_dict:
            ifm_model.load_state_dict(ifm_state_dict)
        else:
            ifm_model.load_state_dict(ifm_checkpoint)
    ifm_model = ifm_model.to(args.device).eval()
    
    model = End2EndModel(graph_model, ifm_model, args.fingerprint_dim).eval()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    
    with torch.no_grad():
        for idx, smiles in enumerate(canonical_smiles):
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, \
            x_full_atom_neighbors, x_full_bond_neighbors, _ = get_smiles_array([smiles], feature_dicts)
            
            x_atom = torch.Tensor(x_atom).to(args.device)
            x_bonds = torch.Tensor(x_bonds).to(args.device)
            x_atom_index = torch.LongTensor(x_atom_index).to(args.device)
            x_bond_index = torch.LongTensor(x_bond_index).to(args.device)
            x_mask = torch.Tensor(x_mask).to(args.device)
            x_full_atom_neighbors = torch.Tensor(x_full_atom_neighbors).to(args.device)
            x_full_bond_neighbors = torch.Tensor(x_full_bond_neighbors).to(args.device)
            finger_feats = torch.Tensor(fingerprints[idx:idx+1]).to(args.device)
            
            output = model(x_atom, x_bonds, x_atom_index, x_bond_index, 
                          x_mask, x_full_atom_neighbors, x_full_bond_neighbors, finger_feats)
            predictions.append(output.cpu().item())
    
    # Save results
    print("Saving results...")
    output_df = smiles_df.iloc[valid_indices].copy().reset_index(drop=True)
    output_df['canonical_smiles'] = canonical_smiles
    output_df['predicted_value'] = predictions
    output_df.to_csv(args.output_file, index=False)
    
    print(f"Done! Results saved to: {args.output_file}")
    print(f"Predicted {len(predictions)} molecules")
    
    return output_df


if __name__ == "__main__":
    main()
