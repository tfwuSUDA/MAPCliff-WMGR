import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from tqdm import tqdm
def read_smiles(csv_path):
    df = pd.read_csv(csv_path)
    smiles = df['smiles'].values 
    return smiles

def compute_ecfp(smiles_list, radius=2, n_bits=1024):
    ecfp_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            ecfp_list.append(np.array(ecfp))
        else:
            ecfp_list.append(np.zeros(n_bits))  # 处理无法解析的SMILES
    return np.array(ecfp_list)

def load_descriptors(numpy_path):
    return np.load(numpy_path)

def concatenate_descriptors(ecfp_array, descriptors_array):
    return np.hstack((ecfp_array, descriptors_array))

def save_combined_descriptors(output_path, combined_array):
    np.save(output_path, combined_array)

# 示例用法
if __name__ == "__main__":
    dir_path = "./mol_feature/"
    task_list = [f[:-4] for f in os.listdir(dir_path) if f.endswith(".csv")]
    for task in tqdm(task_list):
        csv_path = f"./benchmark1/{task}/{task}.csv"
        numpy_path = "./mol_fingers_npy/" + task + ".npy"
        output_path = "./add_ECFP/" + task + ".npy"
        smiles = read_smiles(csv_path)
        ecfp_array = compute_ecfp(smiles)
        descriptors_array = load_descriptors(numpy_path)
        combined_array = concatenate_descriptors(ecfp_array, descriptors_array)
        save_combined_descriptors(output_path, combined_array)
