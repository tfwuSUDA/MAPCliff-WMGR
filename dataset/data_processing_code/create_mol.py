import os
import pandas as pd
from rdkit import Chem
import logging
from tqdm import tqdm
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def create_mol_files(dataset, csv_path, output_dir):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError("The CSV file must contain a 'smiles' column.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for index, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mol_file_path = os.path.join(output_dir, f'{dataset}_molecule_{index}.mol')
        with open(mol_file_path, 'w') as mol_file:
            mol_file.write(Chem.MolToMolBlock(mol))

dir_path = "./benchmark1/"
dataset_path = os.listdir(dir_path)
for dataset in tqdm(dataset_path):
    csv_path = f"{dir_path}{dataset}/{dataset}.csv"
    output_dir = f"./mol_total/"
    create_mol_files(dataset, csv_path, output_dir)
