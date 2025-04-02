import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def replace_values_in_csv(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file != "tox21.csv":
                continue
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    # 处理除第一列之外的所有列
                    cols = df.columns[1:]  # 假设第一列是smiles，跳过它
                    
                    # 替换NaN值为0
                    df[cols] = df[cols].fillna(0)
                    
                    # 替换Inf和-Infs为0
                    df[cols] = df[cols].replace([np.inf, -np.inf], 0)
                    
                    # 保存修改后的DataFrame
                    df.to_csv(file_path, index=False)
                except PermissionError:
                    print(f"Permission denied when trying to write to {file_path}. File may be open in another program.")