import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


def process_data(task):
    dataset = pd.read_csv(f"./mol_feature/{task}.csv")
    cols_to_process = dataset.columns[1:]
    rm_cols = (
        dataset[cols_to_process]
        .isnull()
        .any()[dataset[cols_to_process].isnull().any()]
        .index
    )
    dataset.drop(columns=rm_cols, inplace=True)
    cols_to_process = dataset.columns[1:]
    data_fea_corr = dataset[cols_to_process].corr()
    del_fea2_ind = set()
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_ind.add(data_fea_corr.columns[j])
    dataset.drop(columns=list(del_fea2_ind), inplace=True)
    cols_to_process = dataset.columns[1:]
    dataset[cols_to_process] = dataset[cols_to_process].apply(standardize, axis=0)
    dataset.dropna(axis=1, how="all", inplace=True)
    print(f"{task}: standardize complete")
    output_folder = "./std_mol_fingers/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    dataset.to_csv(f"{output_folder}/{task}.csv", index=False)
    numpy_output_folder = "./mol_fingers_npy/"
    feature_array = dataset.iloc[:, 1:].values
    np.save(f"{numpy_output_folder}/{task}.npy", feature_array)



def main():
    task_dir = "./mol_feature/"
    task_list = [f[:-4] for f in os.listdir(task_dir) if f.endswith(".csv")]
    num_processes = 10
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_data, task_list), total=len(task_list)))


if __name__ == "__main__":
    main()
