import argparse
import torch
import numpy as np
import pandas as pd
from unit import Meter,MyDataset,EarlyStopping,collate_fn,IFM_DNN_4
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
import sys
import copy
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time

start_time = time.time()
from sklearn.model_selection import train_test_split
import warnings
import os
import shutil
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
label_choose = {"freesolv": ["freesolv"]}

def standardize(col):
    return (col - np.mean(col)) / np.std(col)

def get_pos_weight(Ys):
    Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)
    num_pos = torch.sum(Ys, dim=0)
    num_indices = torch.tensor(len(Ys))
    return (num_indices - num_pos) / num_pos
def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        Xs, Ys, masks = batch_data
        Xs, Ys, masks = Xs.to(args.device), Ys.to(args.device), masks.to(args.device)
        outputs = model(Xs)
        loss = (loss_func(outputs, Ys) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.cpu()
        Ys.cpu()
        masks.cpu()
        loss.cpu()
        train_metric.update(outputs, Ys, masks)
    if args.reg:
        rmse_score = np.mean(
            train_metric.compute_metric(args.metric)
        ) 
        mae_score = np.mean(
            train_metric.compute_metric("mae")
        )
        r2_score = np.mean(train_metric.compute_metric("r2")) 
        return {"rmse": rmse_score, "mae": mae_score, "r2": r2_score}
    else:
        roc_score = np.mean(
            train_metric.compute_metric(args.metric)
        ) 
        prc_score = np.mean(
            train_metric.compute_metric("prc_auc")
        )
        return {"roc_auc": roc_score, "prc_auc": prc_score}


def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            Xs, Ys, masks = batch_data
            Xs, Ys, masks = (
                Xs.to(args.device),
                Ys.to(args.device),
                masks.to(args.device),
            )
            outputs = model(Xs)
            outputs.cpu()
            Ys.cpu()
            masks.cpu()
            eval_metric.update(outputs, Ys, masks)
    if args.reg:
        rmse_score = np.mean(
            eval_metric.compute_metric(args.metric)
        )
        mae_score = np.mean(eval_metric.compute_metric("mae"))
        r2_score = np.mean(eval_metric.compute_metric("r2"))
        return {"rmse": rmse_score, "mae": mae_score, "r2": r2_score}
    else:
        roc_score = np.mean(
            eval_metric.compute_metric(args.metric)
        )
        prc_score = np.mean(
            eval_metric.compute_metric("prc_auc")
        )
        return {"roc_auc": roc_score, "prc_auc": prc_score}


def main(task_name, seed, split_id):
    parser = argparse.ArgumentParser(description="PyTorch implementation")
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--data_label", type=str, default="CHEMBL218_EC50", help="dataset."
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="IFM_4",
        help="Embedding method: None, LE, LSIM, GM, IFM, SIM.",
    )
    parser.add_argument("--epochs", type=int, default=3000, help="running epochs")
    parser.add_argument(
        "--runseed",
        type=int,
        default=66,
        help="Seed for minibatch selection, random initialization.",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="batchsize")
    parser.add_argument("--patience", type=int, default=64, help="patience")
    parser.add_argument("--opt_iters", type=int, default=50, help="optimization_iters")
    parser.add_argument(
        "--repetitions", type=int, default=50, help="splitting repetitions"
    )
    args = parser.parse_args()
    args.data_label = task_name
    labels = label_choose[task_name]
    args.runseed = seed
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    args.device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    args.task = args.data_label
    args.reg = True
    args.metric = "rmse"
    if args.embed == "IFM_4":
        hyper_paras_space = {
            "l2": hp.uniform("l2", 0, 0.01),
            "dropout": hp.uniform("dropout", 0, 0.5),
            "d_out": hp.randint("d_out", 127),
            "omega1": hp.uniform("omega1", 0.001, 1),  # 1
            "omega0": hp.uniform("omega0", 0.001, 1),  # 1
            "sigma": hp.loguniform("sigma", np.log(0.01), np.log(100)),
            "hidden_unit1": hp.choice("hidden_unit1", [64, 128, 256, 512, 1024]),
            "hidden_unit2": hp.choice("hidden_unit2", [64, 128, 256, 512, 1024]),
            "hidden_unit3": hp.choice("hidden_unit3", [64, 128, 256, 512, 1024]),
            "hidden_unit4": hp.choice("hidden_unit4", [64, 128, 256, 512, 1024]),
        }
    else:
        raise ValueError("Invalid Embedding Name")
    total_df = pd.read_csv(f"../dataset/MoleculeNet/{task_name}/{task_name}.csv")
    split_fold_path = f"../dataset/MoleculeNet/{args.task}/splits/scaffold-{split_id}.npy"
    split_fold_npy = np.load(split_fold_path, allow_pickle=True)
    split_test = split_fold_npy[2]
    split_val = split_fold_npy[1]
    split_train = split_fold_npy[0]
    total_df['split'] = 'test'
    total_df.loc[split_train, 'split'] = 'train'
    total_df.loc[split_val, 'split'] = 'val'
    train_idx = total_df[total_df["split"] == "train"].index
    test_idx = total_df[total_df["split"] == "test"].index
    val_idx = total_df[total_df["split"] == "val"].index    
    finger_feature_path = (f"../dataset/MoleculeNet/KPGT/SNN_Finger_feature/{args.task}.npy")
    finger_feature = np.load(finger_feature_path)
    train_feature = finger_feature[train_idx]
    test_feature = finger_feature[test_idx]
    val_feature = finger_feature[val_idx]
    inputs = train_feature.shape[1]
    train_labels = total_df.loc[train_idx, labels].values.reshape(-1, len(labels))
    test_labels = total_df.loc[test_idx, labels].values.reshape(-1, len(labels))
    val_labels = total_df.loc[val_idx, labels].values.reshape(-1, len(labels))
    all_labels = total_df[labels].values.reshape(-1, len(labels))
    train_dataset = MyDataset(train_feature, train_labels)
    test_dataset = MyDataset(test_feature, test_labels)
    val_dataset = MyDataset(val_feature, val_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    validation_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    if not args.reg:
        pos_weights = get_pos_weight(all_labels)
        print("Positive weight for the combined dataset:", pos_weights)
    print("inputs:", inputs)

    def hyper_opt(hyper_paras):
        hidden_units = [
            hyper_paras["hidden_unit1"],
            hyper_paras["hidden_unit2"],
            hyper_paras["hidden_unit3"],
            hyper_paras["hidden_unit4"],
        ]
        if args.embed == "IFM_4":
            my_model = IFM_DNN_4(
                inputs=inputs,
                hidden_units=hidden_units,
                d_out=hyper_paras["d_out"] + 1,
                sigma=hyper_paras["sigma"],
                dp_ratio=hyper_paras["dropout"],
                first_omega_0=hyper_paras["omega0"],
                hidden_omega_0=hyper_paras["omega1"],
                outputs=len(labels),
                reg=args.reg,
            )
        else:
            raise ValueError("Invalid Embedding Name")
        optimizer = torch.optim.Adadelta(
            my_model.parameters(), weight_decay=hyper_paras["l2"]
        )
        file_name = "../save_model/%s_%.4f_%d_%d_%d_%d_%.4f_early_stop.pth" % (
            args.task,
            hyper_paras["dropout"],
            hyper_paras["hidden_unit1"],
            hyper_paras["hidden_unit2"],
            hyper_paras["hidden_unit3"],
            hyper_paras["hidden_unit4"],
            hyper_paras["l2"],
        )
        if args.reg:
            loss_func = MSELoss(reduction="none")
            stopper = EarlyStopping(
                mode="lower", patience=args.patience, filename=file_name
            )
        else:
            loss_func = BCEWithLogitsLoss(
                reduction="none", pos_weight=pos_weights.to(args.device)
            )
            stopper = EarlyStopping(
                mode="higher", patience=args.patience, filename=file_name
            )
        my_model.to(args.device)
        for i in range(args.epochs):
            run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)
            val_scores = run_an_eval_epoch(my_model, validation_loader, args)
            early_stop = stopper.step(val_scores[args.metric], my_model)
            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        val_scores = run_an_eval_epoch(my_model, validation_loader, args)
        feedback_val = (
            val_scores[args.metric] if args.reg else (1 - val_scores[args.metric])
        )
        my_model.cpu()
        gc.collect()
        return feedback_val
    trials = Trials()
    print("******hyper-parameter optimization is starting now******")
    opt_res = fmin(
        hyper_opt,
        hyper_paras_space,
        algo=tpe.suggest,
        max_evals=args.opt_iters,
        trials=trials,
    )
    print("******hyper-parameter optimization is over******")
    print("the best hyper-parameters settings for " + args.task + " are:  ", opt_res)
    hidden_unit1_ls = [64, 128, 256, 512, 1024]
    hidden_unit2_ls = [64, 128, 256, 512, 1024]
    hidden_unit3_ls = [64, 128, 256, 512, 1024]
    hidden_unit4_ls = [64, 128, 256, 512, 1024]
    opt_hidden_units = [
        hidden_unit1_ls[opt_res["hidden_unit1"]],
        hidden_unit2_ls[opt_res["hidden_unit2"]],
        hidden_unit3_ls[opt_res["hidden_unit3"]],
        hidden_unit4_ls[opt_res["hidden_unit4"]],
    ]
    if args.embed == "IFM_4":
        best_model = IFM_DNN_4(
            inputs=inputs,
            hidden_units=opt_hidden_units,
            d_out=opt_res["d_out"] + 1,
            sigma=opt_res["sigma"],
            dp_ratio=opt_res["dropout"],
            first_omega_0=opt_res["omega0"],
            hidden_omega_0=opt_res["omega1"],
            outputs=len(labels),
            reg=args.reg,
        )
    else:
        raise ValueError("Invalid Embedding Name")
    best_hyperparams = {
        "inputs": inputs,
        "hidden_units": opt_hidden_units,
        "d_out": opt_res["d_out"] + 1,
        "sigma": opt_res["sigma"],
        "dropout": opt_res["dropout"],
        "first_omega_0": opt_res["omega0"],
        "hidden_omega_0": opt_res["omega1"],
        "l2": opt_res["l2"],
    }
    best_file_name = "../save_model/%s_%.4f_%d_%d_%d_%d_%.4f_early_stop.pth" % (
        args.task,
        opt_res["dropout"],
        hidden_unit1_ls[opt_res["hidden_unit1"]],
        hidden_unit1_ls[opt_res["hidden_unit2"]],
        hidden_unit1_ls[opt_res["hidden_unit3"]],
        hidden_unit1_ls[opt_res["hidden_unit4"]],
        opt_res["l2"],
    )
    best_model.load_state_dict(torch.load(best_file_name, map_location=args.device)["model_state_dict"])
    best_model.to(args.device)
    test_scores = run_an_eval_epoch(best_model, test_loader, args)
    print("test set:", test_scores)
    return best_file_name, test_scores, best_hyperparams



if __name__ == "__main__":
    task = "freesolv"
    split_list = [0, 1, 2]
    seed = 1
    for split_id in split_list:
        main(task_name=task, seed=seed, split_id=split_id)
