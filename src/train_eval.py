"""
train_fixed_optimized.py
-----------------------------------------------------
Full optimized training loop for HighD Trajectory Prediction

 80/20 train/val split
 Exponential scheduled sampling
 Velocity + Acceleration loss
 Early stopping
 Extended curriculum
 Constant-velocity baseline
 Comprehensive evaluation (metrics + plots)
"""

import os
import sys
import time
import json
import random
from collections import defaultdict

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from highd_dataloader import make_dataloader_fixed
from utils import ade_fde, cumulate_deltas, torch_cumulate_deltas, plot_prediction_one, save_json
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from evaluate import evaluate_model_comprehensive, compute_comprehensive_metrics


#  General setup
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

if torch.backends.mps.is_available():
    device = torch.device("cpu")
    print("Using CPU")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA")


#  Loss: Position + Velocity + Acceleration
def trajectory_loss(pred, gt, alpha_v=0.2, alpha_a=0.1):
    pos_loss = nn.functional.l1_loss(pred, gt)
    vel_loss = nn.functional.l1_loss(pred[:,1:] - pred[:,:-1],
                                     gt[:,1:] - gt[:,:-1])
    acc_loss = nn.functional.l1_loss(
        (pred[:,2:] - 2*pred[:,1:-1] + pred[:,:-2]),
        (gt[:,2:] - 2*gt[:,1:-1] + gt[:,:-2])
    )
    return pos_loss + alpha_v*vel_loss + alpha_a*acc_loss

#  Helper functions
def split_tracks_by_vehicle(tracks_df, val_frac=0.2, seed=SEED):
    vids = sorted(tracks_df['id'].unique())
    random.Random(seed).shuffle(vids)
    n_val = int(len(vids) * val_frac)
    val_vids = set(vids[:n_val])
    train_vids = set(vids[n_val:])
    train_df = tracks_df[tracks_df['id'].isin(train_vids)].reset_index(drop=True)
    val_df = tracks_df[tracks_df['id'].isin(val_vids)].reset_index(drop=True)
    return train_df, val_df

def exponential_scheduled_sampling(epoch, total_epochs, start=1.0, end=0.1, k=5):
    return end + (start - end) * np.exp(-k * epoch / total_epochs)

def constant_velocity_baseline(gt):
    v = gt[1] - gt[0]
    pred = gt[0] + np.arange(gt.shape[0])[:,None] * v
    return pred


#  Core Training Loop

def evaluate_model(model, loader, device, eval_samples=200, pred_len=None, stage_idx=None):
    """
    Evaluate model on validation set with optional curriculum context.
    Handles both LSTM and Transformer models safely.
    """
    model.eval()
    preds_all, gts_all = [], []

    with torch.no_grad():
        for i, sample in enumerate(loader.dataset):
            if i >= eval_samples:
                break

            # Load features
            target = torch.from_numpy(sample['target_feats']).unsqueeze(0).to(device)
            neigh_dyn = torch.from_numpy(sample['neighbors_dyn']).unsqueeze(0).to(device)
            neigh_spatial = torch.from_numpy(sample['neighbors_spatial']).unsqueeze(0).to(device)
            lane = torch.from_numpy(sample['lane_feats']).unsqueeze(0).to(device)
            gt = sample['gt']

            # Determine last observed position (for residual CV baseline)
            last_obs_pos = torch.zeros(1, 2, device=device)
            if target.shape[1] >= 2:
                last_obs_pos = target[:, -1, :2]

            # Forward pass
            if "train_stage" in model.forward.__code__.co_varnames:
                # Transformer with teacher-forced CV warmup support
                pred = model(
                    target, neigh_dyn, neigh_spatial, lane,
                    last_obs_pos=last_obs_pos,
                    pred_len=pred_len,
                    train_stage=stage_idx + 1 if stage_idx is not None else None
                )[0].cpu().numpy()
            else:
                # LSTM baseline (no warmup argument)
                pred = model(
                    target, neigh_dyn, neigh_spatial, lane,
                    pred_len=pred_len
                )[0].cpu().numpy()

            preds_all.append(pred)
            gts_all.append(gt)

    # Stack predictions and ground truths safely
    preds_all = np.stack(preds_all, axis=0)
    gts_all = np.stack(gts_all, axis=0)

    # Compute all metrics
    metrics = compute_comprehensive_metrics(preds_all, gts_all)
    return metrics, preds_all, gts_all


def train_loop(tracks_df, save_dir='./results/checkpoints',
               model_type='transformer', curriculum=None,
               obs_len=10, batch_size=32, k_neighbors=8,
               val_frac=0.2, patience=5):

    os.makedirs(save_dir, exist_ok=True)

    train_df, val_df = split_tracks_by_vehicle(tracks_df, val_frac)
    print(f"Train vehicles: {train_df['id'].nunique()} | Val vehicles: {val_df['id'].nunique()}")

    # Select model
    if model_type == "lstm":
        model = SimpleSLSTM(input_dim=7, hidden_dim=256, output_dim=2,
                            obs_len=obs_len, pred_len=curriculum[-1][0]).to(device)
    else:
        model = ImprovedTrajectoryTransformer(d_model=256, nhead=8, num_layers=4,
                                              pred_len=curriculum[-1][0],
                                              k_neighbors=k_neighbors).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = None
    best_ade, wait = float("inf"), 0
    history = defaultdict(list)

    for stage_idx, (pred_len, lr, epochs) in enumerate(curriculum):
        print(f"\n{'='*60}\nStage {stage_idx+1}: pred_len={pred_len}, lr={lr}, epochs={epochs}\n{'='*60}")
        for pg in opt.param_groups: pg["lr"] = lr
        model.pred_len = pred_len
        train_loader = make_dataloader_fixed(train_df, batch_size=batch_size, shuffle=True,
                                             obs_len=obs_len, pred_len=pred_len, downsample=5,
                                             k_neighbors=k_neighbors)
        val_loader = make_dataloader_fixed(val_df, batch_size=1, shuffle=False,
                                           obs_len=obs_len, pred_len=pred_len, downsample=5,
                                           k_neighbors=k_neighbors)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr*0.1)

        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            ss_p = exponential_scheduled_sampling(epoch, epochs)
            pbar = tqdm(train_loader, desc=f"Stage{stage_idx+1} Epoch {epoch}/{epochs}")
            for batch in pbar:
                target = batch["target"].to(device)
                neigh_dyn = batch["neigh_dyn"].to(device)
                neigh_spatial = batch["neigh_spatial"].to(device)
                lane = batch["lane"].to(device)
                gt = batch["gt"].to(device)
                opt.zero_grad()
                pred = model(target, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)
                loss = trajectory_loss(pred, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", ss_p=f"{ss_p:.2f}")
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} | Loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.1e}")

            # Validation
            metrics, preds, gts = evaluate_model(model, val_loader, device)
            ADE, FDE = metrics["ADE"], metrics["FDE"]
            print(f"Validation: ADE={ADE:.3f} FDE={FDE:.3f}")

            history["stage"].append(stage_idx+1)
            history["epoch"].append(epoch + sum([c[2] for c in curriculum[:stage_idx]]))
            history["ADE"].append(ADE)
            history["FDE"].append(FDE)
            history["MAE"].append(metrics["MAE"])
            history["RMSE"].append(metrics["RMSE"])

            # Early stopping
            if ADE < best_ade:
                best_ade, wait = ADE, 0
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(" Best model updated.")
            else:
                wait += 1
                if wait >= patience:
                    print(" Early stopping triggered.")
                    # break   # remove this line
                    wait = 0  # reset patience and continue next stage

            # Plot sample trajectory
            sample = val_loader.dataset[0]
            with torch.no_grad():
                target = torch.from_numpy(sample['target_feats']).unsqueeze(0).to(device)
                neigh_dyn = torch.from_numpy(sample['neighbors_dyn']).unsqueeze(0).to(device)
                neigh_spatial = torch.from_numpy(sample['neighbors_spatial']).unsqueeze(0).to(device)
                lane = torch.from_numpy(sample['lane_feats']).unsqueeze(0).to(device)
                pred_sample = model(target, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)[0].cpu().numpy()
            plot_prediction_one(sample['target_feats'][:,:2], sample['gt'], pred_sample,
                                save_path=os.path.join(save_dir, f"stage{stage_idx+1}_epoch{epoch:02d}.png"))

        if wait >= patience:
            break


    # Save training curves
    save_json(history, os.path.join(save_dir, "train_history.json"))
    plt.figure(figsize=(6,4))
    plt.plot(history["ADE"], label="ADE")
    plt.plot(history["FDE"], label="FDE")
    plt.xlabel("Evaluation step")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.title("Validation ADE/FDE over training")
    plt.savefig(os.path.join(save_dir, "ade_fde_curve.png"), dpi=150)
    plt.close()
    print("Training complete. Best ADE:", best_ade)
    return model, history


#  Main entry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_csv")
    parser.add_argument("--model_type", choices=["transformer", "lstm"], default="transformer")
    parser.add_argument("--save_dir", default="./results/checkpoints")
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.tracks_csv)
    print("Loaded rows:", len(df), "unique vehicles:", df["id"].nunique())

    model, history = train_loop(
        df, save_dir=args.save_dir, model_type=args.model_type,
        curriculum=[(10, 5e-4, 4), (15, 4e-4, 6), (20, 3e-4, 8), (25, 2e-4, 10)],
        obs_len=20, batch_size=32, k_neighbors=8, val_frac=args.val_frac
    )

    # Final Evaluation (Comprehensive)

    best_model_path = os.path.join(args.save_dir, "./results/best_model.pt")
    if os.path.exists(best_model_path):
        print(f"\nLoading best model weights from {best_model_path}...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("\n[Warning] Best model checkpoint not found. Using last trained weights.")

    print("\nStarting comprehensive evaluation...")
    metrics, pred_df = evaluate_model_comprehensive(model, df, n_samples=500,
                                                    save_dir=os.path.join(args.save_dir, "./results/eval_results"))

