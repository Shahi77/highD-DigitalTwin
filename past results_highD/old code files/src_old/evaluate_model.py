"""
Evaluate trained trajectory prediction model on HighD dataset.
Computes ADE, FDE, RMSE, MAE, R² — in meters — with visual plots and CSV logging.
"""

import os, sys, csv
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.model.transformer_light import TransformerLight
from src.model.multichannel_attention import MultiChannelAttention
from src.data_loader import load_highd_scene
from src.sync import align_scene
from src.feature_extraction import extract_features
from src.model_input import build_dataloader


# --------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=19, d_model=128, nhead=4, num_layers=2, pred_horizon=75):
        super().__init__()
        self.attn = MultiChannelAttention(in_dim=input_dim)
        self.transformer = TransformerLight(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, pred_horizon=pred_horizon
        )

    def forward(self, x):
        x, _ = self.attn(x)
        return self.transformer(x)


# --------------------------------------------------------------
# Metrics Computation
# --------------------------------------------------------------
def compute_metrics(preds, gt):
    ade = np.mean(np.linalg.norm(preds - gt, axis=2))
    fde = np.mean(np.linalg.norm(preds[:, -1, :] - gt[:, -1, :], axis=1))
    rmse = np.sqrt(mean_squared_error(gt.reshape(-1, 2), preds.reshape(-1, 2)))
    mae = mean_absolute_error(gt.reshape(-1, 2), preds.reshape(-1, 2))
    r2 = r2_score(gt.reshape(-1, 2), preds.reshape(-1, 2))
    return ade, fde, rmse, mae, r2


# --------------------------------------------------------------
# Visualization
# --------------------------------------------------------------
def plot_trajectories(gt, preds, num_samples=5, save_path=None):
    plt.figure(figsize=(7, 6))
    idxs = np.random.choice(len(gt), min(num_samples, len(gt)), replace=False)
    for i in idxs:
        plt.plot(gt[i][:, 0], gt[i][:, 1], 'b-', label='Ground Truth' if i == idxs[0] else "")
        plt.plot(preds[i][:, 0], preds[i][:, 1], 'r--', label='Prediction' if i == idxs[0] else "")
    plt.xlabel("ΔX (m)")
    plt.ylabel("ΔY (m)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title("Predicted vs Actual Trajectories")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


# --------------------------------------------------------------
# CSV Logging
# --------------------------------------------------------------
def log_metrics(save_path, model_name, scene, ade, fde, rmse, mae, r2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    new = not os.path.exists(save_path)
    with open(save_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["Time", "Model", "Scene", "ADE", "FDE", "RMSE", "MAE", "R2"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name, scene, ade, fde, rmse, mae, r2
        ])


# --------------------------------------------------------------
# Evaluation Routine
# --------------------------------------------------------------
def evaluate(model_path="./results/best_model.pt", model_name="AdaptiveDT",
             scene_idx=1, device="cpu"):
    print(f"\n🔍 Loading model from {model_path}")
    model = TrajectoryPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval().to(device)

    print(f"Loading scene {scene_idx}...")
    scene = load_highd_scene("./data/highd/dataset", scene_idx)
    aligned = align_scene(scene, Δt=0.04)  # Match training
    features, _ = extract_features(aligned, include_neighbors=True, k_neighbors=3)
    dataset, _ = build_dataloader(features, hist_len=50, fut_len=75, step=10, batch_size=64)
    
    xb, yb = next(iter(dataset))   # sample from dataset
    print("xb shape, yb shape:", xb.shape, yb.shape)

    # send through model
    xb_t = xb.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(xb_t).cpu().numpy()  # expected (1, fut_len, 2)
    pred = pred.squeeze(0)

    print("\n--- DEBUG SAMPLE ---")
    print("pred shape:", pred.shape)
    print("pred (first 6):\n", np.round(pred[:6], 4))
    print("yb (first 6):\n", np.round(yb.numpy()[:6], 4))
    print("pred mean,std:", pred.mean(), pred.std())
    print("yb mean,std:", yb.numpy().mean(), yb.numpy().std())
    print("first pred vs first gt:", pred[0], yb.numpy()[0])
    print("pred max-min (per axis):", pred.max(axis=0) - pred.min(axis=0))
    print("--- END DEBUG ---\n")


    preds_all, gt_all = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dataset, desc="Evaluating"):
            xb = xb.unsqueeze(0).to(device)
            pred = model(xb).cpu().numpy()
            preds_all.append(pred.squeeze(0))
            gt_all.append(yb.numpy())

    preds_all, gt_all = np.stack(preds_all), np.stack(gt_all)

    # ----------------------------------------------------------
    # Normalization handling
    # ----------------------------------------------------------
    stats_path_npz = "./results/norm_stats.npz"
    denorm_applied = False

    if os.path.exists(stats_path_npz):
        stats = np.load(stats_path_npz)
        mean, std = stats["mean"], stats["std"]
        preds_all = preds_all * std[:2] + mean[:2]
        gt_all = gt_all * std[:2] + mean[:2]
        denorm_applied = True
        print(f"Applied normalization from .npz (mean[:2]={mean[:2]}, std[:2]={std[:2]})")
    else:
        print(" norm_stats.npz not found — using raw normalized coords.")

    # ----------------------------------------------------------
    # Convert to relative motion (Δx, Δy)
    # ----------------------------------------------------------
    preds_all = preds_all - preds_all[:, 0:1, :]
    gt_all = gt_all - gt_all[:, 0:1, :]

    # Apply approximate scaling (if normalization skipped)
    if not denorm_applied:
        scale = 10.0  # convert normalized range (~0–1) to meters
        preds_all *= scale
        gt_all *= scale
    
    print("preds_all mean,std:", preds_all.mean(), preds_all.std())
    print("gt_all mean,std:", gt_all.mean(), gt_all.std())    

    # ----------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------
    ade, fde, rmse, mae, r2 = compute_metrics(preds_all, gt_all)
    print(f"Evaluation Metrics for {model_name} (Scene {scene_idx}):")
    print(f"  ADE  = {ade:.3f} m")
    print(f"  FDE  = {fde:.3f} m")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  R²   = {r2:.3f}")

    # ----------------------------------------------------------
    # Save results
    # ----------------------------------------------------------
    log_metrics("./results/metrics_summary.csv", model_name, scene_idx, ade, fde, rmse, mae, r2)
    plot_trajectories(gt_all, preds_all, save_path=f"./results/scene_{scene_idx}_trajectories.png")
    print("\n Metrics and trajectory plot saved in ./results/")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(model_path="./results/best_model.pt", scene_idx=1, device=device)
