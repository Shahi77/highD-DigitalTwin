"""
Training pipeline for Adaptive Digital Twin vehicle trajectory prediction
using Multi-Channel Attention + Lightweight Transformer
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.model.transformer_light import TransformerLight
from src.model.multichannel_attention import MultiChannelAttention
from src.model_input import build_dataloader
from src.data_loader import load_highd_scene
from src.sync import align_scene
from src.feature_extraction import extract_features


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=19, d_model=128, nhead=4, num_layers=2, pred_horizon=75):
        super().__init__()
        self.attn = MultiChannelAttention(in_dim=input_dim)
        self.transformer = TransformerLight(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pred_horizon=pred_horizon
        )

    def forward(self, x):
        x, weights = self.attn(x)
        preds = self.transformer(x)
        return preds, weights


def train_model(features, hist_len=50, fut_len=75, batch_size=8, epochs=10, lr=1e-3, device="cpu"):
    print("\n Normalizing features...")
    mean = np.mean(features, axis=(0, 1))
    std = np.std(features, axis=(0, 1)) + 1e-6
    os.makedirs("./results", exist_ok=True)
    np.savez("./results/norm_stats.npz", mean=mean, std=std)
    print(" Normalization stats saved to ./results/norm_stats.npz")

    features_norm = (features - mean) / std
    torch.save({"mean": mean, "std": std}, "./results/norm_stats.pt")

    dataset, loader = build_dataloader(
        features_norm, hist_len=hist_len, fut_len=fut_len, step=10, batch_size=batch_size
    )

    # Split 80/20 -  train/val
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = TrajectoryPredictor(input_dim=features.shape[-1], pred_horizon=fut_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                xb, yb = xb.to(device), yb.to(device)
                preds, _ = model(xb)
                val_loss += loss_fn(preds, yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f}")

        torch.save(model.state_dict(), f"./results/model_epoch_{epoch}.pt")

    # Plot training curves
    os.makedirs("./results", exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Progress (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results/training_curve.png", dpi=200)
    plt.close()

    print("\n Training complete. Model and normalization stats saved in ./results/")
    return model



if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # ---- Load and prepare data ----
    scene_dir = "./data/highd/dataset"
    scene = load_highd_scene(scene_dir, 1)
    scene_aligned = align_scene(scene, Δt=0.04)  # 25 Hz
    features, ids = extract_features(scene_aligned, include_neighbors=True, k_neighbors=3)

    # ---- Train the model ----
    model = train_model(features, hist_len=50, fut_len=75, batch_size=8, epochs=1, device=device)
    torch.save(model.state_dict(), "./results/best_model.pt")
