# train_eval_highd.py
import os, json, random, numpy as np, pandas as pd, torch, torch.nn as nn
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from utils import ade_fde, combined_loss, save_json
from evaluate import compute_comprehensive_metrics, plot_error_distribution, plot_diverse_samples
from highd_dataloader import make_dataloader_highd, load_highd_scenes, make_dataloader_highd_multiscene

# ---------------- Setup ----------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Eval Helper ----------------
def evaluate(model, loader, name="val"):
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            obs = b["target"].to(device)
            fut = b["gt"].to(device)
            nd = b["neighbors_dyn"].to(device)
            ns = b["neighbors_spatial"].to(device)
            lane = b["lane"].to(device)
            pred, _ = model(obs, nd, ns, lane) if hasattr(model, "multi_att") else (model(obs, nd, ns, lane), None)
            preds.append(pred.cpu().numpy())
            gts.append(fut.cpu().numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    ADE, FDE = ade_fde(preds, gts)
    print(f"{name} ADE={ADE:.3f}  FDE={FDE:.3f}")
    return ADE, FDE, preds, gts

# ---------------- Training ----------------
def train_highd(csv_path, save_dir="./results_highd", model_type="transformer",
                train_ratio=0.8, epochs=10):
    os.makedirs(save_dir, exist_ok=True)

    obs_len, pred_len, batch_size = 20, 25, 32

    # --- Automatically handle single-scene or multi-scene ---
    if os.path.isdir(csv_path):
        train_files, val_files = load_highd_scenes(csv_path, train_ratio=train_ratio)
        train_loader = make_dataloader_highd_multiscene(train_files, batch_size, obs_len, pred_len, shuffle=True)
        val_loader   = make_dataloader_highd_multiscene(val_files,   batch_size, obs_len, pred_len, shuffle=False)
        print(f"Loaded {len(train_files)} training scenes and {len(val_files)} validation scenes.")
    else:
        res = make_dataloader_highd(csv_path, batch_size=batch_size,
                                    obs_len=obs_len, pred_len=pred_len, shuffle=True)
        if isinstance(res, tuple):
            train_loader, val_loader = res
            print("Auto-split multi-scene dataset detected.")
        else:
            train_loader = res
            val_loader = make_dataloader_highd(csv_path, batch_size=batch_size,
                                               obs_len=obs_len, pred_len=pred_len, shuffle=False)
            print(f"Loaded single-scene HighD file: {csv_path}")

    # --- Model setup ---
    model = SimpleSLSTM(pred_len=pred_len) if model_type == "slstm" else ImprovedTrajectoryTransformer(pred_len=pred_len)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_ade = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_ADE": []}
    patience, wait = 4, 0

    print(f"\nTraining {model_type.upper()} on HighD for {epochs} epochs ({train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% val)\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for b in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            obs = b["target"].to(device)
            fut = b["gt"].to(device)
            nd = b["neighbors_dyn"].to(device)
            ns = b["neighbors_spatial"].to(device)
            lane = b["lane"].to(device)

            opt.zero_grad()
            pred, _ = model(obs, nd, ns, lane) if hasattr(model, "multi_att") else (model(obs, nd, ns, lane), None)
            loss, _ = combined_loss(pred, fut, w_pos=1.0, w_vel=0.3, w_acc=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                obs = b["target"].to(device)
                fut = b["gt"].to(device)
                nd = b["neighbors_dyn"].to(device)
                ns = b["neighbors_spatial"].to(device)
                lane = b["lane"].to(device)
                pred, _ = model(obs, nd, ns, lane) if hasattr(model, "multi_att") else (model(obs, nd, ns, lane), None)
                loss, _ = combined_loss(pred, fut, w_pos=1.0, w_vel=0.3, w_acc=0.1)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        # ---- Evaluation metrics ----
        train_ADE, train_FDE, _, _ = evaluate(model, train_loader, "Train")
        val_ADE, val_FDE, _, _ = evaluate(model, val_loader, "Val")
        history["val_ADE"].append(val_ADE)
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, TrainADE={train_ADE:.3f}, ValADE={val_ADE:.3f}")

        # ---- Early stopping ----
        if val_ADE < best_ade:
            best_ade, wait = val_ADE, 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_type}.pt"))
        else:
            wait += 1
        if wait >= patience:
            print("⏸ Early stopping triggered."); break

        # ---- Visualization ----
        if epoch == epochs // 5 or epoch == epochs:
            obs_np = obs[0, :, :2].cpu().numpy()
            gt_np = fut[0].cpu().numpy()
            pred_np = pred[0].detach().cpu().numpy()
            plt.figure(figsize=(6, 5))
            plt.plot(obs_np[:, 0], obs_np[:, 1], "ko-", label="Observed")
            plt.plot(gt_np[:, 0], gt_np[:, 1], "g-", label="Ground Truth")
            plt.plot(pred_np[:, 0], pred_np[:, 1], "r--", label="Predicted")
            plt.legend(); plt.grid(True); plt.axis("equal")
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_viz.png"), dpi=150)
            plt.close()

    # ---- Final Evaluation ----
    model.load_state_dict(torch.load(os.path.join(save_dir, f"best_{model_type}.pt"), map_location=device))
    model.eval()
    _, _, preds, gts = evaluate(model, val_loader, "Final Val")

    metrics = compute_comprehensive_metrics(preds, gts)
    print(json.dumps(metrics, indent=2))
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    plot_error_distribution(preds, gts, save_dir)
    plot_diverse_samples(preds, gts, [np.zeros((obs_len, 2))]*len(preds), save_dir)
    print(f"All evaluation artifacts saved to: {save_dir}")

    return model, metrics

# ---------------- Main ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tracks_csv")
    parser.add_argument("--model_type", choices=["transformer", "slstm"], default="transformer")
    parser.add_argument("--save_dir", default="./results_highd")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_highd(
        args.tracks_csv,
        save_dir=args.save_dir,
        model_type=args.model_type,
        train_ratio=args.train_ratio,
        epochs=args.epochs
    )
