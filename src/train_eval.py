# train_eval.py
"""
HighD Training and Evaluation Script with Accuracy & Precision Metrics
-----------------------------------------------------------------------
• Automatically detects single- or multi-scene datasets
• Supports: Transformer, S-LSTM, Vanilla LSTM, CS-LSTM, Social-GAN, GNN
• Includes CV residual warm-up for Transformer
• Handles early stopping and adaptive learning rate scheduling
• Outputs comprehensive metrics including accuracy and precision
• Compatible with DT_fixed_simulation.py pipeline
"""

import os, json, random, numpy as np, pandas as pd, torch, torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (
    ImprovedTrajectoryTransformer, 
    SimpleSLSTM,
    VanillaLSTM,
    CSLSTM,
    SocialGAN,
    GNNTrajectoryPredictor
)
from utils import (
    ade_fde, 
    combined_loss, 
    save_json,
    compute_accuracy_precision,
    compute_classification_metrics
)
from evaluate import compute_comprehensive_metrics, plot_error_distribution, plot_diverse_samples
from highd_dataloader import make_dataloader_highd, load_highd_scenes, make_dataloader_highd_multiscene


# -------------------- Setup --------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# -------------------- Model Factory --------------------
def create_model(model_type, obs_len=20, pred_len=25, k_neighbors=8):
    """Factory function to create models based on type"""
    model_type = model_type.lower()
    
    if model_type == "transformer":
        return ImprovedTrajectoryTransformer(pred_len=pred_len, k_neighbors=k_neighbors)
    elif model_type == "slstm":
        return SimpleSLSTM(obs_len=obs_len, pred_len=pred_len, k_neighbors=k_neighbors)
    elif model_type == "lstm":
        return VanillaLSTM(obs_len=obs_len, pred_len=pred_len)
    elif model_type == "cslstm":
        return CSLSTM(obs_len=obs_len, pred_len=pred_len, k_neighbors=k_neighbors)
    elif model_type == "social-gan" or model_type == "sgan":
        return SocialGAN(obs_len=obs_len, pred_len=pred_len, k_neighbors=k_neighbors)
    elif model_type == "gnn":
        return GNNTrajectoryPredictor(obs_len=obs_len, pred_len=pred_len, k_neighbors=k_neighbors)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: transformer, slstm, lstm, cslstm, social-gan, gnn")


# -------------------- Forward Wrapper --------------------
def forward_model(model, obs, nd, ns, lane, teacher_forced=False):
    """
    Unified forward pass for all models.
    Includes teacher-forced CV residual warm-up for transformer.
    """
    model_name = model.__class__.__name__
    
    if model_name == "ImprovedTrajectoryTransformer":
        last_obs_pos = obs[:, -1, :2]
        pred = model(obs, nd, ns, lane, last_obs_pos=last_obs_pos)
        if teacher_forced:
            # Residual warm-up (CV baseline blending)
            with torch.no_grad():
                v_last = obs[:, -1, 2:4]  # last observed velocity
                t = torch.arange(model.pred_len, device=obs.device).float().view(1, -1, 1)
                cv_baseline = last_obs_pos.unsqueeze(1) + t * v_last.unsqueeze(1)
            alpha = 0.1  # blending ratio
            pred = (1 - alpha) * pred + alpha * cv_baseline
        return pred
    elif model_name == "SocialGAN":
        # For Social-GAN, we can optionally inject noise during training
        if model.training:
            noise = torch.randn(obs.shape[0], model.noise_dim, device=obs.device)
            return model(obs, nd, ns, lane, noise=noise)
        else:
            # At inference, use zero noise for deterministic prediction
            noise = torch.zeros(obs.shape[0], model.noise_dim, device=obs.device)
            return model(obs, nd, ns, lane, noise=noise)
    else:
        # All other models (LSTM, S-LSTM, CS-LSTM, GNN)
        return model(obs, nd, ns, lane)


# -------------------- Evaluation Helper --------------------
def evaluate(model, loader, name="val", compute_detailed_metrics=False):
    """
    Evaluate model and compute comprehensive metrics including accuracy/precision.
    
    Args:
        model: trained model
        loader: data loader
        name: dataset name for logging
        compute_detailed_metrics: if True, compute full accuracy/precision metrics
    
    Returns:
        tuple of (ADE, FDE, predictions, ground_truths, detailed_metrics)
    """
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            obs = b["target"].to(device)
            fut = b["gt"].to(device)
            nd = b["neighbors_dyn"].to(device)
            ns = b["neighbors_spatial"].to(device)
            lane = b["lane"].to(device)
            pred = forward_model(model, obs, nd, ns, lane)
            preds.append(pred.cpu().numpy())
            gts.append(fut.cpu().numpy())
    
    preds, gts = np.concatenate(preds), np.concatenate(gts)
    ADE, FDE = ade_fde(preds, gts)
    
    detailed_metrics = {}
    if compute_detailed_metrics:
        # Compute accuracy and precision at multiple thresholds
        acc_metrics = compute_accuracy_precision(preds, gts, thresholds=[0.5, 1.0, 2.0])
        detailed_metrics.update(acc_metrics)
        
        # Compute classification-style metrics
        class_metrics = compute_classification_metrics(preds, gts, threshold=1.0)
        detailed_metrics.update(class_metrics)
    
    print(f"{name} ADE={ADE:.3f}  FDE={FDE:.3f}", end="")
    if compute_detailed_metrics:
        print(f"  Acc@1m={detailed_metrics['accuracy_1.0m']:.3f}  Prec@1m={detailed_metrics['precision_1.0m']:.3f}")
    else:
        print()
    
    return ADE, FDE, preds, gts, detailed_metrics


# -------------------- Training Core --------------------
def train_highd(csv_path, save_dir="./results_highd", model_type="transformer",
                train_ratio=0.8, epochs=10, batch_size=32, learning_rate=5e-4):
    os.makedirs(save_dir, exist_ok=True)
    obs_len, pred_len = 20, 25

    # --- Automatic multi-scene detection ---
    if os.path.isdir(csv_path):
        train_files, val_files = load_highd_scenes(csv_path, train_ratio=train_ratio)
        train_loader = make_dataloader_highd_multiscene(
            train_files, batch_size, obs_len, pred_len,
            shuffle=True
        )
        val_loader = make_dataloader_highd_multiscene(
            val_files, batch_size, obs_len, pred_len,
            shuffle=False
        )
        print(f"Detected {len(train_files) + len(val_files)} scenes → train={len(train_files)}, val={len(val_files)}")
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
    model = create_model(model_type, obs_len=obs_len, pred_len=pred_len)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_type.upper()} Model:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # --- Optimizer and scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    best_ade = float("inf")
    history = {
        "train_loss": [], "val_loss": [], 
        "train_ADE": [], "train_FDE": [],
        "val_ADE": [], "val_FDE": [],
        "train_accuracy_1m": [], "val_accuracy_1m": [],
        "train_precision_1m": [], "val_precision_1m": []
    }
    patience, wait = 5, 0

    print(f"Training {model_type.upper()} on HighD for {epochs} epochs "
          f"({train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% val)\n")

    # --- Epoch Loop ---
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
            
            # Teacher forcing only for transformer in first epoch
            teacher_forced = (model.__class__.__name__ == "ImprovedTrajectoryTransformer" and epoch == 1)
            pred = forward_model(model, obs, nd, ns, lane, teacher_forced=teacher_forced)
            
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
                pred = forward_model(model, obs, nd, ns, lane)
                loss, _ = combined_loss(pred, fut, w_pos=1.0, w_vel=0.3, w_acc=0.1)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        # ---- Evaluation Metrics (with accuracy/precision) ----
        train_ADE, train_FDE, _, _, train_metrics = evaluate(
            model, train_loader, "Train", compute_detailed_metrics=True
        )
        val_ADE, val_FDE, _, _, val_metrics = evaluate(
            model, val_loader, "Val", compute_detailed_metrics=True
        )
        
        # Store metrics in history
        history["train_ADE"].append(train_ADE)
        history["train_FDE"].append(train_FDE)
        history["val_ADE"].append(val_ADE)
        history["val_FDE"].append(val_FDE)
        history["train_accuracy_1m"].append(train_metrics.get('accuracy_1.0m', 0))
        history["val_accuracy_1m"].append(val_metrics.get('accuracy_1.0m', 0))
        history["train_precision_1m"].append(train_metrics.get('precision_1.0m', 0))
        history["val_precision_1m"].append(val_metrics.get('precision_1.0m', 0))
        
        print(f"Epoch {epoch}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"           TrainADE={train_ADE:.3f}, ValADE={val_ADE:.3f}, ValFDE={val_FDE:.3f}")

        # ---- Early Stopping ----
        if val_ADE < best_ade:
            best_ade, wait = val_ADE, 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_type}.pt"))
            print(f"✓ Best model saved (ADE: {best_ade:.3f})")
        else:
            wait += 1
        if wait >= patience:
            print(f"⏸ Early stopping triggered after {epoch} epochs")
            break

        # ---- Visualization ----
        if epoch in {1, epochs // 2, epochs} or wait == 0:
            try:
                obs_np = obs[0, :, :2].cpu().numpy()
                gt_np  = fut[0].cpu().numpy()
                pred_np = pred[0].detach().cpu().numpy()
                
                plt.figure(figsize=(8, 6))
                plt.plot(obs_np[:, 0], obs_np[:, 1], "ko-", label="Observed", linewidth=2)
                plt.plot(gt_np[:, 0], gt_np[:, 1], "g-", label="Ground Truth", linewidth=2)
                plt.plot(pred_np[:, 0], pred_np[:, 1], "r--", label="Predicted", linewidth=2)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.axis("equal")
                plt.title(f"{model_type.upper()} - Epoch {epoch}", fontsize=14)
                plt.xlabel("X (m)", fontsize=12)
                plt.ylabel("Y (m)", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_viz.png"), dpi=150)
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create visualization: {e}")

    # ---- Plot Training History ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plots
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].plot(history["val_loss"], label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Training Loss")
    
    # ADE/FDE plots
    axes[0, 1].plot(history["train_ADE"], label="Train ADE", color='blue')
    axes[0, 1].plot(history["val_ADE"], label="Val ADE", color='cyan')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("ADE (m)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title("Average Displacement Error")
    
    axes[0, 2].plot(history["train_FDE"], label="Train FDE", color='red')
    axes[0, 2].plot(history["val_FDE"], label="Val FDE", color='orange')
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("FDE (m)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_title("Final Displacement Error")
    
    # Accuracy plots
    axes[1, 0].plot(history["train_accuracy_1m"], label="Train Acc@1m", color='green')
    axes[1, 0].plot(history["val_accuracy_1m"], label="Val Acc@1m", color='lime')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title("Accuracy @ 1m Threshold")
    
    # Precision plots
    axes[1, 1].plot(history["train_precision_1m"], label="Train Prec@1m", color='purple')
    axes[1, 1].plot(history["val_precision_1m"], label="Val Prec@1m", color='magenta')
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title("Precision @ 1m Threshold")
    
    # Combined metrics
    axes[1, 2].plot(history["val_ADE"], label="Val ADE", color='blue')
    axes[1, 2].plot(history["val_FDE"], label="Val FDE", color='red')
    ax2 = axes[1, 2].twinx()
    ax2.plot(history["val_accuracy_1m"], label="Val Acc@1m", color='green', linestyle='--')
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Error (m)")
    ax2.set_ylabel("Accuracy")
    axes[1, 2].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_title("Combined Metrics")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()

    # ---- Final Evaluation ----
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, f"best_{model_type}.pt"), 
                                     map_location=device))
    model.eval()
    
    # Final comprehensive evaluation
    final_ADE, final_FDE, preds, gts, final_metrics = evaluate(
        model, val_loader, "Final Val", compute_detailed_metrics=True
    )

    # Compute additional comprehensive metrics
    comprehensive_metrics = compute_comprehensive_metrics(preds, gts)
    
    # Merge all metrics
    metrics = {
        "model_type": model_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs_trained": epoch,
        "best_val_ADE": best_ade,
        "final_ADE": final_ADE,
        "final_FDE": final_FDE,
        **final_metrics,
        **comprehensive_metrics,
        "training_time_epochs": epoch,
    }
    
    # Add final training metrics
    if len(history["train_ADE"]) > 0:
        metrics["final_train_ADE"] = history["train_ADE"][-1]
        metrics["final_train_FDE"] = history["train_FDE"][-1]
        metrics["final_train_accuracy_1m"] = history["train_accuracy_1m"][-1]
        metrics["final_train_precision_1m"] = history["train_precision_1m"][-1]
    
    print("\n" + json.dumps(metrics, indent=2))
    save_json(metrics, os.path.join(save_dir, "metrics.json"))
    save_json(history, os.path.join(save_dir, "training_history.json"))
    
    # Create comprehensive evaluation plots
    plot_error_distribution(preds, gts, save_dir)
    plot_diverse_samples(preds, gts, [np.zeros((obs_len, 2))] * len(preds), save_dir)
    
    print(f"\n✓ All evaluation artifacts saved to: {save_dir}")
    print("="*60 + "\n")

    return model, metrics


# -------------------- Comparison Utility --------------------
def compare_models(csv_path, save_dir="./results_comparison", models=None, 
                  train_ratio=0.8, epochs=10):
    """
    Train and compare multiple models on the same dataset.
    
    Args:
        csv_path: Path to dataset
        save_dir: Directory to save comparison results
        models: List of model types to compare. If None, compares all.
        train_ratio: Train/val split ratio
        epochs: Number of epochs to train each model
    """
    if models is None:
        models = ["lstm", "slstm", "cslstm", "social-gan", "gnn", "transformer"]
    
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    print("\n" + "="*60)
    print(f"COMPARING {len(models)} MODELS")
    print("="*60 + "\n")
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print('='*60 + "\n")
        
        model_dir = os.path.join(save_dir, model_type)
        try:
            _, metrics = train_highd(
                csv_path, 
                save_dir=model_dir,
                model_type=model_type,
                train_ratio=train_ratio,
                epochs=epochs
            )
            results[model_type] = metrics
        except Exception as e:
            print(f"\n⚠ Error training {model_type}: {e}\n")
            results[model_type] = {"error": str(e)}
    
    # Create comprehensive comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60 + "\n")
    
    comparison_df = pd.DataFrame([
        {
            "Model": model,
            "ADE": results[model].get("final_ADE", float('nan')),
            "FDE": results[model].get("final_FDE", float('nan')),
            "Acc@0.5m": results[model].get("accuracy_0.5m", float('nan')),
            "Acc@1m": results[model].get("accuracy_1.0m", float('nan')),
            "Acc@2m": results[model].get("accuracy_2.0m", float('nan')),
            "Prec@1m": results[model].get("precision_1.0m", float('nan')),
            "F1": results[model].get("f1_score", float('nan')),
            "Success": results[model].get("success_rate_2m", float('nan')),
            "Params": results[model].get("total_params", 0),
            "Epochs": results[model].get("epochs_trained", 0)
        }
        for model in models if "error" not in results[model]
    ])
    
    comparison_df = comparison_df.sort_values("ADE")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(save_dir, "comparison_table.csv"), index=False)
    
    # Plot comprehensive comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    x = range(len(comparison_df))
    models_list = comparison_df["Model"].tolist()
    
    # ADE
    axes[0, 0].bar(x, comparison_df["ADE"], color='blue', alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_list, rotation=45)
    axes[0, 0].set_ylabel("ADE (m)")
    axes[0, 0].set_title("Average Displacement Error")
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # FDE
    axes[0, 1].bar(x, comparison_df["FDE"], color='red', alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models_list, rotation=45)
    axes[0, 1].set_ylabel("FDE (m)")
    axes[0, 1].set_title("Final Displacement Error")
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Accuracy @ 1m
    axes[0, 2].bar(x, comparison_df["Acc@1m"], color='green', alpha=0.7)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models_list, rotation=45)
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].set_title("Accuracy @ 1m Threshold")
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_ylim([0, 1])
    
    # Precision @ 1m
    axes[1, 0].bar(x, comparison_df["Prec@1m"], color='purple', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models_list, rotation=45)
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision @ 1m Threshold")
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    
    # Model Size
    axes[1, 1].bar(x, comparison_df["Params"] / 1000, color='orange', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models_list, rotation=45)
    axes[1, 1].set_ylabel("Parameters (K)")
    axes[1, 1].set_title("Model Size")
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Success Rate
    axes[1, 2].bar(x, comparison_df["Success"], color='cyan', alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models_list, rotation=45)
    axes[1, 2].set_ylabel("Success Rate")
    axes[1, 2].set_title("Success Rate (FDE < 2m)")
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()
    
    save_json(results, os.path.join(save_dir, "all_results.json"))
    print(f"\n✓ Comparison results saved to: {save_dir}\n")
    
    return results


# -------------------- Main Entry --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train trajectory prediction models on HighD dataset")
    parser.add_argument("tracks_csv", help="Path to CSV file or directory containing *_tracks.csv files")
    parser.add_argument("--model_type", 
                       choices=["transformer", "slstm", "lstm", "cslstm", "social-gan", "gnn"],
                       default="transformer",
                       help="Model architecture to use")
    parser.add_argument("--save_dir", default="./results/whole_dataset_lstm",
                       help="Directory to save results")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Train/validation split ratio")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all models")
    parser.add_argument("--compare_models", nargs="+",
                       help="Specific models to compare (e.g., lstm slstm transformer)")
    args = parser.parse_args()

    if args.compare or args.compare_models:
        compare_models(
            args.tracks_csv,
            save_dir=args.save_dir,
            models=args.compare_models,
            train_ratio=args.train_ratio,
            epochs=args.epochs
        )
    else:
        train_highd(
            args.tracks_csv,
            save_dir=args.save_dir,
            model_type=args.model_type,
            train_ratio=args.train_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )