# train_eval.py
"""
FINAL FIXED VERSION with proper relative coordinates and comprehensive monitoring
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import ImprovedTrajectoryTransformer
from highd_dataloader import make_dataloader_fixed
from utils import ade_fde

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def trajectory_loss_improved(pred, gt, lambda_vel=0.5, lambda_acc=0.2, lambda_diversity=0.1):
    """Enhanced loss with diversity enforcement"""
    pos_loss = nn.functional.mse_loss(pred, gt)
    
    if pred.shape[1] > 1:
        pred_vel = pred[:, 1:, :] - pred[:, :-1, :]
        gt_vel = gt[:, 1:, :] - gt[:, :-1, :]
        vel_loss = nn.functional.mse_loss(pred_vel, gt_vel)
    else:
        vel_loss = torch.tensor(0.0, device=pred.device)
    
    if pred.shape[1] > 2:
        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        gt_acc = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]
        acc_loss = nn.functional.mse_loss(pred_acc, gt_acc)
    else:
        acc_loss = torch.tensor(0.0, device=pred.device)
    
    pred_std = pred.std(dim=0).mean()
    gt_std = gt.std(dim=0).mean()
    diversity_loss = torch.abs(pred_std - gt_std)
    
    total_loss = pos_loss + lambda_vel * vel_loss + lambda_acc * acc_loss + lambda_diversity * diversity_loss
    
    return total_loss, {
        'pos': pos_loss.item(),
        'vel': vel_loss.item() if isinstance(vel_loss, torch.Tensor) else 0,
        'acc': acc_loss.item() if isinstance(acc_loss, torch.Tensor) else 0,
        'div': diversity_loss.item()
    }

def normalize_tracks(df):
    df = df.copy()
    df['xVelocity'] = df['xVelocity'].clip(-50, 50)
    df['yVelocity'] = df['yVelocity'].clip(-50, 50)
    df['xAcceleration'] = df['xAcceleration'].clip(-10, 10)
    df['yAcceleration'] = df['yAcceleration'].clip(-10, 10)
    return df

def plot_sample_improved(sample, pred_agent_frame, save_path, title="Prediction"):
    """Plot in world frame for interpretability"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Agent frame
    obs_agent = sample['target_feats'][:, :2]
    gt_agent = sample['gt']
    
    ax1.plot(obs_agent[:, 0], obs_agent[:, 1], 'bo-', label='Observed', markersize=4, alpha=0.7)
    ax1.plot(gt_agent[:, 0], gt_agent[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax1.plot(pred_agent_frame[:, 0], pred_agent_frame[:, 1], 'r--', label='Predicted', linewidth=2)
    ax1.scatter([0], [0], c='orange', s=100, marker='X', label='Agent Origin', zorder=5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Agent Frame (Relative)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Right: World frame
    meta = sample['meta']
    origin = meta['origin']
    yaw = meta['yaw']
    
    # Transform back to world
    c, s = np.cos(yaw), np.sin(yaw)
    def to_world(pts_agent):
        xw = c * pts_agent[:, 0] - s * pts_agent[:, 1] + origin[0]
        yw = s * pts_agent[:, 0] + c * pts_agent[:, 1] + origin[1]
        return np.stack([xw, yw], axis=-1)
    
    obs_world = meta['obs_world']
    gt_world = meta['fut_world'] + origin  # fut_world is already world coords
    pred_world = to_world(pred_agent_frame)
    
    ax2.plot(obs_world[:, 0], obs_world[:, 1], 'bo-', label='Observed', markersize=4, alpha=0.7)
    ax2.plot(gt_world[:, 0], gt_world[:, 1], 'g-', label='Ground Truth', linewidth=2)
    ax2.plot(pred_world[:, 0], pred_world[:, 1], 'r--', label='Predicted', linewidth=2)
    ax2.scatter([origin[0]], [origin[1]], c='orange', s=100, marker='X', label='Last Obs', zorder=5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("World Frame (Absolute)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_model_fixed(tracks_df, save_dir='checkpoints_fixed'):
    os.makedirs(save_dir, exist_ok=True)
    
    curriculum = [
        (10, 5e-4, 6),
        (15, 3e-4, 10),
        (25, 2e-4, 15)
    ]
    
    obs_len = 10
    batch_size = 32
    k_neighbors = 8
    
    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=k_neighbors
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_ade = float('inf')
    
    for stage, (pred_len, lr, epochs) in enumerate(curriculum):
        print(f"\n{'='*60}")
        print(f"Stage {stage+1}: pred_len={pred_len} ({pred_len/5:.1f}s), lr={lr:.0e}, epochs={epochs}")
        print(f"{'='*60}")
        
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        model.pred_len = pred_len
        loader = make_dataloader_fixed(
            tracks_df, batch_size=batch_size, shuffle=True,
            obs_len=obs_len, pred_len=pred_len, downsample=5, k_neighbors=k_neighbors
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
        
        for epoch in range(1, epochs+1):
            model.train()
            epoch_loss = 0.0
            loss_components = {'pos': 0, 'vel': 0, 'acc': 0, 'div': 0}
            
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
            
            for batch in pbar:
                target = batch['target'].to(device)
                neigh_dyn = batch['neigh_dyn'].to(device)
                neigh_spatial = batch['neigh_spatial'].to(device)
                lane = batch['lane'].to(device)
                gt = batch['gt'].to(device)
                
                optimizer.zero_grad()
                pred = model(target, neigh_dyn, neigh_spatial, lane, last_obs_pos=None, pred_len=pred_len)
                
                loss, loss_dict = trajectory_loss_improved(pred, gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                for k in loss_components:
                    loss_components[k] += loss_dict[k]
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pos': f'{loss_dict["pos"]:.3f}',
                    'div': f'{loss_dict["div"]:.3f}'
                })
            
            scheduler.step()
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch} avg_loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
            print(f"  Components: pos={loss_components['pos']/len(loader):.4f} "
                  f"vel={loss_components['vel']/len(loader):.4f} "
                  f"acc={loss_components['acc']/len(loader):.4f} "
                  f"div={loss_components['div']/len(loader):.4f}")
            
            # Evaluate
            if epoch % 2 == 0 or epoch == epochs:
                model.eval()
                eval_samples = 100
                preds_all, gts_all = [], []
                
                with torch.no_grad():
                    for i, s in enumerate(loader.dataset):
                        if i >= eval_samples:
                            break
                        target = torch.from_numpy(s['target_feats']).unsqueeze(0).to(device)
                        neigh_dyn = torch.from_numpy(s['neighbors_dyn']).unsqueeze(0).to(device)
                        neigh_spatial = torch.from_numpy(s['neighbors_spatial']).unsqueeze(0).to(device)
                        lane = torch.from_numpy(s['lane_feats']).unsqueeze(0).to(device)
                        gt = s['gt']
                        
                        pred = model(target, neigh_dyn, neigh_spatial, lane, last_obs_pos=None, pred_len=pred_len)
                        pred_np = pred[0].cpu().numpy()
                        
                        preds_all.append(pred_np)
                        gts_all.append(gt)
                
                preds_all = np.array(preds_all)
                gts_all = np.array(gts_all)
                ADE, FDE = ade_fde(preds_all, gts_all)
                
                # Diversity check
                pred_var = preds_all.var(axis=0).mean()
                gt_var = gts_all.var(axis=0).mean()
                var_ratio = pred_var / (gt_var + 1e-8)
                
                print(f"  Eval (n={eval_samples}): ADE={ADE:.4f}m FDE={FDE:.4f}m VarRatio={var_ratio:.3f}")
                
                if ADE < best_ade:
                    best_ade = ADE
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    print(f"  *** New best ADE: {best_ade:.4f}m ***")
                
                # Plot sample
                sample = loader.dataset[0]
                with torch.no_grad():
                    target = torch.from_numpy(sample['target_feats']).unsqueeze(0).to(device)
                    neigh_dyn = torch.from_numpy(sample['neighbors_dyn']).unsqueeze(0).to(device)
                    neigh_spatial = torch.from_numpy(sample['neighbors_spatial']).unsqueeze(0).to(device)
                    lane = torch.from_numpy(sample['lane_feats']).unsqueeze(0).to(device)
                    pred = model(target, neigh_dyn, neigh_spatial, lane, pred_len=pred_len)[0].cpu().numpy()
                
                plot_sample_improved(
                    sample, pred,
                    save_path=os.path.join(save_dir, f'stage{stage+1}_epoch{epoch:02d}.png'),
                    title=f"Stage {stage+1} Epoch {epoch} (ADE={ADE:.3f}m, Var={var_ratio:.2f})"
                )
        
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_stage{stage+1}.pt'))
    
    return model

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python train_fixed.py path/to/tracks.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    df = normalize_tracks(df)
    print(f"Loaded {len(df)} rows, {df['id'].nunique()} unique vehicles")
    
    print("\n  RETRAINING WITH FIXED DATALOADER")
    print("Key fix: Ground truth is now displacement from last observed position")
    print("This should resolve the underprediction issue.\n")
    
    model = train_model_fixed(df, save_dir='checkpoints_fixed')
    print("\n Training complete! Now run comprehensive evaluation.")

