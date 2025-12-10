"""
Visualization using ACTUAL training data format
This bypasses the conversion and uses data exactly as the model was trained
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import sys

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from highd_dataloader import make_dataloader_highd

# Configuration
MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
MODEL_TYPE = "slstm"
HIGHD_CSV = "./data/highd/dataset/02_tracks.csv"  # Original highD data
SAVE_DIR = "./prediction_visualizations"
NUM_VIZ = 15

device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")

print(f"Using device: {device}\n")


def load_model(model_path, model_type="slstm", pred_len=25):
    """Load trained model"""
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded {model_type.upper()} model\n")
    return model


def visualize_trajectory(obs, pred, gt, vehicle_id, lane_id, save_path):
    """Create visualization"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Draw road
    road_width = 30
    lane_width = 10
    ax.add_patch(patches.Rectangle(
        (obs[0, 0] - 50, -road_width/2), 
        obs[-1, 0] - obs[0, 0] + pred[-1, 0] - obs[-1, 0] + 100, 
        road_width,
        facecolor='#7f8c8d', edgecolor='black', linewidth=2
    ))
    
    # Lane markings
    x_start = obs[0, 0] - 50
    x_end = pred[-1, 0] + 50
    for i in range(1, 3):
        y_pos = -road_width/2 + i * lane_width
        n_dashes = 20
        dash_length = (x_end - x_start) / (2 * n_dashes)
        for j in range(n_dashes):
            x_dash = x_start + j * 2 * dash_length
            ax.plot([x_dash, x_dash + dash_length], [y_pos, y_pos], 
                   'w--', linewidth=2, alpha=0.8)
    
    lane_y = -road_width/2 + (lane_id + 0.5) * lane_width
    
    # Plot trajectories
    obs_x = obs[:, 0]
    obs_y = obs[:, 1] if obs.shape[1] > 1 else np.full_like(obs_x, lane_y)
    ax.plot(obs_x, obs_y, 'o-', color='#f1c40f', linewidth=3, 
           markersize=6, label='Observation', zorder=5)
    
    gt_x = np.concatenate([obs[-1:, 0], gt[:, 0]])
    gt_y = np.concatenate([obs[-1:, 1] if obs.shape[1] > 1 else [lane_y], 
                          gt[:, 1] if gt.shape[1] > 1 else np.full(len(gt), lane_y)])
    ax.plot(gt_x, gt_y, 'o-', color='#e74c3c', linewidth=3, 
           markersize=6, label='Ground Truth', zorder=4, linestyle='--', dashes=(5, 3))
    
    pred_x = np.concatenate([obs[-1:, 0], pred[:, 0]])
    pred_y = np.concatenate([obs[-1:, 1] if obs.shape[1] > 1 else [lane_y], pred[:, 1]])
    ax.plot(pred_x, pred_y, 's-', color='#2ecc71', linewidth=3, 
           markersize=5, label='Prediction', zorder=6)
    
    # Vehicle boxes
    vehicle_length, vehicle_width = 4.5, 2.0
    for (x, y, color, alpha) in [
        (obs[-1, 0], obs_y[-1], '#3498db', 1.0),
        (pred[-1, 0], pred_y[-1], '#2ecc71', 0.6),
        (gt[-1, 0], gt_y[-1], '#e74c3c', 0.6)
    ]:
        ax.add_patch(patches.Rectangle(
            (x - vehicle_length/2, y - vehicle_width/2),
            vehicle_length, vehicle_width,
            facecolor=color, edgecolor='black', linewidth=1.5, 
            alpha=alpha, zorder=7
        ))
    
    # Styling
    ax.set_xlim(obs[0, 0] - 50, pred[-1, 0] + 50)
    ax.set_ylim(-road_width/2 - 5, road_width/2 + 5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'Vehicle {vehicle_id} - Trajectory Prediction', 
                fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    """Generate visualizations from actual dataloader"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load model
    model = load_model(MODEL_PATH, MODEL_TYPE, pred_len=25)
    
    # Load data using the SAME dataloader used during training
    print(f"Loading data from {HIGHD_CSV}...")
    try:
        dataloader = make_dataloader_highd(
            HIGHD_CSV, 
            batch_size=1,  # One at a time for visualization
            obs_len=20, 
            pred_len=25,
            shuffle=False
        )
        print(f"Loaded dataloader\n")
    except Exception as e:
        print(f"✗ Error loading dataloader: {e}")
        print("Trying directory path...")
        dataloader = make_dataloader_highd(
            "./data/highd/dataset/",
            batch_size=1,
            obs_len=20,
            pred_len=25,
            shuffle=False
        )
    
    print(f"Generating {NUM_VIZ} visualizations...\n")
    
    viz_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if viz_count >= NUM_VIZ:
            break
        
        try:
            # Get data from batch (already in correct format!)
            obs = batch["target"].to(device)  # [1, obs_len, features]
            gt = batch["gt"].to(device)  # [1, pred_len, 2]
            nd = batch["neighbors_dyn"].to(device)
            ns = batch["neighbors_spatial"].to(device)
            lane = batch["lane"].to(device)
            
            # Make prediction
            with torch.no_grad():
                if hasattr(model, "multi_att"):
                    last_obs_pos = obs[:, -1, :2]
                    pred = model(obs, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = model(obs, nd, ns, lane)
            
            # Convert to numpy
            obs_np = obs[0].cpu().numpy()  # Take only x, y
            gt_np = gt[0].cpu().numpy()
            pred_np = pred[0].cpu().numpy()
            
            # Get lane info (from observation or default)
            lane_id = int(obs[0, 0, 6].item()) if obs.shape[2] > 6 else 1
            
            # Save visualization
            save_path = os.path.join(SAVE_DIR, f'trajectory_batch_{batch_idx}.png')
            visualize_trajectory(obs_np, pred_np, gt_np, batch_idx, lane_id, save_path)
            
            viz_count += 1
            print(f"[{viz_count}/{NUM_VIZ}] Generated: trajectory_batch_{batch_idx}.png")
            
        except Exception as e:
            print(f"Error with batch {batch_idx}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated {viz_count} trajectory visualizations")
    print(f"Saved to: {SAVE_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()