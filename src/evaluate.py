# evaluate_comprehensive.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import json

from models import ImprovedTrajectoryTransformer
from highd_dataloader import make_dataloader_fixed as make_dataloader
from utils import ade_fde

def compute_comprehensive_metrics(preds, gts):
    """
    Compute comprehensive metrics beyond ADE/FDE
    preds, gts: (N, T, 2) numpy arrays
    """
    assert preds.shape == gts.shape
    N, T, _ = preds.shape
    
    # 1. ADE & FDE (standard)
    dists = np.linalg.norm(preds - gts, axis=-1)  # (N, T)
    ADE = dists.mean()
    FDE = dists[:, -1].mean()
    
    # 2. MAE (Mean Absolute Error) - per coordinate
    mae_x = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    mae_y = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()
    MAE = (mae_x + mae_y) / 2
    
    # 3. RMSE (Root Mean Square Error)
    mse = ((preds - gts) ** 2).mean()
    RMSE = np.sqrt(mse)
    
    # 4. Velocity consistency (check if predicted motion is realistic)
    pred_vel = np.diff(preds, axis=1)  # (N, T-1, 2)
    gt_vel = np.diff(gts, axis=1)
    vel_error = np.linalg.norm(pred_vel - gt_vel, axis=-1).mean()
    
    # 5. Acceleration consistency
    if T > 2:
        pred_acc = np.diff(pred_vel, axis=1)
        gt_acc = np.diff(gt_vel, axis=1)
        acc_error = np.linalg.norm(pred_acc - gt_acc, axis=-1).mean()
    else:
        acc_error = 0.0
    
    # 6. Direction error (heading consistency)
    pred_angles = np.arctan2(pred_vel[:, :, 1], pred_vel[:, :, 0])
    gt_angles = np.arctan2(gt_vel[:, :, 1], gt_vel[:, :, 0])
    angle_diff = np.abs(pred_angles - gt_angles)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # wrap
    direction_error = np.degrees(angle_diff.mean())
    
    # 7. Miss Rate (how many predictions are > 2m from ground truth at final step)
    miss_threshold = 2.0  # meters
    miss_rate = (dists[:, -1] > miss_threshold).mean() * 100
    
    # 8. Lateral vs Longitudinal error breakdown
    # Assume x is longitudinal (highway direction), y is lateral
    lon_error = np.abs(preds[:, :, 0] - gts[:, :, 0]).mean()
    lat_error = np.abs(preds[:, :, 1] - gts[:, :, 1]).mean()
    
    # 9. Per-timestep error (for plotting error evolution)
    per_step_error = dists.mean(axis=0)  # (T,)
    
    # 10. Model collapse detection (variance check)
    pred_variance = preds.var(axis=0).mean()  # should be non-zero
    gt_variance = gts.var(axis=0).mean()
    variance_ratio = pred_variance / (gt_variance + 1e-8)
    
    metrics = {
        'ADE': float(ADE),
        'FDE': float(FDE),
        'MAE': float(MAE),
        'RMSE': float(RMSE),
        'Velocity_Error': float(vel_error),
        'Acceleration_Error': float(acc_error),
        'Direction_Error_deg': float(direction_error),
        'Miss_Rate_%': float(miss_rate),
        'Longitudinal_Error': float(lon_error),
        'Lateral_Error': float(lat_error),
        'Variance_Ratio': float(variance_ratio),
        'Per_Step_Error': per_step_error.tolist()
    }
    
    return metrics

def save_predictions_csv(preds, gts, meta_list, save_path='predictions.csv'):
    """
    Save predictions vs ground truth to CSV for detailed analysis
    """
    rows = []
    for i, (pred, gt, meta) in enumerate(zip(preds, gts, meta_list)):
        for t in range(len(pred)):
            rows.append({
                'sample_idx': i,
                'vehicle_id': meta['vid'],
                'timestep': t,
                'pred_x': pred[t, 0],
                'pred_y': pred[t, 1],
                'gt_x': gt[t, 0],
                'gt_y': gt[t, 1],
                'error': np.linalg.norm(pred[t] - gt[t])
            })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions CSV to {save_path}")
    return df

def plot_error_distribution(preds, gts, save_dir='eval_plots'):
    """
    Plot error distributions and diagnostics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    dists = np.linalg.norm(preds - gts, axis=-1)  # (N, T)
    final_errors = dists[:, -1]
    
    # 1. Error histogram
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Final Displacement Error (m)')
    plt.ylabel('Frequency')
    plt.title('FDE Distribution')
    plt.axvline(final_errors.mean(), color='r', linestyle='--', label=f'Mean: {final_errors.mean():.2f}m')
    plt.legend()
    
    # 2. Error over time
    plt.subplot(1, 2, 2)
    per_step_error = dists.mean(axis=0)
    timesteps = np.arange(len(per_step_error)) * 0.2  # 5Hz = 0.2s
    plt.plot(timesteps, per_step_error, 'b-', linewidth=2)
    plt.fill_between(timesteps, 
                     dists.min(axis=0), 
                     dists.max(axis=0), 
                     alpha=0.2)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement Error (m)')
    plt.title('Error Evolution Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150)
    plt.close()
    
    # 3. Lateral vs Longitudinal error
    plt.figure(figsize=(8, 6))
    lon_errors = np.abs(preds[:, :, 0] - gts[:, :, 0]).flatten()
    lat_errors = np.abs(preds[:, :, 1] - gts[:, :, 1]).flatten()
    
    plt.scatter(lon_errors, lat_errors, alpha=0.1, s=1)
    plt.xlabel('Longitudinal Error (m)')
    plt.ylabel('Lateral Error (m)')
    plt.title('Longitudinal vs Lateral Error')
    plt.plot([0, lon_errors.max()], [0, lon_errors.max()], 'r--', alpha=0.5, label='x=y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'lon_vs_lat_error.png'), dpi=150)
    plt.close()
    
    # 4. Prediction variance check (model collapse detection)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    pred_std_x = preds[:, :, 0].std(axis=0)
    gt_std_x = gts[:, :, 0].std(axis=0)
    plt.plot(timesteps, pred_std_x, 'r-', label='Predicted X std', linewidth=2)
    plt.plot(timesteps, gt_std_x, 'g-', label='Ground Truth X std', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m)')
    plt.title('X-axis Variance (Model Collapse Check)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    pred_std_y = preds[:, :, 1].std(axis=0)
    gt_std_y = gts[:, :, 1].std(axis=0)
    plt.plot(timesteps, pred_std_y, 'r-', label='Predicted Y std', linewidth=2)
    plt.plot(timesteps, gt_std_y, 'g-', label='Ground Truth Y std', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Standard Deviation (m)')
    plt.title('Y-axis Variance (Model Collapse Check)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variance_check.png'), dpi=150)
    plt.close()
    
    print(f"Saved diagnostic plots to {save_dir}/")

def plot_diverse_samples(preds, gts, obs_list, save_dir='eval_plots', n_samples=10):
    """
    Plot diverse prediction samples (worst, best, median)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute final errors
    final_errors = np.linalg.norm(preds[:, -1, :] - gts[:, -1, :], axis=-1)
    
    # Find indices: worst, median, best
    worst_idx = np.argsort(final_errors)[-n_samples//3:]
    median_idx = np.argsort(final_errors)[len(final_errors)//2 - n_samples//3: len(final_errors)//2]
    best_idx = np.argsort(final_errors)[:n_samples//3]
    
    categories = [
        ('worst', worst_idx),
        ('median', median_idx),
        ('best', best_idx)
    ]
    
    for cat_name, indices in categories:
        fig, axes = plt.subplots(1, len(indices), figsize=(5*len(indices), 4))
        if len(indices) == 1:
            axes = [axes]
        
        for ax_idx, sample_idx in enumerate(indices):
            ax = axes[ax_idx]
            obs = obs_list[sample_idx][:, :2]
            gt = gts[sample_idx]
            pred = preds[sample_idx]
            error = final_errors[sample_idx]
            
            ax.plot(obs[:, 0], obs[:, 1], 'bo-', label='Observed', markersize=3, alpha=0.7)
            ax.plot(gt[:, 0], gt[:, 1], 'g-', label='Ground Truth', linewidth=2)
            ax.plot(pred[:, 0], pred[:, 1], 'r--', label='Predicted', linewidth=2)
            ax.scatter([0], [0], c='orange', s=100, marker='X', zorder=5)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'FDE={error:.2f}m')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
        
        plt.suptitle(f'{cat_name.upper()} Predictions', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_{cat_name}.png'), dpi=150)
        plt.close()

def evaluate_model_comprehensive(model, tracks_df, n_samples=1000, save_dir='eval_results'):
    """
    Comprehensive evaluation with all metrics and visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    loader = make_dataloader(
        tracks_df, batch_size=1, shuffle=False,
        obs_len=10, pred_len=25, downsample=5, k_neighbors=8
    )
    
    preds_all, gts_all, obs_all, meta_all = [], [], [], []
    
    print(f"Evaluating on {min(n_samples, len(loader))} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=min(n_samples, len(loader)))):
            if i >= n_samples:
                break
            
            target = batch['target'].to(next(model.parameters()).device)
            neigh_dyn = batch['neigh_dyn'].to(next(model.parameters()).device)
            neigh_spatial = batch['neigh_spatial'].to(next(model.parameters()).device)
            lane = batch['lane'].to(next(model.parameters()).device)
            gt = batch['gt'][0].cpu().numpy()
            
            pred = model(target, neigh_dyn, neigh_spatial, lane)[0].cpu().numpy()
            
            preds_all.append(pred)
            gts_all.append(gt)
            obs_all.append(target[0].cpu().numpy())
            meta_all.append(batch['meta'][0])
    
    preds_all = np.array(preds_all)
    gts_all = np.array(gts_all)
    obs_all = np.array(obs_all)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    # Compute all metrics
    metrics = compute_comprehensive_metrics(preds_all, gts_all)
    
    # Print metrics
    print(f"\n Standard Metrics:")
    print(f"  ADE: {metrics['ADE']:.4f} m")
    print(f"  FDE: {metrics['FDE']:.4f} m")
    print(f"  MAE: {metrics['MAE']:.4f} m")
    print(f"  RMSE: {metrics['RMSE']:.4f} m")
    
    print(f"\n Motion Metrics:")
    print(f"  Velocity Error: {metrics['Velocity_Error']:.4f} m/s")
    print(f"  Acceleration Error: {metrics['Acceleration_Error']:.4f} m/s²")
    print(f"  Direction Error: {metrics['Direction_Error_deg']:.2f}°")
    
    print(f"\n Spatial Breakdown:")
    print(f"  Longitudinal Error: {metrics['Longitudinal_Error']:.4f} m")
    print(f"  Lateral Error: {metrics['Lateral_Error']:.4f} m")
    
    print(f"\n  Quality Checks:")
    print(f"  Miss Rate (>2m): {metrics['Miss_Rate_%']:.2f}%")
    print(f"  Variance Ratio: {metrics['Variance_Ratio']:.4f}")
    
    # WARNING: Check for model collapse
    if metrics['Variance_Ratio'] < 0.1:
        print("\n WARNING: MODEL COLLAPSE DETECTED!")
        print("   Predicted trajectories have very low variance.")
        print("   Model is likely predicting near-constant values.")
        print("   → Need to increase diversity in predictions!")
    elif metrics['Variance_Ratio'] < 0.5:
        print("\n  WARNING: Low prediction diversity")
        print("   Consider adding diversity loss or increasing model capacity")
    else:
        print("\n Prediction diversity looks healthy")
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualizations
    print(f"\nGenerating diagnostic plots...")
    plot_error_distribution(preds_all, gts_all, save_dir)
    plot_diverse_samples(preds_all, gts_all, obs_all, save_dir, n_samples=9)
    
    # Save predictions CSV
    print(f"\nSaving predictions CSV...")
    pred_df = save_predictions_csv(preds_all, gts_all, meta_all, 
                                   save_path=os.path.join(save_dir, 'predictions.csv'))
    
    # Compare with paper benchmarks
    print(f"\n" + "="*60)
    print("COMPARISON WITH PAPER (INTERACTION Dataset)")
    print("="*60)
    paper_ade_interaction = 1.5  # approximate from paper
    paper_fde_interaction = 3.0
    
    print(f"  Paper ADE: ~{paper_ade_interaction:.2f}m  |  Your ADE: {metrics['ADE']:.4f}m")
    print(f"  Paper FDE: ~{paper_fde_interaction:.2f}m  |  Your FDE: {metrics['FDE']:.4f}m")
    
    if metrics['ADE'] < paper_ade_interaction:
        print(f"\n Your ADE is better (but HighD is simpler than INTERACTION)")
    
    print(f"\nNotes:")
    print(f"  - HighD (highway) is simpler than INTERACTION (urban)")
    print(f"  - Your low errors might indicate straight-line predictions")
    print(f"  - Check variance ratio and visual plots for true performance")
    
    print(f"\n" + "="*60)
    print(f"Evaluation complete! Results saved to {save_dir}/")
    print("="*60)
    
    return metrics, pred_df

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate_comprehensive.py path/to/tracks.csv path/to/model.pt")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    model_path = sys.argv[2]
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Normalize same as training
    df['xVelocity'] = df['xVelocity'].clip(-50, 50)
    df['yVelocity'] = df['yVelocity'].clip(-50, 50)
    df['xAcceleration'] = df['xAcceleration'].clip(-10, 10)
    df['yAcceleration'] = df['yAcceleration'].clip(-10, 10)
    
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    evaluate_model_comprehensive(model, df, n_samples=1000, save_dir='eval_results')