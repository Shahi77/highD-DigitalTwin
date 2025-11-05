# diagnose_model.py
"""
Quick diagnosis script to check if model has collapsed
and provide recommendations
"""
import torch
import numpy as np
import pandas as pd
from models import ImprovedTrajectoryTransformer
from highd_dataloader import make_dataloader_fixed as make_dataloader


def diagnose_model(model_path, tracks_csv, n_samples=100):
    """
    Quick diagnosis of trained model
    """
    print("\n" + "="*70)
    print(" MODEL DIAGNOSIS")
    print("="*70)
    
    # Load data
    df = pd.read_csv(tracks_csv)
    df['xVelocity'] = df['xVelocity'].clip(-50, 50)
    df['yVelocity'] = df['yVelocity'].clip(-50, 50)
    df['xAcceleration'] = df['xAcceleration'].clip(-10, 10)
    df['yAcceleration'] = df['yAcceleration'].clip(-10, 10)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedTrajectoryTransformer(
        d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get loader
    loader = make_dataloader(df, batch_size=1, shuffle=False,
                                      obs_len=10, pred_len=25, downsample=5)
    
    preds, gts = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            target = batch['target'].to(device)
            neigh_dyn = batch['neigh_dyn'].to(device)
            neigh_spatial = batch['neigh_spatial'].to(device)
            lane = batch['lane'].to(device)
            gt = batch['gt'][0].cpu().numpy()
            
            pred = model(target, neigh_dyn, neigh_spatial, lane)[0].cpu().numpy()
            preds.append(pred)
            gts.append(gt)
    
    preds = np.array(preds)
    gts = np.array(gts)
    
    # Diagnostics
    print(f"\n Statistics on {n_samples} samples:\n")
    
    # 1. Prediction variance
    pred_var = preds.var(axis=0).mean()
    gt_var = gts.var(axis=0).mean()
    var_ratio = pred_var / (gt_var + 1e-8)
    
    print(f"1. VARIANCE CHECK:")
    print(f"Predicted variance: {pred_var:.6f}")
    print(f"Ground truth variance: {gt_var:.6f}")
    print(f"Ratio: {var_ratio:.4f}")
    
    if var_ratio < 0.1:
        print(f" CRITICAL: Model has collapsed! Predictions are nearly constant.")
        status = "COLLAPSED"
    elif var_ratio < 0.5:
        print(f" WARNING: Low diversity in predictions")
        status = "LOW_DIVERSITY"
    else:
        print(f"Variance looks healthy")
        status = "HEALTHY"
    
    # 2. Mean prediction magnitude
    pred_mean_disp = np.abs(preds).mean()
    gt_mean_disp = np.abs(gts).mean()
    
    print(f"\n2. DISPLACEMENT MAGNITUDE:")
    print(f"Predicted mean |displacement|: {pred_mean_disp:.4f} m")
    print(f"Ground truth mean |displacement|: {gt_mean_disp:.4f} m")
    
    if pred_mean_disp < 1.0:
        print(f" Predictions are very small (model predicting minimal motion)")
    
    # 3. Direction check
    pred_final = preds[:, -1, :]
    gt_final = gts[:, -1, :]
    
    pred_angles = np.arctan2(pred_final[:, 1], pred_final[:, 0])
    gt_angles = np.arctan2(gt_final[:, 1], gt_final[:, 0])
    
    angle_std_pred = np.std(pred_angles)
    angle_std_gt = np.std(gt_angles)
    
    print(f"\n3. DIRECTION DIVERSITY:")
    print(f"   Predicted angle std: {np.degrees(angle_std_pred):.2f}°")
    print(f"   Ground truth angle std: {np.degrees(angle_std_gt):.2f}°")
    
    if angle_std_pred < 0.1:
        print(f"All predictions point in nearly same direction!")
    
    # 4. Errors
    errors = np.linalg.norm(preds - gts, axis=-1)
    ade = errors.mean()
    fde = errors[:, -1].mean()
    
    print(f"\n4. ERRORS:")
    print(f"ADE: {ade:.4f} m")
    print(f"FDE: {fde:.4f} m")
    
    # Final diagnosis
    print(f"\n" + "="*70)
    print(f" DIAGNOSIS: {status}")
    print(f"="*70)
    
    if status == "COLLAPSED":
        print(f"\n MODEL HAS COLLAPSED - Urgent fixes needed:\n")
        print(f"  1. Add diversity loss to training (implemented in train_improved.py)")
        print(f"  2. Increase learning rate slightly (try 5e-4)")
        print(f"  3. Reduce regularization (lower weight_decay)")
        print(f"  4. Add noise augmentation to inputs")
        print(f"  5. Check if model outputs are clipped/saturated")
        print(f"\n  CAN NOT PROCEED TO SUMO - Retrain first!")
        
    elif status == "LOW_DIVERSITY":
        print(f"\n LOW DIVERSITY - Improvements recommended:\n")
        print(f"  1. Retrain with improved loss function")
        print(f"  2. Test on lane-change & turn scenarios specifically")
        print(f"  3. Add data augmentation")
        print(f"\n Can proceed to SUMO cautiously, but expect issues with complex maneuvers")
        
    else:
        print(f"\n MODEL LOOKS HEALTHY\n")
        print(f"  Recommendations:")
        print(f"  1. Test on other HighD scenes (02-60)")
        print(f"  2. Evaluate on lane-change specific samples")
        print(f"  3. Check performance degradation over longer horizons")
        print(f"\n  Ready to proceed to SUMO integration!")
    
    print(f"\n" + "="*70)
    
    return {
        'status': status,
        'variance_ratio': var_ratio,
        'ade': ade,
        'fde': fde,
        'pred_var': pred_var,
        'gt_var': gt_var
    }

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python diagnose_model.py path/to/model.pt path/to/tracks.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    tracks_csv = sys.argv[2]
    
    results = diagnose_model(model_path, tracks_csv, n_samples=200)