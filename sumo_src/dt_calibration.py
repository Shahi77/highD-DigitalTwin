"""
Digital Twin Model Calibration Script
--------------------------------------
This script helps calibrate the model output to match expected physical scales.
Run this BEFORE the full simulation to find the right scaling factors.
"""

import sys, numpy as np, torch, pandas as pd
sys.path.append("/Users/shahi/Developer/Project-highD/src")

from models import SimpleSLSTM, ImprovedTrajectoryTransformer
from highd_dataloader import make_dataloader_highd

device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")


def analyze_training_data(data_path: str, num_samples: int = 100):
    """Analyze the training data to understand typical scales"""
    print("\n" + "="*70)
    print("ANALYZING TRAINING DATA")
    print("="*70)
    
    try:
        if data_path.endswith('.csv'):
            loader = make_dataloader_highd(data_path, batch_size=32, shuffle=False)
        else:
            raise ValueError("Provide path to tracks CSV file")
        
        all_gt = []
        all_obs = []
        
        for i, batch in enumerate(loader):
            if i >= num_samples // 32:
                break
            
            gt = batch['gt'].numpy()  # (B, pred_len, 2)
            obs = batch['target'].numpy()  # (B, obs_len, 7)
            
            all_gt.append(gt)
            all_obs.append(obs)
        
        all_gt = np.concatenate(all_gt, axis=0)
        all_obs = np.concatenate(all_obs, axis=0)
        
        # Analyze ground truth trajectories
        gt_final_displacements = np.linalg.norm(all_gt[:, -1, :], axis=1)
        gt_total_distances = np.sum(np.linalg.norm(np.diff(all_gt, axis=1), axis=2), axis=1)
        
        # Analyze observation velocities
        obs_velocities = np.linalg.norm(all_obs[:, :, 2:4], axis=2)
        
        print(f"\n📊 Training Data Statistics (n={len(all_gt)} samples):")
        print(f"\nGround Truth Trajectories (5 seconds @ 4Hz):")
        print(f"  Final displacement: {np.mean(gt_final_displacements):.2f} ± {np.std(gt_final_displacements):.2f} m")
        print(f"  Median: {np.median(gt_final_displacements):.2f} m")
        print(f"  Range: [{np.min(gt_final_displacements):.2f}, {np.max(gt_final_displacements):.2f}] m")
        print(f"  Total path length: {np.mean(gt_total_distances):.2f} ± {np.std(gt_total_distances):.2f} m")
        
        print(f"\nObservation Velocities:")
        print(f"  Mean velocity: {np.mean(obs_velocities):.2f} ± {np.std(obs_velocities):.2f} m/s")
        print(f"  Typical speed: {np.mean(obs_velocities) * 3.6:.1f} km/h")
        
        # Expected displacement for 5 seconds
        expected_displacement_5s = np.mean(obs_velocities) * 5.0
        print(f"\nExpected 5s displacement at mean velocity: {expected_displacement_5s:.2f} m")
        
        return {
            'gt_mean_final_disp': float(np.mean(gt_final_displacements)),
            'gt_median_final_disp': float(np.median(gt_final_displacements)),
            'expected_5s_disp': float(expected_displacement_5s),
            'mean_velocity': float(np.mean(obs_velocities))
        }
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None


def analyze_model_predictions(model_path: str, model_type: str, 
                              data_path: str, num_samples: int = 100):
    """Analyze what the model actually predicts"""
    print("\n" + "="*70)
    print("ANALYZING MODEL PREDICTIONS")
    print("="*70)
    
    # Load model
    pred_len = 20
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded {model_type.upper()} model")
    
    # Load data
    loader = make_dataloader_highd(data_path, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples // 32:
                break
            
            obs = batch['target'].to(device)
            gt = batch['gt'].to(device)
            nd = batch['neighbors_dyn'].to(device)
            ns = batch['neighbors_spatial'].to(device)
            lane = batch['lane'].to(device)
            
            if hasattr(model, "multi_att"):
                last_obs_pos = obs[:, -1, :2]
                pred = model(obs, nd, ns, lane, last_obs_pos=last_obs_pos)
            else:
                pred = model(obs, nd, ns, lane)
            
            all_predictions.append(pred.cpu().numpy())
            all_ground_truth.append(gt.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    
    # Analyze prediction scale
    pred_final_displacements = np.linalg.norm(all_predictions[:, -1, :], axis=1)
    gt_final_displacements = np.linalg.norm(all_ground_truth[:, -1, :], axis=1)
    
    # Calculate errors
    gt_aligned = all_ground_truth[:, :all_predictions.shape[1], :]
    errors = np.linalg.norm(all_predictions - gt_aligned, axis=2)
    ade = np.mean(errors)
    fde = np.mean(errors[:, -1])
    
    print(f"\n📊 Model Prediction Statistics (n={len(all_predictions)} samples):")
    print(f"\nPredicted Final Displacements:")
    print(f"  Mean: {np.mean(pred_final_displacements):.2f} ± {np.std(pred_final_displacements):.2f} m")
    print(f"  Median: {np.median(pred_final_displacements):.2f} m")
    print(f"  Range: [{np.min(pred_final_displacements):.2f}, {np.max(pred_final_displacements):.2f}] m")
    
    print(f"\nGround Truth Final Displacements:")
    print(f"  Mean: {np.mean(gt_final_displacements):.2f} ± {np.std(gt_final_displacements):.2f} m")
    
    print(f"\nTraining Set Performance:")
    print(f"  ADE: {ade:.3f} m")
    print(f"  FDE: {fde:.3f} m")
    
    # Calculate scale ratio
    scale_ratio = np.mean(gt_final_displacements) / np.mean(pred_final_displacements)
    print(f"\n🎯 Recommended Scale Factor: {scale_ratio:.3f}")
    print(f"   (Multiply model outputs by this to match ground truth scale)")
    
    return {
        'pred_mean_final_disp': float(np.mean(pred_final_displacements)),
        'gt_mean_final_disp': float(np.mean(gt_final_displacements)),
        'scale_ratio': float(scale_ratio),
        'ade': float(ade),
        'fde': float(fde)
    }


def check_coordinate_system(data_path: str, num_samples: int = 10):
    """Verify coordinate system used in training"""
    print("\n" + "="*70)
    print("CHECKING COORDINATE SYSTEM")
    print("="*70)
    
    loader = make_dataloader_highd(data_path, batch_size=1, shuffle=False)
    
    print("\nAnalyzing first few samples...")
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        
        obs = batch['target'].numpy()[0]  # (obs_len, 7)
        gt = batch['gt'].numpy()[0]  # (pred_len, 2)
        
        # Check if last observation is at origin
        last_obs_pos = obs[-1, :2]
        
        print(f"\nSample {i+1}:")
        print(f"  Last obs position: ({last_obs_pos[0]:.3f}, {last_obs_pos[1]:.3f})")
        print(f"  First GT position: ({gt[0, 0]:.3f}, {gt[0, 1]:.3f})")
        print(f"  GT displacement range: x=[{gt[:, 0].min():.2f}, {gt[:, 0].max():.2f}], "
              f"y=[{gt[:, 1].min():.2f}, {gt[:, 1].max():.2f}]")
        
        # Check if it's agent-centric
        if np.allclose(last_obs_pos, 0.0, atol=0.01):
            print(f"  ✓ Agent-centric: Last obs at origin")
        else:
            print(f"  ✗ Not agent-centric: Last obs NOT at origin")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calibrate Digital Twin Model")
    parser.add_argument("--data_path", required=True,
                       help="Path to training data CSV")
    parser.add_argument("--model_path", required=True,
                       help="Path to trained model")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--num_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DIGITAL TWIN MODEL CALIBRATION")
    print("="*70)
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"Type: {args.model_type.upper()}")
    print("="*70)
    
    # Step 1: Analyze training data
    data_stats = analyze_training_data(args.data_path, args.num_samples)
    
    # Step 2: Check coordinate system
    check_coordinate_system(args.data_path, num_samples=5)
    
    # Step 3: Analyze model predictions
    model_stats = analyze_model_predictions(
        args.model_path, args.model_type, 
        args.data_path, args.num_samples
    )
    
    # Step 4: Generate recommendations
    print("\n" + "="*70)
    print("CALIBRATION RECOMMENDATIONS")
    print("="*70)
    
    if model_stats and data_stats:
        print(f"\n1. Expected vs Predicted Displacement:")
        print(f"   Ground truth mean: {model_stats['gt_mean_final_disp']:.2f} m")
        print(f"   Model prediction mean: {model_stats['pred_mean_final_disp']:.2f} m")
        print(f"   Ratio: {model_stats['scale_ratio']:.3f}")
        
        print(f"\n2. Training Performance:")
        print(f"   ADE: {model_stats['ade']:.3f} m")
        print(f"   FDE: {model_stats['fde']:.3f} m")
        
        if model_stats['ade'] < 2.0:
            print(f"   ✓ Good training performance (ADE < 2m)")
        else:
            print(f"     High training error - model may need retraining")
        
        print(f"\n3. Recommended Settings for Simulation:")
        print(f"   --velocity_scale {model_stats['scale_ratio']:.3f}")
        if model_stats['scale_ratio'] < 0.5 or model_stats['scale_ratio'] > 2.0:
            print(f"     Unusual scale ratio - check coordinate system!")
        
        print(f"\n4. Expected Simulation Results:")
        print(f"   If calibrated correctly, expect:")
        print(f"   - ADE: {model_stats['ade']:.2f} - {model_stats['ade']*2:.2f} m")
        print(f"   - FDE: {model_stats['fde']:.2f} - {model_stats['fde']*2:.2f} m")
        print(f"   (Domain shift may cause 1.5-2x increase)")
    
    print("\n" + "="*70)
    print("Run simulation with: python dt_fixed_v2.py --velocity_scale <value>")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()