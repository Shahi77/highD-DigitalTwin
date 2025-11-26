"""
Digital Twin Diagnostic Analyzer
---------------------------------
Verifies if your simulation results are trustworthy by checking:
1. Prediction diversity (are predictions actually varying?)
2. Physical plausibility (do predictions follow road geometry?)
3. Comparison with training performance
4. Coordinate system verification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(filepath: str):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def analyze_prediction_diversity(metrics_data):
    """Check if predictions are actually diverse"""
    print("\n" + "="*70)
    print("PREDICTION DIVERSITY ANALYSIS")
    print("="*70)
    
    raw_preds = metrics_data.get('raw_predictions', [])
    if not raw_preds:
        print("⚠️  No raw predictions available!")
        return
    
    distances = [p['pred_distance'] for p in raw_preds]
    ades = [p['ade'] for p in raw_preds]
    fdes = [p['fde'] for p in raw_preds]
    
    print(f"\n📊 Prediction Distance Statistics (n={len(distances)}):")
    print(f"  Mean: {np.mean(distances):.2f}m")
    print(f"  Std:  {np.std(distances):.2f}m")
    print(f"  CV (coefficient of variation): {np.std(distances)/np.mean(distances)*100:.2f}%")
    
    # Low CV indicates all predictions are similar
    cv = np.std(distances) / np.mean(distances) * 100
    if cv < 5:
        print(f"\n⚠️  WARNING: Very low diversity (CV={cv:.2f}%)")
        print(f"  This suggests model is producing nearly identical predictions!")
        print(f"  Expected CV for real traffic: 15-30%")
    elif cv < 10:
        print(f"\n⚠️  CAUTION: Low diversity (CV={cv:.2f}%)")
        print(f"  Predictions may not be adapting to different scenarios")
    else:
        print(f"\n✓ Good diversity (CV={cv:.2f}%)")
    
    # Check per-vehicle variation
    vehicle_distances = {}
    for p in raw_preds:
        vid = p['vehicle_id']
        if vid not in vehicle_distances:
            vehicle_distances[vid] = []
        vehicle_distances[vid].append(p['pred_distance'])
    
    within_vehicle_cvs = []
    for vid, dists in vehicle_distances.items():
        if len(dists) > 1:
            cv = np.std(dists) / np.mean(dists) * 100 if np.mean(dists) > 0 else 0
            within_vehicle_cvs.append(cv)
    
    if within_vehicle_cvs:
        print(f"\n📊 Within-Vehicle Prediction Variation:")
        print(f"  Mean CV: {np.mean(within_vehicle_cvs):.2f}%")
        if np.mean(within_vehicle_cvs) < 2:
            print(f"  ⚠️  Same vehicle getting nearly identical predictions!")


def analyze_physical_plausibility(metrics_data):
    """Check if predictions are physically plausible"""
    print("\n" + "="*70)
    print("PHYSICAL PLAUSIBILITY ANALYSIS")
    print("="*70)
    
    debug_stats = metrics_data.get('debug_stats', {})
    avg_dist = debug_stats.get('avg_pred_distance', 0)
    
    # Expected values for highway traffic
    # At 4Hz, 20 frames = 5 seconds
    # Highway speeds: 25-35 m/s (90-126 km/h)
    # Expected distance in 5s: 125-175m
    
    print(f"\n🚗 Expected vs Actual:")
    print(f"  Prediction horizon: 5 seconds (20 frames @ 4Hz)")
    print(f"  Expected highway distance: 125-175m")
    print(f"  Your average distance: {avg_dist:.2f}m")
    
    if avg_dist < 100:
        print(f"\n⚠️  WARNING: Predictions too short!")
        print(f"  Possible causes:")
        print(f"  - Model underestimating velocities")
        print(f"  - Wrong coordinate system")
        print(f"  - Time scaling mismatch")
    elif avg_dist > 200:
        print(f"\n⚠️  WARNING: Predictions too long!")
        print(f"  Possible causes:")
        print(f"  - Velocity scaling applied incorrectly")
        print(f"  - Coordinate system error")
    else:
        print(f"\n✓ Distance within expected range")
    
    # Check ADE/FDE ratio
    traj_acc = metrics_data.get('trajectory_accuracy', {})
    ade = traj_acc.get('ADE_mean', 0)
    fde = traj_acc.get('FDE_mean', 0)
    
    if ade > 0:
        ratio = fde / ade
        print(f"\n📊 Error Growth Analysis:")
        print(f"  FDE/ADE ratio: {ratio:.2f}")
        print(f"  Expected ratio: 1.5-2.5 (errors grow over time)")
        
        if ratio < 1.2:
            print(f"  ⚠️  Ratio too low - might indicate coordinate issues")
        elif ratio > 3.0:
            print(f"  ⚠️  Ratio too high - errors growing too fast")
        else:
            print(f"  ✓ Error growth pattern looks reasonable")


def compare_with_training(metrics_data, training_ade=0.795, training_fde=5.322):
    """Compare simulation results with training performance"""
    print("\n" + "="*70)
    print("TRAINING VS SIMULATION COMPARISON")
    print("="*70)
    
    traj_acc = metrics_data.get('trajectory_accuracy', {})
    sim_ade = traj_acc.get('ADE_mean', 0)
    sim_fde = traj_acc.get('FDE_mean', 0)
    
    print(f"\n📊 Performance Comparison:")
    print(f"\n  Training (on highD dataset):")
    print(f"    ADE: {training_ade:.3f}m")
    print(f"    FDE: {training_fde:.3f}m")
    
    print(f"\n  Simulation (on SUMO):")
    print(f"    ADE: {sim_ade:.3f}m")
    print(f"    FDE: {sim_fde:.3f}m")
    
    ade_ratio = sim_ade / training_ade if training_ade > 0 else 0
    fde_ratio = sim_fde / training_fde if training_fde > 0 else 0
    
    print(f"\n  Degradation:")
    print(f"    ADE: {ade_ratio:.2f}x worse")
    print(f"    FDE: {fde_ratio:.2f}x worse")
    
    print(f"\n🔍 Interpretation:")
    if ade_ratio < 1.5:
        print(f"  ✓ Excellent - minimal domain shift")
    elif ade_ratio < 2.5:
        print(f"  ✓ Good - expected domain shift range")
    elif ade_ratio < 4.0:
        print(f"  ⚠️  Moderate degradation - investigate causes")
    else:
        print(f"  ❌ High degradation - likely systematic error!")
        print(f"\n  Possible causes:")
        print(f"  - Coordinate system mismatch")
        print(f"  - Wrong data preprocessing")
        print(f"  - SUMO vs real data distribution mismatch")
        print(f"  - Scaling factors applied incorrectly")


def check_velocity_scaling_effect(metrics_data, scale_factor=1.309):
    """Analyze if velocity scaling is helping or hiding issues"""
    print("\n" + "="*70)
    print("VELOCITY SCALING ANALYSIS")
    print("="*70)
    
    if scale_factor == 1.0:
        print("\n✓ No velocity scaling applied")
        return
    
    print(f"\n⚠️  Velocity scaling factor: {scale_factor}")
    
    debug_stats = metrics_data.get('debug_stats', {})
    avg_dist = debug_stats.get('avg_pred_distance', 0)
    
    # What would distance be without scaling?
    original_dist = avg_dist / scale_factor
    
    print(f"\n📊 Effect of Scaling:")
    print(f"  Original prediction distance: {original_dist:.2f}m")
    print(f"  After {scale_factor}x scaling: {avg_dist:.2f}m")
    
    print(f"\n🔍 Important Questions:")
    print(f"  1. Why does model predict {original_dist:.2f}m instead of ~130m?")
    print(f"  2. Is this a training issue or simulation issue?")
    print(f"  3. Should we retrain instead of scaling?")
    
    traj_acc = metrics_data.get('trajectory_accuracy', {})
    ade = traj_acc.get('ADE_mean', 0)
    
    if ade > 2.0:
        print(f"\n⚠️  Even after scaling, ADE={ade:.2f}m is high")
        print(f"  This suggests scaling is not fixing the root problem!")


def plot_error_distribution(metrics_data, output_path=None):
    """Plot error distributions"""
    raw_preds = metrics_data.get('raw_predictions', [])
    if not raw_preds:
        return
    
    ades = [p['ade'] for p in raw_preds]
    fdes = [p['fde'] for p in raw_preds]
    distances = [p['pred_distance'] for p in raw_preds]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(ades, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(ades), color='r', linestyle='--', label=f'Mean: {np.mean(ades):.2f}')
    axes[0].set_xlabel('ADE (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Average Displacement Error')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(fdes, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(fdes), color='r', linestyle='--', label=f'Mean: {np.mean(fdes):.2f}')
    axes[1].set_xlabel('FDE (m)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Final Displacement Error')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(distances, bins=30, edgecolor='black', alpha=0.7)
    axes[2].axvline(np.mean(distances), color='r', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    axes[2].axvline(130, color='g', linestyle='--', label='Expected: ~130m')
    axes[2].set_xlabel('Prediction Distance (m)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Predicted Travel Distance')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved plot to {output_path}")
    else:
        plt.savefig('dt_diagnostic_plot.png', dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved plot to dt_diagnostic_plot.png")
    
    plt.close()


def generate_recommendations(metrics_data, training_ade=0.795):
    """Generate actionable recommendations"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    traj_acc = metrics_data.get('trajectory_accuracy', {})
    sim_ade = traj_acc.get('ADE_mean', 0)
    debug_stats = metrics_data.get('debug_stats', {})
    avg_dist = debug_stats.get('avg_pred_distance', 0)
    std_dist = debug_stats.get('std_pred_distance', 0)
    
    issues_found = []
    
    # Check diversity
    cv = (std_dist / avg_dist * 100) if avg_dist > 0 else 0
    if cv < 5:
        issues_found.append("low_diversity")
    
    # Check distance
    if avg_dist < 100 or avg_dist > 200:
        issues_found.append("wrong_distance")
    
    # Check degradation
    degradation = sim_ade / training_ade if training_ade > 0 else 0
    if degradation > 4.0:
        issues_found.append("high_degradation")
    
    if not issues_found:
        print("\n✅ Results look reasonable!")
        print("\nNext steps:")
        print("  1. Visualize predictions in SUMO GUI")
        print("  2. Compare with baseline methods")
        print("  3. Run longer simulations for stability")
    else:
        print("\n⚠️  Issues detected! Recommendations:")
        
        if "low_diversity" in issues_found:
            print("\n1. LOW DIVERSITY ISSUE:")
            print("   - Model producing similar predictions for all vehicles")
            print("   - Actions:")
            print("     • Check if observation data is varying")
            print("     • Verify velocity/acceleration inputs")
            print("     • Try without velocity scaling first")
        
        if "wrong_distance" in issues_found:
            print("\n2. DISTANCE MISMATCH:")
            print("   - Predictions not matching expected travel distance")
            print("   - Actions:")
            print("     • Verify time horizon (5s @ 4Hz = 20 frames)")
            print("     • Check coordinate system transformation")
            print("     • Compare with training data statistics")
        
        if "high_degradation" in issues_found:
            print("\n3. HIGH ERROR DEGRADATION:")
            print("   - Performance much worse than training")
            print("   - Actions:")
            print("     • Run WITHOUT velocity scaling first")
            print("     • Check coordinate normalization")
            print("     • Verify rotation transformation matches training")
            print("     • Consider retraining on SUMO data")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic analysis of DT simulation")
    parser.add_argument("--metrics", required=True, help="Path to metrics JSON file")
    parser.add_argument("--training_ade", type=float, default=0.795)
    parser.add_argument("--training_fde", type=float, default=5.322)
    parser.add_argument("--velocity_scale", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DIGITAL TWIN DIAGNOSTIC ANALYZER")
    print("="*70)
    print(f"Metrics file: {args.metrics}")
    print("="*70)
    
    try:
        metrics_data = load_metrics(args.metrics)
    except Exception as e:
        print(f"\n❌ Error loading metrics: {e}")
        return
    
    # Run analyses
    analyze_prediction_diversity(metrics_data)
    analyze_physical_plausibility(metrics_data)
    compare_with_training(metrics_data, args.training_ade, args.training_fde)
    check_velocity_scaling_effect(metrics_data, args.velocity_scale)
    
    if args.plot:
        plot_error_distribution(metrics_data)
    
    generate_recommendations(metrics_data, args.training_ade)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()