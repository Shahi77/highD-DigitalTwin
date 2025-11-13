"""
Enhanced Digital Twin SUMO Simulation with Real-time Metrics
------------------------------------------------------------
Implements proper DT architecture with:
- Real-time prediction every 0.5s (500ms)
- Comprehensive metrics collection (ADE, FDE, MAE, RMSE)
- Inference timing and computational efficiency tracking
- Clean visualization with dotted prediction/true trajectories
- Comparison between DT-enabled and baseline (replay-only) modes
"""

import os
import sys
import csv
import random
import numpy as np
import torch
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from utils import check_sumo_env, start_sumo, running

check_sumo_env()
import traci
from traci import constants as tc


# ==================== Configuration ====================
@dataclass
class DTConfig:
    """Digital Twin Configuration"""
    # Model settings
    MODEL_TYPE: str = "slstm"
    MODEL_PATH: str = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    
    # Prediction settings (from paper)
    OBS_LEN: int = 20  # 5 seconds at 4Hz (20 frames)
    PRED_LEN: int = 20  # 5 seconds prediction horizon (as per paper)
    K_NEIGHBORS: int = 8
    PREDICTION_INTERVAL_MS: int = 500  # Predict every 0.5 seconds
    
    # Visualization settings (clean, short trajectories)
    DRAW_PREDICTIONS: bool = True
    PRED_DISPLAY_LEN: int = 12  # Only show 3 seconds of prediction (12 frames @ 4Hz)
    LINE_SPACING: int = 3  # Every 3rd point for dotted effect
    PRED_LINE_WIDTH: float = 1.0
    TRUE_LINE_WIDTH: float = 1.0
    PRED_DISPLAY_LEN: int = 8      # Shorter, ~2s
    LINE_SPACING: int = 4          # More spaced/dotted
    MAX_PREDICTIONS_SHOWN: int = 3 # Show only a few vehicles for clarity
    
    # Simulation settings
    GUI: bool = True
    TOTAL_TIME: int = 4000  # ~16 minutes simulation
    START_STEP: int = 100
    
    # DT mode
    USE_DT_PREDICTION: bool = True  # Set to False for baseline replay-only mode
    
    # Output paths
    METRICS_OUTPUT: str = "./dt_metrics.json"
    RESULTS_DIR: str = "./dt_results"


config = DTConfig()


# ==================== Device Setup ====================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# ==================== Metrics Collector ====================
@dataclass
class PredictionMetrics:
    """Stores metrics for a single prediction"""
    vehicle_id: str
    timestamp: float
    ade: float  # Average Displacement Error
    fde: float  # Final Displacement Error
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    inference_time_ms: float  # Time to compute prediction
    speed_mae: float  # Speed prediction error
    acc_mae: float  # Acceleration prediction error


class MetricsCollector:
    """Collects and computes DT performance metrics"""
    
    def __init__(self, config: DTConfig):
        self.config = config
        self.predictions: List[PredictionMetrics] = []
        self.inference_times: List[float] = []
        self.start_time = time.time()
        self.total_predictions = 0
        self.failed_predictions = 0
        
    def add_prediction(self, vehicle_id: str, timestamp: float, 
                      pred_traj: np.ndarray, true_traj: np.ndarray,
                      pred_speed: np.ndarray, true_speed: np.ndarray,
                      pred_acc: np.ndarray, true_acc: np.ndarray,
                      inference_time_ms: float):
        """Add prediction metrics"""
        
        # Compute trajectory errors
        displacements = np.linalg.norm(pred_traj - true_traj, axis=1)
        ade = np.mean(displacements)
        fde = displacements[-1]
        mae = np.mean(np.abs(pred_traj - true_traj))
        rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
        
        # Compute speed/acceleration errors
        speed_mae = np.mean(np.abs(pred_speed - true_speed))
        acc_mae = np.mean(np.abs(pred_acc - true_acc))
        
        metrics = PredictionMetrics(
            vehicle_id=vehicle_id,
            timestamp=timestamp,
            ade=ade,
            fde=fde,
            mae=mae,
            rmse=rmse,
            inference_time_ms=inference_time_ms,
            speed_mae=speed_mae,
            acc_mae=acc_mae
        )
        
        self.predictions.append(metrics)
        self.inference_times.append(inference_time_ms)
        self.total_predictions += 1
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.predictions:
            return {}
        
        ades = [p.ade for p in self.predictions]
        fdes = [p.fde for p in self.predictions]
        maes = [p.mae for p in self.predictions]
        rmses = [p.rmse for p in self.predictions]
        speed_maes = [p.speed_mae for p in self.predictions]
        acc_maes = [p.acc_mae for p in self.predictions]
        
        elapsed_time = time.time() - self.start_time
        
        return {
            "trajectory_accuracy": {
                "ADE_mean": float(np.mean(ades)),
                "ADE_std": float(np.std(ades)),
                "FDE_mean": float(np.mean(fdes)),
                "FDE_std": float(np.std(fdes)),
                "MAE_mean": float(np.mean(maes)),
                "RMSE_mean": float(np.mean(rmses)),
            },
            "temporal_consistency": {
                "speed_MAE_mean": float(np.mean(speed_maes)),
                "acc_MAE_mean": float(np.mean(acc_maes)),
            },
            "computational_efficiency": {
                "inference_time_mean_ms": float(np.mean(self.inference_times)),
                "inference_time_std_ms": float(np.std(self.inference_times)),
                "inference_time_p50_ms": float(np.percentile(self.inference_times, 50)),
                "inference_time_p95_ms": float(np.percentile(self.inference_times, 95)),
                "inference_time_p99_ms": float(np.percentile(self.inference_times, 99)),
                "total_predictions": self.total_predictions,
                "failed_predictions": self.failed_predictions,
                "throughput_pred_per_sec": self.total_predictions / elapsed_time if elapsed_time > 0 else 0,
            },
            "simulation_info": {
                "total_runtime_sec": elapsed_time,
                "dt_mode_enabled": config.USE_DT_PREDICTION,
            }
        }
    
    def save_results(self, filepath: str):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        summary = self.get_summary()
        summary["raw_predictions"] = [asdict(p) for p in self.predictions[-100:]]  # Last 100
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n Metrics saved to {filepath}")


# ==================== Model Loading ====================
def load_model(model_path: str, model_type: str, pred_len: int):
    """Load trained prediction model"""
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f" Loaded {model_type.upper()} model (k={model.k} neighbors)")
    return model


# ==================== Vehicle Trajectory Tracking ====================
class TrajectoryTracker:
    """Tracks vehicle trajectories for prediction"""
    
    def __init__(self, obs_len: int = 20):
        self.obs_len = obs_len
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        self.last_prediction_time = {}
        
    def update(self, vehicle_id: str, position: Tuple[float, float], 
               velocity: float, acceleration: float, lane_id: int):
        """Add observation"""
        self.trajectories[vehicle_id].append({
            'x': position[0],
            'y': position[1],
            'vx': velocity,
            'vy': 0.0,
            'ax': acceleration,
            'ay': 0.0,
            'lane_id': lane_id,
            'timestamp': traci.simulation.getTime()
        })
    
    def get_observation(self, vehicle_id: str) -> Optional[np.ndarray]:
        """Get observation sequence [obs_len, 7]"""
        if vehicle_id not in self.trajectories:
            return None
        
        traj = list(self.trajectories[vehicle_id])
        if len(traj) < self.obs_len:
            return None
        
        obs = np.zeros((self.obs_len, 7))
        for i, frame in enumerate(traj):
            obs[i, 0] = frame['x']
            obs[i, 1] = frame['y']
            obs[i, 2] = frame['vx']
            obs[i, 3] = frame['vy']
            obs[i, 4] = frame['ax']
            obs[i, 5] = frame['ay']
            obs[i, 6] = frame['lane_id']
        
        return obs
    
    def should_predict(self, vehicle_id: str, current_time: float, 
                      interval_ms: int) -> bool:
        """Check if prediction should be made (every interval_ms)"""
        if vehicle_id not in self.last_prediction_time:
            return True
        
        time_since_last = (current_time - self.last_prediction_time[vehicle_id]) * 1000
        return time_since_last >= interval_ms
    
    def mark_predicted(self, vehicle_id: str, current_time: float):
        """Mark that prediction was made"""
        self.last_prediction_time[vehicle_id] = current_time
    
    def has_enough_history(self, vehicle_id: str) -> bool:
        """Check if vehicle has enough history"""
        return (vehicle_id in self.trajectories and 
                len(self.trajectories[vehicle_id]) >= self.obs_len)


# ==================== Digital Twin Prediction Manager ====================
class DigitalTwinPredictor:
    """Manages DT predictions with metrics collection"""
    
    def __init__(self, model, tracker: TrajectoryTracker, 
                 metrics: MetricsCollector, config: DTConfig):
        self.model = model
        self.tracker = tracker
        self.metrics = metrics
        self.config = config
        self.active_predictions = {}
        self.drawn_polygons = set()
    
    def _safe_remove_polygon(self, poly_id: str):
        """Safely remove polygon"""
        if poly_id in self.drawn_polygons:
            try:
                traci.polygon.remove(poly_id)
                self.drawn_polygons.remove(poly_id)
            except:
                self.drawn_polygons.discard(poly_id)
    
    def _safe_add_polygon(self, poly_id: str, points: List[Tuple[float, float]], 
                         color: Tuple[int, int, int, int], layer: int, lineWidth: float):
        """Safely add polygon"""
        self._safe_remove_polygon(poly_id)
        try:
            traci.polygon.add(poly_id, points, color=color, fill=False, 
                            layer=layer, lineWidth=lineWidth)
            self.drawn_polygons.add(poly_id)
        except:
            pass
    
    def make_prediction(self, vehicle_id: str) -> Optional[Tuple[np.ndarray, float]]:
        """Make prediction and return (prediction, inference_time_ms)"""
        obs = self.tracker.get_observation(vehicle_id)
        if obs is None:
            return None
        
        # Prepare input
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        nd = torch.zeros(1, self.model.k, self.config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, self.model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        # Time the inference
        start_time = time.time()
        
        try:
            with torch.no_grad():
                if hasattr(self.model, "multi_att"):  # Transformer
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = self.model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:  # SLSTM
                    pred = self.model(obs_tensor, nd, ns, lane)
            
            inference_time_ms = (time.time() - start_time) * 1000
            pred_np = pred.cpu().numpy()[0]  # [pred_len, 2]
            
            return pred_np, inference_time_ms
            
        except Exception as e:
            self.metrics.failed_predictions += 1
            return None
    
    def collect_ground_truth(self, vehicle_id: str, pred_len: int) -> Optional[Dict]:
        """Collect ground truth for metrics (future trajectory)"""
        try:
            current_pos = traci.vehicle.getPosition(vehicle_id)
            current_speed = traci.vehicle.getSpeed(vehicle_id)
            current_acc = traci.vehicle.getAcceleration(vehicle_id)
            current_angle = traci.vehicle.getAngle(vehicle_id)
            
            # Estimate future trajectory using constant velocity model
            angle_rad = np.radians(90 - current_angle)
            vx = current_speed * np.cos(angle_rad)
            vy = current_speed * np.sin(angle_rad)
            
            dt = 0.25  # 4Hz
            true_traj = np.zeros((pred_len, 2))
            true_speed = np.full(pred_len, current_speed)
            true_acc = np.full(pred_len, current_acc)
            
            for i in range(pred_len):
                true_traj[i, 0] = vx * dt * (i + 1)
                true_traj[i, 1] = vy * dt * (i + 1)
            
            return {
                'start_pos': current_pos,
                'true_traj': true_traj,
                'true_speed': true_speed,
                'true_acc': true_acc,
                'timestamp': traci.simulation.getTime()
            }
        except:
            return None
    
    def update_predictions(self, vehicle_ids: List[str], current_time: float):
        """Update predictions for eligible vehicles"""
        # Clean up old predictions
        vehicles_to_remove = [vid for vid in self.active_predictions.keys() 
                            if vid not in vehicle_ids]
        for vid in vehicles_to_remove:
            for prefix in ["pred_", "gt_"]:
                self._safe_remove_polygon(f"{prefix}{vid}")
            del self.active_predictions[vid]
        
        # Make new predictions (only for vehicles due for prediction)
        eligible = [vid for vid in vehicle_ids 
                   if self.tracker.has_enough_history(vid) and
                      self.tracker.should_predict(vid, current_time, 
                                                 self.config.PREDICTION_INTERVAL_MS)]
        
        # Limit to max shown
        eligible = eligible[:self.config.MAX_PREDICTIONS_SHOWN]
        
        for vid in eligible:
            result = self.make_prediction(vid)
            if result is None:
                continue
            
            pred, inference_time_ms = result
            gt_data = self.collect_ground_truth(vid, self.config.PRED_LEN)
            
            if gt_data is None:
                continue
            
            # Store prediction
            self.active_predictions[vid] = {
                'prediction': pred,
                'ground_truth': gt_data,
                'inference_time_ms': inference_time_ms
            }
            
            # Mark as predicted
            self.tracker.mark_predicted(vid, current_time)
            
            # Collect metrics (use display length for fair comparison)
            display_len = min(self.config.PRED_DISPLAY_LEN, self.config.PRED_LEN)
            self.metrics.add_prediction(
                vehicle_id=vid,
                timestamp=current_time,
                pred_traj=pred[:display_len],
                true_traj=gt_data['true_traj'][:display_len],
                pred_speed=np.linalg.norm(np.diff(pred[:display_len], axis=0), axis=1),
                true_speed=gt_data['true_speed'][:display_len-1],
                pred_acc=np.zeros(display_len-1),  # Simplified
                true_acc=gt_data['true_acc'][:display_len-1],
                inference_time_ms=inference_time_ms
            )
        
    def draw_predictions(self):
            """Draw truly dotted, separated, short trajectories for a few vehicles"""
            if not self.config.GUI or not self.config.DRAW_PREDICTIONS:
                return

            offset = 1.0
            # Remove previous polygons (all dots)
            for poly_id in list(self.drawn_polygons):
                self._safe_remove_polygon(poly_id)
            # Draw new dots for each vehicle
            for vid, data in self.active_predictions.items():
                if vid not in traci.vehicle.getIDList():
                    continue
                current_pos = traci.vehicle.getPosition(vid)
                pred = data['prediction']
                gt_data = data['ground_truth']
                display_len = min(self.config.PRED_DISPLAY_LEN, len(pred))

                # Dotted green dots for predicted
                pred_cumsum_x = np.cumsum(pred[:display_len, 0])
                pred_cumsum_y = np.cumsum(pred[:display_len, 1])
                for i in range(0, len(pred_cumsum_x), self.config.LINE_SPACING):
                    px = current_pos[0] + pred_cumsum_x[i] + offset
                    py = current_pos[1] + pred_cumsum_y[i]
                    dot_id = f"pred_dot_{vid}_{i}"
                    # as a filled polygon with one point (SUMO docs: a circle for one pt)
                    tiny = 0.5  # meters for dot size
                    pts = [
                        (px, py),
                        (px+tiny, py),
                        (px, py+tiny)
                    ]
                    traci.polygon.add(dot_id, pts, color=(0,255,0,255), fill=True, layer=101, lineWidth=3.0)
                    self.drawn_polygons.add(dot_id)

                # Dotted red dots for actual
                gt_traj = gt_data['true_traj'][:display_len]
                for i in range(0, len(gt_traj), self.config.LINE_SPACING):
                    gx = current_pos[0] + gt_traj[i, 0] - offset
                    gy = current_pos[1] + gt_traj[i, 1]
                    dot_id = f"gt_dot_{vid}_{i}"
                    tiny = 0.5  # meters for dot size
                    pts = [
                        (px, py),
                        (px+tiny, py),
                        (px, py+tiny)
                    ]
                    traci.polygon.add(dot_id, pts, color=(0,255,0,255), fill=True, layer=101, lineWidth=3.0)
                    self.drawn_polygons.add(dot_id)


                # Optional: Color vehicle for easier ID
                traci.vehicle.setColor(vid, (0,255,255,255))

    def clear_visualizations(self):
        """Clear all polygons"""
        if not self.config.GUI:
            return
        for poly_id in list(self.drawn_polygons):
            self._safe_remove_polygon(poly_id)


# ==================== Main DT Simulation ====================
def run_dt_simulation(config: DTConfig):
    """Run Digital Twin-enabled SUMO simulation"""
    
    print("\n" + "="*70)
    print("DIGITAL TWIN SUMO SIMULATION")
    print("="*70)
    print(f"Mode: {'DT-ENABLED' if config.USE_DT_PREDICTION else 'BASELINE (Replay-only)'}")
    print(f"Model: {config.MODEL_TYPE.upper()}")
    print(f"Prediction interval: {config.PREDICTION_INTERVAL_MS}ms")
    print(f"Prediction horizon: {config.PRED_LEN} frames ({config.PRED_LEN * 0.25:.1f}s)")
    print(f"Display length: {config.PRED_DISPLAY_LEN} frames ({config.PRED_DISPLAY_LEN * 0.25:.1f}s)")
    print("="*70 + "\n")
    
    # Load model (if DT mode)
    model = None
    if config.USE_DT_PREDICTION:
        model = load_model(config.MODEL_PATH, config.MODEL_TYPE, config.PRED_LEN)
    
    # Initialize components
    tracker = TrajectoryTracker(obs_len=config.OBS_LEN)
    metrics = MetricsCollector(config)
    
    predictor = None
    if config.USE_DT_PREDICTION and model is not None:
        predictor = DigitalTwinPredictor(model, tracker, metrics, config)
    
    # Load highD data
    from main import (trajectory_tracking, aggregate_vehicles, gene_config, 
                     has_vehicle_entered, AVAILABLE_CAR_TYPES, AVAILABLE_TRUCK_TYPES,
                     CHECK_ALL, LAN_CHANGE_MODE)
    
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
    # Setup simulation
    cfg_file = gene_config()
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=config.GUI)
    
    times = 0
    random.seed(7)
    vehicles_added = 0
    
    print("Starting simulation...\n")
    
    try:
        while running(True, times, config.TOTAL_TIME + 1):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Add vehicles from highD
            if times > 0 and times % 4 == 0:
                current_step = int(times / 4)
                
                if has_vehicle_entered(current_step, vehicles_to_enter):
                    for data in vehicles_to_enter[current_step]:
                        vehicle_class = data["class"].lower()
                        
                        if "truck" in vehicle_class or "bus" in vehicle_class:
                            type_id = random.choice(AVAILABLE_TRUCK_TYPES)
                            depart_speed = random.uniform(24, 25)
                        else:
                            type_id = random.choice(AVAILABLE_CAR_TYPES)
                            depart_speed = random.uniform(31, 33)
                        
                        lane_id = max(0, min(2, int(data.get("laneId", 1)) - 1))
                        depart_pos = random.uniform(10, 30)
                        direction = data.get("drivingDirection", 1)
                        
                        route_id = "route_direction1" if direction == 1 else "route_direction2"
                        vehicle_id = f"d{direction}_{data['id']}"
                        
                        try:
                            traci.vehicle.add(
                                vehID=vehicle_id,
                                routeID=route_id,
                                typeID=type_id,
                                departSpeed=depart_speed,
                                departPos=depart_pos,
                                departLane=lane_id,
                            )
                            traci.vehicle.setSpeedMode(vehicle_id, CHECK_ALL)
                            traci.vehicle.setLaneChangeMode(vehicle_id, LAN_CHANGE_MODE)
                            vehicles_added += 1
                        except:
                            pass
            
            # Get active vehicles
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update trajectory tracking
            for vid in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    lane = traci.vehicle.getLaneIndex(vid)
                    tracker.update(vid, pos, speed, acc, lane)
                except:
                    continue
            
            # DT predictions (every 0.5s as per config)
            if config.USE_DT_PREDICTION and predictor and times > config.START_STEP:
                predictor.update_predictions(vehicle_ids, current_time)
                predictor.draw_predictions()
            
            # Progress
            if times % 500 == 0 and times > 0:
                num_active = len(vehicle_ids)
                num_predictions = len(predictor.active_predictions) if predictor else 0
                avg_inference = np.mean(metrics.inference_times[-100:]) if len(metrics.inference_times) > 0 else 0
                
                print(f"Step {times:5d} ({current_time:6.1f}s): "
                      f"{num_active:3d} vehicles | "
                      f"{num_predictions:3d} predictions | "
                      f"Inference: {avg_inference:.2f}ms")
            
            if times >= config.TOTAL_TIME:
                print(f"\n Reached target time: {config.TOTAL_TIME} steps")
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Simulation interrupted")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up
        if predictor:
            predictor.clear_visualizations()
        try:
            traci.close()
        except:
            pass
        
        # Force metrics save even if simulation was interrupted
        print("\n Saving metrics...")
        time.sleep(1)  # Give time for final updates
    
    # Save metrics
    print("\n" + "="*70)
    print("SIMULATION COMPLETE - FINAL METRICS")
    print("="*70)
    
    summary = metrics.get_summary()
    
    if summary and summary.get('computational_efficiency', {}).get('total_predictions', 0) > 0:
        print("\n Trajectory Accuracy:")
        print(f"  ADE: {summary['trajectory_accuracy']['ADE_mean']:.4f} ± {summary['trajectory_accuracy']['ADE_std']:.4f} m")
        print(f"  FDE: {summary['trajectory_accuracy']['FDE_mean']:.4f} ± {summary['trajectory_accuracy']['FDE_std']:.4f} m")
        print(f"  MAE: {summary['trajectory_accuracy']['MAE_mean']:.4f} m")
        print(f"  RMSE: {summary['trajectory_accuracy']['RMSE_mean']:.4f} m")
        
        print("\n  Computational Efficiency:")
        print(f"  Mean inference time: {summary['computational_efficiency']['inference_time_mean_ms']:.2f} ms")
        print(f"  P50 inference time: {summary['computational_efficiency']['inference_time_p50_ms']:.2f} ms")
        print(f"  P95 inference time: {summary['computational_efficiency']['inference_time_p95_ms']:.2f} ms")
        print(f"  P99 inference time: {summary['computational_efficiency']['inference_time_p99_ms']:.2f} ms")
        print(f"  Throughput: {summary['computational_efficiency']['throughput_pred_per_sec']:.2f} pred/s")
        print(f"  Total predictions: {summary['computational_efficiency']['total_predictions']}")
        print(f"  Failed predictions: {summary['computational_efficiency']['failed_predictions']}")
        
        # Save to file
        metrics.save_results(config.METRICS_OUTPUT)
    else:
        print("\n No predictions were recorded during simulation")
        print("   This is expected for baseline mode")
        
        # Still save empty metrics for baseline
        if not config.USE_DT_PREDICTION:
            empty_summary = {
                "trajectory_accuracy": {},
                "computational_efficiency": {
                    "total_predictions": 0,
                    "inference_time_mean_ms": 0,
                    "throughput_pred_per_sec": 0
                },
                "simulation_info": {
                    "total_runtime_sec": time.time() - metrics.start_time,
                    "dt_mode_enabled": False
                }
            }
            metrics.save_results(config.METRICS_OUTPUT)
    
    print(f"\n Total vehicles added: {vehicles_added}")
    print("="*70 + "\n")
    
    return summary


# ==================== Entry Point ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Digital Twin SUMO Simulation")
    parser.add_argument("--mode", choices=["dt", "baseline"], default="dt",
                       help="Run with DT predictions or baseline replay")
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--total_time", type=int, default=4000)
    parser.add_argument("--pred_interval_ms", type=int, default=500,
                       help="Prediction interval in milliseconds")

    args = parser.parse_args()

    config.USE_DT_PREDICTION = (args.mode == "dt")
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.GUI = args.gui
    config.TOTAL_TIME = args.total_time
    config.PREDICTION_INTERVAL_MS = args.pred_interval_ms
    config.METRICS_OUTPUT = f"./dt_metrics_{args.mode}.json"

    run_dt_simulation(config)
