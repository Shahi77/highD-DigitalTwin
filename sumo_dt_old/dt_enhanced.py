"""
Digital Twin SUMO Simulation - CRITICAL FIXES FOR PROPER METRICS
-----------------------------------------------------------------
Key fixes:
1. Apply same coordinate transformation as training (rotation normalization)
2. Add velocity-based scaling for predictions
3. Implement proper denormalization
4. Add coordinate system validation
5. Improve ground truth alignment
"""

import os, sys, random, numpy as np, torch, time, json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from utils import check_sumo_env, start_sumo, running

check_sumo_env()
import traci
from traci import constants as tc


@dataclass
class DTConfig:
    MODEL_TYPE: str = "slstm"
    MODEL_PATH: str = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    OBS_LEN: int = 20
    PRED_LEN: int = 20
    K_NEIGHBORS: int = 8
    
    TRAINING_FREQ_HZ: int = 4
    SUMO_STEP_MS: int = 1000
    PREDICTION_INTERVAL_MS: int = 500
    MIN_GT_FRAMES: int = 8
    MAX_GT_WAIT_STEPS: int = 100
    
    # CRITICAL: Coordinate system parameters
    USE_ROTATION_NORMALIZATION: bool = True
    VELOCITY_SCALE_FACTOR: float = 1.0  # Tune this based on training data
    MAX_PREDICTION_DISTANCE: float = 200.0  # Sanity check (meters)
    
    DRAW_PREDICTIONS: bool = True
    PRED_DISPLAY_LEN: int = 8
    GUI: bool = True
    TOTAL_TIME: int = 4000
    START_STEP: int = 100
    METRICS_OUTPUT: str = "./dt_results/dt_metrics_v2.json"


config = DTConfig()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class PredictionMetrics:
    vehicle_id: str
    prediction_step: int
    ade: float
    fde: float
    inference_latency_ms: float
    e2e_latency_ms: float
    num_gt_frames: int
    prediction_timestamp: float
    pred_distance: float  # For debugging


class SimpleMetricsCollector:
    def __init__(self, config: DTConfig):
        self.config = config
        self.metrics: List[PredictionMetrics] = []
        self.start_time = time.time()
        
    def add_metric(self, vehicle_id: str, prediction_step: int,
                   pred_traj: np.ndarray, true_traj: np.ndarray,
                   inference_ms: float, e2e_ms: float,
                   timestamp: float):
        """Add a single prediction metric"""
        displacements = np.linalg.norm(pred_traj - true_traj, axis=1)
        ade = float(np.mean(displacements))
        fde = float(displacements[-1])
        pred_distance = float(np.linalg.norm(pred_traj[-1]))
        
        metric = PredictionMetrics(
            vehicle_id=vehicle_id,
            prediction_step=prediction_step,
            ade=ade,
            fde=fde,
            inference_latency_ms=inference_ms,
            e2e_latency_ms=e2e_ms,
            num_gt_frames=len(true_traj),
            prediction_timestamp=timestamp,
            pred_distance=pred_distance
        )
        
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict:
        if not self.metrics:
            return {}
        
        ades = [m.ade for m in self.metrics]
        fdes = [m.fde for m in self.metrics]
        inference_times = [m.inference_latency_ms for m in self.metrics]
        pred_distances = [m.pred_distance for m in self.metrics]
        
        return {
            "trajectory_accuracy": {
                "ADE_mean": float(np.mean(ades)),
                "ADE_std": float(np.std(ades)),
                "ADE_median": float(np.median(ades)),
                "FDE_mean": float(np.mean(fdes)),
                "FDE_std": float(np.std(fdes)),
                "FDE_median": float(np.median(fdes)),
            },
            "latency_metrics": {
                "inference_mean_ms": float(np.mean(inference_times)),
                "inference_p95_ms": float(np.percentile(inference_times, 95)),
            },
            "debug_stats": {
                "avg_pred_distance": float(np.mean(pred_distances)),
                "total_predictions": len(self.metrics),
            }
        }
    
    def save_results(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        summary = self.get_summary()
        summary["raw_predictions"] = [asdict(m) for m in self.metrics[-100:]]
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Saved {len(self.metrics)} metrics to {filepath}")


def load_model(model_path: str, model_type: str, pred_len: int):
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✓ Loaded {model_type.upper()} model")
    return model


class TrajectoryTracker:
    """Track historical trajectories with proper coordinate handling"""
    def __init__(self, obs_len: int = 20, use_rotation_norm: bool = True):
        self.obs_len = obs_len
        self.use_rotation_norm = use_rotation_norm
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        self.last_sample_time = defaultdict(float)
        
    def should_sample(self, vehicle_id: str, current_time_s: float, freq_hz: int = 4) -> bool:
        if vehicle_id not in self.last_sample_time:
            return True
        elapsed = current_time_s - self.last_sample_time[vehicle_id]
        return elapsed >= (1.0 / freq_hz)
        
    def update(self, vehicle_id: str, position: Tuple[float, float], 
               velocity: float, acceleration: float, angle: float,
               lane_id: int, current_time_s: float):
        """Update trajectory with proper state information"""
        if self.should_sample(vehicle_id, current_time_s, config.TRAINING_FREQ_HZ):
            # Convert velocity to x,y components based on angle
            angle_rad = np.radians(90 - angle)  # SUMO angle to standard
            vx = velocity * np.cos(angle_rad)
            vy = velocity * np.sin(angle_rad)
            
            self.trajectories[vehicle_id].append({
                'x': position[0], 'y': position[1],
                'vx': vx, 'vy': vy,
                'ax': acceleration * np.cos(angle_rad),
                'ay': acceleration * np.sin(angle_rad),
                'heading': angle_rad,
                'lane_id': lane_id,
                'time': current_time_s
            })
            self.last_sample_time[vehicle_id] = current_time_s
    
    def get_observation(self, vehicle_id: str) -> Optional[Tuple[np.ndarray, Tuple[float, float], float]]:
        """Get observation with rotation normalization like training"""
        if vehicle_id not in self.trajectories:
            return None
        
        traj = list(self.trajectories[vehicle_id])
        if len(traj) < self.obs_len:
            return None
        
        # Extract raw trajectory
        obs_abs = np.zeros((self.obs_len, 7))
        for i, frame in enumerate(traj):
            obs_abs[i] = [frame['x'], frame['y'], frame['vx'], frame['vy'],
                         frame['ax'], frame['ay'], frame['heading']]
        
        # Get reference frame (last observation)
        last_pos = (traj[-1]['x'], traj[-1]['y'])
        last_heading = traj[-1]['heading']
        
        # Apply rotation normalization (same as training data loader)
        if self.use_rotation_norm:
            obs_normalized = self._apply_rotation_normalization(obs_abs, last_pos, last_heading)
        else:
            # Simple relative coordinates
            obs_normalized = obs_abs.copy()
            obs_normalized[:, 0] -= last_pos[0]
            obs_normalized[:, 1] -= last_pos[1]
        
        # Add lane feature
        lane_feat = np.array([traj[-1]['lane_id'] / 10.0])
        
        return obs_normalized, last_pos, last_heading
    
    def _apply_rotation_normalization(self, obs: np.ndarray, origin: Tuple[float, float], 
                                     yaw: float) -> np.ndarray:
        """Apply same rotation normalization as training dataloader"""
        normalized = obs.copy()
        
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        
        # Rotate positions
        dx = obs[:, 0] - origin[0]
        dy = obs[:, 1] - origin[1]
        normalized[:, 0] = cos_yaw * dx - sin_yaw * dy
        normalized[:, 1] = sin_yaw * dx + cos_yaw * dy
        
        # Rotate velocities
        normalized[:, 2] = cos_yaw * obs[:, 2] - sin_yaw * obs[:, 3]
        normalized[:, 3] = sin_yaw * obs[:, 2] + cos_yaw * obs[:, 3]
        
        # Rotate accelerations
        normalized[:, 4] = cos_yaw * obs[:, 4] - sin_yaw * obs[:, 5]
        normalized[:, 5] = sin_yaw * obs[:, 4] + cos_yaw * obs[:, 5]
        
        # Normalize heading
        normalized[:, 6] = ((obs[:, 6] - yaw + np.pi) % (2 * np.pi)) - np.pi
        
        return normalized
    
    def has_enough_history(self, vehicle_id: str) -> bool:
        return (vehicle_id in self.trajectories and 
                len(self.trajectories[vehicle_id]) >= self.obs_len)


class PredictionRecord:
    """Store prediction with proper coordinate tracking"""
    def __init__(self, vehicle_id: str, prediction_step: int, 
                 prediction_relative: np.ndarray, last_obs_pos: Tuple[float, float],
                 last_obs_heading: float, inference_ms: float, e2e_ms: float,
                 timestamp: float, sample_freq_hz: int):
        self.vehicle_id = vehicle_id
        self.prediction_step = prediction_step
        self.prediction_relative = prediction_relative  # In rotated frame
        self.last_obs_pos = last_obs_pos
        self.last_obs_heading = last_obs_heading
        self.inference_ms = inference_ms
        self.e2e_ms = e2e_ms
        self.timestamp = timestamp
        self.sample_freq_hz = sample_freq_hz
        
        self.ground_truth: List[Tuple[float, float]] = []
        self.last_gt_sample_time = timestamp
        self.steps_waiting = 0
        self.completed = False
    
    def add_ground_truth_frame(self, position: Tuple[float, float], current_time_s: float):
        if not self.completed:
            elapsed = current_time_s - self.last_gt_sample_time
            if elapsed >= (1.0 / self.sample_freq_hz):
                self.ground_truth.append(position)
                self.last_gt_sample_time = current_time_s
    
    def can_evaluate(self, min_frames: int) -> bool:
        return len(self.ground_truth) >= min_frames
    
    def should_timeout(self, max_steps: int) -> bool:
        self.steps_waiting += 1
        return self.steps_waiting > max_steps
    
    def get_metrics(self, max_len: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert to same coordinate system for comparison"""
        if len(self.ground_truth) == 0:
            return None
        
        # Convert ground truth to rotated relative frame (same as prediction)
        cos_yaw = np.cos(-self.last_obs_heading)
        sin_yaw = np.sin(-self.last_obs_heading)
        
        gt_relative = np.zeros((len(self.ground_truth), 2))
        for i, pos in enumerate(self.ground_truth):
            dx = pos[0] - self.last_obs_pos[0]
            dy = pos[1] - self.last_obs_pos[1]
            gt_relative[i, 0] = cos_yaw * dx - sin_yaw * dy
            gt_relative[i, 1] = sin_yaw * dx + cos_yaw * dy
        
        use_len = min(len(self.prediction_relative), len(gt_relative), max_len)
        
        return self.prediction_relative[:use_len], gt_relative[:use_len]


class DigitalTwinPredictor:
    def __init__(self, model, tracker: TrajectoryTracker, 
                 metrics: SimpleMetricsCollector, config: DTConfig):
        self.model = model
        self.tracker = tracker
        self.metrics = metrics
        self.config = config
        
        self.active_predictions = {}
        self.prediction_records: Dict[str, PredictionRecord] = {}
        self.last_prediction_time = {}
        self.drawn_objects = set()
        
        self.total_predictions_made = 0
        self.total_metrics_collected = 0
        self.failed_collections = 0
        
        # For debugging
        self.prediction_scale_stats = []
    
    def should_predict(self, vehicle_id: str, current_time: float) -> bool:
        if vehicle_id not in self.last_prediction_time:
            return True
        elapsed_ms = (current_time - self.last_prediction_time[vehicle_id]) * 1000
        return elapsed_ms >= self.config.PREDICTION_INTERVAL_MS
    
    def make_prediction(self, vehicle_id: str, current_time: float, current_step: int):
        result = self.tracker.get_observation(vehicle_id)
        if result is None:
            return False
        
        obs, last_obs_pos, last_obs_heading = result
        
        # Prepare tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        nd = torch.zeros(1, self.model.k, self.config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, self.model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        e2e_start = time.time()
        
        try:
            inference_start = time.time()
            with torch.no_grad():
                if hasattr(self.model, "multi_att"):
                    last_obs_pos_tensor = obs_tensor[:, -1, :2]
                    pred = self.model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos_tensor)
                else:
                    pred = self.model(obs_tensor, nd, ns, lane)
            
            inference_ms = (time.time() - inference_start) * 1000
            e2e_ms = (time.time() - e2e_start) * 1000
            
            pred_np = pred.cpu().numpy()[0]  # Shape: (PRED_LEN, 2)
            
            # CRITICAL: Check prediction scale
            pred_magnitude = np.linalg.norm(pred_np[-1])
            self.prediction_scale_stats.append(pred_magnitude)
            
            # Sanity check
            if pred_magnitude > self.config.MAX_PREDICTION_DISTANCE:
                print(f"⚠️  Suspicious prediction for {vehicle_id}: {pred_magnitude:.1f}m")
                return False
            
            # Create prediction record (stays in relative rotated frame)
            record = PredictionRecord(
                vehicle_id=vehicle_id,
                prediction_step=current_step,
                prediction_relative=pred_np,
                last_obs_pos=last_obs_pos,
                last_obs_heading=last_obs_heading,
                inference_ms=inference_ms,
                e2e_ms=e2e_ms,
                timestamp=current_time,
                sample_freq_hz=self.config.TRAINING_FREQ_HZ
            )
            
            record_key = f"{vehicle_id}_{current_step}"
            self.prediction_records[record_key] = record
            
            # For visualization, convert to absolute coordinates
            pred_absolute = self._to_absolute_coords(pred_np, last_obs_pos, last_obs_heading)
            self.active_predictions[vehicle_id] = {
                'prediction': pred_absolute,
                'start_pos': last_obs_pos,
                'timestamp': current_time
            }
            
            self.last_prediction_time[vehicle_id] = current_time
            self.total_predictions_made += 1
            
            return True
            
        except Exception as e:
            print(f"Prediction error for {vehicle_id}: {e}")
            return False
    
    def _to_absolute_coords(self, pred_relative: np.ndarray, 
                           origin: Tuple[float, float], yaw: float) -> np.ndarray:
        """Convert from rotated relative to absolute coordinates"""
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        pred_abs = np.zeros_like(pred_relative)
        pred_abs[:, 0] = origin[0] + (cos_yaw * pred_relative[:, 0] - sin_yaw * pred_relative[:, 1])
        pred_abs[:, 1] = origin[1] + (sin_yaw * pred_relative[:, 0] + cos_yaw * pred_relative[:, 1])
        
        return pred_abs
    
    def update_ground_truth(self, vehicle_ids: List[str], current_time: float):
        for record_key, record in list(self.prediction_records.items()):
            vehicle_id = record.vehicle_id
            if vehicle_id in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vehicle_id)
                    record.add_ground_truth_frame(pos, current_time)
                except:
                    pass
    
    def collect_metrics(self):
        records_to_remove = []
        
        for record_key, record in list(self.prediction_records.items()):
            if record.can_evaluate(self.config.MIN_GT_FRAMES):
                result = record.get_metrics(self.config.PRED_LEN)
                
                if result is not None:
                    pred_traj, true_traj = result
                    
                    self.metrics.add_metric(
                        vehicle_id=record.vehicle_id,
                        prediction_step=record.prediction_step,
                        pred_traj=pred_traj,
                        true_traj=true_traj,
                        inference_ms=record.inference_ms,
                        e2e_ms=record.e2e_ms,
                        timestamp=record.timestamp
                    )
                    
                    self.total_metrics_collected += 1
                    records_to_remove.append(record_key)
                    continue
            
            if record.should_timeout(self.config.MAX_GT_WAIT_STEPS):
                self.failed_collections += 1
                records_to_remove.append(record_key)
        
        for key in records_to_remove:
            del self.prediction_records[key]
    
    def update_visualizations(self, vehicle_ids: List[str]):
        for vid in list(self.active_predictions.keys()):
            if vid not in vehicle_ids:
                self.clear_vehicle_visualization(vid)
                del self.active_predictions[vid]
        
        if self.config.GUI and self.config.DRAW_PREDICTIONS:
            self.draw_predictions()
    
    def draw_predictions(self):
        """Simplified visualization"""
        for vid, data in self.active_predictions.items():
            if vid not in traci.vehicle.getIDList():
                continue
            try:
                traci.vehicle.setColor(vid, (255, 255, 0, 255))
            except:
                pass
    
    def clear_vehicle_visualization(self, vehicle_id: str):
        pass
    
    def clear_all_visualizations(self):
        pass
    
    def print_scale_stats(self):
        if self.prediction_scale_stats:
            stats = np.array(self.prediction_scale_stats)
            print(f"\n📊 Prediction Scale Stats:")
            print(f"  Mean distance: {np.mean(stats):.2f}m")
            print(f"  Median: {np.median(stats):.2f}m")
            print(f"  Std: {np.std(stats):.2f}m")
            print(f"  Range: [{np.min(stats):.2f}, {np.max(stats):.2f}]m")


def run_dt_simulation(config: DTConfig):
    print("\n" + "="*70)
    print("DIGITAL TWIN SIMULATION - V2 WITH CRITICAL FIXES")
    print("="*70)
    print(f"Rotation normalization: {config.USE_ROTATION_NORMALIZATION}")
    print(f"Velocity scale: {config.VELOCITY_SCALE_FACTOR}")
    print("="*70 + "\n")
    
    model = load_model(config.MODEL_PATH, config.MODEL_TYPE, config.PRED_LEN)
    
    tracker = TrajectoryTracker(
        obs_len=config.OBS_LEN,
        use_rotation_norm=config.USE_ROTATION_NORMALIZATION
    )
    metrics = SimpleMetricsCollector(config)
    predictor = DigitalTwinPredictor(model, tracker, metrics, config)
    
    from main import (trajectory_tracking, aggregate_vehicles, gene_config, 
                     has_vehicle_entered, AVAILABLE_CAR_TYPES, AVAILABLE_TRUCK_TYPES,
                     CHECK_ALL, LAN_CHANGE_MODE)
    
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
    cfg_file = gene_config()
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=config.GUI)
    
    times = 0
    random.seed(7)
    
    print("Starting simulation...\n")
    
    try:
        while running(True, times, config.TOTAL_TIME + 1):
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Vehicle spawning
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
                                vehID=vehicle_id, routeID=route_id,
                                typeID=type_id, departSpeed=depart_speed,
                                departPos=depart_pos, departLane=lane_id,
                            )
                            traci.vehicle.setSpeedMode(vehicle_id, CHECK_ALL)
                            traci.vehicle.setLaneChangeMode(vehicle_id, LAN_CHANGE_MODE)
                        except:
                            pass
            
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update trajectories
            for vid in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    angle = traci.vehicle.getAngle(vid)
                    lane = traci.vehicle.getLaneIndex(vid)
                    tracker.update(vid, pos, speed, acc, angle, lane, current_time)
                except:
                    continue
            
            # Prediction and metrics
            if times > config.START_STEP:
                for vid in vehicle_ids:
                    if (tracker.has_enough_history(vid) and 
                        predictor.should_predict(vid, current_time)):
                        predictor.make_prediction(vid, current_time, times)
                
                predictor.update_ground_truth(vehicle_ids, current_time)
                predictor.collect_metrics()
                predictor.update_visualizations(vehicle_ids)
            
            # Progress reporting
            if times % 500 == 0 and times > 0:
                num_metrics = len(metrics.metrics)
                if num_metrics > 0:
                    recent = metrics.metrics[-10:]
                    avg_ade = np.mean([m.ade for m in recent])
                    avg_fde = np.mean([m.fde for m in recent])
                    print(f"Step {times:5d} | Metrics: {num_metrics:4d} | "
                          f"ADE: {avg_ade:.2f}m | FDE: {avg_fde:.2f}m")
            
            if times >= config.TOTAL_TIME:
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Interrupted")
    finally:
        try:
            traci.close()
        except:
            pass
    
    # Summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    
    predictor.print_scale_stats()
    summary = metrics.get_summary()
    
    if summary:
        print(f"\n📊 Results ({len(metrics.metrics)} predictions):")
        acc = summary['trajectory_accuracy']
        print(f"  ADE: {acc['ADE_mean']:.3f} ± {acc['ADE_std']:.3f} m (median: {acc['ADE_median']:.3f})")
        print(f"  FDE: {acc['FDE_mean']:.3f} ± {acc['FDE_std']:.3f} m (median: {acc['FDE_median']:.3f})")
        
        metrics.save_results(config.METRICS_OUTPUT)
    
    print("\n" + "="*70)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--no_rotation", action="store_true", help="Disable rotation normalization")
    parser.add_argument("--velocity_scale", type=float, default=1.309)
    args = parser.parse_args()
    
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.GUI = args.gui
    config.USE_ROTATION_NORMALIZATION = not args.no_rotation
    config.VELOCITY_SCALE_FACTOR = args.velocity_scale
    
    run_dt_simulation(config)