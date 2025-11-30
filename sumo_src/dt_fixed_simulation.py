"""
Digital Twin SUMO Simulation - SYSTEM-LEVEL METRICS + ENHANCED VISUALIZATION
-----------------------------------------------------------------------------
Adds comprehensive system performance tracking:
- Streaming delay (observation collection time)
- Prediction-to-TraCI injection delay
- End-to-end cycle time
- Throughput metrics
- Synchronization stability
- Temporal drift analysis
- Fixed ground truth visualization (red lines)
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
    
    USE_ROTATION_NORMALIZATION: bool = True
    VELOCITY_SCALE_FACTOR: float = 1.0
    MAX_PREDICTION_DISTANCE: float = 200.0
    
    # Visualization
    DRAW_PREDICTIONS: bool = True
    DRAW_GROUND_TRUTH: bool = True  # NEW: Explicit GT control
    PRED_DISPLAY_LEN: int = 8
    DASH_LENGTH: float = 0.5
    DASH_GAP: float = 0.3
    LINE_WIDTH: float = 0.4
    LATERAL_OFFSET: float = 0.9
    PRED_LINE_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 255)  # Green
    GT_LINE_COLOR: Tuple[int, int, int, int] = (255, 50, 50, 255)    # Bright Red
    VEHICLE_COLOR: Tuple[int, int, int, int] = (255, 255, 0, 255)
    
    # Adaptive visualization
    ADAPTIVE_VIZ: bool = True
    HIGH_DENSITY_THRESHOLD: int = 50
    REDUCED_DISPLAY_LEN: int = 5
    
    GUI: bool = True
    TOTAL_TIME: int = 4000
    START_STEP: int = 100
    METRICS_OUTPUT: str = "./dt_results/dt_system_metrics.json"


config = DTConfig()
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    step: int
    timestamp: float
    
    # Streaming metrics
    observation_collection_ms: float
    data_preprocessing_ms: float
    
    # Prediction metrics
    inference_latency_ms: float
    prediction_processing_ms: float
    prediction_loop_overhead_ms: float
    
    # TraCI injection metrics
    traci_update_ms: float
    visualization_update_ms: float
    
    # SUMO simulation
    sumo_step_ms: float
    
    # End-to-end metrics
    total_cycle_ms: float
    
    # Throughput metrics
    active_vehicles: int
    predictions_made: int
    gt_updates: int
    
    # Synchronization metrics
    prediction_queue_depth: int
    pending_gt_collection: int
    queue_variance: float
    
    # Drift metrics
    sim_time: float
    wall_time: float
    time_ratio: float


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
    pred_distance: float


class EnhancedMetricsCollector:
    """Collects both trajectory and system-level metrics"""
    def __init__(self, config: DTConfig):
        self.config = config
        self.trajectory_metrics: List[PredictionMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.start_time = time.time()
        self.start_sim_time = 0.0
        
        # Training data statistics for validation
        self.training_pred_distances: List[float] = []
        
        # Queue variance tracking
        self.queue_history = deque(maxlen=100)
        
    def add_trajectory_metric(self, vehicle_id: str, prediction_step: int,
                             pred_traj: np.ndarray, true_traj: np.ndarray,
                             inference_ms: float, e2e_ms: float,
                             timestamp: float):
        displacements = np.linalg.norm(pred_traj - true_traj, axis=1)
        ade = float(np.mean(displacements))
        fde = float(displacements[-1])
        pred_distance = float(np.linalg.norm(pred_traj[-1]))
        eval_distance = float(np.linalg.norm(true_traj[-1]))
        
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
        
        self.trajectory_metrics.append(metric)
        self.training_pred_distances.append(pred_distance)
    
    def add_system_metric(self, step: int, timestamp: float,
                         obs_collect_ms: float, preprocess_ms: float,
                         inference_ms: float, pred_process_ms: float,
                         pred_loop_ms: float, traci_ms: float, viz_ms: float,
                         sumo_step_ms: float, total_cycle_ms: float,
                         active_vehicles: int, predictions_made: int,
                         gt_updates: int, queue_depth: int,
                         pending_gt: int):
        
        wall_time = time.time() - self.start_time
        if self.start_sim_time == 0.0:
            self.start_sim_time = timestamp
        
        sim_elapsed = timestamp - self.start_sim_time
        time_ratio = sim_elapsed / wall_time if wall_time > 0 else 0.0
        
        # Track queue variance
        self.queue_history.append(queue_depth)
        queue_variance = float(np.var(list(self.queue_history))) if len(self.queue_history) > 1 else 0.0
        
        metric = SystemMetrics(
            step=step,
            timestamp=timestamp,
            observation_collection_ms=obs_collect_ms,
            data_preprocessing_ms=preprocess_ms,
            inference_latency_ms=inference_ms,
            prediction_processing_ms=pred_process_ms,
            prediction_loop_overhead_ms=pred_loop_ms,
            traci_update_ms=traci_ms,
            visualization_update_ms=viz_ms,
            sumo_step_ms=sumo_step_ms,
            total_cycle_ms=total_cycle_ms,
            active_vehicles=active_vehicles,
            predictions_made=predictions_made,
            gt_updates=gt_updates,
            prediction_queue_depth=queue_depth,
            pending_gt_collection=pending_gt,
            queue_variance=queue_variance,
            sim_time=timestamp,
            wall_time=wall_time,
            time_ratio=time_ratio
        )
        
        self.system_metrics.append(metric)
    
    def get_trajectory_summary(self) -> Dict:
        if not self.trajectory_metrics:
            return {}
        
        ades = [m.ade for m in self.trajectory_metrics]
        fdes = [m.fde for m in self.trajectory_metrics]
        inference_times = [m.inference_latency_ms for m in self.trajectory_metrics]
        pred_distances = [m.pred_distance for m in self.trajectory_metrics]
        
        return {
            "trajectory_accuracy": {
                "ADE_mean": float(np.mean(ades)),
                "ADE_std": float(np.std(ades)),
                "ADE_median": float(np.median(ades)),
                "ADE_min": float(np.min(ades)),
                "ADE_max": float(np.max(ades)),
                "FDE_mean": float(np.mean(fdes)),
                "FDE_std": float(np.std(fdes)),
                "FDE_median": float(np.median(fdes)),
                "FDE_min": float(np.min(fdes)),
                "FDE_max": float(np.max(fdes)),
            },
            "inference_latency": {
                "mean_ms": float(np.mean(inference_times)),
                "std_ms": float(np.std(inference_times)),
                "p95_ms": float(np.percentile(inference_times, 95)),
                "p99_ms": float(np.percentile(inference_times, 99)),
            },
            "prediction_stats": {
                "avg_distance": float(np.mean(pred_distances)),
                "std_distance": float(np.std(pred_distances)),
                "total_predictions": len(self.trajectory_metrics),
            }
        }
    
    def get_system_summary(self) -> Dict:
        if not self.system_metrics:
            return {}
        
        # Calculate statistics for each metric
        obs_times = [m.observation_collection_ms for m in self.system_metrics]
        inference_times = [m.inference_latency_ms for m in self.system_metrics if m.inference_latency_ms > 0]
        pred_loop_times = [m.prediction_loop_overhead_ms for m in self.system_metrics]
        traci_times = [m.traci_update_ms for m in self.system_metrics]
        viz_times = [m.visualization_update_ms for m in self.system_metrics]
        sumo_times = [m.sumo_step_ms for m in self.system_metrics]
        total_times = [m.total_cycle_ms for m in self.system_metrics]
        
        active_vehicles = [m.active_vehicles for m in self.system_metrics]
        predictions_made = [m.predictions_made for m in self.system_metrics]
        queue_depths = [m.prediction_queue_depth for m in self.system_metrics]
        queue_variances = [m.queue_variance for m in self.system_metrics]
        time_ratios = [m.time_ratio for m in self.system_metrics if m.time_ratio > 0]
        
        # Calculate unaccounted time
        avg_total = np.mean(total_times)
        avg_accounted = (np.mean(obs_times) + np.mean(inference_times) + 
                        np.mean(pred_loop_times) + np.mean(traci_times) + 
                        np.mean(viz_times) + np.mean(sumo_times))
        unaccounted_ms = avg_total - avg_accounted
        unaccounted_pct = (unaccounted_ms / avg_total * 100) if avg_total > 0 else 0
        
        return {
            "streaming_delay": {
                "observation_collection_mean_ms": float(np.mean(obs_times)),
                "observation_collection_std_ms": float(np.std(obs_times)),
                "observation_collection_p95_ms": float(np.percentile(obs_times, 95)),
            },
            "prediction_to_traci_delay": {
                "traci_update_mean_ms": float(np.mean(traci_times)),
                "traci_update_std_ms": float(np.std(traci_times)),
                "traci_update_p95_ms": float(np.percentile(traci_times, 95)),
            },
            "end_to_end_cycle": {
                "total_cycle_mean_ms": float(np.mean(total_times)),
                "total_cycle_std_ms": float(np.std(total_times)),
                "total_cycle_p95_ms": float(np.percentile(total_times, 95)),
                "total_cycle_max_ms": float(np.max(total_times)),
            },
            "sumo_simulation": {
                "sumo_step_mean_ms": float(np.mean(sumo_times)),
                "sumo_step_std_ms": float(np.std(sumo_times)),
                "sumo_step_p95_ms": float(np.percentile(sumo_times, 95)),
            },
            "prediction_overhead": {
                "pred_loop_mean_ms": float(np.mean(pred_loop_times)),
                "pred_loop_std_ms": float(np.std(pred_loop_times)),
                "pred_loop_p95_ms": float(np.percentile(pred_loop_times, 95)),
            },
            "visualization_overhead": {
                "viz_update_mean_ms": float(np.mean(viz_times)),
                "viz_update_std_ms": float(np.std(viz_times)),
                "viz_update_p95_ms": float(np.percentile(viz_times, 95)),
            },
            "throughput": {
                "avg_vehicles_per_step": float(np.mean(active_vehicles)),
                "max_vehicles_per_step": int(np.max(active_vehicles)),
                "avg_predictions_per_step": float(np.mean(predictions_made)),
                "total_predictions": int(np.sum(predictions_made)),
            },
            "synchronization": {
                "avg_queue_depth": float(np.mean(queue_depths)),
                "max_queue_depth": int(np.max(queue_depths)),
                "min_queue_depth": int(np.min(queue_depths)),
                "stability_variance": float(np.var(queue_depths)),
                "avg_rolling_variance": float(np.mean(queue_variances)),
                "queue_fluctuation_range": int(np.max(queue_depths) - np.min(queue_depths)),
            },
            "temporal_drift": {
                "avg_time_ratio": float(np.mean(time_ratios)),
                "time_ratio_variance": float(np.var(time_ratios)),
                "sim_faster_than_realtime": float(np.mean(time_ratios)) > 1.0,
            },
            "breakdown_percentages": {
                "observation_pct": float(np.mean(obs_times) / avg_total * 100),
                "inference_pct": float(np.mean(inference_times) / avg_total * 100) if inference_times else 0.0,
                "pred_loop_pct": float(np.mean(pred_loop_times) / avg_total * 100),
                "traci_pct": float(np.mean(traci_times) / avg_total * 100),
                "visualization_pct": float(np.mean(viz_times) / avg_total * 100),
                "sumo_step_pct": float(np.mean(sumo_times) / avg_total * 100),
                "unaccounted_pct": float(unaccounted_pct),
                "unaccounted_ms": float(unaccounted_ms),
            },
            "prediction_scale_validation": {
                "training_pred_mean_m": float(np.mean(self.training_pred_distances)) if self.training_pred_distances else 0.0,
                "training_pred_std_m": float(np.std(self.training_pred_distances)) if self.training_pred_distances else 0.0,
                "training_pred_median_m": float(np.median(self.training_pred_distances)) if self.training_pred_distances else 0.0,
                "total_samples": len(self.training_pred_distances),
            }
        }
    
    def save_results(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        results = {
            "trajectory_metrics": self.get_trajectory_summary(),
            "system_metrics": self.get_system_summary(),
            "recent_trajectory_samples": [asdict(m) for m in self.trajectory_metrics[-50:]],
            "recent_system_samples": [asdict(m) for m in self.system_metrics[-50:]],
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Saved metrics to {filepath}")


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
            angle_rad = np.radians(90 - angle)
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
        
        obs_abs = np.zeros((self.obs_len, 7))
        for i, frame in enumerate(traj):
            obs_abs[i] = [frame['x'], frame['y'], frame['vx'], frame['vy'],
                         frame['ax'], frame['ay'], frame['heading']]
        
        last_pos = (traj[-1]['x'], traj[-1]['y'])
        last_heading = traj[-1]['heading']
        
        if self.use_rotation_norm:
            obs_normalized = self._apply_rotation_normalization(obs_abs, last_pos, last_heading)
        else:
            obs_normalized = obs_abs.copy()
            obs_normalized[:, 0] -= last_pos[0]
            obs_normalized[:, 1] -= last_pos[1]
        
        return obs_normalized, last_pos, last_heading
    
    def _apply_rotation_normalization(self, obs: np.ndarray, origin: Tuple[float, float], 
                                     yaw: float) -> np.ndarray:
        """Apply same rotation normalization as training dataloader"""
        normalized = obs.copy()
        
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        
        dx = obs[:, 0] - origin[0]
        dy = obs[:, 1] - origin[1]
        normalized[:, 0] = cos_yaw * dx - sin_yaw * dy
        normalized[:, 1] = sin_yaw * dx + cos_yaw * dy
        
        normalized[:, 2] = cos_yaw * obs[:, 2] - sin_yaw * obs[:, 3]
        normalized[:, 3] = sin_yaw * obs[:, 2] + cos_yaw * obs[:, 3]
        
        normalized[:, 4] = cos_yaw * obs[:, 4] - sin_yaw * obs[:, 5]
        normalized[:, 5] = sin_yaw * obs[:, 4] + cos_yaw * obs[:, 5]
        
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
        self.prediction_relative = prediction_relative
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
                 metrics: EnhancedMetricsCollector, config: DTConfig):
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
        
        self.prediction_scale_stats = []
        
        # System metrics tracking
        self.cycle_start_time = 0.0
    
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
            
            pred_np = pred.cpu().numpy()[0]
            # pred_travel_distance = np.linalg.norm(pred_np[-1])
            # print(f"[{vehicle_id}] Model output distance: {pred_travel_distance:.1f}m")
            
            if self.config.VELOCITY_SCALE_FACTOR != 1.0:
                pred_np = pred_np * self.config.VELOCITY_SCALE_FACTOR
            
            pred_magnitude = np.linalg.norm(pred_np[-1])
            self.prediction_scale_stats.append(pred_magnitude)
            
            if pred_magnitude > self.config.MAX_PREDICTION_DISTANCE:
                print(f"⚠️  Suspicious prediction for {vehicle_id}: {pred_magnitude:.1f}m")
                return False
            
            # Create prediction record
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
            
            # For visualization
            pred_absolute = self._to_absolute_coords(pred_np, last_obs_pos, last_obs_heading)
            # absolute_travel = np.linalg.norm(pred_absolute[-1] - pred_absolute[0])
            # print(f"[{vehicle_id}] Absolute travel distance: {absolute_travel:.1f}m")   
            self.active_predictions[vehicle_id] = {
                'prediction': pred_absolute,
                'start_pos': last_obs_pos,
                'timestamp': current_time,
                'step': current_step,
                'record_key': record_key  # Link to record
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
                    
                    self.metrics.add_trajectory_metric(
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
        # Clean up vehicles that no longer exist
        for vid in list(self.active_predictions.keys()):
            if vid not in vehicle_ids:
                self.clear_vehicle_visualization(vid)
                del self.active_predictions[vid]
        
        # ALSO: Clean up predictions that are too old
        current_time = traci.simulation.getTime()
        stale_predictions = []
        for vid, data in self.active_predictions.items():
            if current_time - data['timestamp'] > 5.0:  # 5 seconds old
                stale_predictions.append(vid)
        
        for vid in stale_predictions:
            self.clear_vehicle_visualization(vid)
            del self.active_predictions[vid]
        
        if self.config.GUI and self.config.DRAW_PREDICTIONS:
            self.draw_predictions()

    def draw_predictions(self):
        """Draw prediction and ground truth lines parallel to vehicle trajectory"""
        current_density = len(self.active_predictions)
        if self.config.ADAPTIVE_VIZ and current_density > self.config.HIGH_DENSITY_THRESHOLD:
            display_len = self.config.REDUCED_DISPLAY_LEN
        else:
            display_len = self.config.PRED_DISPLAY_LEN
        
        for vid, data in self.active_predictions.items():
            if vid not in traci.vehicle.getIDList():
                continue

            try:
                current_pos = traci.vehicle.getPosition(vid)
                angle = traci.vehicle.getAngle(vid)
                speed = traci.vehicle.getSpeed(vid)
                pred_absolute = data['prediction']
                use_len = min(display_len, len(pred_absolute))

                traci.vehicle.setColor(vid, self.config.VEHICLE_COLOR)

                # Get vehicle length
                vehicle_length = traci.vehicle.getLength(vid)
                
                angle_rad = np.radians(90 - angle)
                
                # Detect driving direction
                is_leftward = vid.startswith('d2')

                # Calculate forward offset to start lines ahead of vehicle
                # Start lines from front bumper + small gap
                forward_offset = (vehicle_length / 2.0) + 1.0  # Half length + 1m gap

                # Convert prediction from absolute to vehicle-local coordinates
                pred_rel = pred_absolute[:use_len] - pred_absolute[0]
                
                # Rotate to vehicle's local frame
                dx = pred_rel[:, 0]
                dy = pred_rel[:, 1]
                x_local = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                y_local = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
                
                # Flip if leftward traffic
                if is_leftward:
                    x_local = -x_local
                    forward_offset = -forward_offset  # Flip offset direction too
                
                # Shift entire trajectory forward by vehicle length
                x_local = x_local + forward_offset
                
                # Apply offset in local space
                pred_local = np.column_stack((x_local, y_local + self.config.LATERAL_OFFSET))
                
                # Skip first point AND filter out points behind vehicle
                pred_local_filtered = pred_local[pred_local[:, 0] > 0] if not is_leftward else pred_local[pred_local[:, 0] < 0]
                
                if len(pred_local_filtered) > 1:
                    self._draw_dashed_line(
                        vid, "pred",
                        current_pos[0],
                        current_pos[1],
                        pred_local_filtered,
                        self.config.PRED_LINE_COLOR
                    )
                
                # Draw ground truth trajectory (red)
                if self.config.DRAW_GROUND_TRUTH:
                    try:
                        # Create GT in local space - start ahead of vehicle
                        true_traj = np.zeros((use_len, 2))
                        for i in range(use_len):
                            dt = (i + 1) * 0.25
                            forward_distance = speed * dt + forward_offset
                            
                            # Flip if leftward
                            if is_leftward:
                                forward_distance = -forward_distance
                            
                            true_traj[i, 0] = forward_distance
                            true_traj[i, 1] = -self.config.LATERAL_OFFSET
                        
                        # Filter out points behind vehicle
                        true_traj_filtered = true_traj[true_traj[:, 0] > 0] if not is_leftward else true_traj[true_traj[:, 0] < 0]

                        if len(true_traj_filtered) > 1:
                            self._draw_dashed_line(
                                vid, "gt",
                                current_pos[0],
                                current_pos[1],
                                true_traj_filtered,
                                self.config.GT_LINE_COLOR
                            )
                    except:
                        pass
            except Exception as e:
                continue
    def _draw_dashed_line_absolute(self, vehicle_id, prefix, points, color):
        """Draw a dashed line from absolute trajectory points (parallel to path)"""
        if len(points) < 2:
            return
        
        dash_len = self.config.DASH_LENGTH
        gap_len = self.config.DASH_GAP
        width = self.config.LINE_WIDTH

        seg_id = 0
        acc_dist = 0.0
        dash_active = True
        seg_points = [points[0]]
        
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            step_dist = np.hypot(dx, dy)
            
            if dash_active:
                seg_points.append(points[i])
            
            acc_dist += step_dist

            if dash_active and acc_dist >= dash_len:
                if len(seg_points) >= 2:
                    poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
                    self._safe_remove_object(poly_id)
                    try:
                        traci.polygon.add(poly_id, shape=seg_points, color=color, 
                                        fill=False, lineWidth=width, layer=102)
                        self.drawn_objects.add(poly_id)
                    except:
                        pass
                    seg_id += 1
                dash_active = False
                acc_dist = 0.0
                seg_points = []
            elif not dash_active and acc_dist >= gap_len:
                dash_active = True
                acc_dist = 0.0
                seg_points = [points[i]]
        
        # Draw final segment
        if dash_active and len(seg_points) >= 2:
            poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
            self._safe_remove_object(poly_id)
            try:
                traci.polygon.add(poly_id, shape=seg_points, color=color, 
                                fill=False, lineWidth=width, layer=102)
                self.drawn_objects.add(poly_id)
            except:
                pass
    
    def _draw_dashed_line(self, vehicle_id, prefix, start_x, start_y, trajectory, color):
        """Draw a dashed line with improved visibility"""
        dash_len = self.config.DASH_LENGTH
        gap_len = self.config.DASH_GAP
        width = self.config.LINE_WIDTH

        seg_id = 0
        acc_dist = 0.0
        dash_active = True
        seg_points = [(start_x, start_y)]  # Always start with initial point
        last_x, last_y = start_x, start_y

        for i in range(len(trajectory)):
            x = start_x + trajectory[i, 0]
            y = start_y + trajectory[i, 1]

            if dash_active:
                seg_points.append((x, y))

            dx, dy = x - last_x, y - last_y
            step_dist = np.hypot(dx, dy)
            acc_dist += step_dist

            if dash_active and acc_dist >= dash_len:
                if len(seg_points) >= 2:
                    poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
                    self._safe_remove_object(poly_id)
                    try:
                        traci.polygon.add(poly_id, shape=seg_points, color=color, 
                                        fill=False, lineWidth=width, layer=102)
                        self.drawn_objects.add(poly_id)
                    except:
                        pass
                    seg_id += 1
                dash_active = False
                acc_dist = 0.0
                seg_points = []
            elif not dash_active and acc_dist >= gap_len:
                dash_active = True
                acc_dist = 0.0
                seg_points = [(x, y)]

            last_x, last_y = x, y
        
        # Draw final segment if there are remaining points
        if dash_active and len(seg_points) >= 2:
            poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
            self._safe_remove_object(poly_id)
            try:
                traci.polygon.add(poly_id, shape=seg_points, color=color, 
                                fill=False, lineWidth=width, layer=102)
                self.drawn_objects.add(poly_id)
            except:
                pass
    
    def _safe_remove_object(self, obj_id: str):
        if obj_id in self.drawn_objects:
            try:
                traci.polygon.remove(obj_id)
            except:
                pass
            self.drawn_objects.discard(obj_id)
    
    def clear_vehicle_visualization(self, vehicle_id: str):
        """Clear all visualization objects for a vehicle"""
        for prefix in ["pred", "gt"]:
            for i in range(300):  # Increased from 200 to be safe
                obj_id = f"{prefix}_{vehicle_id}_{i}"
                self._safe_remove_object(obj_id)
    
    def clear_all_visualizations(self):
        for obj_id in list(self.drawn_objects):
            self._safe_remove_object(obj_id)
    
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
    print("DIGITAL TWIN SIMULATION - COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"Rotation normalization: {config.USE_ROTATION_NORMALIZATION}")
    print(f"Velocity scale: {config.VELOCITY_SCALE_FACTOR}")
    print(f"Visualization: Predictions={config.DRAW_PREDICTIONS}, GT={config.DRAW_GROUND_TRUTH}")
    print(f"Adaptive viz: {config.ADAPTIVE_VIZ} (threshold: {config.HIGH_DENSITY_THRESHOLD})")
    print("="*70 + "\n")
    
    model = load_model(config.MODEL_PATH, config.MODEL_TYPE, config.PRED_LEN)
    
    tracker = TrajectoryTracker(
        obs_len=config.OBS_LEN,
        use_rotation_norm=config.USE_ROTATION_NORMALIZATION
    )
    metrics = EnhancedMetricsCollector(config)
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
    
    print("Starting simulation with comprehensive profiling...\n")
    
    try:
        while running(True, times, config.TOTAL_TIME + 1):
            cycle_start = time.time()
            
            # Step 1: SUMO simulation step
            sumo_start = time.time()
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            sumo_step_ms = (time.time() - sumo_start) * 1000
            
            # Step 2: Vehicle spawning
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
            
            # Step 3: Observation collection (streaming delay)
            obs_start = time.time()
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
            obs_collect_ms = (time.time() - obs_start) * 1000
            
            # Step 4: Prediction and metrics (only after START_STEP)
            predictions_made = 0
            total_inference_ms = 0.0
            pred_loop_overhead_ms = 0.0
            
            if times > config.START_STEP:
                pred_loop_start = time.time()
                pred_start = time.time()
                
                for vid in vehicle_ids:
                    if (tracker.has_enough_history(vid) and 
                        predictor.should_predict(vid, current_time)):
                        if predictor.make_prediction(vid, current_time, times):
                            predictions_made += 1
                
                pred_process_ms = (time.time() - pred_start) * 1000
                
                # Average inference time from recent predictions
                if predictor.prediction_scale_stats:
                    total_inference_ms = pred_process_ms / max(predictions_made, 1)
                
                # Step 5: Ground truth update (TraCI injection delay)
                traci_start = time.time()
                predictor.update_ground_truth(vehicle_ids, current_time)
                predictor.collect_metrics()
                traci_update_ms = (time.time() - traci_start) * 1000
                
                # Step 6: Visualization update
                viz_start = time.time()
                predictor.update_visualizations(vehicle_ids)
                viz_update_ms = (time.time() - viz_start) * 1000
                
                # Calculate prediction loop overhead
                pred_loop_overhead_ms = (time.time() - pred_loop_start) * 1000 - (pred_process_ms + traci_update_ms + viz_update_ms)
            else:
                pred_process_ms = 0.0
                traci_update_ms = 0.0
                viz_update_ms = 0.0
            
            # Step 7: Calculate total cycle time
            total_cycle_ms = (time.time() - cycle_start) * 1000
            
            # Step 8: Record system metrics (every 10 steps to reduce overhead)
            if times % 10 == 0 and times > config.START_STEP:
                metrics.add_system_metric(
                    step=times,
                    timestamp=current_time,
                    obs_collect_ms=obs_collect_ms,
                    preprocess_ms=0.0,
                    inference_ms=total_inference_ms,
                    pred_process_ms=pred_process_ms,
                    pred_loop_ms=pred_loop_overhead_ms,
                    traci_ms=traci_update_ms,
                    viz_ms=viz_update_ms,
                    sumo_step_ms=sumo_step_ms,
                    total_cycle_ms=total_cycle_ms,
                    active_vehicles=len(vehicle_ids),
                    predictions_made=predictions_made,
                    gt_updates=len([r for r in predictor.prediction_records.values() if len(r.ground_truth) > 0]),
                    queue_depth=len(predictor.active_predictions),
                    pending_gt=len(predictor.prediction_records)
                )
            
            # Progress reporting
            if times % 500 == 0 and times > 0:
                num_vehicles = len(vehicle_ids)
                num_displaying = len(predictor.active_predictions)
                num_records = len(predictor.prediction_records)
                num_metrics = len(metrics.trajectory_metrics)
                
                if num_metrics > 0:
                    recent = metrics.trajectory_metrics[-10:]
                    avg_ade = np.mean([m.ade for m in recent])
                    avg_fde = np.mean([m.fde for m in recent])
                    avg_pred_dist = np.mean([m.pred_distance for m in recent])
                    
                    # System metrics
                    if metrics.system_metrics:
                        recent_sys = metrics.system_metrics[-10:]
                        avg_cycle = np.mean([m.total_cycle_ms for m in recent_sys])
                        avg_ratio = np.mean([m.time_ratio for m in recent_sys if m.time_ratio > 0])
                        avg_queue_var = np.mean([m.queue_variance for m in recent_sys])
                        
                        print(f"Step {times:5d} | Veh: {num_vehicles:3d} | Display: {num_displaying:3d} | "
                              f"Queue: {num_records:3d} (σ²={avg_queue_var:.1f}) | Metrics: {num_metrics:4d}")
                        print(f"  └─ ADE: {avg_ade:.2f}m | FDE: {avg_fde:.2f}m | PredDist: {avg_pred_dist:.1f}m | "
                              f"Cycle: {avg_cycle:.1f}ms | TimeRatio: {avg_ratio:.2f}x")
                    else:
                        print(f"Step {times:5d} | Veh: {num_vehicles:3d} | Display: {num_displaying:3d} | "
                              f"Queue: {num_records:3d} | Metrics: {num_metrics:4d} | "
                              f"ADE: {avg_ade:.2f}m | FDE: {avg_fde:.2f}m | PredDist: {avg_pred_dist:.1f}m")
                else:
                    print(f"Step {times:5d} | Veh: {num_vehicles:3d} | Display: {num_displaying:3d} | "
                          f"Queue: {num_records:3d} | Metrics: {num_metrics:4d} (collecting...)")
            
            if times >= config.TOTAL_TIME:
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Interrupted")
    finally:
        predictor.clear_all_visualizations()
        try:
            traci.close()
        except:
            pass
    
    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE SIMULATION ANALYSIS")
    print("="*70)
    
    predictor.print_scale_stats()
    
    traj_summary = metrics.get_trajectory_summary()
    sys_summary = metrics.get_system_summary()
    
    if traj_summary:
        print(f"\n TRAJECTORY METRICS ({len(metrics.trajectory_metrics)} predictions):")
        print(f"  Total predictions made: {predictor.total_predictions_made}")
        print(f"  Metrics collected: {predictor.total_metrics_collected}")
        print(f"  Failed collections: {predictor.failed_collections}")
        if predictor.total_predictions_made > 0:
            print(f"  Collection rate: {predictor.total_metrics_collected/predictor.total_predictions_made*100:.1f}%")
        
        acc = traj_summary['trajectory_accuracy']
        print(f"\n Trajectory Accuracy:")
        print(f"  ADE: {acc['ADE_mean']:.3f} ± {acc['ADE_std']:.3f} m")
        print(f"       (median: {acc['ADE_median']:.3f}, range: [{acc['ADE_min']:.3f}, {acc['ADE_max']:.3f}])")
        print(f"  FDE: {acc['FDE_mean']:.3f} ± {acc['FDE_std']:.3f} m")
        print(f"       (median: {acc['FDE_median']:.3f}, range: [{acc['FDE_min']:.3f}, {acc['FDE_max']:.3f}])")
    
    if sys_summary:
        print(f"\n SYSTEM PERFORMANCE METRICS:")
        
        # Prediction scale validation
        pred_scale = sys_summary['prediction_scale_validation']
        print(f"\n  Prediction Scale Validation ({pred_scale['total_samples']} samples):")
        print(f"    Mean prediction distance: {pred_scale['training_pred_mean_m']:.2f}m")
        print(f"    Std: {pred_scale['training_pred_std_m']:.2f}m")
        print(f"    Median: {pred_scale['training_pred_median_m']:.2f}m")
        
        stream = sys_summary['streaming_delay']
        print(f"\n  Streaming Delay (Observation Collection):")
        print(f"    Mean: {stream['observation_collection_mean_ms']:.2f} ± {stream['observation_collection_std_ms']:.2f} ms")
        print(f"    P95: {stream['observation_collection_p95_ms']:.2f} ms")
        
        sumo = sys_summary['sumo_simulation']
        print(f"\n  SUMO Simulation Step:")
        print(f"    Mean: {sumo['sumo_step_mean_ms']:.2f} ± {sumo['sumo_step_std_ms']:.2f} ms")
        print(f"    P95: {sumo['sumo_step_p95_ms']:.2f} ms")
        
        pred_overhead = sys_summary['prediction_overhead']
        print(f"\n Prediction Loop Overhead:")
        print(f"    Mean: {pred_overhead['pred_loop_mean_ms']:.2f} ± {pred_overhead['pred_loop_std_ms']:.2f} ms")
        print(f"    P95: {pred_overhead['pred_loop_p95_ms']:.2f} ms")
        
        traci_delay = sys_summary['prediction_to_traci_delay']
        print(f"\n  Prediction-to-TraCI Injection Delay:")
        print(f"    Mean: {traci_delay['traci_update_mean_ms']:.2f} ± {traci_delay['traci_update_std_ms']:.2f} ms")
        print(f"    P95: {traci_delay['traci_update_p95_ms']:.2f} ms")
        
        cycle = sys_summary['end_to_end_cycle']
        print(f"\n  End-to-End Cycle Time:")
        print(f"    Mean: {cycle['total_cycle_mean_ms']:.2f} ± {cycle['total_cycle_std_ms']:.2f} ms")
        print(f"    P95: {cycle['total_cycle_p95_ms']:.2f} ms")
        print(f"    Max: {cycle['total_cycle_max_ms']:.2f} ms")
        
        viz = sys_summary['visualization_overhead']
        print(f"\n  Visualization Overhead:")
        print(f"    Mean: {viz['viz_update_mean_ms']:.2f} ± {viz['viz_update_std_ms']:.2f} ms")
        print(f"    P95: {viz['viz_update_p95_ms']:.2f} ms")
        
        throughput = sys_summary['throughput']
        print(f"\n  Throughput:")
        print(f"    Avg vehicles/step: {throughput['avg_vehicles_per_step']:.1f}")
        print(f"    Max vehicles/step: {throughput['max_vehicles_per_step']}")
        print(f"    Avg predictions/step: {throughput['avg_predictions_per_step']:.2f}")
        print(f"    Total predictions: {throughput['total_predictions']}")
        
        sync = sys_summary['synchronization']
        print(f"\n  Synchronization & Stability:")
        print(f"    Avg queue depth: {sync['avg_queue_depth']:.1f}")
        print(f"    Queue range: [{sync['min_queue_depth']}, {sync['max_queue_depth']}] (Δ={sync['queue_fluctuation_range']})")
        print(f"    Global variance: {sync['stability_variance']:.2f}")
        print(f"    Rolling variance: {sync['avg_rolling_variance']:.2f}")
        print(f"    Analysis: High fluctuation ({sync['queue_fluctuation_range']} vehicles) indicates")
        print(f"        variable traffic flow or vehicle lifetime patterns")
        
        drift = sys_summary['temporal_drift']
        print(f"\n Temporal Drift Analysis:")
        print(f"    Avg time ratio (sim/wall): {drift['avg_time_ratio']:.3f}x")
        print(f"    Time ratio variance: {drift['time_ratio_variance']:.6f}")
        print(f"    Sim faster than realtime: {drift['sim_faster_than_realtime']}")
        
        breakdown = sys_summary['breakdown_percentages']
        print(f"\n Cycle Time Breakdown:")
        print(f"    Observation:      {breakdown['observation_pct']:5.1f}%")
        print(f"    Inference:        {breakdown['inference_pct']:5.1f}%")
        print(f"    Pred Loop:        {breakdown['pred_loop_pct']:5.1f}%")
        print(f"    TraCI:            {breakdown['traci_pct']:5.1f}%")
        print(f"    Visualization:    {breakdown['visualization_pct']:5.1f}%")
        print(f"    SUMO Step:        {breakdown['sumo_step_pct']:5.1f}%")
        print(f"    Unaccounted:      {breakdown['unaccounted_pct']:5.1f}% ({breakdown['unaccounted_ms']:.1f}ms)")
        
        
        metrics.save_results(config.METRICS_OUTPUT)
    
    return traj_summary, sys_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--no_rotation", action="store_true", help="Disable rotation normalization")
    parser.add_argument("--velocity_scale", type=float, default=1.0, 
                       help="Scale factor for predictions")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")
    parser.add_argument("--no_gt_viz", action="store_true", help="Disable ground truth visualization")
    parser.add_argument("--total_time", type=int, default=4000)
    parser.add_argument("--output", type=str, default="./dt_results/dt_system_metrics.json")
    
    args = parser.parse_args()
    
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.GUI = args.gui
    config.USE_ROTATION_NORMALIZATION = not args.no_rotation
    config.VELOCITY_SCALE_FACTOR = args.velocity_scale
    config.DRAW_PREDICTIONS = not args.no_viz
    config.DRAW_GROUND_TRUTH = not args.no_gt_viz
    config.TOTAL_TIME = args.total_time
    config.METRICS_OUTPUT = args.output

    run_dt_simulation(config)