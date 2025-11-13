"""
Fixed Digital Twin SUMO Simulation - Final Version
--------------------------------------------------
- Proper coordinate transformation
- Close-spaced dashed lines
- Working metrics collection
- Real-time predictions for all vehicles
"""

import os
import sys
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


@dataclass
class DTConfig:
    MODEL_TYPE: str = "slstm"
    MODEL_PATH: str = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    OBS_LEN: int = 20
    PRED_LEN: int = 20
    K_NEIGHBORS: int = 8
    PREDICTION_INTERVAL_MS: int = 500
    
    # Visualization - much tighter, closer dashes
    DRAW_PREDICTIONS: bool = True
    PRED_DISPLAY_LEN: int = 8  # Shorter - 2 seconds only
    DASH_LENGTH: float = 0.3  # Short dashes
    DASH_GAP: float = 0.15  # Small gaps
    LINE_WIDTH: float = 0.25  # Thinner
    LATERAL_OFFSET: float = 0.35  # Very close to vehicle
    
    # Colors
    PRED_LINE_COLOR: Tuple[int, int, int, int] = (0, 255, 0, 255)  # Green
    TRUE_LINE_COLOR: Tuple[int, int, int, int] = (255, 0, 0, 255)  # Red
    VEHICLE_COLOR: Tuple[int, int, int, int] = (255, 255, 0, 255)  # Yellow
    
    GUI: bool = True
    TOTAL_TIME: int = 4000
    START_STEP: int = 100
    USE_DT_PREDICTION: bool = True
    METRICS_OUTPUT: str = "./dt_metrics_fixed.json"


config = DTConfig()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class EnhancedPredictionMetrics:
    vehicle_id: str
    timestamp: float
    ade: float
    fde: float
    mae: float
    rmse: float
    inference_latency_ms: float
    e2e_latency_ms: float
    speed_mae: float
    acc_mae: float
    frame_alignment: float
    temporal_drift_ms: float


class EnhancedMetricsCollector:
    def __init__(self, config: DTConfig):
        self.config = config
        self.predictions: List[EnhancedPredictionMetrics] = []
        self.latency_samples: List[float] = []
        self.e2e_latency_samples: List[float] = []
        self.frame_alignments: List[float] = []
        self.temporal_drifts: List[float] = []
        self.prediction_timestamps: List[float] = []
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.start_time = time.time()
        
    def add_prediction(self, vehicle_id: str, timestamp: float,
                      pred_traj: np.ndarray, true_traj: np.ndarray,
                      pred_speed: np.ndarray, true_speed: np.ndarray,
                      pred_acc: np.ndarray, true_acc: np.ndarray,
                      inference_time_ms: float, e2e_time_ms: float,
                      frame_alignment: float, temporal_drift_ms: float):
        
        displacements = np.linalg.norm(pred_traj - true_traj, axis=1)
        ade = np.mean(displacements)
        fde = displacements[-1]
        mae = np.mean(np.abs(pred_traj - true_traj))
        rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
        
        speed_mae = np.mean(np.abs(pred_speed - true_speed))
        acc_mae = np.mean(np.abs(pred_acc - true_acc))
        
        metrics = EnhancedPredictionMetrics(
            vehicle_id=vehicle_id, timestamp=timestamp,
            ade=ade, fde=fde, mae=mae, rmse=rmse,
            inference_latency_ms=inference_time_ms,
            e2e_latency_ms=e2e_time_ms,
            speed_mae=speed_mae, acc_mae=acc_mae,
            frame_alignment=frame_alignment,
            temporal_drift_ms=temporal_drift_ms
        )
        
        self.predictions.append(metrics)
        self.latency_samples.append(inference_time_ms)
        self.e2e_latency_samples.append(e2e_time_ms)
        self.frame_alignments.append(frame_alignment)
        self.temporal_drifts.append(temporal_drift_ms)
        self.prediction_timestamps.append(timestamp)
        self.successful_predictions += 1
    
    def calculate_advanced_metrics(self) -> Dict:
        if not self.predictions:
            return {}
        
        ades = [p.ade for p in self.predictions]
        fdes = [p.fde for p in self.predictions]
        
        latency_jitter = np.std(self.latency_samples) if len(self.latency_samples) > 1 else 0
        far = np.mean(self.frame_alignments) if self.frame_alignments else 0
        avg_temporal_drift = np.mean(self.temporal_drifts) if self.temporal_drifts else 0
        
        if len(self.prediction_timestamps) > 1:
            intervals = np.diff(self.prediction_timestamps) * 1000
            expected_interval = self.config.PREDICTION_INTERVAL_MS
            usi = 1.0 - np.mean(np.abs(intervals - expected_interval) / expected_interval)
            usi = max(0.0, min(1.0, usi))
        else:
            usi = 1.0
        
        prediction_refresh_rate = len(self.predictions) / (time.time() - self.start_time)
        adaptation_success_ratio = self.successful_predictions / (
            self.successful_predictions + self.failed_predictions
        ) if (self.successful_predictions + self.failed_predictions) > 0 else 0
        
        accuracy_score = 1.0 / (1.0 + np.mean(ades))
        dtci = 0.4 * accuracy_score + 0.3 * far + 0.3 * usi
        
        total_compute_time = sum(self.latency_samples) / 1000
        ce = self.successful_predictions / total_compute_time if total_compute_time > 0 else 0
        
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
                "inference_latency_mean_ms": float(np.mean(self.latency_samples)),
                "inference_latency_p95_ms": float(np.percentile(self.latency_samples, 95)),
                "e2e_latency_mean_ms": float(np.mean(self.e2e_latency_samples)),
                "latency_jitter_ms": float(latency_jitter),
            },
            "synchronization_metrics": {
                "frame_alignment_ratio": float(far),
                "temporal_drift_mean_ms": float(avg_temporal_drift),
                "update_synchrony_index": float(usi),
            },
            "responsiveness_metrics": {
                "prediction_refresh_rate_hz": float(prediction_refresh_rate),
                "adaptation_success_ratio": float(adaptation_success_ratio),
            },
            "composite_metrics": {
                "digital_twin_coherence_index": float(dtci),
                "computational_efficiency_pred_per_sec": float(ce),
            },
            "performance_summary": {
                "total_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "success_rate": float(adaptation_success_ratio),
            }
        }
    
    def save_results(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        summary = self.calculate_advanced_metrics()
        summary["raw_predictions_sample"] = [asdict(p) for p in self.predictions[-50:]]
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Metrics saved to {filepath}")


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


class GroundTruthTracker:
    def __init__(self, config: DTConfig):
        self.config = config
        self.future_trajectories = {}
        
    def start_tracking(self, vehicle_id: str, start_time: float):
        if vehicle_id not in self.future_trajectories:
            self.future_trajectories[vehicle_id] = {}
        self.future_trajectories[vehicle_id][start_time] = []
    
    def update_position(self, vehicle_id: str, position: Tuple[float, float], 
                       speed: float, acceleration: float):
        if vehicle_id not in self.future_trajectories:
            return
        
        for start_time in list(self.future_trajectories[vehicle_id].keys()):
            self.future_trajectories[vehicle_id][start_time].append({
                'pos': position, 'speed': speed, 'acc': acceleration
            })
    
    def get_ground_truth(self, vehicle_id: str, start_time: float, 
                        start_pos: Tuple[float, float]) -> Optional[Dict]:
        if (vehicle_id not in self.future_trajectories or 
            start_time not in self.future_trajectories[vehicle_id]):
            return None
        
        traj_data = self.future_trajectories[vehicle_id][start_time]
        if len(traj_data) < self.config.PRED_LEN:
            return None
        
        # Convert to RELATIVE coordinates
        true_traj = np.zeros((self.config.PRED_LEN, 2))
        true_speeds = np.zeros(self.config.PRED_LEN)
        true_accs = np.zeros(self.config.PRED_LEN)
        
        for i in range(self.config.PRED_LEN):
            true_traj[i, 0] = traj_data[i]['pos'][0] - start_pos[0]
            true_traj[i, 1] = traj_data[i]['pos'][1] - start_pos[1]
            true_speeds[i] = traj_data[i]['speed']
            true_accs[i] = traj_data[i]['acc']
        
        del self.future_trajectories[vehicle_id][start_time]
        
        return {
            'trajectory': true_traj,
            'speeds': true_speeds,
            'accelerations': true_accs
        }


class TrajectoryTracker:
    def __init__(self, obs_len: int = 20):
        self.obs_len = obs_len
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        self.last_prediction_time = {}
        
    def update(self, vehicle_id: str, position: Tuple[float, float], 
               velocity: float, acceleration: float, lane_id: int):
        self.trajectories[vehicle_id].append({
            'x': position[0], 'y': position[1],
            'vx': velocity, 'vy': 0.0,
            'ax': acceleration, 'ay': 0.0,
            'lane_id': lane_id,
            'timestamp': traci.simulation.getTime()
        })
    
    def get_observation(self, vehicle_id: str) -> Optional[np.ndarray]:
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
    
    def should_predict(self, vehicle_id: str, current_time: float, interval_ms: int) -> bool:
        if vehicle_id not in self.last_prediction_time:
            return True
        time_since_last = (current_time - self.last_prediction_time[vehicle_id]) * 1000
        return time_since_last >= interval_ms
    
    def mark_predicted(self, vehicle_id: str, current_time: float):
        self.last_prediction_time[vehicle_id] = current_time
    
    def has_enough_history(self, vehicle_id: str) -> bool:
        return (vehicle_id in self.trajectories and 
                len(self.trajectories[vehicle_id]) >= self.obs_len)


class DigitalTwinPredictor:
    def __init__(self, model, tracker: TrajectoryTracker, 
                 gt_tracker: GroundTruthTracker,
                 metrics: EnhancedMetricsCollector, config: DTConfig):
        self.model = model
        self.tracker = tracker
        self.gt_tracker = gt_tracker
        self.metrics = metrics
        self.config = config
        self.active_predictions = {}
        self.drawn_objects = set()
        self.prediction_start_times = {}
    
    def _safe_remove_object(self, obj_id: str):
        if obj_id in self.drawn_objects:
            try:
                traci.polygon.remove(obj_id)
            except:
                pass
            self.drawn_objects.discard(obj_id)
    
    def make_prediction(self, vehicle_id: str, current_time: float):
        e2e_start = time.time()
        
        obs = self.tracker.get_observation(vehicle_id)
        if obs is None:
            return None
        
        try:
            start_pos = traci.vehicle.getPosition(vehicle_id)
        except:
            return None
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        nd = torch.zeros(1, self.model.k, self.config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, self.model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        inference_start = time.time()
        
        try:
            with torch.no_grad():
                if hasattr(self.model, "multi_att"):
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = self.model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = self.model(obs_tensor, nd, ns, lane)
            
            inference_time_ms = (time.time() - inference_start) * 1000
            e2e_time_ms = (time.time() - e2e_start) * 1000
            pred_np = pred.cpu().numpy()[0]
            
            self.gt_tracker.start_tracking(vehicle_id, current_time)
            self.prediction_start_times[vehicle_id] = (current_time, start_pos)
            
            return pred_np, inference_time_ms, e2e_time_ms, start_pos
            
        except Exception as e:
            self.metrics.failed_predictions += 1
            return None
    
    def update_predictions(self, vehicle_ids: List[str], current_time: float):
        # Clean up
        vehicles_to_remove = [vid for vid in self.active_predictions.keys() 
                            if vid not in vehicle_ids]
        for vid in vehicles_to_remove:
            self.clear_vehicle_visualization(vid)
            del self.active_predictions[vid]
            if vid in self.prediction_start_times:
                del self.prediction_start_times[vid]
        
        # Make predictions for ALL vehicles
        eligible = [vid for vid in vehicle_ids 
                   if self.tracker.has_enough_history(vid) and
                      self.tracker.should_predict(vid, current_time, 
                                                 self.config.PREDICTION_INTERVAL_MS)]
        
        for vid in eligible:
            result = self.make_prediction(vid, current_time)
            if result is None:
                continue
            
            pred, inference_time_ms, e2e_time_ms, start_pos = result
            
            self.active_predictions[vid] = {
                'prediction': pred,
                'start_pos': start_pos,
                'inference_time_ms': inference_time_ms,
                'e2e_time_ms': e2e_time_ms,
                'timestamp': current_time
            }
            
            self.tracker.mark_predicted(vid, current_time)
    
    def collect_metrics(self, vehicle_ids: List[str], current_time: float):
        for vid in list(self.prediction_start_times.keys()):
            pred_start_time, start_pos = self.prediction_start_times[vid]
            time_elapsed = current_time - pred_start_time
            
            # Collect GT after just 4 seconds (16 frames) - more realistic for highway
            if time_elapsed >= 4.0:  
                gt_data = self.gt_tracker.get_ground_truth(vid, pred_start_time, start_pos)
                
                if gt_data and vid in self.active_predictions:
                    pred_data = self.active_predictions[vid]
                    pred_traj = pred_data['prediction']
                    
                    # Use only what we have (16 frames minimum)
                    actual_len = len(gt_data['trajectory'])
                    display_len = min(self.config.PRED_DISPLAY_LEN, actual_len, len(pred_traj))
                    
                    if display_len >= 8:  # Only if we have at least 8 frames (2 seconds)
                        frame_alignment = min(actual_len / self.config.PRED_LEN, 1.0)
                        expected_time_ms = display_len * 250  # Adjusted for actual length
                        actual_time_ms = time_elapsed * 1000
                        temporal_drift_ms = abs(actual_time_ms - expected_time_ms)
                        
                        pred_speeds = np.linalg.norm(np.diff(pred_traj[:display_len], axis=0), axis=1) / 0.25
                        true_speeds = gt_data['speeds'][:display_len-1]
                        
                        self.metrics.add_prediction(
                            vehicle_id=vid,
                            timestamp=pred_start_time,
                            pred_traj=pred_traj[:display_len],
                            true_traj=gt_data['trajectory'][:display_len],
                            pred_speed=pred_speeds,
                            true_speed=true_speeds,
                            pred_acc=np.zeros(display_len-1),
                            true_acc=gt_data['accelerations'][:display_len-1],
                            inference_time_ms=pred_data['inference_time_ms'],
                            e2e_time_ms=pred_data['e2e_time_ms'],
                            frame_alignment=frame_alignment,
                            temporal_drift_ms=temporal_drift_ms
                        )
                    
                    del self.prediction_start_times[vid]
        
    def draw_predictions(self):
        if not self.config.GUI or not self.config.DRAW_PREDICTIONS:
            return
        
        for vid, data in self.active_predictions.items():
            if vid not in traci.vehicle.getIDList():
                continue

            try:
                current_pos = traci.vehicle.getPosition(vid)
                angle = traci.vehicle.getAngle(vid)
                pred = data['prediction']
                display_len = min(self.config.PRED_DISPLAY_LEN, len(pred))

                # highlight vehicle
                traci.vehicle.setColor(vid, self.config.VEHICLE_COLOR)

                # perpendicular offset
                angle_rad = np.radians(90 - angle)
                perp_x = -np.sin(angle_rad) * self.config.LATERAL_OFFSET
                perp_y = np.cos(angle_rad) * self.config.LATERAL_OFFSET

                # --- FIX 1: convert to relative ---
                traj_world = pred[:display_len]
                pred_rel = traj_world - traj_world[0]

                # --- FIX 2: rotate world-relative → vehicle-local ---
                dx = pred_rel[:, 0]
                dy = pred_rel[:, 1]

                x_local =  dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
                y_local = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

                pred_local = np.column_stack((x_local, y_local))

                # draw predicted path (green)
                self._draw_dashed_line(
                    vid,
                    "pred",
                    current_pos[0] + perp_x,
                    current_pos[1] + perp_y,
                    pred_local,
                    self.config.PRED_LINE_COLOR
                )
                # debug for first vehicle

                # draw simple ground truth forward line (red)
                try:
                    speed = traci.vehicle.getSpeed(vid)
                    vx = speed * np.cos(angle_rad)
                    vy = speed * np.sin(angle_rad)

                    true_traj = np.zeros((display_len, 2))
                    for i in range(display_len):
                        dt = (i + 1) * 0.25
                        true_traj[i, 0] = vx * dt
                        true_traj[i, 1] = vy * dt

                    self._draw_dashed_line(
                        vid,
                        "true",
                        current_pos[0] - perp_x,
                        current_pos[1] - perp_y,
                        true_traj,
                        self.config.TRUE_LINE_COLOR
                    )
                except:
                    pass

            except:
                continue

    def _draw_dashed_line(self, vehicle_id, prefix, start_x, start_y, trajectory, color):
        dash_len = self.config.DASH_LENGTH
        gap_len = self.config.DASH_GAP
        width = self.config.LINE_WIDTH

        seg_id = 0
        acc_dist = 0.0
        dash_active = True
        seg_points = []
        last_x = start_x
        last_y = start_y

        for i in range(len(trajectory)):
            x = start_x + trajectory[i, 0]
            y = start_y + trajectory[i, 1]

            dx = x - last_x
            dy = y - last_y
            dist = np.hypot(dx, dy)

            if dash_active:
                seg_points.append((x, y))

            acc_dist += dist

            if dash_active and acc_dist >= dash_len:

                if len(seg_points) >= 2:
                    poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
                    self._safe_remove_object(poly_id)

                    traci.polygon.add(
                        poly_id,
                        shape=seg_points,
                        color=color,
                        fill=False,
                        lineWidth=width,
                        layer=102
                    )
                    self.drawn_objects.add(poly_id)

                    seg_id += 1

                dash_active = False
                acc_dist = 0.0
                seg_points = []

            elif (not dash_active) and acc_dist >= gap_len:
                dash_active = True
                acc_dist = 0.0
                seg_points = [(x, y)]

            last_x = x
            last_y = y

        # Final dash
        if dash_active and len(seg_points) >= 2:
            poly_id = f"{prefix}_{vehicle_id}_{seg_id}"
            self._safe_remove_object(poly_id)

            traci.polygon.add(
                poly_id,
                shape=seg_points,
                color=color,
                fill=False,
                lineWidth=width,
                layer=102
            )
            self.drawn_objects.add(poly_id)

    def clear_vehicle_visualization(self, vehicle_id: str):
        for prefix in ["pred_", "true_"]:
            for i in range(200):
                self._safe_remove_object(f"{prefix}{vehicle_id}_{i}")
    
    def clear_all_visualizations(self):
        for obj_id in list(self.drawn_objects):
            self._safe_remove_object(obj_id)


def run_dt_simulation(config: DTConfig):
    print("\n" + "="*70)
    print("REAL-TIME DIGITAL TWIN SUMO SIMULATION")
    print("="*70)
    print(f"Mode: DT-ENABLED with Continuous Predictions")
    print(f"Visualization: Green=Predicted | Red=Actual | Yellow=Vehicle")
    print("="*70 + "\n")
    
    model = None
    if config.USE_DT_PREDICTION:
        model = load_model(config.MODEL_PATH, config.MODEL_TYPE, config.PRED_LEN)
    
    tracker = TrajectoryTracker(obs_len=config.OBS_LEN)
    gt_tracker = GroundTruthTracker(config)
    metrics = EnhancedMetricsCollector(config)
    
    predictor = None
    if config.USE_DT_PREDICTION and model is not None:
        predictor = DigitalTwinPredictor(model, tracker, gt_tracker, metrics, config)
    
    from main import (trajectory_tracking, aggregate_vehicles, gene_config, 
                     has_vehicle_entered, AVAILABLE_CAR_TYPES, AVAILABLE_TRUCK_TYPES,
                     CHECK_ALL, LAN_CHANGE_MODE)
    
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
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
            
            # Add vehicles
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
                            vehicles_added += 1
                        except:
                            pass
            
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update tracking
            for vid in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    lane = traci.vehicle.getLaneIndex(vid)
                    tracker.update(vid, pos, speed, acc, lane)
                    gt_tracker.update_position(vid, pos, speed, acc)
                except:
                    continue
            
            # DT predictions (for ALL vehicles continuously)
            if config.USE_DT_PREDICTION and predictor and times > config.START_STEP:
                predictor.update_predictions(vehicle_ids, current_time)
                predictor.collect_metrics(vehicle_ids, current_time)
                predictor.draw_predictions()
            
            if times % 500 == 0 and times > 0:
                num_active = len(vehicle_ids)
                num_predictions = len(predictor.active_predictions) if predictor else 0
                num_metrics = len(metrics.predictions) if predictor else 0
                
                # Show real-time metrics
                if num_metrics > 0:
                    recent_ades = [p.ade for p in metrics.predictions[-10:]]
                    recent_latency = metrics.latency_samples[-10:] if len(metrics.latency_samples) > 0 else [0]
                    print(f"Step {times:5d}: {num_active:3d} vehicles | {num_predictions:3d} active | "
                        f"{num_metrics:4d} metrics collected | "
                        f"ADE: {np.mean(recent_ades):.2f}m | "
                        f"Lat: {np.mean(recent_latency):.2f}ms")
                else:
                    print(f"Step {times:5d}: {num_active:3d} vehicles | {num_predictions:3d} active | "
                        f"{num_metrics:4d} metrics (waiting for GT...)")
                
            if times >= config.TOTAL_TIME:
                print(f"\n✓ Reached target time: {config.TOTAL_TIME} steps")
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Simulation interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if predictor:
            predictor.clear_all_visualizations()
        try:
            traci.close()
        except:
            pass
        print("\n💾 Saving metrics...")
        time.sleep(1)
    
    # Print results
    print("\n" + "="*70)
    print("SIMULATION COMPLETE - METRICS SUMMARY")
    print("="*70)
    
    summary = metrics.calculate_advanced_metrics()

    if summary and summary.get('performance_summary', {}).get('total_predictions', 0) > 0:
        print("\n Trajectory Accuracy:")
        acc = summary['trajectory_accuracy']
        print(f"  ADE: {acc['ADE_mean']:.3f} ± {acc['ADE_std']:.3f} m (median: {acc['ADE_median']:.3f})")
        print(f"  FDE: {acc['FDE_mean']:.3f} ± {acc['FDE_std']:.3f} m (median: {acc['FDE_median']:.3f})")
        
        print("\n Latency Metrics:")
        lat = summary['latency_metrics']
        print(f"  Inference Latency (Lᵢ): {lat['inference_latency_mean_ms']:.2f} ms")
        print(f"  E2E Latency (Lₑ): {lat['e2e_latency_mean_ms']:.2f} ms")
        print(f"  Latency Jitter (σₗ): {lat['latency_jitter_ms']:.2f} ms")
        print(f"  P95 Latency: {lat['inference_latency_p95_ms']:.2f} ms")
        
        print("\n Synchronization Metrics:")
        sync = summary['synchronization_metrics']
        print(f"  Frame Alignment Ratio (FAR): {sync['frame_alignment_ratio']:.4f}")
        print(f"  Temporal Drift (Δt_d): {sync['temporal_drift_mean_ms']:.2f} ms")
        print(f"  Update Synchrony Index (USI): {sync['update_synchrony_index']:.4f}")
        
        print("\n Responsiveness Metrics:")
        resp = summary['responsiveness_metrics']
        print(f"  Prediction Refresh Rate (PRR): {resp['prediction_refresh_rate_hz']:.2f} Hz")
        print(f"  Adaptation Success Ratio (ASR): {resp['adaptation_success_ratio']:.4f}")
        
        print("\n Composite Metrics:")
        comp = summary['composite_metrics']
        print(f"  Digital Twin Coherence Index (DTCI): {comp['digital_twin_coherence_index']:.4f}")
        print(f"  Computational Efficiency (CE): {comp['computational_efficiency_pred_per_sec']:.2f} pred/s")
        
        print("\nPerformance Summary:")
        perf = summary['performance_summary']
        print(f"  Total predictions: {perf['total_predictions']}")
        print(f"  Failed predictions: {perf['failed_predictions']}")
        print(f"  Success rate: {perf['success_rate']*100:.1f}%")
        
        metrics.save_results(config.METRICS_OUTPUT)
    else:
        print("\nNo full summary available — saving raw metrics instead.\n")

        try:
            if len(metrics.predictions) > 0:
                with open("dt_metrics.json", "w") as f:
                    json.dump([m.to_dict() for m in metrics.predictions], f, indent=2)
                print(f"Saved dt_metrics.json with {len(metrics.predictions)} entries")
            else:
                print("No metrics to save")
        except Exception as e:
            print("Could not write JSON:", e)
            print("Dumping metrics to console...")
            for m in metrics.predictions:
                print(m.to_dict())

    
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fixed Digital Twin SUMO Simulation")
    parser.add_argument("--mode", choices=["dt", "baseline"], default="dt")
    parser.add_argument("--model_path", type=str,
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--total_time", type=int, default=4000)
    parser.add_argument("--output", type=str, default="./dt_metrics_fixed.json")

    args = parser.parse_args()

    config.USE_DT_PREDICTION = (args.mode == "dt")
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.GUI = args.gui
    config.TOTAL_TIME = args.total_time
    config.METRICS_OUTPUT = args.output

    run_dt_simulation(config)