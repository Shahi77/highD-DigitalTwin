#!/usr/bin/env python3
"""
dt_new.py – Enhanced Digital Twin with Complete Metrics & Proper Visualization

Improvements over original:
 ✓ Per-step ADE/FDE metrics at each prediction horizon
 ✓ Precision/Accuracy at 0.5m, 1.0m, 2.0m thresholds
 ✓ Detailed latency breakdown: observation, inference, communication, E2E
 ✓ Simplified parallel-line visualization matching reference code
 ✓ Comprehensive metrics summary with statistics
"""

import os, sys, time, json, random, argparse
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM
from utils import check_sumo_env, start_sumo, running

check_sumo_env()
import traci

# ==================== CONFIG ====================
@dataclass
class DTConfig:
    MODEL_TYPE: str = "slstm"
    MODEL_PATH: str = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    
    OBS_LEN: int = 20
    PRED_LEN: int = 20
    K_NEIGHBORS: int = 8
    
    TRAINING_FREQ_HZ: int = 4
    PREDICTION_INTERVAL_MS: int = 500
    MIN_GT_FRAMES: int = 8
    MAX_GT_WAIT_STEPS: int = 100
    
    USE_ROTATION_NORMALIZATION: bool = True
    POSITION_SCALE: float = 1.0
    
    # Viz
    DRAW_PREDICTIONS: bool = True
    DRAW_GROUND_TRUTH: bool = True
    PRED_DISPLAY_LEN: int = 8
    DASH_LENGTH: float = 0.5
    DASH_GAP: float = 0.3
    LINE_WIDTH: float = 0.4
    LATERAL_OFFSET: float = 0.8
    PRED_LINE_COLOR: Tuple = (0, 255, 0, 255)
    GT_LINE_COLOR: Tuple = (255, 50, 50, 255)
    VEHICLE_COLOR: Tuple = (255, 255, 0, 255)
    
    GUI: bool = True
    TOTAL_TIME: int = 4000
    START_STEP: int = 100
    METRICS_OUTPUT: str = "./dt_results/dt_system_metrics.json"
    VIZ_INTERVAL: int = 4
    ACC_WINDOW: int = 200
    REPORT_INTERVAL: int = 500

config = DTConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ==================== HELPERS ====================
def sumo_angle_to_heading(angle_deg: float) -> float:
    return np.radians(90.0 - angle_deg)

def relative_to_absolute(pred_rel: np.ndarray, origin: Tuple, yaw: float, scale: float = 1.0) -> np.ndarray:
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    pred_abs = np.zeros_like(pred_rel)
    pred_abs[:, 0] = origin[0] + (cos_y * pred_rel[:, 0] - sin_y * pred_rel[:, 1])
    pred_abs[:, 1] = origin[1] + (sin_y * pred_rel[:, 0] + cos_y * pred_rel[:, 1])
    return pred_abs

# ==================== ENHANCED METRICS ====================
@dataclass
class PredictionMetrics:
    vehicle_id: str
    prediction_step: int
    ade_per_step: List[float]  # ADE at horizons 1,2,...,H
    fde_per_step: List[float]  # FDE at horizons 1,2,...,H
    ade: float
    fde: float
    precision_05m: float
    precision_10m: float
    precision_20m: float
    observation_latency_ms: float
    inference_latency_ms: float
    communication_latency_ms: float
    e2e_latency_ms: float
    num_gt_frames: int
    prediction_timestamp: float
    pred_distance: float

class MetricsCollector:
    def __init__(self, cfg: DTConfig):
        self.cfg = cfg
        self.trajectory_metrics: List[PredictionMetrics] = []
        self.system_metrics: List[Dict] = []
        self.pred_distances: List[float] = []

    def add_trajectory_metric(
        self,
        vehicle_id,
        step,
        pred_rel,
        gt_rel,
        obs_ms,
        inf_ms,
        comm_ms,
        e2e_ms,
        ts,
    ):
        dists = np.linalg.norm(pred_rel - gt_rel, axis=1)
        ade_per_step = [float(np.mean(dists[: i + 1])) for i in range(len(dists))]
        fde_per_step = [float(dists[i]) for i in range(len(dists))]
        ade, fde = float(np.mean(dists)), float(dists[-1])
        prec_05 = float(np.mean(dists <= 0.5))
        prec_10 = float(np.mean(dists <= 1.0))
        prec_20 = float(np.mean(dists <= 2.0))
        pred_dist = float(np.linalg.norm(pred_rel[-1]))
        self.pred_distances.append(pred_dist)
        
        pm = PredictionMetrics(
            vehicle_id,
            step,
            ade_per_step,
            fde_per_step,
            ade,
            fde,
            prec_05,
            prec_10,
            prec_20,
            obs_ms,
            inf_ms,
            comm_ms,
            e2e_ms,
            len(gt_rel),
            ts,
            pred_dist,
        )
        self.trajectory_metrics.append(pm)

    def add_system_metric(self, entry: Dict):
        self.system_metrics.append(entry)

    def save(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        
        if self.trajectory_metrics:
            m = self.trajectory_metrics
            summary = {
                "total_predictions": len(m),
                "ade_mean": float(np.mean([x.ade for x in m])),
                "ade_std": float(np.std([x.ade for x in m])),
                "fde_mean": float(np.mean([x.fde for x in m])),
                "fde_std": float(np.std([x.fde for x in m])),
                "precision_05m": float(np.mean([x.precision_05m for x in m])),
                "precision_10m": float(np.mean([x.precision_10m for x in m])),
                "precision_20m": float(np.mean([x.precision_20m for x in m])),
                "observation_latency_mean_ms": float(np.mean([x.observation_latency_ms for x in m])),
                "inference_latency_mean_ms": float(np.mean([x.inference_latency_ms for x in m])),
                "inference_latency_std_ms": float(np.std([x.inference_latency_ms for x in m])),
                "communication_latency_mean_ms": float(np.mean([x.communication_latency_ms for x in m])),
                "e2e_latency_mean_ms": float(np.mean([x.e2e_latency_ms for x in m])),
                "e2e_latency_std_ms": float(np.std([x.e2e_latency_ms for x in m])),
            }
            
            # Per-step aggregate
            max_len = max(len(x.ade_per_step) for x in m)
            ade_by_step = []
            fde_by_step = []
            for i in range(max_len):
                ades = [x.ade_per_step[i] for x in m if i < len(x.ade_per_step)]
                fdes = [x.fde_per_step[i] for x in m if i < len(x.fde_per_step)]
                ade_by_step.append(
                    {"step": i + 1, "mean": float(np.mean(ades)), "std": float(np.std(ades))}
                )
                fde_by_step.append(
                    {"step": i + 1, "mean": float(np.mean(fdes)), "std": float(np.std(fdes))}
                )
            summary["ade_by_step"] = ade_by_step
            summary["fde_by_step"] = fde_by_step
        else:
            summary = {}
        
        out = {
            "summary": summary,
            "trajectory_metrics": [asdict(x) for x in self.trajectory_metrics],
            "system_metrics": self.system_metrics[-500:],
            "pred_scale": {
                "mean": float(np.mean(self.pred_distances)) if self.pred_distances else 0.0,
                "std": float(np.std(self.pred_distances)) if self.pred_distances else 0.0,
            },
        }
        
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n✓ Saved metrics to {out_path}")
        
        if summary:
            print("\n" + "=" * 70)
            print(" " * 20 + "FINAL METRICS SUMMARY")
            print("=" * 70)
            print(f"Total Predictions: {summary['total_predictions']}")
            print(f"\nAccuracy Metrics:")
            print(f"  ADE: {summary['ade_mean']:.3f} ± {summary['ade_std']:.3f} m")
            print(f"  FDE: {summary['fde_mean']:.3f} ± {summary['fde_std']:.3f} m")
            print(f"\nPrecision (% within threshold):")
            print(f"  @0.5m: {summary['precision_05m'] * 100:6.2f}%")
            print(f"  @1.0m: {summary['precision_10m'] * 100:6.2f}%")
            print(f"  @2.0m: {summary['precision_20m'] * 100:6.2f}%")
            print(f"\nLatency Breakdown:")
            print(f"  Observation:    {summary['observation_latency_mean_ms']:6.2f} ms")
            print(
                f"  Inference:      {summary['inference_latency_mean_ms']:6.2f} ± "
                f"{summary['inference_latency_std_ms']:.2f} ms"
            )
            print(
                f"  Communication:  {summary['communication_latency_mean_ms']:6.2f} ms"
            )
            print(
                f"  End-to-End:     {summary['e2e_latency_mean_ms']:6.2f} ± "
                f"{summary['e2e_latency_std_ms']:.2f} ms"
            )
            print("=" * 70 + "\n")

# ==================== TRACKER ====================
class TrajectoryTracker:
    def __init__(self, obs_len=20, use_rot_norm=True, scale=1.0):
        self.obs_len, self.use_rot_norm, self.scale = obs_len, use_rot_norm, scale
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        self.last_sample_time = defaultdict(float)

    def should_sample(self, vid, t, freq_hz=4):
        return vid not in self.last_sample_time or (t - self.last_sample_time[vid]) >= (1.0 / freq_hz)

    def update(self, vid, pos, speed, acc, angle_deg, lane_idx, t):
        if not self.should_sample(vid, t, config.TRAINING_FREQ_HZ):
            return
        heading = sumo_angle_to_heading(angle_deg)
        vx, vy = speed * np.cos(heading), speed * np.sin(heading)
        ax, ay = acc * np.cos(heading), acc * np.sin(heading)
        x, y = pos[0] * config.POSITION_SCALE, pos[1] * config.POSITION_SCALE
        self.trajectories[vid].append(
            {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "heading": heading,
                "lane_id": lane_idx,
                "time": t,
            }
        )
        self.last_sample_time[vid] = t

    def has_enough_history(self, vid):
        return vid in self.trajectories and len(self.trajectories[vid]) >= self.obs_len

    def get_observation(self, vid):
        if not self.has_enough_history(vid):
            return None
        traj = list(self.trajectories[vid])
        obs = np.array(
            [[f["x"], f["y"], f["vx"], f["vy"], f["ax"], f["ay"], f["heading"]] for f in traj],
            dtype=np.float32,
        )
        last = traj[-1]
        last_pos, last_heading = (last["x"], last["y"]), last["heading"]
        if self.use_rot_norm:
            c, s = np.cos(-last_heading), np.sin(-last_heading)
            norm = obs.copy()
            dx, dy = obs[:, 0] - last_pos[0], obs[:, 1] - last_pos[1]
            norm[:, 0], norm[:, 1] = c * dx - s * dy, s * dx + c * dy
            norm[:, 2], norm[:, 3] = c * obs[:, 2] - s * obs[:, 3], s * obs[:, 2] + c * obs[:, 3]
            norm[:, 4], norm[:, 5] = c * obs[:, 4] - s * obs[:, 5], s * obs[:, 4] + c * obs[:, 5]
            norm[:, 6] = ((obs[:, 6] - last_heading + np.pi) % (2 * np.pi)) - np.pi
            return norm, last_pos, last_heading
        else:
            obs[:, :2] -= last_pos
            return obs, last_pos, last_heading

# ==================== PREDICTION RECORD ====================
class PredictionRecord:
    def __init__(
        self,
        vid,
        step,
        pred_rel,
        last_pos,
        last_heading,
        obs_ms,
        inf_ms,
        comm_ms,
        e2e_ms,
        ts,
        hz,
    ):
        self.vehicle_id, self.step, self.pred_relative = vid, step, pred_rel
        self.last_pos, self.last_heading = last_pos, last_heading
        self.obs_latency_ms, self.inference_ms = obs_ms, inf_ms
        self.comm_latency_ms, self.e2e_ms = comm_ms, e2e_ms
        self.timestamp, self.sample_hz = ts, hz
        self.ground_truth: List[Tuple[float, float]] = []
        self.last_gt_time, self.wait_steps = ts, 0

    def add_ground_truth_frame(self, pos, cur_t, scale):
        if cur_t - self.last_gt_time >= (1.0 / self.sample_hz):
            self.ground_truth.append((pos[0] * scale, pos[1] * scale))
            self.last_gt_time = cur_t

    def can_evaluate(self, min_frames):
        return len(self.ground_truth) >= min_frames

    def should_timeout(self, max_steps):
        self.wait_steps += 1
        return self.wait_steps > max_steps

    def get_metrics(self, max_len):
        if not self.ground_truth:
            return None
        c, s = np.cos(-self.last_heading), np.sin(-self.last_heading)
        gt_rel = np.zeros((len(self.ground_truth), 2), dtype=np.float32)
        for i, p in enumerate(self.ground_truth):
            dx, dy = p[0] - self.last_pos[0], p[1] - self.last_pos[1]
            gt_rel[i] = [c * dx - s * dy, s * dx + c * dy]
        use_len = min(len(self.pred_relative), len(gt_rel), max_len)
        return self.pred_relative[:use_len], gt_rel[:use_len]

# ==================== MODEL LOADER ====================
def load_model(path, model_type, pred_len):
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len, k_neighbors=config.K_NEIGHBORS)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len, k_neighbors=config.K_NEIGHBORS)

    state = torch.load(path, map_location=device)
    # state can be a plain state_dict or a checkpoint object
    if isinstance(state, dict) and all(k.startswith("module.") or "." in k for k in state.keys()):
        model.load_state_dict(state)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        # fallback if it was saved as model object
        try:
            model.load_state_dict(state.state_dict())
        except Exception:
            model.load_state_dict(state)

    model.to(device).eval()
    print(f"✓ Loaded {model_type.upper()} model on {device}")
    return model

# ==================== PREDICTOR ====================
class DigitalTwinPredictor:
    def __init__(self, model, tracker, metrics, cfg: DTConfig):
        self.model, self.tracker, self.metrics, self.cfg = model, tracker, metrics, cfg
        self.pred_records: Dict[str, PredictionRecord] = {}
        self.active_predictions: Dict[str, Dict] = {}
        self.last_pred_time: Dict[str, float] = {}
        self.drawn_objects = set()
        self.total_predictions = 0
        self.collected_metrics = 0
        self.failed_collections = 0

    def should_predict(self, vid, cur_time):
        return vid not in self.last_pred_time or (cur_time - self.last_pred_time[vid]) * 1000 >= self.cfg.PREDICTION_INTERVAL_MS

    def make_prediction(self, vid, cur_time, step):
        latency = {"obs": 0.0, "inf": 0.0, "comm": 0.0, "e2e": 0.0}
        e2e_start = time.time()
        
        obs_start = time.time()
        res = self.tracker.get_observation(vid)
        if not res:
            return False, latency
        obs_norm, last_pos, last_heading = res
        latency["obs"] = (time.time() - obs_start) * 1000.0
        
        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(device)
        nd = torch.zeros(1, self.cfg.K_NEIGHBORS, self.cfg.OBS_LEN, 7, device=device)
        ns = torch.zeros(1, self.cfg.K_NEIGHBORS, self.cfg.OBS_LEN, 18, device=device)
        lane = torch.zeros(1, self.cfg.OBS_LEN, 1, device=device)

        try:
            inf_start = time.time()
            if self.cfg.MODEL_TYPE == "transformer":
                with torch.no_grad():
                    preds = self.model(
                        obs_t,
                        nd,
                        ns,
                        lane,
                        last_obs_pos=torch.as_tensor([last_pos], dtype=torch.float32, device=device),
                    )
                pred_abs = preds.cpu().numpy()[0]
                c, s = np.cos(-last_heading), np.sin(-last_heading)
                diffs = pred_abs - np.array(last_pos)
                pred_rel = np.stack(
                    [c * diffs[:, 0] - s * diffs[:, 1], s * diffs[:, 0] + c * diffs[:, 1]],
                    axis=-1,
                )
            else:
                with torch.no_grad():
                    preds = self.model(obs_t, nd, ns, lane)
                pred_rel = preds.cpu().numpy()[0]
                pred_abs = relative_to_absolute(pred_rel, last_pos, last_heading, self.cfg.POSITION_SCALE)
            latency["inf"] = (time.time() - inf_start) * 1000.0
            
            comm_start = time.time()
            # artificial communication delay for DT pipeline
            time.sleep(0.01)
            latency["comm"] = (time.time() - comm_start) * 1000.0
            latency["e2e"] = (time.time() - e2e_start) * 1000.0

            # sanity check
            if np.linalg.norm(pred_rel[-1]) > 1e4:
                return False, latency

            rec = PredictionRecord(
                vid,
                step,
                pred_rel,
                last_pos,
                last_heading,
                latency["obs"],
                latency["inf"],
                latency["comm"],
                latency["e2e"],
                cur_time,
                self.cfg.TRAINING_FREQ_HZ,
            )
            key = f"{vid}_{step}"
            self.pred_records[key] = rec
            self.active_predictions[vid] = {
                "pred_abs": pred_abs,
                "record_key": key,
                "timestamp": cur_time,
            }
            self.last_pred_time[vid] = cur_time
            self.total_predictions += 1
            return True, latency
        except Exception as e:
            print(f"[ERR] Prediction for {vid} failed: {e}")
            return False, latency

    def update_ground_truth(self, vids, cur_time):
        for k, rec in list(self.pred_records.items()):
            if rec.vehicle_id in vids:
                try:
                    pos = traci.vehicle.getPosition(rec.vehicle_id)
                    rec.add_ground_truth_frame(pos, cur_time, self.cfg.POSITION_SCALE)
                except Exception:
                    pass

    def collect_metrics(self):
        to_remove = []
        for k, rec in list(self.pred_records.items()):
            if rec.can_evaluate(self.cfg.MIN_GT_FRAMES):
                out = rec.get_metrics(self.cfg.PRED_LEN)
                if out:
                    pred_rel, gt_rel = out
                    self.metrics.add_trajectory_metric(
                        rec.vehicle_id,
                        rec.step,
                        pred_rel,
                        gt_rel,
                        rec.obs_latency_ms,
                        rec.inference_ms,
                        rec.comm_latency_ms,
                        rec.e2e_ms,
                        rec.timestamp,
                    )
                    self.collected_metrics += 1
                to_remove.append(k)
            elif rec.should_timeout(self.cfg.MAX_GT_WAIT_STEPS):
                self.failed_collections += 1
                to_remove.append(k)
        for k in to_remove:
            self.pred_records.pop(k, None)

    def update_visualizations(self, vids):
        # clear stale
        for vid in list(self.active_predictions.keys()):
            if vid not in vids:
                self.clear_vehicle_visualization(vid)
                self.active_predictions.pop(vid, None)

        if not self.cfg.GUI or not self.cfg.DRAW_PREDICTIONS:
            return

        for vid, data in self.active_predictions.items():
            try:
                if vid not in traci.vehicle.getIDList():
                    continue
                pos = traci.vehicle.getPosition(vid)
                angle = traci.vehicle.getAngle(vid)
                speed = traci.vehicle.getSpeed(vid)
                vlen = traci.vehicle.getLength(vid)

                ang_rad = np.radians(angle)
                dx, dy = np.sin(ang_rad), np.cos(ang_rad)
                perp_dx, perp_dy = -dy, dx
                front_off = vlen / 2.0 + 1.0
                sx, sy = pos[0] + dx * front_off, pos[1] + dy * front_off
                use_len = min(self.cfg.PRED_DISPLAY_LEN, self.cfg.PRED_LEN)
                
                # prediction line
                pred_pts = [
                    (
                        sx + dx * speed * (i + 1) * 0.25 + perp_dx * self.cfg.LATERAL_OFFSET,
                        sy + dy * speed * (i + 1) * 0.25 + perp_dy * self.cfg.LATERAL_OFFSET,
                    )
                    for i in range(use_len)
                ]
                if len(pred_pts) > 1:
                    self._draw_dashed_line_absolute(vid, "pred", pred_pts, self.cfg.PRED_LINE_COLOR)
                
                # ground-truth reference line (parallel, lower)
                if self.cfg.DRAW_GROUND_TRUTH:
                    gt_pts = [
                        (
                            sx + dx * speed * (i + 1) * 0.25 - perp_dx * self.cfg.LATERAL_OFFSET,
                            sy + dy * speed * (i + 1) * 0.25 - perp_dy * self.cfg.LATERAL_OFFSET,
                        )
                        for i in range(use_len)
                    ]
                    if len(gt_pts) > 1:
                        self._draw_dashed_line_absolute(vid, "gt", gt_pts, self.cfg.GT_LINE_COLOR)
                
                traci.vehicle.setColor(vid, self.cfg.VEHICLE_COLOR)
            except Exception:
                pass

    def _draw_dashed_line_absolute(self, vid, prefix, pts, color):
        if len(pts) < 2:
            return
        seg_id, acc = 0, 0.0
        dash_on = True
        seg_pts = [pts[0]]
        for i in range(1, len(pts)):
            dist = np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            if dash_on:
                seg_pts.append(pts[i])
            acc += dist
            if dash_on and acc >= self.cfg.DASH_LENGTH:
                if len(seg_pts) >= 2:
                    pid = f"{prefix}_{vid}_{seg_id}"
                    self._safe_remove_object(pid)
                    try:
                        traci.polygon.add(
                            pid,
                            shape=seg_pts,
                            color=color,
                            fill=False,
                            lineWidth=self.cfg.LINE_WIDTH,
                            layer=102,
                        )
                        self.drawn_objects.add(pid)
                    except Exception:
                        pass
                    seg_id += 1
                dash_on, acc, seg_pts = False, 0.0, []
            elif not dash_on and acc >= self.cfg.DASH_GAP:
                dash_on, acc, seg_pts = True, 0.0, [pts[i]]
        if dash_on and len(seg_pts) >= 2:
            pid = f"{prefix}_{vid}_{seg_id}"
            self._safe_remove_object(pid)
            try:
                traci.polygon.add(
                    pid,
                    shape=seg_pts,
                    color=color,
                    fill=False,
                    lineWidth=self.cfg.LINE_WIDTH,
                    layer=102,
                )
                self.drawn_objects.add(pid)
            except Exception:
                pass

    def _safe_remove_object(self, oid):
        if oid in self.drawn_objects:
            try:
                traci.polygon.remove(oid)
            except Exception:
                pass
            self.drawn_objects.discard(oid)

    def clear_vehicle_visualization(self, vid):
        for pfx in ("pred", "gt"):
            for i in range(300):
                self._safe_remove_object(f"{pfx}_{vid}_{i}")

    def clear_all_visualizations(self):
        for obj in list(self.drawn_objects):
            self._safe_remove_object(obj)

# ==================== MAIN LOOP ====================
def run_dt_simulation(cfg: DTConfig):
    print("\n" + "=" * 70)
    print(" " * 15 + "DIGITAL TWIN SIMULATION (fixed coords + conventions)")
    print("=" * 70 + "\n")
    
    print(f" Loaded {cfg.MODEL_TYPE.upper()} model from {cfg.MODEL_PATH} on {device}")
    print(f" SUMO_HOME: {os.environ.get('SUMO_HOME', 'NOT SET')}")
    
    model = load_model(cfg.MODEL_PATH, cfg.MODEL_TYPE, cfg.PRED_LEN)
    tracker = TrajectoryTracker(cfg.OBS_LEN, cfg.USE_ROTATION_NORMALIZATION, cfg.POSITION_SCALE)
    metrics = MetricsCollector(cfg)
    predictor = DigitalTwinPredictor(model, tracker, metrics, cfg)

    # import helpers from your existing highD DT pipeline
    from main import (
        trajectory_tracking,
        aggregate_vehicles,
        gene_config,
        has_vehicle_entered,
        AVAILABLE_CAR_TYPES,
        AVAILABLE_TRUCK_TYPES,
        CHECK_ALL,
        LAN_CHANGE_MODE,
    )

    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    cfg_file = gene_config()
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=cfg.GUI)

    print("Starting simulation...")
    times = 0
    random.seed(7)
    
    try:
        while running(True, times, cfg.TOTAL_TIME + 1):
            cycle_start = time.time()
            sumo_start = time.time()
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            sumo_step_ms = (time.time() - sumo_start) * 1000.0

            # spawn vehicles (every 4 steps -> 4 Hz)
            if times > 0 and times % 4 == 0:
                step = int(times / 4)
                if has_vehicle_entered(step, vehicles_to_enter):
                    for data in vehicles_to_enter[step]:
                        vc = data["class"].lower()
                        if "truck" in vc or "bus" in vc:
                            type_id = random.choice(AVAILABLE_TRUCK_TYPES)
                            ds = random.uniform(24.0, 25.0)
                        else:
                            type_id = random.choice(AVAILABLE_CAR_TYPES)
                            ds = random.uniform(31.0, 33.0)
                        lane_id = max(0, min(2, int(data.get("laneId", 1)) - 1))
                        dp = random.uniform(10.0, 30.0)
                        direction = data.get("drivingDirection", 1)
                        rid = f"route_direction{direction}"
                        vid = f"d{direction}_{data['id']}"
                        try:
                            traci.vehicle.add(
                                vid,
                                rid,
                                type_id,
                                departSpeed=ds,
                                departPos=dp,
                                departLane=lane_id,
                            )
                            traci.vehicle.setSpeedMode(vid, CHECK_ALL)
                            traci.vehicle.setLaneChangeMode(vid, LAN_CHANGE_MODE)
                        except Exception:
                            pass

            vids = traci.vehicle.getIDList()

            # observation collection
            obs_start = time.time()
            for vid in vids:
                try:
                    tracker.update(
                        vid,
                        traci.vehicle.getPosition(vid),
                        traci.vehicle.getSpeed(vid),
                        traci.vehicle.getAcceleration(vid),
                        traci.vehicle.getAngle(vid),
                        traci.vehicle.getLaneIndex(vid),
                        current_time,
                    )
                except Exception:
                    pass
            obs_ms = (time.time() - obs_start) * 1000.0

            preds_made = 0
            inf_times: List[float] = []
            traci_ms = 0.0
            viz_ms = 0.0
            pred_overhead_ms = 0.0

            if times > cfg.START_STEP:
                pred_block_start = time.time()

                # predictions
                for vid in vids:
                    if tracker.has_enough_history(vid) and predictor.should_predict(vid, current_time):
                        ok, lat = predictor.make_prediction(vid, current_time, times)
                        if ok:
                            preds_made += 1
                            inf_times.append(lat["inf"])

                # ground-truth accumulation and metric collection
                traci_start = time.time()
                predictor.update_ground_truth(vids, current_time)
                predictor.collect_metrics()
                traci_ms = (time.time() - traci_start) * 1000.0

                # visualization
                viz_start = time.time()
                if times % cfg.VIZ_INTERVAL == 0:
                    predictor.update_visualizations(vids)
                viz_ms = (time.time() - viz_start) * 1000.0

                pred_overhead_ms = (time.time() - pred_block_start) * 1000.0 - (traci_ms + viz_ms)

            # full cycle latency
            cycle_ms = (time.time() - cycle_start) * 1000.0
            mean_inf = float(np.mean(inf_times)) if inf_times else 0.0

            # system-level metrics per step
            metrics.add_system_metric(
                {
                    "step": int(times),
                    "sim_time": float(current_time),
                    "num_vehicles": int(len(vids)),
                    "predictions_made": int(preds_made),
                    "sumo_step_ms": float(sumo_step_ms),
                    "obs_collection_ms": float(obs_ms),
                    "inference_mean_ms": float(mean_inf),
                    "prediction_overhead_ms": float(pred_overhead_ms),
                    "traci_bookkeeping_ms": float(traci_ms),
                    "visualization_ms": float(viz_ms),
                    "cycle_total_ms": float(cycle_ms),
                }
            )

            # progress reporting
            if times > cfg.START_STEP and times % cfg.REPORT_INTERVAL == 0:
                recent = metrics.trajectory_metrics[-cfg.ACC_WINDOW :] if metrics.trajectory_metrics else []
                if recent:
                    ade_mean = float(np.mean([m.ade for m in recent]))
                    fde_mean = float(np.mean([m.fde for m in recent]))
                    p05 = float(np.mean([m.precision_05m for m in recent])) * 100.0
                    p10 = float(np.mean([m.precision_10m for m in recent])) * 100.0
                    p20 = float(np.mean([m.precision_20m for m in recent])) * 100.0
                    lat_mean = float(np.mean([m.inference_latency_ms for m in recent]))

                    print(
                        f"Step {times:5d} | Veh {len(vids):4d} | PredMade {preds_made:4d} | "
                        f"Metrics {len(metrics.trajectory_metrics):5d}"
                    )
                    print(
                        f"  ADE {ade_mean:.3f} m | FDE {fde_mean:.3f} m | "
                        f"Acc@0.5/1/2m: {p05:4.2f}/{p10:4.2f}/{p20:4.2f}% | "
                        f"Lat {lat_mean:.2f} ms"
                    )

            times += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down...")
    finally:
        print("\nCleaning up visualization and closing SUMO...")
        try:
            predictor.clear_all_visualizations()
        except Exception:
            pass
        try:
            traci.close(False)
        except Exception:
            pass

        print(
            f"\nStep {times:5d} | TotalPred {predictor.total_predictions} | "
            f"Collected {predictor.collected_metrics} | Failed {predictor.failed_collections}"
        )
        metrics.save(cfg.METRICS_OUTPUT)

# ==================== CLI ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Digital Twin simulation for highD with metrics")
    parser.add_argument("--model_path", type=str, default=config.MODEL_PATH)
    parser.add_argument("--model_type", type=str, choices=["slstm", "transformer"], default=config.MODEL_TYPE)
    parser.add_argument("--obs_len", type=int, default=config.OBS_LEN)
    parser.add_argument("--pred_len", type=int, default=config.PRED_LEN)
    parser.add_argument("--total_time", type=int, default=config.TOTAL_TIME)
    parser.add_argument("--start_step", type=int, default=config.START_STEP)
    parser.add_argument("--metrics_output", type=str, default=config.METRICS_OUTPUT)
    parser.add_argument("--no-rot-norm", action="store_true", help="disable rotation normalization")
    parser.add_argument("--gui", dest="gui", action="store_true", help="enable SUMO GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="disable SUMO GUI")
    parser.set_defaults(gui=config.GUI)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.OBS_LEN = args.obs_len
    config.PRED_LEN = args.pred_len
    config.TOTAL_TIME = args.total_time
    config.START_STEP = args.start_step
    config.METRICS_OUTPUT = args.metrics_output
    config.USE_ROTATION_NORMALIZATION = not args.no_rot_norm
    config.GUI = args.gui

    run_dt_simulation(config)
