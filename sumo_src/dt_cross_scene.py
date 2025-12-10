#!/usr/bin/env python3
"""
Proper Digital Twin Evaluation Pipeline
========================================
Train on Scene 02 → Test on Scene 09 (cross-scene generalization)

Key Differences from Current Code:
✓ Uses GT playback (not closed-loop predictions)
✓ Tests cross-scene generalization
✓ Computes spatial accuracy & precision
✓ Per-step latency tracking (summary every few hundred steps)
✓ Generates simple GT vs prediction plots
"""

import os
import sys
import time
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# change this if your src folder is elsewhere
sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import SimpleSLSTM, ImprovedTrajectoryTransformer


# ==================== CONFIG ====================

@dataclass
class DTConfig:
    TRAIN_SCENE: str = "02"
    TEST_SCENE: str = "09"

    MODEL_PATH: str = (
        "/Users/shahi/Developer/Project-highD/"
        "results/results_scene02_lstm/checkpoints/best_model.pt"
    )
    MODEL_TYPE: str = "slstm"  # "slstm" or "transformer"

    TEST_TRACKS: str = (
        "/Users/shahi/Developer/Project-highD/data/highd/dataset/09_tracks.csv"
    )

    OBS_LEN: int = 20
    PRED_LEN: int = 20
    MIN_GT_LEN: int = 8  # minimum future steps to compute metrics

    DT_HZ: int = 4  # digital twin update frequency
    HIGHD_FPS: int = 25  # original highD frame rate

    METRICS_OUTPUT: str = "./dt_results_cross_scene/dt_cross_scene_metrics.json"
    VIZ_OUTPUT_DIR: str = "./dt_results_cross_scene/plots"

    REPORT_INTERVAL_FRAMES: int = 500  # how often to print status
    MAX_FRAMES: int = -1  # -1 = full sequence

    SEED: int = 7
    VIZ_SAMPLES: int = 30  # how many predictions to visualize


config = DTConfig()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ==================== HELPERS ====================

def angle_from_velocity(vx: float, vy: float) -> float:
    """Return heading angle in radians from velocity components."""
    if abs(vx) < 1e-4 and abs(vy) < 1e-4:
        return 0.0
    return float(np.arctan2(vy, vx))


def relative_to_absolute(pred_rel: np.ndarray, origin: Tuple[float, float], yaw: float) -> np.ndarray:
    """Convert relative ego frame coords back to global frame."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    pred_abs = np.zeros_like(pred_rel)
    pred_abs[:, 0] = origin[0] + (cos_y * pred_rel[:, 0] - sin_y * pred_rel[:, 1])
    pred_abs[:, 1] = origin[1] + (sin_y * pred_rel[:, 0] + cos_y * pred_rel[:, 1])
    return pred_abs


# ==================== METRICS ====================

@dataclass
class PredictionMetrics:
    vehicle_id: int
    frame: int
    ade: float
    fde: float
    precision_05m: float
    precision_10m: float
    precision_20m: float
    horizon: int
    obs_latency_ms: float
    inf_latency_ms: float
    e2e_latency_ms: float


class MetricsCollector:
    def __init__(self):
        self.prediction_metrics: List[PredictionMetrics] = []
        self.viz_samples: List[Dict] = []

    def add_prediction(
        self,
        vehicle_id: int,
        frame: int,
        pred_rel: np.ndarray,
        gt_rel: np.ndarray,
        obs_ms: float,
        inf_ms: float,
        e2e_ms: float,
        last_pos: Tuple[float, float],
        last_heading: float,
    ):
        dists = np.linalg.norm(pred_rel - gt_rel, axis=1)
        ade = float(np.mean(dists))
        fde = float(dists[-1])
        p05 = float(np.mean(dists <= 0.5))
        p10 = float(np.mean(dists <= 1.0))
        p20 = float(np.mean(dists <= 2.0))

        pm = PredictionMetrics(
            vehicle_id=vehicle_id,
            frame=frame,
            ade=ade,
            fde=fde,
            precision_05m=p05,
            precision_10m=p10,
            precision_20m=p20,
            horizon=len(dists),
            obs_latency_ms=obs_ms,
            inf_latency_ms=inf_ms,
            e2e_latency_ms=e2e_ms,
        )
        self.prediction_metrics.append(pm)

        # store for visualization later
        self.viz_samples.append(
            {
                "vehicle_id": vehicle_id,
                "frame": frame,
                "pred_rel": pred_rel.copy(),
                "gt_rel": gt_rel.copy(),
                "last_pos": last_pos,
                "last_heading": last_heading,
            }
        )

    def summarize(self) -> Dict:
        if not self.prediction_metrics:
            return {}

        m = self.prediction_metrics
        summary = {
            "total_predictions": len(m),
            "ade_mean": float(np.mean([x.ade for x in m])),
            "ade_std": float(np.std([x.ade for x in m])),
            "fde_mean": float(np.mean([x.fde for x in m])),
            "fde_std": float(np.std([x.fde for x in m])),
            "precision_05m": float(np.mean([x.precision_05m for x in m])),
            "precision_10m": float(np.mean([x.precision_10m for x in m])),
            "precision_20m": float(np.mean([x.precision_20m for x in m])),
            "obs_latency_mean_ms": float(np.mean([x.obs_latency_ms for x in m])),
            "inf_latency_mean_ms": float(np.mean([x.inf_latency_ms for x in m])),
            "inf_latency_std_ms": float(np.std([x.inf_latency_ms for x in m])),
            "e2e_latency_mean_ms": float(np.mean([x.e2e_latency_ms for x in m])),
            "e2e_latency_std_ms": float(np.std([x.e2e_latency_ms for x in m])),
        }
        return summary

    def save(self, out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        summary = self.summarize()
        out = {
            "summary": summary,
            "prediction_metrics": [asdict(x) for x in self.prediction_metrics],
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print("\n==============================================================")
        print("                   CROSS-SCENE METRICS SUMMARY                ")
        print("==============================================================")
        if summary:
            print(f"Total Predictions: {summary['total_predictions']}")
            print(f"\nAccuracy:")
            print(f"  ADE: {summary['ade_mean']:.3f} ± {summary['ade_std']:.3f} m")
            print(f"  FDE: {summary['fde_mean']:.3f} ± {summary['fde_std']:.3f} m")
            print(f"\nPrecision (% within threshold):")
            print(f"  @0.5m: {summary['precision_05m'] * 100:6.2f}%")
            print(f"  @1.0m: {summary['precision_10m'] * 100:6.2f}%")
            print(f"  @2.0m: {summary['precision_20m'] * 100:6.2f}%")
            print(f"\nLatency:")
            print(f"  Observation: {summary['obs_latency_mean_ms']:.3f} ms")
            print(
                f"  Inference:   {summary['inf_latency_mean_ms']:.3f} ± "
                f"{summary['inf_latency_std_ms']:.3f} ms"
            )
            print(
                f"  E2E:         {summary['e2e_latency_mean_ms']:.3f} ± "
                f"{summary['e2e_latency_std_ms']:.3f} ms"
            )
        print("==============================================================\n")


# ==================== TRACKER ====================

class TrajectoryTracker:
    """
    Simple per-vehicle sliding window of past states in global frame,
    then normalized to ego frame at inference time.
    """

    def __init__(self, obs_len: int):
        self.obs_len = obs_len
        self.traj: Dict[int, List[Dict]] = {}
        self.last_heading: Dict[int, float] = {}

    def update(
        self,
        vid: int,
        frame: int,
        x: float,
        y: float,
        vx: float,
        vy: float,
        ax: float,
        ay: float,
    ):
        heading = angle_from_velocity(vx, vy)
        if vid not in self.traj:
            self.traj[vid] = []
        self.traj[vid].append(
            {
                "frame": frame,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "heading": heading,
            }
        )
        if len(self.traj[vid]) > self.obs_len:
            self.traj[vid].pop(0)
        self.last_heading[vid] = heading

    def has_enough(self, vid: int) -> bool:
        return vid in self.traj and len(self.traj[vid]) >= self.obs_len

    def get_observation(self, vid: int):
        """
        Returns normalized observation, last_pos, last_heading
        obs shape: (obs_len, 7)
        """
        if not self.has_enough(vid):
            return None, None, None

        traj = self.traj[vid]
        obs = np.array(
            [
                [
                    s["x"],
                    s["y"],
                    s["vx"],
                    s["vy"],
                    s["ax"],
                    s["ay"],
                    s["heading"],
                ]
                for s in traj
            ],
            dtype=np.float32,
        )
        last = traj[-1]
        last_pos = (last["x"], last["y"])
        last_heading = last["heading"]

        # ego centering and rotation normalization
        c, s = np.cos(-last_heading), np.sin(-last_heading)
        norm = obs.copy()
        dx = obs[:, 0] - last_pos[0]
        dy = obs[:, 1] - last_pos[1]
        norm[:, 0] = c * dx - s * dy
        norm[:, 1] = s * dx + c * dy
        norm[:, 2] = c * obs[:, 2] - s * obs[:, 3]
        norm[:, 3] = s * obs[:, 2] + c * obs[:, 3]
        norm[:, 4] = c * obs[:, 4] - s * obs[:, 5]
        norm[:, 5] = s * obs[:, 4] + c * obs[:, 5]
        norm[:, 6] = ((obs[:, 6] - last_heading + np.pi) % (2 * np.pi)) - np.pi

        return norm, last_pos, last_heading


# ==================== MODEL LOADER ====================

def load_model(cfg: DTConfig):
    if cfg.MODEL_TYPE == "slstm":
        model = SimpleSLSTM(pred_len=cfg.PRED_LEN, k_neighbors=8)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=cfg.PRED_LEN, k_neighbors=8)

    state = torch.load(cfg.MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state.state_dict())

    model.to(device).eval()
    print(f"Loaded {cfg.MODEL_TYPE} model from {cfg.MODEL_PATH} on {device}")
    return model


# ==================== HIGH-D DATA PLAYER ====================

class HighDPlayer:
    """
    Replays a single highD scene and provides per-frame vehicle states.

    Supports both naming conventions:
      x/localX, y/localY, xVelocity, yVelocity, xAcceleration, yAcceleration
    """
    def __init__(self, csv_path: str, fps: int):
        self.csv_path = csv_path
        self.fps = fps
        self.df = pd.read_csv(csv_path)

        # decide column names dynamically
        self.col_x = "localX" if "localX" in self.df.columns else "x"
        self.col_y = "localY" if "localY" in self.df.columns else "y"

        # velocities and accelerations are standard in highD
        # if someone preprocessed and renamed, you can add fallbacks here
        self.col_vx = "xVelocity"
        self.col_vy = "yVelocity"
        self.col_ax = "xAcceleration"
        self.col_ay = "yAcceleration"

        required = [
            "id",
            "frame",
            self.col_x,
            self.col_y,
            self.col_vx,
            self.col_vy,
            self.col_ax,
            self.col_ay,
        ]
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"Expected column {c} in {csv_path}")

        self.df.sort_values(["frame", "id"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # group by vehicle
        self.by_vehicle: Dict[int, pd.DataFrame] = {}
        for vid, g in self.df.groupby("id"):
            self.by_vehicle[int(vid)] = g.sort_values("frame").reset_index(drop=True)

        # mapping frame → list of (vid, row_idx)
        self.frames_sorted = sorted(self.df["frame"].unique())
        self.frame_to_entries: Dict[int, List[Tuple[int, int]]] = {f: [] for f in self.frames_sorted}

        for vid, g in self.by_vehicle.items():
            for idx, row in g.iterrows():
                fr = int(row["frame"])
                self.frame_to_entries[fr].append((vid, idx))

    def iter_frames(self):
        """
        Yields (global_step_idx, frame_id, list of (vid, row_df_idx)).
        """
        for step_idx, fr in enumerate(self.frames_sorted):
            yield step_idx, fr, self.frame_to_entries[fr]


# ==================== CORE EVALUATION LOOP ====================

def run_cross_scene_dt(cfg: DTConfig):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    print("\n==============================================================")
    print("         CROSS-SCENE DIGITAL TWIN EVALUATION (GT playback)    ")
    print("==============================================================")
    print(f"Train scene: {cfg.TRAIN_SCENE}")
    print(f"Test scene:  {cfg.TEST_SCENE}")
    print(f"Test tracks: {cfg.TEST_TRACKS}")
    print("==============================================================\n")

    model = load_model(cfg)
    tracker = TrajectoryTracker(cfg.OBS_LEN)
    metrics = MetricsCollector()
    player = HighDPlayer(cfg.TEST_TRACKS, cfg.HIGHD_FPS)

    step_to_next_dt = max(1, int(round(cfg.HIGHD_FPS / cfg.DT_HZ)))
    print(f"HighD fps: {cfg.HIGHD_FPS}, DT freq: {cfg.DT_HZ} Hz, "
          f"reading every {step_to_next_dt} frames")

    last_dt_step = -1
    total_predictions = 0

    for global_step, frame_id, entries in player.iter_frames():
        if cfg.MAX_FRAMES > 0 and global_step >= cfg.MAX_FRAMES:
            break

        # sample every step_to_next_dt frames to emulate DT at cfg.DT_HZ
        if global_step % step_to_next_dt != 0:
            continue

        dt_step = global_step // step_to_next_dt

        # update tracker with states at this frame
        for vid, row_idx in entries:
            row = player.by_vehicle[vid].iloc[row_idx]
            x = float(row[player.col_x])
            y = float(row[player.col_y])
            vx = float(row[player.col_vx])
            vy = float(row[player.col_vy])
            ax = float(row[player.col_ax])
            ay = float(row[player.col_ay])
            tracker.update(vid, frame_id, x, y, vx, vy, ax, ay)


        # predictions at this step
        for vid, row_idx in entries:
            if not tracker.has_enough(vid):
                continue

            # avoid multiple predictions for same vid in same dt_step
            # (not strictly necessary but keeps counts clean)
            if last_dt_step == dt_step:
                pass

            obs_start = time.time()
            obs_norm, last_pos, last_heading = tracker.get_observation(vid)
            obs_ms = (time.time() - obs_start) * 1000.0
            if obs_norm is None:
                continue

            # prepare input tensors
            hist = torch.from_numpy(obs_norm).unsqueeze(0).to(device)  # (1, obs_len, 7)
            neigh = torch.zeros(1, 8, cfg.OBS_LEN, 7, device=device)
            neigh_state = torch.zeros(1, 8, cfg.OBS_LEN, 18, device=device)
            lane = torch.zeros(1, cfg.OBS_LEN, 1, device=device)

            e2e_start = time.time()
            inf_start = time.time()
            with torch.no_grad():
                if cfg.MODEL_TYPE == "transformer":
                    preds_abs = model(
                        hist,
                        neigh,
                        neigh_state,
                        lane,
                        last_obs_pos=torch.as_tensor(
                            [last_pos], dtype=torch.float32, device=device
                        ),
                    )
                    preds_abs = preds_abs.cpu().numpy()[0]
                    # convert absolute to relative
                    c, s = np.cos(-last_heading), np.sin(-last_heading)
                    diffs = preds_abs - np.array(last_pos)
                    pred_rel = np.stack(
                        [c * diffs[:, 0] - s * diffs[:, 1],
                         s * diffs[:, 0] + c * diffs[:, 1]],
                        axis=-1,
                    )
                else:
                    preds_rel = model(hist, neigh, neigh_state, lane)
                    pred_rel = preds_rel.cpu().numpy()[0]
            inf_ms = (time.time() - inf_start) * 1000.0
            e2e_ms = (time.time() - e2e_start) * 1000.0

            # get ground truth future for this vehicle
            veh_df = player.by_vehicle[vid]
            # locate current row in that df
            cur_frame = frame_id
            cur_idx_list = veh_df.index[veh_df["frame"] == cur_frame].tolist()
            if not cur_idx_list:
                continue
            cur_idx = cur_idx_list[0]
            fut_df = veh_df.iloc[cur_idx + 1 : cur_idx + 1 + cfg.PRED_LEN]
            if len(fut_df) < cfg.MIN_GT_LEN:
                continue

            gt_abs = np.stack(
                [fut_df[player.col_x].values, fut_df[player.col_y].values],
                axis=-1,
            ).astype(np.float32)


            # convert gt to same relative ego frame
            c, s = np.cos(-last_heading), np.sin(-last_heading)
            gt_rel = np.zeros_like(gt_abs)
            dx = gt_abs[:, 0] - last_pos[0]
            dy = gt_abs[:, 1] - last_pos[1]
            gt_rel[:, 0] = c * dx - s * dy
            gt_rel[:, 1] = s * dx + c * dy

            use_len = min(len(pred_rel), len(gt_rel))
            pred_rel_use = pred_rel[:use_len]
            gt_rel_use = gt_rel[:use_len]

            metrics.add_prediction(
                vehicle_id=vid,
                frame=int(frame_id),
                pred_rel=pred_rel_use,
                gt_rel=gt_rel_use,
                obs_ms=obs_ms,
                inf_ms=inf_ms,
                e2e_ms=e2e_ms,
                last_pos=last_pos,
                last_heading=last_heading,
            )
            total_predictions += 1

        if dt_step > 0 and dt_step % (config.REPORT_INTERVAL_FRAMES // step_to_next_dt) == 0:
            recent = metrics.prediction_metrics[-1000:] if metrics.prediction_metrics else []
            if recent:
                ade_mean = float(np.mean([m.ade for m in recent]))
                fde_mean = float(np.mean([m.fde for m in recent]))
                p05 = float(np.mean([m.precision_05m for m in recent]) * 100.0)
                p10 = float(np.mean([m.precision_10m for m in recent]) * 100.0)
                p20 = float(np.mean([m.precision_20m for m in recent]) * 100.0)
                lat_mean = float(np.mean([m.inf_latency_ms for m in recent]))
                print(
                    f"Frame {frame_id:6d} | DT step {dt_step:5d} | "
                    f"Pred {len(recent):6d} (recent) | "
                    f"ADE {ade_mean:.3f} | FDE {fde_mean:.3f} | "
                    f"Acc@0.5/1/2m: {p05:.2f}/{p10:.2f}/{p20:.2f}% | "
                    f"Lat {lat_mean:.2f} ms"
                )

        last_dt_step = dt_step

    print(f"\nTotal predictions evaluated: {total_predictions}")
    os.makedirs(os.path.dirname(cfg.METRICS_OUTPUT) or ".", exist_ok=True)
    metrics.save(cfg.METRICS_OUTPUT)

    # visualization
    save_visualizations(metrics, cfg)


# ==================== VISUALIZATION ====================

def save_visualizations(metrics: MetricsCollector, cfg: DTConfig):
    os.makedirs(cfg.VIZ_OUTPUT_DIR, exist_ok=True)
    samples = metrics.viz_samples
    if not samples:
        print("No samples to visualize.")
        return

    random.shuffle(samples)
    samples = samples[: cfg.VIZ_SAMPLES]

    print(f"Saving {len(samples)} GT vs prediction plots to {cfg.VIZ_OUTPUT_DIR}")

    for i, s in enumerate(samples):
        vid = s["vehicle_id"]
        frame = s["frame"]
        pred_rel = s["pred_rel"]
        gt_rel = s["gt_rel"]
        last_pos = s["last_pos"]
        last_heading = s["last_heading"]

        # convert back to absolute for plotting in scene coordinates
        pred_abs = relative_to_absolute(pred_rel, last_pos, last_heading)
        gt_abs = relative_to_absolute(gt_rel, last_pos, last_heading)

        plt.figure(figsize=(4, 4))
        plt.plot(
            gt_abs[:, 0],
            gt_abs[:, 1],
            label="GT future",
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            pred_abs[:, 0],
            pred_abs[:, 1],
            label="Prediction",
            linestyle="--",
            linewidth=2,
        )
        plt.scatter(
            [last_pos[0]],
            [last_pos[1]],
            marker="o",
        )
        plt.title(f"Veh {vid} at frame {frame}")
        plt.xlabel("localX (m)")
        plt.ylabel("localY (m)")
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        out_path = os.path.join(cfg.VIZ_OUTPUT_DIR, f"veh{vid}_frame{frame}_sample{i}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


# ==================== CLI ====================

def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Cross-scene DT evaluation on highD with GT playback"
    )
    p.add_argument("--train_scene", default=config.TRAIN_SCENE)
    p.add_argument("--test_scene", default=config.TEST_SCENE)
    p.add_argument(
        "--model_path",
        default=config.MODEL_PATH,
        help="Path to trained model checkpoint (e.g., scene02 SLSTM)",
    )
    p.add_argument(
        "--model_type",
        choices=["slstm", "transformer"],
        default=config.MODEL_TYPE,
    )
    p.add_argument(
        "--test_csv",
        default=config.TEST_TRACKS,
        help="Path to highD tracks CSV for test scene",
    )
    p.add_argument(
        "--metrics_output",
        default=config.METRICS_OUTPUT,
    )
    p.add_argument(
        "--viz_output_dir",
        default=config.VIZ_OUTPUT_DIR,
    )
    p.add_argument("--obs_len", type=int, default=config.OBS_LEN)
    p.add_argument("--pred_len", type=int, default=config.PRED_LEN)
    p.add_argument("--min_gt_len", type=int, default=config.MIN_GT_LEN)
    p.add_argument("--dt_hz", type=int, default=config.DT_HZ)
    p.add_argument("--highd_fps", type=int, default=config.HIGHD_FPS)
    p.add_argument("--max_frames", type=int, default=config.MAX_FRAMES)
    p.add_argument("--viz_samples", type=int, default=config.VIZ_SAMPLES)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.TRAIN_SCENE = args.train_scene
    config.TEST_SCENE = args.test_scene
    config.MODEL_PATH = args.model_path
    config.MODEL_TYPE = args.model_type
    config.TEST_TRACKS = args.test_csv
    config.METRICS_OUTPUT = args.metrics_output
    config.VIZ_OUTPUT_DIR = args.viz_output_dir
    config.OBS_LEN = args.obs_len
    config.PRED_LEN = args.pred_len
    config.MIN_GT_LEN = args.min_gt_len
    config.DT_HZ = args.dt_hz
    config.HIGHD_FPS = args.highd_fps
    config.MAX_FRAMES = args.max_frames
    config.VIZ_SAMPLES = args.viz_samples

    run_cross_scene_dt(config)
