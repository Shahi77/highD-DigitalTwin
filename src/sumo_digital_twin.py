#!/usr/bin/env python3
"""
sumo_digital_twin.py — Digital Twin Visualization for HighD (fixed & robust)
Notes:
 - Waits for SUMO TraCI server to be ready after starting SUMO.
 - DOES NOT close the connection prematurely.
 - Calls traci.close() only when simulation loop completes or on error.
"""

import argparse
import os
import math
import time
import numpy as np
import pandas as pd
import torch

try:
    import traci
    from traci import vehicle as traci_vehicle
    from traci.exceptions import FatalTraCIError
except Exception as e:
    raise RuntimeError(f"Could not import traci. Ensure SUMO_HOME is set and traci is installed. Error: {e}")

from models import TrajectoryTransformer

# ----------------------------
# Config
# ----------------------------
SUMO_BINARY = "sumo-gui"            # or "sumo" for headless
OBS_LEN = 10
PRED_LEN = 25
DOWNSAMPLE = 5
SIM_STEP_S = 0.2
SMOOTH_ALPHA = 0.3
SUMO_VEH_TYPE = "car"
LOG_PRED_CSV = "checkpoints_01/sumo_dt_predictions_extended.csv"

# SUMO net extents (must match your highway.net.xml convBoundary)
SUMO_X_MIN, SUMO_X_MAX = 0.0, 600.0
SUMO_Y_MIN, SUMO_Y_MAX = 0.0, 30.0

# ----------------------------
# Model loader
# ----------------------------
def load_model(checkpoint_path, device='cpu'):
    model = TrajectoryTransformer(in_dim=7, d_model=256, nhead=8, num_layers=4, pred_len=PRED_LEN)
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Warning: checkpoint {checkpoint_path} not found. Running without model (visual test only).")
        model.eval()
        return model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ----------------------------
# SUMO start helper (robust)
# ----------------------------
def start_sumo(sumo_cfg, sumo_binary=SUMO_BINARY, step_length=SIM_STEP_S, wait_timeout_s=10.0):
    """
    Start SUMO and wait until TraCI server responds to commands.
    Returns previous cwd to restore later.
    Raises RuntimeError if TraCI server doesn't become available.
    """
    sumo_cfg = os.path.abspath(sumo_cfg)
    sumo_dir = os.path.dirname(sumo_cfg)
    cfg_basename = os.path.basename(sumo_cfg)

    cwd_before = os.getcwd()
    try:
        os.chdir(sumo_dir)
    except Exception as e:
        print(f"Could not change directory to SUMO dir {sumo_dir}: {e}")

    cmd = [
        sumo_binary,
        "-c", cfg_basename,
        "--step-length", str(step_length),
        "--start",
        "--no-warnings",
        "--time-to-teleport", "-1",
        "--collision.action", "none"
    ]
    print("🟢 Starting SUMO:", " ".join(cmd))
    try:
        traci.start(cmd)
    except Exception as e:
        # restore cwd if starting failed
        try:
            os.chdir(cwd_before)
        except Exception:
            pass
        raise RuntimeError(f"Failed to start SUMO with command {cmd}: {e}")

    # Wait until TraCI server accepts commands (getVersion)
    start_t = time.time()
    while True:
        try:
            # getVersion will succeed only if TraCI server is ready
            _ = traci.getVersion()
            break
        except Exception:
            if time.time() - start_t > wait_timeout_s:
                # restore cwd before raising
                try:
                    os.chdir(cwd_before)
                except Exception:
                    pass
                raise RuntimeError("Timed out waiting for SUMO TraCI server to become ready.")
            time.sleep(0.1)
    # small extra pause to let GUI draw
    time.sleep(0.15)
    return cwd_before

# ----------------------------
# HighD <-> SUMO mapping + spawn helpers
# ----------------------------
def compute_highd_bounds(df):
    x_min, x_max = float(df['x'].min()), float(df['x'].max())
    y_min, y_max = float(df['y'].min()), float(df['y'].max())
    return x_min, x_max, y_min, y_max

def highd_to_sumo(x_h, y_h, bounds):
    x_min, x_max, y_min, y_max = bounds
    sx = (SUMO_X_MAX - SUMO_X_MIN) / (x_max - x_min) if (x_max - x_min) > 1e-6 else 1.0
    sy = (SUMO_Y_MAX - SUMO_Y_MIN) / (y_max - y_min) if (y_max - y_min) > 1e-6 else 1.0
    s = min(sx, sy)
    x_s = SUMO_X_MIN + (x_h - x_min) * s
    y_s = SUMO_Y_MIN + (y_h - y_min) * s
    x_s = max(SUMO_X_MIN + 0.5, min(SUMO_X_MAX - 0.5, x_s))
    y_s = max(SUMO_Y_MIN + 0.5, min(SUMO_Y_MAX - 0.5, y_s))
    return float(x_s), float(y_s)

def spawn_vehicle(vid, x, y, color=(0,255,0,255), route_id="r0", type_id=SUMO_VEH_TYPE, preferred_lane_index=1):
    try:
        current_ids = []
        try:
            current_ids = traci.vehicle.getIDList()
        except FatalTraCIError:
            # connection not available
            print(f"  ⚠️ TraCI connection not available when trying to spawn {vid}.")
            return False

        if str(vid) in current_ids:
            traci.vehicle.moveToXY(str(vid), edgeID="E0", lane=preferred_lane_index,
                                   x=float(x), y=float(y), angle=0, keepRoute=True)
            traci.vehicle.setColor(str(vid), color)
            return True

        traci.vehicle.add(vehID=str(vid), routeID=route_id, typeID=type_id,
                          depart="now", departPos="0", departLane="best")
        # move to exact coords
        traci.vehicle.moveToXY(str(vid), edgeID="E0", lane=preferred_lane_index,
                               x=float(x), y=float(y), angle=0, keepRoute=True)
        traci.vehicle.setSpeed(str(vid), 0.01)
        traci.vehicle.setColor(str(vid), color)
        print(f"  ✅ Spawned vehicle {vid} at ({x:.2f},{y:.2f})")
        return True
    except FatalTraCIError as e:
        print(f"  ⚠️ TraCI fatal error spawning {vid}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️ Failed to spawn {vid}: {e}")
        return False

def safe_move_vehicle_to(vid, x, y, alpha=SMOOTH_ALPHA, lane_index=1):
    try:
        if str(vid) not in traci.vehicle.getIDList():
            return
        cur_x, cur_y = traci.vehicle.getPosition(str(vid))
        new_x = float(cur_x) * (1 - alpha) + float(x) * alpha
        new_y = float(cur_y) * (1 - alpha) + float(y) * alpha
        new_x = max(SUMO_X_MIN + 0.2, min(SUMO_X_MAX - 0.2, new_x))
        new_y = max(SUMO_Y_MIN + 0.2, min(SUMO_Y_MAX - 0.2, new_y))
        traci.vehicle.moveToXY(str(vid), edgeID="E0", lane=lane_index,
                               x=new_x, y=new_y, angle=0, keepRoute=True)
        traci.vehicle.setSpeed(str(vid), 0.01)
    except FatalTraCIError:
        # connection dead — caller will handle
        raise
    except Exception:
        pass

# ----------------------------
# Main
# ----------------------------
def run_extended_digital_twin(tracks_csv, model_checkpoint, sumo_cfg, device='cpu', sim_steps=1000):
    if not os.path.exists(tracks_csv):
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv}")

    df = pd.read_csv(tracks_csv)
    print(f"Loaded {len(df)} rows from {tracks_csv}")
    print(f"HighD range X=[{df['x'].min():.2f}, {df['x'].max():.2f}] Y=[{df['y'].min():.2f}, {df['y'].max():.2f}]")

    bounds = compute_highd_bounds(df)
    model = load_model(model_checkpoint, device)

    counts = df['id'].value_counts()
    min_frames_needed = (OBS_LEN + PRED_LEN) * DOWNSAMPLE
    sample_vids = counts[counts > min_frames_needed].index.tolist()[:3]
    if len(sample_vids) == 0:
        raise RuntimeError("No vehicles with enough frames found in dataset.")
    print("Vehicles selected:", sample_vids)

    groups = {vid: g.sort_values('frame').reset_index(drop=True) for vid, g in df.groupby('id')}

    # Start SUMO and wait for TraCI server
    cwd_before = start_sumo(sumo_cfg)

    # make sure simulation is initialized
    try:
        traci.simulationStep()
    except Exception as e:
        # If we cannot perform a step, close and raise
        try:
            traci.close()
        except Exception:
            pass
        raise RuntimeError(f"traci.simulationStep failed after start: {e}")

    print("\nSpawning vehicles...")
    for vid in sample_vids:
        arr = groups[vid]
        x_world, y_world = float(arr.iloc[0]['x']), float(arr.iloc[0]['y'])
        xs, ys = highd_to_sumo(x_world, y_world, bounds)
        spawn_vehicle(str(vid), xs, ys, color=(0,255,0,255))
        spawn_vehicle(f"ghost_{vid}", xs + 1.5, ys, color=(255,0,0,255))

    print("\n🚀 Starting simulation loop...")
    try:
        for step in range(sim_steps):
            if step % 100 == 0:
                try:
                    sim_time = traci.simulation.getTime()
                except FatalTraCIError:
                    raise RuntimeError("TraCI connection closed unexpectedly during loop.")
                print(f"  Step {step}/{sim_steps}, sim_time={sim_time:.1f}s")

            for vid in sample_vids:
                x_target_h = bounds[0] + (step * 0.5)
                y_target_h = (groups[vid].iloc[0]['y'])
                xs, ys = highd_to_sumo(x_target_h, y_target_h, bounds)
                safe_move_vehicle_to(str(vid), xs, ys, alpha=SMOOTH_ALPHA)
                safe_move_vehicle_to(f"ghost_{vid}", xs + 2.0, ys, alpha=SMOOTH_ALPHA)

            traci.simulationStep()
    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    except Exception as e:
        print("❌ Error during simulation loop:", e)
    finally:
        # gracefully close TraCI (this tells SUMO we are done)
        try:
            traci.close()
        except Exception:
            pass
        # restore cwd
        try:
            os.chdir(cwd_before)
        except Exception:
            pass

    print("✅ Simulation finished. Close SUMO manually if GUI still open.")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_csv", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--sumo_cfg", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sim_steps", type=int, default=1000)
    args = parser.parse_args()

    run_extended_digital_twin(
        args.tracks_csv, args.model_checkpoint, args.sumo_cfg,
        device=args.device, sim_steps=args.sim_steps
    )
