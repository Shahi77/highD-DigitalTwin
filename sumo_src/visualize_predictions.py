"""
Trajectory Prediction Visualization on SUMO Simulation
------------------------------------------------------
Loads trained LSTM/Transformer model and visualizes predictions
on the running SUMO simulation with ground truth comparison.
"""

import os
import sys
import csv
import random
import numpy as np
import torch
from collections import defaultdict, deque

sys.path.append("/Users/shahi/Developer/Project-highD/src")
from models import ImprovedTrajectoryTransformer, SimpleSLSTM

from utils import check_sumo_env, start_sumo, running
check_sumo_env()

import traci
from traci import constants as tc


# ==================== Configuration ====================
class Config:
    # Model settings
    MODEL_TYPE = "slstm"  # or "transformer"
    MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
    
    # Prediction settings
    OBS_LEN = 20  # Observation length (frames)
    PRED_LEN = 25  # Prediction length (frames)
    K_NEIGHBORS = 8  # Number of neighbors (must match training)
    
    # Visualization settings
    PREDICT_EVERY_N_STEPS = 4  # Make predictions every 4 simulation steps (1 second)
    MAX_PREDICTIONS_SHOWN = 5  # Show predictions for N vehicles at a time
    DRAW_PREDICTIONS = True  # Draw prediction lines in SUMO GUI
    
    # Simulation settings
    GUI = True
    TOTAL_TIME = 1000  # Run for limited time for testing
    START_STEP = 100  # Start predictions after some vehicles are in


# ==================== Device Setup ====================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# ==================== Model Loading ====================
def load_model(model_path, model_type="slstm", pred_len=25):
    """Load trained prediction model"""
    if model_type == "slstm":
        model = SimpleSLSTM(pred_len=pred_len)
    else:
        model = ImprovedTrajectoryTransformer(pred_len=pred_len)
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f" Loaded {model_type.upper()} model from checkpoint")
    print(f"  Model expects k={model.k} neighbors")
    return model


# ==================== Vehicle Trajectory Tracking ====================
class TrajectoryTracker:
    """Tracks vehicle positions over time for prediction"""
    
    def __init__(self, obs_len=20):
        self.obs_len = obs_len
        self.trajectories = defaultdict(lambda: deque(maxlen=obs_len))
        
    def update(self, vehicle_id, position, velocity, acceleration, lane_id):
        """Add new observation for a vehicle"""
        # Store [x, y, vx, vy, ax, ay, lane_id] for each timestep
        self.trajectories[vehicle_id].append({
            'x': position[0],
            'y': position[1],
            'vx': velocity[0] if isinstance(velocity, tuple) else velocity,
            'vy': 0.0,
            'ax': acceleration,
            'ay': 0.0,
            'lane_id': lane_id
        })
    
    def get_observation(self, vehicle_id):
        """Get observation sequence for prediction [obs_len, 7]"""
        if vehicle_id not in self.trajectories:
            return None
        
        traj = list(self.trajectories[vehicle_id])
        if len(traj) < self.obs_len:
            return None  # Not enough history
        
        # Convert to numpy array [obs_len, 7] (x, y, vx, vy, ax, ay, lane_id)
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
    
    def has_enough_history(self, vehicle_id):
        """Check if vehicle has enough history for prediction"""
        return (vehicle_id in self.trajectories and 
                len(self.trajectories[vehicle_id]) >= self.obs_len)
    
    def remove(self, vehicle_id):
        """Remove vehicle from tracking"""
        if vehicle_id in self.trajectories:
            del self.trajectories[vehicle_id]


# ==================== Prediction Manager ====================
class PredictionVisualizer:
    """Manages predictions and visualization in SUMO"""
    
    def __init__(self, model, tracker, pred_len=25, k=8):
        self.model = model
        self.tracker = tracker
        self.pred_len = pred_len
        self.k = k
        self.active_predictions = {}  # vehicle_id -> prediction data
        
    def make_prediction(self, vehicle_id):
        """Make trajectory prediction for a vehicle"""
        obs = self.tracker.get_observation(vehicle_id)
        if obs is None:
            return None
        
        # Prepare input tensors [1, obs_len, 7]
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Create dummy neighbor and lane data with CORRECT shapes
        # Model expects: nd shape [batch, k, obs_len, 7]
        nd = torch.zeros(1, self.k, Config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, self.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        # Make prediction
        try:
            with torch.no_grad():
                if hasattr(self.model, "multi_att"):  # Transformer
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = self.model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:  # SLSTM
                    pred = self.model(obs_tensor, nd, ns, lane)
            
            # Convert to numpy [pred_len, 2]
            pred_np = pred.cpu().numpy()[0]
            return pred_np
            
        except Exception as e:
            print(f"  Warning: Prediction failed for {vehicle_id}: {e}")
            return None
    
    def update_predictions(self, vehicle_ids):
        """Update predictions for active vehicles"""
        # Remove predictions for vehicles that no longer exist
        for vid in list(self.active_predictions.keys()):
            if vid not in vehicle_ids:
                del self.active_predictions[vid]
        
        # Make predictions for vehicles with enough history
        count = 0
        for vid in vehicle_ids:
            if count >= Config.MAX_PREDICTIONS_SHOWN:
                break
            
            if self.tracker.has_enough_history(vid):
                pred = self.make_prediction(vid)
                if pred is not None:
                    try:
                        current_pos = traci.vehicle.getPosition(vid)
                        self.active_predictions[vid] = {
                            'prediction': pred,
                            'start_pos': current_pos,
                            'timestamp': traci.simulation.getTime()
                        }
                        count += 1
                    except:
                        continue
    
    def draw_predictions(self):
        """Draw prediction and ground truth trajectories as lines in SUMO GUI"""
        if not Config.GUI or not Config.DRAW_PREDICTIONS:
            return
        
        for vid, data in self.active_predictions.items():
            try:
                # Check if vehicle still exists
                if vid not in traci.vehicle.getIDList():
                    continue
                
                # Highlight vehicle with prediction in CYAN (more visible)
                traci.vehicle.setColor(vid, (0, 255, 255, 255))
                
                # Get current vehicle position and angle
                try:
                    current_pos = traci.vehicle.getPosition(vid)
                    current_angle = traci.vehicle.getAngle(vid)
                    current_speed = traci.vehicle.getSpeed(vid)
                    lane_idx = traci.vehicle.getLaneIndex(vid)
                except:
                    continue
                
                # Get prediction and observation
                pred = data['prediction']
                obs = self.tracker.get_observation(vid)
                
                if obs is None:
                    continue
                
                # Calculate lateral offset based on lane to separate trajectories visually
                # Each lane gets a small offset perpendicular to direction
                angle_rad = np.radians(90 - current_angle)
                lateral_offset = 1.5  # meters offset per lane
                offset_x = -np.sin(angle_rad) * lateral_offset * (lane_idx - 1)
                offset_y = np.cos(angle_rad) * lateral_offset * (lane_idx - 1)
                
                # === Draw OBSERVATION line (YELLOW - Historical path) ===
                # Use only last 10 points for clarity
                obs_display_len = min(10, len(obs))
                obs_points = []
                for i in range(len(obs) - obs_display_len, len(obs), 2):  # Every 2nd point
                    obs_points.append((obs[i, 0] + offset_x, obs[i, 1] + offset_y))
                
                if len(obs_points) > 1:
                    poly_obs_id = f"obs_{vid}"
                    try:
                        traci.polygon.remove(poly_obs_id)
                    except:
                        pass
                    
                    # Create dotted effect by drawing small line segments
                    for i in range(len(obs_points) - 1):
                        segment_id = f"obs_{vid}_seg{i}"
                        try:
                            traci.polygon.remove(segment_id)
                        except:
                            pass
                        traci.polygon.add(
                            segment_id,
                            [obs_points[i], obs_points[i + 1]],
                            color=(255, 255, 0, 255),  # Yellow
                            fill=False,
                            layer=100,
                            lineWidth=1.5
                        )
                
                # === Draw PREDICTED trajectory line (GREEN - Model prediction) ===
                # Limit prediction display to reasonable distance (e.g., 15 points = 3.75 seconds)
                pred_display_len = min(15, self.pred_len)
                
                # Calculate absolute positions with lateral offset
                cumsum_x = np.cumsum(pred[:pred_display_len, 0])
                cumsum_y = np.cumsum(pred[:pred_display_len, 1])
                
                pred_points = []
                pred_points.append((current_pos[0] + offset_x, current_pos[1] + offset_y))
                
                # Add prediction points (every other point for dotted effect)
                for i in range(0, len(cumsum_x), 2):
                    pred_x = current_pos[0] + cumsum_x[i] + offset_x
                    pred_y = current_pos[1] + cumsum_y[i] + offset_y
                    pred_points.append((pred_x, pred_y))
                
                # Draw as dotted line (small segments)
                for i in range(len(pred_points) - 1):
                    segment_id = f"pred_{vid}_seg{i}"
                    try:
                        traci.polygon.remove(segment_id)
                    except:
                        pass
                    traci.polygon.add(
                        segment_id,
                        [pred_points[i], pred_points[i + 1]],
                        color=(0, 255, 0, 255),  # Green
                        fill=False,
                        layer=101,
                        lineWidth=2.0
                    )
                
                # === Draw GROUND TRUTH trajectory line (RED - Actual future path) ===
                # Estimate future ground truth positions using constant velocity model
                # Also limit to same display length as prediction
                try:
                    # Convert angle to radians (SUMO uses degrees from north, clockwise)
                    angle_rad = np.radians(90 - current_angle)
                    vx = current_speed * np.cos(angle_rad)
                    vy = current_speed * np.sin(angle_rad)
                    
                    gt_points = []
                    gt_points.append((current_pos[0] + offset_x, current_pos[1] + offset_y))
                    
                    dt = 0.25  # Time step (4 steps per second)
                    
                    # Generate ground truth points (every 2nd for dotted effect)
                    for i in range(2, pred_display_len, 2):
                        next_x = current_pos[0] + vx * dt * i + offset_x
                        next_y = current_pos[1] + vy * dt * i + offset_y
                        gt_points.append((next_x, next_y))
                    
                    # Draw as dashed line (red)
                    for i in range(len(gt_points) - 1):
                        segment_id = f"gt_{vid}_seg{i}"
                        try:
                            traci.polygon.remove(segment_id)
                        except:
                            pass
                        traci.polygon.add(
                            segment_id,
                            [gt_points[i], gt_points[i + 1]],
                            color=(255, 0, 0, 255),  # Red
                            fill=False,
                            layer=99,
                            lineWidth=2.0
                        )
                except:
                    pass
                
            except traci.exceptions.TraCIException:
                continue
            except Exception as e:
                continue


    def clear_visualizations(self):
        """Clear all drawn trajectories"""
        try:
            # Get all polygon IDs and remove trajectory-related ones
            all_polygons = traci.polygon.getIDList()
            for poly_id in all_polygons:
                if any(prefix in poly_id for prefix in ["pred_", "obs_", "gt_"]):
                    try:
                        traci.polygon.remove(poly_id)
                    except:
                        pass
        except:
            pass
        
        try:
            # Clear POIs as well
            all_pois = traci.poi.getIDList()
            for poi_id in all_pois:
                if poi_id.startswith("pred_"):
                    try:
                        traci.poi.remove(poi_id)
                    except:
                        pass
        except:
            pass


    # Also update the Config class for better visualization control
    class Config:
        # Model settings
        MODEL_TYPE = "slstm"  # or "transformer"
        MODEL_PATH = "/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt"
        
        # Prediction settings
        OBS_LEN = 20  # Observation length (frames)
        PRED_LEN = 25  # Prediction length (frames)
        K_NEIGHBORS = 8  # Number of neighbors (must match training)
        
        # Visualization settings
        PREDICT_EVERY_N_STEPS = 4  # Make predictions every 4 simulation steps (1 second)
        MAX_PREDICTIONS_SHOWN = 3  # Show predictions for 3 vehicles at a time (reduced for clarity)
        DRAW_PREDICTIONS = True  # Draw prediction lines in SUMO GUI
        
        # Simulation settings
        GUI = True
        TOTAL_TIME = 1000  # Run for limited time for testing
        START_STEP = 100  # Start predictions after some vehicles are in
        
# ==================== Main Simulation with Predictions ====================
def main_with_predictions():
    """Run SUMO simulation with trajectory predictions"""
    
    # Load trained model
    model = load_model(Config.MODEL_PATH, Config.MODEL_TYPE, Config.PRED_LEN)
    
    # Initialize trajectory tracker
    tracker = TrajectoryTracker(obs_len=Config.OBS_LEN)
    
    # Initialize prediction visualizer
    visualizer = PredictionVisualizer(model, tracker, Config.PRED_LEN, model.k)
    
    # Load highD data
    from main import trajectory_tracking, aggregate_vehicles, gene_config, init_csv_file, has_vehicle_entered
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
    # Setup simulation
    cfg_file = gene_config()
    
    # Start SUMO
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=Config.GUI)
    
    times = 0
    random.seed(7)
    
    print("\n" + "="*60)
    print("SUMO Simulation with Live Trajectory Predictions")
    print("="*60)
    print(f"Model: {Config.MODEL_TYPE.upper()}")
    print(f"Observation length: {Config.OBS_LEN} frames")
    print(f"Prediction length: {Config.PRED_LEN} frames")
    print(f"Number of neighbors: {model.k}")
    print(f"Running for {Config.TOTAL_TIME} timesteps")
    print("="*60 + "\n")
    
    # For adding vehicles from highD dataset
    from main import AVAILABLE_CAR_TYPES, AVAILABLE_TRUCK_TYPES, CHECK_ALL, LAN_CHANGE_MODE
    vehicles_added = 0
    
    try:
        while running(True, times, Config.TOTAL_TIME + 1):
            traci.simulationStep()
            
            # Add vehicles from highD dataset (same as main.py)
            if times > 0 and times % 4 == 0:
                current_step = int(times / 4)
                
                if has_vehicle_entered(current_step, vehicles_to_enter):
                    for data in vehicles_to_enter[current_step]:
                        if vehicles_added >= 20:  # Limit for testing
                            break
                            
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
                        if direction == 1:
                            route_id = "route_direction1"
                            vehicle_id = f"d1_{data['id']}"
                        else:
                            route_id = "route_direction2"
                            vehicle_id = f"d2_{data['id']}"
                        
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
            
            # Get all active vehicles
            vehicle_ids = traci.vehicle.getIDList()
            
            # Update trajectory history for all vehicles
            for vid in vehicle_ids:
                try:
                    pos = traci.vehicle.getPosition(vid)
                    speed = traci.vehicle.getSpeed(vid)
                    acc = traci.vehicle.getAcceleration(vid)
                    lane = traci.vehicle.getLaneIndex(vid)
                    
                    # Velocity in x direction (assuming straight road)
                    vel = (speed, 0.0)
                    
                    tracker.update(vid, pos, vel, acc, lane)
                except traci.exceptions.TraCIException:
                    continue
            
            # Make predictions periodically
            if times > Config.START_STEP and times % Config.PREDICT_EVERY_N_STEPS == 0:
                # Clear old visualizations
                visualizer.clear_visualizations()
                
                # Update predictions
                visualizer.update_predictions(vehicle_ids)
                
                # Draw predictions
                visualizer.draw_predictions()
                
                if times % 40 == 0:
                    print(f"Step {times} ({times/4:.0f}s): "
                          f"{len(vehicle_ids)} vehicles, "
                          f"{len(visualizer.active_predictions)} predictions shown")
            
            # End condition
            if times >= Config.TOTAL_TIME:
                print("\n Simulation complete!")
                break
            
            times += 1
    
    except KeyboardInterrupt:
        print("\n⏸ Simulation interrupted by user")
    except Exception as e:
        print(f"\n Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        visualizer.clear_visualizations()
        traci.close()
    
    print("\n Simulation ended")
    print(f"Total vehicles added: {vehicles_added}")
    print(f"Predictions made for: {len(visualizer.active_predictions)} vehicles")


# ==================== Standalone Prediction Visualization ====================
def visualize_static_predictions(csv_path, model_path, model_type="slstm", 
                                 num_samples=5, save_dir="./prediction_viz"):
    """
    Create static visualizations comparing predictions vs ground truth
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, model_type, Config.PRED_LEN)
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Group by vehicle ID
    vehicle_groups = df.groupby('id')
    
    print(f"\nGenerating {num_samples} prediction visualizations...")
    
    sample_count = 0
    for vid, group in vehicle_groups:
        if sample_count >= num_samples:
            break
        
        # Sort by frame
        group = group.sort_values('frame')
        
        if len(group) < Config.OBS_LEN + Config.PRED_LEN:
            continue  # Not enough data
        
        # Extract observation and ground truth
        obs_data = group.iloc[:Config.OBS_LEN]
        gt_data = group.iloc[Config.OBS_LEN:Config.OBS_LEN + Config.PRED_LEN]
        
        # Prepare observation tensor [obs_len, 7]
        obs = np.zeros((Config.OBS_LEN, 7))
        obs[:, 0] = obs_data['x'].values
        obs[:, 1] = 0  # y position
        obs[:, 2] = obs_data['v'].values
        obs[:, 3] = 0  # vy
        obs[:, 4] = obs_data['acc'].values if 'acc' in obs_data.columns else 0
        obs[:, 5] = 0  # ay
        obs[:, 6] = obs_data['lane_index'].values
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Correct neighbor dimensions [batch, k, obs_len, 7]
        nd = torch.zeros(1, model.k, Config.OBS_LEN, 7).to(device)
        ns = torch.zeros(1, model.k, 2).to(device)
        lane = torch.zeros(1, 3).to(device)
        
        # Make prediction
        try:
            with torch.no_grad():
                if hasattr(model, "multi_att"):
                    last_obs_pos = obs_tensor[:, -1, :2]
                    pred = model(obs_tensor, nd, ns, lane, last_obs_pos=last_obs_pos)
                else:
                    pred = model(obs_tensor, nd, ns, lane)
            
            pred_np = pred.cpu().numpy()[0]
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Draw road background
            road_width = 30
            lane_width = 10
            ax.add_patch(patches.Rectangle(
                (obs[0, 0] - 50, -road_width/2), 
                obs[-1, 0] - obs[0, 0] + 200,
                road_width,
                facecolor='#7f8c8d', edgecolor='black', linewidth=2
            ))
            
            # Lane markings
            x_start = obs[0, 0] - 50
            x_end = obs[-1, 0] + 150
            for i in range(1, 3):
                y_pos = -road_width/2 + i * lane_width
                n_dashes = 20
                dash_length = (x_end - x_start) / (2 * n_dashes)
                for j in range(n_dashes):
                    x_dash = x_start + j * 2 * dash_length
                    ax.plot([x_dash, x_dash + dash_length], [y_pos, y_pos], 
                           'w--', linewidth=2, alpha=0.8)
            
            lane_id = int(obs[0, 6])
            lane_y = -road_width/2 + (lane_id + 0.5) * lane_width
            
            # Plot observation (yellow)
            obs_x = obs[:, 0]
            obs_y = np.full_like(obs_x, lane_y)
            ax.plot(obs_x, obs_y, 'o-', color='#f1c40f', linewidth=3, 
                   markersize=6, label='Observation', zorder=5)
            
            # Plot ground truth (red)
            gt_x = np.concatenate([obs[-1:, 0], gt_data['x'].values])
            gt_y = np.full_like(gt_x, lane_y)
            ax.plot(gt_x, gt_y, 'o-', color='#e74c3c', linewidth=3, 
                   markersize=6, label='Ground Truth', zorder=4, linestyle='--')
            
            # Plot prediction (green)
            # Predictions are relative to last observation
            pred_x = obs[-1, 0] + np.cumsum(np.concatenate([[0], pred_np[:, 0]]))
            pred_y = lane_y + np.cumsum(np.concatenate([[0], pred_np[:, 1]]))
            ax.plot(pred_x, pred_y, 's-', color='#2ecc71', linewidth=3, 
                   markersize=5, label='Prediction', zorder=6)
            
            # Styling
            ax.set_xlim(obs[0, 0] - 50, max(gt_x[-1], pred_x[-1]) + 50)
            ax.set_ylim(-road_width/2 - 5, road_width/2 + 5)
            ax.set_aspect('equal')
            ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
            ax.set_title(f'Vehicle {vid} - Trajectory Prediction', 
                        fontsize=15, fontweight='bold')
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{vid}.png'), dpi=200)
            plt.close()
            
            sample_count += 1
            print(f"  [{sample_count}/{num_samples}] Generated: prediction_{vid}.png")
            
        except Exception as e:
            print(f"   Error with vehicle {vid}: {e}")
            continue
    
    print(f"\n Saved {sample_count} visualizations to {save_dir}")


# ==================== Entry Point ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trajectory predictions in SUMO")
    parser.add_argument("--mode", choices=["live", "static"], default="live",
                       help="Live simulation or static visualizations")
    parser.add_argument("--model_path", type=str, 
                       default="/Users/shahi/Developer/Project-highD/results/results_scene02_lstm/checkpoints/best_model.pt")
    parser.add_argument("--model_type", choices=["slstm", "transformer"], default="slstm")
    parser.add_argument("--csv_path", type=str, 
                       default="./simulated/data/sumo_direction1.csv",
                       help="CSV file with simulation data")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to visualize (static mode)")
    parser.add_argument("--save_dir", type=str, default="./DT_LivePredictions")
    
    args = parser.parse_args()
    
    # Update config
    Config.MODEL_PATH = args.model_path
    Config.MODEL_TYPE = args.model_type
    
    if args.mode == "live":
        # Live simulation with real-time predictions
        main_with_predictions()
    else:
        # Static visualizations from saved CSV
        visualize_static_predictions(
            args.csv_path,
            args.model_path,
            args.model_type,
            args.num_samples,
            args.save_dir
        )