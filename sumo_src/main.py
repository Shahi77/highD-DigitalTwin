from math import sqrt
import random
import os
import sys
import time
import csv
import shutil

from utils import check_sumo_env, start_sumo, running

check_sumo_env()

import traci
from traci import constants as tc

TOTAL_TIME = 22539 * 4 + 2600 * 4
GUI = True
START_STEP = 0
CHECK_ALL = 0b01111
LAN_CHANGE_MODE = 0b011001011000

AVAILABLE_CAR_TYPES = [f"car{i}" for i in range(1000)]
AVAILABLE_TRUCK_TYPES = [f"truck{i}" for i in range(1000)]


def get_veh_info(edge_id, writer, step):
    try:
        vehicle_list = traci.edge.getLastStepVehicleIDs(edge_id)
        if not vehicle_list:
            return
        
        p_vehicle_list = [v for v in vehicle_list if v.startswith("p.")]
        vehicle_sum = len(vehicle_list)
        p_vehicle_sum = len(p_vehicle_list)

        for vid in vehicle_list:
            try:
                v_speed = traci.vehicle.getSpeed(vid)
                v_acc = traci.vehicle.getAcceleration(vid)
                v_lane_pos = traci.vehicle.getLanePosition(vid)
                v_type = traci.vehicle.getTypeID(vid)
                v_lane = traci.vehicle.getLaneIndex(vid)

                writer.writerow([
                    vid,
                    int(step / 4) + 1,
                    v_type,
                    round(v_speed, 4),
                    round(v_acc, 4),
                    round(v_lane_pos, 4),
                    v_lane,
                    vehicle_sum,
                    p_vehicle_sum,
                ])
            except traci.TraCIException:
                continue
    except Exception as e:
        print(f"Error getting vehicle info: {e}")


def init_csv_file(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "id",
        "frame",
        "idv_type",
        "v",
        "acc",
        "x",
        "lane_index",
        "vehicle_sum",
        "p_vehicle_sum",
    ])
    return f, writer


def gene_config():
    out_dir = "simulated"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree("./sumo_cfg", out_dir)
    return out_dir


def trajectory_tracking():
    """Load highD trajectory data for BOTH directions"""
    tracks_meta_path = "./data/highd/dataset/02_tracksMeta.csv"
    tracks_path = "./data/highd/dataset/02_tracks.csv"

    tracks_meta = {}
    try:
        with open(tracks_meta_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Load BOTH directions (1 and 2)
                direction = int(row["drivingDirection"])
                if direction in [1, 2]:
                    tracks_meta[int(row["id"])] = {
                        "initialFrame": int(row["initialFrame"]),
                        "class": row["class"],
                        "drivingDirection": direction,  # Store direction!
                    }

        with open(tracks_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                vid = int(row["id"])
                if vid in tracks_meta and tracks_meta[vid].get("found") is None:
                    tracks_meta[vid].update({
                        "found": True,
                        "xVelocity": float(row["xVelocity"]),
                        "x": float(row["x"]),
                        "laneId": int(row["laneId"]),
                    })
                    
    except FileNotFoundError as e:
        print(f"Warning: highD data not found: {e}")
        print("Continuing with empty vehicle set")
        return {}

    return tracks_meta


def aggregate_vehicles(tracks_meta):
    """Convert highD frames to simulation steps, separated by direction"""
    vehicles_to_enter = {}
    for vid, data in tracks_meta.items():
        if data.get("found"):
            data["id"] = vid
            frame = data["initialFrame"]
            sim_step = int(frame / 25 * 4)
            
            if sim_step not in vehicles_to_enter:
                vehicles_to_enter[sim_step] = []
            vehicles_to_enter[sim_step].append(data)
    
    return vehicles_to_enter


def has_vehicle_entered(step, vehicles_to_enter):
    return step in vehicles_to_enter


def main(demo_mode=True, real_engine=False, setter=None):
    # Load highD trajectory data
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    
    # Count vehicles by direction
    dir1_count = sum(1 for v in tracks_meta.values() if v.get("drivingDirection") == 1)
    dir2_count = sum(1 for v in tracks_meta.values() if v.get("drivingDirection") == 2)
    
    print(f"Loaded {len(tracks_meta)} vehicles from highD dataset")
    print(f"  Direction 1 (upper highway): {dir1_count} vehicles")
    print(f"  Direction 2 (lower highway): {dir2_count} vehicles")
    print(f"Vehicles will enter over {len(vehicles_to_enter)} simulation steps")
    
    # Setup simulation directory
    cfg_file = gene_config()
    
    # Start SUMO
    start_sumo(cfg_file + "/freeway.sumo.cfg", False, gui=GUI)
    
    step = 0
    times = 0
    random.seed(7)
    
    # Initialize CSV outputs for both directions
    f1, writer_dir1 = init_csv_file(cfg_file + "/data/sumo_direction1.csv")
    f2, writer_dir2 = init_csv_file(cfg_file + "/data/sumo_direction2.csv")
    
    vehicles_added = 0
    vehicles_failed = 0
    vehicles_dir1 = 0
    vehicles_dir2 = 0
    
    print("Starting simulation...")
    print(f"Will run for {TOTAL_TIME} timesteps (~{TOTAL_TIME/4/3600:.1f} hours sim time)")
    
    try:
        while running(demo_mode, times, TOTAL_TIME + 1):
            traci.simulationStep()
            
            # End simulation
            if demo_mode and times == TOTAL_TIME:
                print(f"\nSimulation complete!")
                print(f"Direction 1 vehicles: {vehicles_dir1}")
                print(f"Direction 2 vehicles: {vehicles_dir2}")
                print(f"Total vehicles added: {vehicles_added}")
                print(f"Vehicles failed: {vehicles_failed}")
                f1.close()
                f2.close()
                
                # Copy results
                result_dir = "../data/" + cfg_file
                if os.path.exists(result_dir):
                    shutil.rmtree(result_dir)
                shutil.copytree(cfg_file + "/data", result_dir)
                
                time.sleep(2)
                traci.close()
                break
            
            # Collect vehicle data every 4th timestep
            if times > START_STEP and times % 4 == 0:
                get_veh_info("E0", writer_dir1, times)  # Upper highway
                get_veh_info("E1", writer_dir2, times)  # Lower highway
            
            # Add vehicles from highD dataset
            if times > START_STEP and times % 4 == 0:
                current_step = int(times / 4)
                
                if has_vehicle_entered(current_step, vehicles_to_enter):
                    for data in vehicles_to_enter[current_step]:
                        # Determine vehicle type
                        vehicle_class = data["class"].lower()
                        
                        # Select appropriate vehicle type
                        if "truck" in vehicle_class or "bus" in vehicle_class:
                            type_id = random.choice(AVAILABLE_TRUCK_TYPES)
                            depart_speed = random.uniform(24, 25)
                        else:
                            type_id = random.choice(AVAILABLE_CAR_TYPES)
                            depart_speed = random.uniform(31, 33)
                        
                        # Map highD lanes (1-3) to SUMO lanes (0-2)
                        lane_id = max(0, min(2, int(data.get("laneId", 1)) - 1))
                        
                        # Starting position
                        depart_pos = random.uniform(10, 30)
                        
                        # Get driving direction and assign appropriate route
                        direction = data.get("drivingDirection", 1)
                        
                        if direction == 1:
                            route_id = "route_direction1"  # Upper highway (left to right)
                            vehicle_id = f"d1_{data['id']}"
                            vehicles_dir1 += 1
                        else:  # direction == 2
                            route_id = "route_direction2"  # Lower highway (right to left)
                            vehicle_id = f"d2_{data['id']}"
                            vehicles_dir2 += 1
                        
                        try:
                            traci.vehicle.add(
                                vehID=vehicle_id,
                                routeID=route_id,
                                typeID=type_id,
                                departSpeed=depart_speed,
                                departPos=depart_pos,
                                departLane=lane_id,
                            )
                            
                            # Set driving behavior
                            traci.vehicle.setSpeedMode(vehicle_id, CHECK_ALL)
                            traci.vehicle.setLaneChangeMode(vehicle_id, LAN_CHANGE_MODE)
                            
                            vehicles_added += 1
                            
                        except traci.exceptions.TraCIException as e:
                            vehicles_failed += 1
                            if vehicles_failed <= 10:
                                print(f"Failed to add vehicle {vehicle_id}: {e}")
                        except Exception as e:
                            vehicles_failed += 1
                            if vehicles_failed <= 10:
                                print(f"Unexpected error adding vehicle {vehicle_id}: {e}")
            
            # Progress indicator
            if times % 1000 == 0 and times > 0:
                current_vehicles = len(traci.vehicle.getIDList())
                print(f"Step {times} ({times/4:.0f}s): "
                      f"{current_vehicles} vehicles active, "
                      f"Dir1: {vehicles_dir1}, Dir2: {vehicles_dir2}")
            
            times += 1
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            f1.close()
            f2.close()
        except:
            pass
        try:
            traci.close()
        except:
            pass
    
    print("\nSimulation ended")
    print(f"Direction 1 vehicles: {vehicles_dir1}")
    print(f"Direction 2 vehicles: {vehicles_dir2}")
    print(f"Total vehicles added: {vehicles_added}")
    print(f"Total vehicles failed: {vehicles_failed}")


if __name__ == "__main__":
    main(True, False)