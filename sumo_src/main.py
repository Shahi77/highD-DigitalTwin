# main.py

from math import sqrt
import random
import os
import sys
import time
import csv
import glob
import shutil

from utils import check_sumo_env, start_sumo, running

check_sumo_env()

import traci
from traci import constants as tc

TOTAL_TIME = 22539 * 4 + 2600 * 4
GUI = True
START_STEP = 0
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
    shutil.copytree("./sumo_cfg", out_dir, dirs_exist_ok=True)
    return out_dir

def trajectory_tracking():
    tracks_meta_path = "./data/highd/dataset/02_tracksMeta.csv"
    tracks_path = "./data/highd/dataset/02_tracks.csv"

    tracks_meta = {}
    try:
        with open(tracks_meta_path, newline="") as csvfile:
            import csv as _csv
            reader = _csv.DictReader(csvfile)
            for row in reader:
                if int(row["drivingDirection"]) == 1:
                    tracks_meta[int(row["id"])] = {
                        "initialFrame": int(row["initialFrame"]),
                        "class": row["class"],
                    }

        with open(tracks_path, newline="") as csvfile:
            import csv as _csv
            reader = _csv.DictReader(csvfile)
            for row in reader:
                vid = int(row["id"])
                if vid in tracks_meta and tracks_meta[vid].get("found") is None:
                    tracks_meta[vid].update(
                        {
                            "found": True,
                            "xVelocity": float(row["xVelocity"]),
                            "x": float(row["x"]),
                            "laneId": int(row["laneId"]),
                        }
                    )
    except FileNotFoundError as e:
        print(f"Warning: highD data not found: {e}")
        print("Continuing with empty vehicle set")
        return {}

    return tracks_meta

def aggregate_vehicles(tracks_meta):
    vehicles_to_enter = {}
    for vid, data in tracks_meta.items():
        if data.get("found"):
            data["id"] = vid
            frame = int(data['initialFrame'] / 25 * 4)
            vehicles_to_enter.setdefault(frame, []).append(data)
    return vehicles_to_enter

def build_sumo_cmd(sim_dir, use_gui):
    # single net source of truth under simulated/
    net = os.path.join(sim_dir, "net.xml")
    assert os.path.isfile(net), f"Missing net {net}"

    # optional files
    routes = os.path.join(sim_dir, "route.xml")
    add_files = []

    # include vTypeDistributions if present
    vtypes = os.path.join(sim_dir, "vTypeDistributions.add.xml")
    if os.path.isfile(vtypes):
        add_files.append(vtypes)

    # include any other *.add.xml under simulated, except duplicates
    for p in glob.glob(os.path.join(sim_dir, "*.add.xml")):
        if p != vtypes:
            add_files.append(p)

    cmd = [
        "sumo-gui" if use_gui else "sumo",
        "-n",
        net,
    ]

    if os.path.isfile(routes):
        cmd += ["-r", routes]

    if add_files:
        cmd += ["-a", ",".join(add_files)]

    # standard settings
    cmd += [
        "--step-length",
        "0.25",
        "--no-step-log",
        "true",
        "--time-to-teleport",
        "-1",
    ]

    return cmd

def choose_type_from_class(clazz):
    if clazz.lower().startswith("truck") or clazz.lower().startswith("bus"):
        return random.choice(AVAILABLE_TRUCK_TYPES)
    return random.choice(AVAILABLE_CAR_TYPES)

def safe_add_vehicle(veh_id, edge_id, lane_index, vtype_id, depart_speed="max"):
    try:
        # if a route file exists, prefer a route id if you have one
        # fallback is edge-only with "addFull" that builds a default route of one edge
        traci.vehicle.addFull(
            vehID=veh_id,
            routeID="",
            typeID=vtype_id,
            depart=traci.simulation.getTime(),
            departLane=str(lane_index),
            departSpeed=depart_speed,
            departPos="base",
            arrivalLane="current",
            arrivalPos="max",
            arrivalSpeed="current",
        )
        # set the first edge if needed when no route is provided
        if edge_id:
            traci.vehicle.setRoute(veh_id, [edge_id])
    except traci.TraCIException as e:
        # as a backup, place directly on a concrete lane if known
        try:
            lane_id = f"{edge_id}_{lane_index}"
            traci.vehicle.add(
                vehID=veh_id,
                routeID="",
                typeID=vtype_id,
                depart=traci.simulation.getTime(),
                departSpeed=depart_speed,
                departLane=lane_id,
            )
        except traci.TraCIException as e2:
            print(f"spawn failed for {veh_id}: {e} | {e2}")

def main(demo_mode=False, real_engine=False, setter=None):
    sim_dir = gene_config()  # copies sumo_cfg -> simulated
    sumo_cmd = build_sumo_cmd(sim_dir, use_gui=GUI)

    print("Starting SUMO with:", " ".join(sumo_cmd))
    traci.start(sumo_cmd)
    safe_add_vehicle("debug0", "E0", 1, random.choice(AVAILABLE_CAR_TYPES))

    # write telemetry on the first main edge if present, else skip
    telem_edge = "E0" if "E0" in traci.edge.getIDList() else None


    # prepare CSV
    out_file, writer = init_csv_file("sim_out/telemetry.csv")

    # load highD-derived entry schedule if available
    tracks_meta = trajectory_tracking()
    vehicles_to_enter = aggregate_vehicles(tracks_meta)
    print(f"Loaded {len(tracks_meta)} vehicles from highD dataset")
    print(f"Frames with entries: {len(vehicles_to_enter)}")

    try:
        step = START_STEP
        next_internal_spawn = 0

        while running() and step <= TOTAL_TIME:
            # schedule from highD frames
            if step in vehicles_to_enter:
                for rec in vehicles_to_enter[step]:
                    vid = f"h{rec['id']}"
                    vtype = choose_type_from_class(rec.get("class", "car"))
                    # naïve mapping to a starting edge and lane
                    start_edge = "E0" if traci.edge.exists("E0") else None
                    lane_index = max(0, min(2, rec.get("laneId", 0) - 1))
                    safe_add_vehicle(vid, start_edge, lane_index, vtype)

            # small internal trickle, helps when no routes exist
            if traci.simulation.getMinExpectedNumber() < 100 and step >= next_internal_spawn:
                if traci.edge.exists("E0"):
                    vid = f"test_{int(step)}"
                    vtype = random.choice(AVAILABLE_CAR_TYPES)
                    lane_index = random.randint(0, max(0, traci.lane.getNumLanes("E0_0") - 1)) if traci.lane.exists("E0_0") else 0
                    safe_add_vehicle(vid, "E0", lane_index, vtype)
                next_internal_spawn = step + 50

            # advance sim
            traci.simulationStep()

            # telemetry
            if telem_edge is not None:
                get_veh_info(telem_edge, writer, step)

            step += 1

        print("Simulation loop finished")

    except Exception as e:
        print(f"Fatal error in loop: {e}")
    finally:
        try:
            traci.close(False)
        except Exception:
            pass
        out_file.close()

if __name__ == "__main__":
    main(False, False)
