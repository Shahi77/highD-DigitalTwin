import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_highd_scene(scene_dir, recording_id):
    """
    Load one HighD recording (tracks, trackMeta, recordingMeta)
    and return per-agent trajectories with metric coordinates (meters).
    """
    prefix = f"{int(recording_id):02d}"
    track_file = os.path.join(scene_dir, f"{prefix}_tracks.csv")
    trackmeta_file = os.path.join(scene_dir, f"{prefix}_tracksMeta.csv")
    recordmeta_file = os.path.join(scene_dir, f"{prefix}_recordingMeta.csv")

    for f in [track_file, trackmeta_file, recordmeta_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")

    df_tracks = pd.read_csv(track_file)
    df_trackmeta = pd.read_csv(trackmeta_file)
    df_recordmeta = pd.read_csv(recordmeta_file)
    meta = df_recordmeta.iloc[0].to_dict()

    fps = int(meta.get("frameRate", 25))  # 25 Hz
    dt = 1.0 / fps

    agents = {}
    for vid, g in df_tracks.groupby("id"):
        x, y = g["x"].values, g["y"].values
        vx, vy = g["xVelocity"].values, g["yVelocity"].values
        ax, ay = g["xAcceleration"].values, g["yAcceleration"].values
        heading = np.degrees(np.arctan2(vy, vx))
        t = g["frame"].values * dt
        agents[int(vid)] = {
            "t": t, "x": x, "y": y,
            "vx": vx, "vy": vy, "ax": ax, "ay": ay,
            "heading": heading,
            "laneId": g["laneId"].values
        }

    for _, row in df_trackmeta.iterrows():
        vid = int(row["id"])
        if vid in agents:
            agents[vid]["meta"] = row.to_dict()

    return {"recording_id": prefix, "fps": fps, "dt": dt, "meta": meta, "agents": agents}


def load_all_scenes(scene_dir):
    scenes = {}
    print(f"Loading HighD dataset from: {scene_dir}")
    for i in tqdm(range(1, 61), desc="Loading scenes"):
        try:
            scenes[f"{i:02d}"] = load_highd_scene(scene_dir, i)
        except Exception as e:
            print(f"Skipping {i:02d}: {e}")
    print(f"Loaded {len(scenes)} scenes.")
    return scenes


if __name__ == "__main__":
    data_dir = "./data/highd/dataset"   
    scenes = load_all_scenes(data_dir)
    print(f"\nTotal scenes loaded: {len(scenes)}")
