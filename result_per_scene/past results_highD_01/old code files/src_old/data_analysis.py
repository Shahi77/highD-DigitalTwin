import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_scene(scene_dir, scene_id=1):
    prefix = f"{int(scene_id):02d}"
    track_file = os.path.join(scene_dir, f"{prefix}_tracks.csv")
    meta_file = os.path.join(scene_dir, f"{prefix}_recordingMeta.csv")
    trackmeta_file = os.path.join(scene_dir, f"{prefix}_tracksMeta.csv")

    print(f"\n🔍 Analyzing Scene {prefix}")
    if not os.path.exists(track_file):
        raise FileNotFoundError(f"Missing {track_file}")

    df_tracks = pd.read_csv(track_file)
    df_meta = pd.read_csv(meta_file)
    df_trackmeta = pd.read_csv(trackmeta_file)

    # ------------------------------
    # Basic Info
    # ------------------------------
    print("\n📊 Basic Info")
    print(f"Tracks shape: {df_tracks.shape}")
    print(f"TrackMeta shape: {df_trackmeta.shape}")
    print(f"RecordingMeta columns: {df_meta.columns.tolist()}")

    fps = int(df_meta.iloc[0]['frameRate'])
    duration = df_meta.iloc[0]['duration']
    print(f"\nFrame rate: {fps} Hz | Duration: {duration:.2f} s")

    total_frames = df_tracks['frame'].max()
    print(f"Total frames in recording: {total_frames}")

    # ------------------------------
    # Vehicles / Trajectories
    # ------------------------------
    n_agents = df_tracks['id'].nunique()
    print(f"\nTotal agents: {n_agents}")

    traj_lengths = df_tracks.groupby("id")["frame"].count().values
    print(f"Avg trajectory length: {np.mean(traj_lengths):.1f} frames "
          f"({np.mean(traj_lengths)/fps:.1f} s)")
    print(f"Min/Max trajectory length: {np.min(traj_lengths)}, {np.max(traj_lengths)}")

    # ------------------------------
    # Coordinate ranges
    # ------------------------------
    x_min, x_max = df_tracks["x"].min(), df_tracks["x"].max()
    y_min, y_max = df_tracks["y"].min(), df_tracks["y"].max()
    print(f"\nX range: {x_min:.1f} → {x_max:.1f} m")
    print(f"Y range: {y_min:.1f} → {y_max:.1f} m")

    # ------------------------------
    # Speed / Acceleration Statistics
    # ------------------------------
    vx = df_tracks["xVelocity"]; vy = df_tracks["yVelocity"]
    speed = np.sqrt(vx**2 + vy**2)
    print(f"\nMean speed: {speed.mean():.2f} m/s ({speed.mean()*3.6:.1f} km/h)")
    print(f"Speed range: {speed.min():.2f}–{speed.max():.2f} m/s")

    ax = df_tracks["xAcceleration"]; ay = df_tracks["yAcceleration"]
    acc = np.sqrt(ax**2 + ay**2)
    print(f"Mean accel: {acc.mean():.2f} m/s² | Max: {acc.max():.2f}")

    # Detect extreme values
    high_speed = (speed > 50).sum()
    high_acc = (acc > 10).sum()
    print(f"🚧 Outliers: {high_speed} speed >50 m/s | {high_acc} accel >10 m/s²")

    # ------------------------------
    # Missing Values
    # ------------------------------
    print("\n🧩 Missing Values:")
    print(df_tracks.isna().sum())

    # ------------------------------
    # Example trajectories
    # ------------------------------
    print("\n🛣️  Plotting 5 random trajectories...")
    sample_ids = np.random.choice(df_tracks["id"].unique(), 5, replace=False)
    plt.figure(figsize=(8,6))
    for vid in sample_ids:
        g = df_tracks[df_tracks["id"] == vid]
        plt.plot(g["x"], g["y"], label=f"id {vid}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Scene {prefix} Sample Trajectories (Global Coordinates)")
    plt.legend()
    plt.grid(True)
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/scene_{prefix}_sample_trajs.png", dpi=200)
    plt.close()
    print(f"Saved plot → ./results/scene_{prefix}_sample_trajs.png")

    # ------------------------------
    # Return useful info
    # ------------------------------
    return {
        "scene": prefix,
        "fps": fps,
        "duration": duration,
        "frames": total_frames,
        "n_agents": n_agents,
        "traj_len_mean": np.mean(traj_lengths),
        "x_range": (x_min, x_max),
        "y_range": (y_min, y_max),
        "speed_mean": speed.mean(),
        "speed_max": speed.max(),
        "accel_max": acc.max(),
        "outliers_speed": int(high_speed),
        "outliers_accel": int(high_acc),
    }


if __name__ == "__main__":
    data_dir = "./data/highd/dataset"
    all_stats = []
    for sid in range(1, 4):  # analyze first 3 scenes
        try:
            info = analyze_scene(data_dir, sid)
            all_stats.append(info)
        except Exception as e:
            print(f"Error in scene {sid}: {e}")
    df = pd.DataFrame(all_stats)
    df.to_csv("./results/highd_dataset_overview.csv", index=False)
    print("\n✅ Summary saved to ./results/highd_dataset_overview.csv")
