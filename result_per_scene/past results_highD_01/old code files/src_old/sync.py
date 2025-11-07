import numpy as np
from tqdm import tqdm

def interpolate(a, b, ratio):
    return a + (b - a) * ratio

def predictive_fill(agent, master_times, dt):
    """Interpolate missing timestamps for one agent."""
    t = agent["t"]; x, y = agent["x"], agent["y"]
    vx, vy = agent["vx"], agent["vy"]
    ax, ay = agent["ax"], agent["ay"]

    aligned, conf = [], []
    for tm in master_times:
        if tm <= t[0]:
            aligned.append([x[0], y[0], vx[0], vy[0], ax[0], ay[0]]); conf.append(1.0)
        elif tm >= t[-1]:
            Δt = tm - t[-1]
            aligned.append([
                x[-1] + vx[-1]*Δt + 0.5*ax[-1]*Δt**2,
                y[-1] + vy[-1]*Δt + 0.5*ay[-1]*Δt**2,
                vx[-1], vy[-1], ax[-1], ay[-1]
            ]); conf.append(0.5)
        else:
            i = np.searchsorted(t, tm) - 1
            r = (tm - t[i]) / (t[i+1] - t[i])
            aligned.append([
                interpolate(x[i], x[i+1], r),
                interpolate(y[i], y[i+1], r),
                interpolate(vx[i], vx[i+1], r),
                interpolate(vy[i], vy[i+1], r),
                interpolate(ax[i], ax[i+1], r),
                interpolate(ay[i], ay[i+1], r)
            ])
            conf.append(1.0)
    return np.array(aligned), np.array(conf)

def align_scene(scene, Δt=0.04):
    """Align to 25 Hz (0.04 s) timestamps and convert to agent-centric coordinates."""
    max_t = max(max(a["t"]) for a in scene["agents"].values())
    master_times = np.arange(0, max_t, Δt)
    aligned_agents = {}
    
    for vid, agent in tqdm(scene["agents"].items(), desc=f"Aligning scene {scene['recording_id']}"):
        s, c = predictive_fill(agent, master_times, Δt)
        
        # --- Convert to agent-centric local coordinates ---
        s[:, 0] -= s[0, 0]  # x -> start from 0
        s[:, 1] -= s[0, 1]  # y -> start from 0
        
        aligned_agents[vid] = {"states": s, "confidence": c, "meta": agent.get("meta", {})}
    
    return {"times": master_times, "agents": aligned_agents, "recording_id": scene["recording_id"]}


if __name__ == "__main__":
    from data_loader import load_highd_scene
    scene_dir = "./data/highd/dataset"
    scene = load_highd_scene(scene_dir, 1)
    aligned = align_scene(scene, Δt=0.04)
    print(f"\nAligned agents: {len(aligned['agents'])}")
    print(f"Aligned time steps: {len(aligned['times'])}")
