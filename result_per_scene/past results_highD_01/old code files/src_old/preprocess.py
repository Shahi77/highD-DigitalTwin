import numpy as np
from tqdm import tqdm

def filter_agents(scene_aligned, min_len=100):
    """Keep agents with ≥min_len frames."""
    return {vid: a for vid,a in scene_aligned["agents"].items() if len(a["states"])>=min_len}

def normalize_agent(agent):
    s = agent["states"].copy()
    origin = s[0,:2]
    s[:,:2] -= origin
    return s

def build_sequences(scene_aligned, hist_len=50, fut_len=75, step=5):
    """Past 2s → future 3s at 25Hz."""
    seqs=[]
    for vid,a in tqdm(scene_aligned["agents"].items(), desc="Building sequences"):
        s = normalize_agent(a); conf = a["confidence"]
        for st in range(0, len(s)-hist_len-fut_len, step):
            seqs.append((s[st:st+hist_len], s[st+hist_len:st+hist_len+fut_len], conf[st:st+hist_len+fut_len]))
    return seqs


if __name__ == "__main__":
    from data_loader import load_highd_scene
    from sync import align_scene

    scene_dir = "./data/highd/dataset"
    scene = load_highd_scene(scene_dir, 1)
    scene_aligned = align_scene(scene, Δt=0.1)

    # Filter & visualize
    scene_aligned["agents"] = filter_agents(scene_aligned, min_len=100)
    # Build training sequences
    sequences = build_sequences(scene_aligned, hist_len=20, fut_len=30, step=5)
    print(f"\n Preprocessing complete: {len(sequences)} usable (past→future) samples.")
