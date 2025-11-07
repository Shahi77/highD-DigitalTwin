import numpy as np
from tqdm import tqdm

def extract_features(scene_aligned, include_neighbors=True, k_neighbors=3):
    agent_ids = list(scene_aligned["agents"].keys())
    positions = {vid: a["states"][:, :2] for vid, a in scene_aligned["agents"].items()}
    feats = []

    for vid in tqdm(agent_ids, desc="Extracting features"):
        a = scene_aligned["agents"][vid]
        base = np.concatenate([a["states"], a["confidence"][:, None]], axis=1)

        if include_neighbors:
            d = []
            for nb, nbpos in positions.items():
                if nb == vid: continue
                d.append((nb, np.mean(np.linalg.norm(nbpos - positions[vid], axis=1))))
            nearest = sorted(d, key=lambda x: x[1])[:k_neighbors]
            neigh = []
            for nb, _ in nearest:
                nb_s = scene_aligned["agents"][nb]["states"]
                rel = nb_s[:, :4] - a["states"][:, :4]
                neigh.append(rel)
            if neigh:
                base = np.concatenate([base, np.concatenate(neigh, axis=1)], axis=1)

        # --- Safety: enforce agent-centric local (0,0) origin ---
        base[:, :2] -= base[0, :2]
        feats.append(base)

    return np.stack(feats), agent_ids


if __name__ == "__main__":
    from data_loader import load_highd_scene
    from sync import align_scene
    scene_dir = "./data/highd/dataset"
    scene = load_highd_scene(scene_dir, 1)
    aligned = align_scene(scene, Δt=0.04)
    feats, ids = extract_features(aligned)
    print(f"\nExtracted features for {len(ids)} agents with shape: {feats.shape}")
