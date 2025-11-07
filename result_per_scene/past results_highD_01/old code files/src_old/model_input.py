import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, features, hist_len=50, fut_len=75, step=5):
        self.samples=[]
        for traj in features:
            for st in range(0,len(traj)-hist_len-fut_len,step):
                past=traj[st:st+hist_len]
                fut=traj[st+hist_len:st+hist_len+fut_len,:2]
                self.samples.append((past,fut))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,f=self.samples[i]
        return torch.tensor(p,dtype=torch.float32),torch.tensor(f,dtype=torch.float32)

def build_dataloader(features, batch_size=16, hist_len=50, fut_len=75, step=5):
    ds = TrajectoryDataset(features, hist_len, fut_len, step)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

if __name__ == "__main__":
    from data_loader import load_highd_scene
    from sync import align_scene
    from feature_extraction import extract_features

    # Load and prepare features
    scene_dir = "./data/highd/dataset"
    scene = load_highd_scene(scene_dir, 1)
    scene_aligned = align_scene(scene, Δt=0.1)
    features, ids = extract_features(scene_aligned, include_neighbors=True, k_neighbors=3)

    # Build dataset + dataloader
    dataset, loader = build_dataloader(features, hist_len=20, fut_len=30, step=10, batch_size=16)

    # Fetch one batch for model input
    xb, yb = next(iter(loader))
    print(f"\nBatch input shape:  {xb.shape}  (batch, seq, features)")
    print(f"Batch target shape: {yb.shape}  (batch, horizon, 2)")
