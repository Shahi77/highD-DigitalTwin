# highd_dataloader.py
import os, math, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# defaults
DOWNSAMPLE, OBS_LEN, PRED_LEN, K_NEIGH, MAX_SPEED = 1, 20, 20, 8, 50.0

# -------------------------------------------------------------------------
# Scene loading
# -------------------------------------------------------------------------
def load_highd_scenes(folder_path, train_ratio=0.8):
    csvs = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("_tracks.csv")])
    if not csvs:
        raise ValueError(f"No *_tracks.csv found in {folder_path}")
    n_train = int(len(csvs) * train_ratio)
    train, val = csvs[:n_train], csvs[n_train:]
    print(f"Detected {len(csvs)} scenes → train={len(train)}, val={len(val)}")
    return train, val

def make_dataloader_highd_multiscene(files, batch_size=32, obs_len=OBS_LEN, pred_len=PRED_LEN, shuffle=True, **kw):
    datasets = []
    for f in files:
        loader = make_dataloader_highd(f, batch_size=1, obs_len=obs_len, pred_len=pred_len, shuffle=False, **kw)
        datasets.append(loader.dataset)
    if not datasets:
        raise ValueError("No valid datasets from file list")
    combined = ConcatDataset(datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_fixed, drop_last=True)

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class HighDDatasetFixed(Dataset):
    """Agent-centric relative coordinates (future = displacement from last obs)."""
    def __init__(self, tracks_df, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 downsample=DOWNSAMPLE, k_neighbors=K_NEIGH, max_speed=MAX_SPEED):
        super().__init__()
        if isinstance(tracks_df, str):
            tracks_df = pd.read_csv(tracks_df)
        self.obs_len, self.pred_len, self.down, self.k, self.max_speed = obs_len, pred_len, downsample, k_neighbors, max_speed

        id_col = "id" if "id" in tracks_df.columns else ("Vehicle_ID" if "Vehicle_ID" in tracks_df.columns else None)
        if id_col is None:
            raise ValueError("Missing id or Vehicle_ID column")
        self.groups = {}
        for vid, g in tracks_df.groupby(id_col):
            df = g.sort_values(by=g.columns[0])
            if "heading" not in df.columns:
                if {"xVelocity","yVelocity"} <= set(df.columns):
                    df["heading"] = np.arctan2(df["yVelocity"], df["xVelocity"])
                else:
                    df["heading"] = 0.0
            arr = df[["frame","x","y","xVelocity","yVelocity","xAcceleration","yAcceleration","heading","laneId"]].to_numpy()
            self.groups[vid] = arr

        self.samples = []
        for vid, arr in self.groups.items():
            n = len(arr); step = self.down
            need = (self.obs_len + self.pred_len) * step
            if n < need: continue
            for start in range(0, n - need + 1, step):
                self.samples.append((vid, start))
        if not self.samples:
            raise ValueError("No valid samples in dataset")

    def __len__(self): return len(self.samples)

    def _window(self, vid, start):
        arr = self.groups[vid]
        idxs = np.arange(start, start + (self.obs_len + self.pred_len) * self.down, self.down, int)
        return arr[idxs]

    def __getitem__(self, idx):
        vid, start = self.samples[idx]; w = self._window(vid, start)
        x, y, vx, vy, ax, ay, head, lane = w[:,1],w[:,2],w[:,3],w[:,4],w[:,5],w[:,6],w[:,7],w[:,8]
        obs_x, obs_y, fut_x, fut_y = x[:self.obs_len], y[:self.obs_len], x[self.obs_len:], y[self.obs_len:]
        origin, yaw = np.array([obs_x[-1], obs_y[-1]]), float(head[self.obs_len-1])

        def tf(xy,v=None,a=None,h=None):
            c,s=np.cos(-yaw),np.sin(-yaw)
            dx,dy=xy[:,0]-origin[0],xy[:,1]-origin[1]
            r=[np.stack([c*dx-s*dy,s*dx+c*dy],-1)]
            if v is not None: r.append(np.stack([c*v[:,0]-s*v[:,1],s*v[:,0]+c*v[:,1]],-1))
            if a is not None: r.append(np.stack([c*a[:,0]-s*a[:,1],s*a[:,0]+c*a[:,1]],-1))
            if h is not None: r.append(((h-yaw+np.pi)%(2*np.pi)-np.pi).astype(np.float32))
            return r

        obs_xy=np.stack([obs_x,obs_y],-1); obs_v=np.stack([vx[:self.obs_len],vy[:self.obs_len]],-1)
        obs_a=np.stack([ax[:self.obs_len],ay[:self.obs_len]],-1)
        obs_xy_r, obs_v_r, obs_a_r, obs_h_r = tf(obs_xy,obs_v,obs_a,head[:self.obs_len])
        fut_xy_r = tf(np.stack([fut_x,fut_y],-1))[0]

        target_feats=np.concatenate([obs_xy_r,obs_v_r,obs_a_r,obs_h_r.reshape(-1,1)],-1).astype(np.float32)
        lane_feats=(lane[:self.obs_len]/10.0).reshape(-1,1).astype(np.float32)
        gt=fut_xy_r.astype(np.float32)
        neigh_dyn=np.zeros((self.k,self.obs_len,7),np.float32)
        neigh_sp=np.zeros((self.k,self.obs_len,18),np.float32)
        return {"target_feats":target_feats,"neighbors_dyn":neigh_dyn,"neighbors_spatial":neigh_sp,"lane_feats":lane_feats,"gt":gt}
# -------------------------------------------------------------------------
# Collate
# -------------------------------------------------------------------------
def collate_fn_fixed(batch):
    B=len(batch);obs_len=batch[0]["target_feats"].shape[0];pred_len=batch[0]["gt"].shape[0];K=batch[0]["neighbors_dyn"].shape[0]
    t=np.zeros((B,obs_len,7),np.float32);nd=np.zeros((B,K,obs_len,7),np.float32)
    ns=np.zeros((B,K,obs_len,18),np.float32);lane=np.zeros((B,obs_len,1),np.float32);gt=np.zeros((B,pred_len,2),np.float32)
    for i,s in enumerate(batch):
        t[i]=s["target_feats"];nd[i]=s["neighbors_dyn"];ns[i]=s["neighbors_spatial"];lane[i]=s["lane_feats"];gt[i]=s["gt"]
    d={"target":torch.from_numpy(t),"neigh_dyn":torch.from_numpy(nd),
       "neigh_spatial":torch.from_numpy(ns),"lane":torch.from_numpy(lane),"gt":torch.from_numpy(gt)}
    # compatibility aliases
    d["neighbors_dyn"]=d["neigh_dyn"]; d["neighbors_spatial"]=d["neigh_spatial"]
    return d

# -------------------------------------------------------------------------
# Dataloader factories
# -------------------------------------------------------------------------
def make_dataloader_fixed(df_or_path,batch_size=32,shuffle=True,**kw):
    if isinstance(df_or_path,str): df_or_path=pd.read_csv(df_or_path)
    ds=HighDDatasetFixed(df_or_path,**kw)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_fn_fixed)

def make_dataloader_highd(path_or_df,batch_size=32,obs_len=OBS_LEN,pred_len=PRED_LEN,shuffle=True,**kw):
    if isinstance(path_or_df,str):
        if os.path.isdir(path_or_df):
            train,val=load_highd_scenes(path_or_df)
            return (make_dataloader_highd_multiscene(train,batch_size,obs_len,pred_len,shuffle=True,**kw),
                    make_dataloader_highd_multiscene(val,batch_size,obs_len,pred_len,shuffle=False,**kw))
        df=pd.read_csv(path_or_df)
    else: df=path_or_df
    return make_dataloader_fixed(df,batch_size=batch_size,shuffle=shuffle,obs_len=obs_len,pred_len=pred_len,**kw)
