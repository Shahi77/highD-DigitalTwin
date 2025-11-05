# highd_dataloader_fixed.py
"""
FIXED VERSION: Proper relative coordinate handling for trajectory prediction
Key fix: Ground truth is displacement from last observed position, not absolute coords
"""
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader

DOWNSAMPLE = 5
OBS_SECONDS = 2.0
PRED_SECONDS = 5.0
OBS_LEN = int(OBS_SECONDS * 5)
PRED_LEN = int(PRED_SECONDS * 5)
K_NEIGH = 8
MAX_SPEED = 50.0

def compute_8dir_features(target_pos, target_vel, target_acc, target_heading,
                          neigh_pos, neigh_vel, neigh_acc, neigh_heading):
    """Compute 8-directional spatial relationship features"""
    dx = neigh_pos[0] - target_pos[0]
    dy = neigh_pos[1] - target_pos[1]
    dist = np.sqrt(dx**2 + dy**2) + 1e-6
    
    angle = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle) % 360
    
    D_dirs = np.zeros(8)
    V_dirs = np.zeros(8)
    
    bin_idx = int((angle_deg + 22.5) // 45) % 8
    D_dirs[bin_idx] = dist
    
    dvx = neigh_vel[0] - target_vel[0]
    dvy = neigh_vel[1] - target_vel[1]
    V_dirs[bin_idx] = np.sqrt(dvx**2 + dvy**2)
    
    dax = neigh_acc[0] - target_acc[0]
    day = neigh_acc[1] - target_acc[1]
    delta_a = np.sqrt(dax**2 + day**2)
    
    delta_phi = neigh_heading - target_heading
    delta_phi = (delta_phi + np.pi) % (2*np.pi) - np.pi
    
    return np.concatenate([D_dirs, V_dirs, [delta_a, delta_phi]])

class HighDDatasetFixed(Dataset):
    """
    FIXED: Proper relative coordinate handling
    - Observed trajectory: relative to first observation (or absolute, doesn't matter much)
    - Future trajectory: DISPLACEMENT from last observed position
    - This ensures model predicts "how far vehicle will move" not "where it will be"
    """
    def __init__(self, tracks_df, obs_len=OBS_LEN, pred_len=PRED_LEN,
                 downsample=DOWNSAMPLE, k_neighbors=K_NEIGH, max_speed=MAX_SPEED):
        super().__init__()
        self.tracks = tracks_df
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.down = downsample
        self.k = k_neighbors
        self.max_speed = max_speed

        # Group by vehicle
        self.groups = {}
        for vid, g in tracks_df.groupby('id'):
            df = g.sort_values('frame').copy()
            if 'heading' not in df.columns:
                df['heading'] = np.arctan2(df['yVelocity'].values, df['xVelocity'].values)
            arr = df[['frame','x','y','xVelocity','yVelocity',
                      'xAcceleration','yAcceleration','heading','laneId']].values
            self.groups[vid] = arr

        # Build samples
        self.samples = []
        for vid, arr in self.groups.items():
            n = len(arr)
            step = self.down
            total_needed = (self.obs_len + self.pred_len) * step
            if n < total_needed:
                continue
            for start in range(0, n - total_needed + 1, step):
                self.samples.append((vid, start))

        if len(self.samples) == 0:
            raise ValueError("No valid samples!")

    def __len__(self):
        return len(self.samples)

    def _get_window(self, vid, start):
        arr = self.groups[vid]
        idxs = np.arange(start, start + (self.obs_len + self.pred_len)*self.down, self.down)
        return arr[idxs.astype(int), :]

    def __getitem__(self, idx):
        vid, start = self.samples[idx]
        window = self._get_window(vid, start)
        T = window.shape[0]
        
        frames = window[:, 0].astype(int)
        x, y = window[:, 1], window[:, 2]
        vx, vy = window[:, 3], window[:, 4]
        ax, ay = window[:, 5], window[:, 6]
        heading = window[:, 7]
        lane_id = window[:, 8]

        # Clip speeds
        speeds = np.sqrt(vx**2 + vy**2)
        if np.any(speeds > self.max_speed):
            factor = np.minimum(1.0, self.max_speed / (speeds + 1e-6))
            vx *= factor
            vy *= factor

        obs_x, obs_y = x[:self.obs_len], y[:self.obs_len]
        fut_x, fut_y = x[self.obs_len:], y[self.obs_len:]

        # Agent-centric: origin at last observed position
        origin = np.array([obs_x[-1], obs_y[-1]])
        yaw = heading[self.obs_len - 1]

        def transform(xy_arr, vel_arr=None, acc_arr=None, head_arr=None):
            """Transform to agent frame centered at origin with rotation -yaw"""
            c, s = np.cos(-yaw), np.sin(-yaw)
            dx = xy_arr[:, 0] - origin[0]
            dy = xy_arr[:, 1] - origin[1]
            xr = c*dx - s*dy
            yr = s*dx + c*dy
            result = [np.stack([xr, yr], axis=-1)]
            
            if vel_arr is not None:
                vx_r = c*vel_arr[:, 0] - s*vel_arr[:, 1]
                vy_r = s*vel_arr[:, 0] + c*vel_arr[:, 1]
                result.append(np.stack([vx_r, vy_r], axis=-1))
            if acc_arr is not None:
                ax_r = c*acc_arr[:, 0] - s*acc_arr[:, 1]
                ay_r = s*acc_arr[:, 0] + c*acc_arr[:, 1]
                result.append(np.stack([ax_r, ay_r], axis=-1))
            if head_arr is not None:
                result.append((head_arr - yaw + np.pi) % (2*np.pi) - np.pi)
            return result

        obs_xy = np.stack([obs_x, obs_y], axis=-1)
        fut_xy = np.stack([fut_x, fut_y], axis=-1)
        obs_vel = np.stack([vx[:self.obs_len], vy[:self.obs_len]], axis=-1)
        obs_acc = np.stack([ax[:self.obs_len], ay[:self.obs_len]], axis=-1)
        
        obs_xy_rel, obs_vel_rel, obs_acc_rel, obs_head_rel = transform(
            obs_xy, obs_vel, obs_acc, heading[:self.obs_len])
        fut_xy_rel = transform(fut_xy)[0]
        
        # CRITICAL: Last observed position in agent frame is [0, 0]
        # Verify this:
        assert np.allclose(obs_xy_rel[-1], [0, 0], atol=1e-5), "Last obs should be at origin!"
        
        # Future trajectory is already displacement from origin [0,0] (which is last obs position)
        # So fut_xy_rel is correct as-is: it represents "where vehicle will be relative to last obs"
        
        # Target features: [x,y,vx,vy,ax,ay,heading] (7D) in agent frame
        target_feats = np.concatenate([
            obs_xy_rel,
            obs_vel_rel,
            obs_acc_rel,
            obs_head_rel.reshape(-1, 1)
        ], axis=-1).astype(np.float32)

        # Find K neighbors + compute 8-dir features
        last_frame = frames[self.obs_len - 1]
        neighbors_dyn = np.zeros((self.k, self.obs_len, 7), dtype=np.float32)
        neighbors_spatial = np.zeros((self.k, self.obs_len, 18), dtype=np.float32)
        
        candidates = []
        for other_vid, other_arr in self.groups.items():
            if other_vid == vid:
                continue
            matching = np.where(other_arr[:, 0].astype(int) == last_frame)[0]
            if len(matching) == 0:
                continue
            idx = matching[0]
            if idx - (self.obs_len - 1)*self.down < 0:
                continue
            idxs = np.arange(idx - (self.obs_len-1)*self.down, idx+1, self.down).astype(int)
            neigh_win = other_arr[idxs, :]
            
            # Transform neighbor to agent frame
            n_xy = neigh_win[:, 1:3]
            n_vel = neigh_win[:, 3:5]
            n_acc = neigh_win[:, 5:7]
            n_head = neigh_win[:, 7]
            
            n_xy_rel, n_vel_rel, n_acc_rel, n_head_rel = transform(n_xy, n_vel, n_acc, n_head)
            
            dyn_feats = np.concatenate([n_xy_rel, n_vel_rel, n_acc_rel, n_head_rel.reshape(-1,1)], axis=-1)
            
            # Compute 8-directional features at each time step
            spatial_feats = []
            for t in range(self.obs_len):
                feat_8d = compute_8dir_features(
                    obs_xy_rel[t], obs_vel_rel[t], obs_acc_rel[t], obs_head_rel[t],
                    n_xy_rel[t], n_vel_rel[t], n_acc_rel[t], n_head_rel[t]
                )
                spatial_feats.append(feat_8d)
            spatial_feats = np.array(spatial_feats)
            
            dist = np.linalg.norm(n_xy_rel[-1])
            candidates.append((dist, dyn_feats, spatial_feats))
        
        candidates.sort(key=lambda x: x[0])
        for i, (d, dyn, spatial) in enumerate(candidates[:self.k]):
            neighbors_dyn[i] = dyn
            neighbors_spatial[i] = spatial

        # Map features
        lane_feats = (lane_id[:self.obs_len] / 10.0).reshape(-1, 1).astype(np.float32)

        return {
            'target_feats': target_feats,  # (T_obs, 7) in agent frame, last pos is [0,0]
            'neighbors_dyn': neighbors_dyn,
            'neighbors_spatial': neighbors_spatial,
            'lane_feats': lane_feats,
            'gt': fut_xy_rel.astype(np.float32),  # (T_pred, 2) displacement from [0,0]
            'meta': {
                'vid': vid,
                'start': start,
                'origin': origin,  # world coords of last obs position
                'yaw': yaw,
                'obs_world': obs_xy,  # for visualization
                'fut_world': fut_xy  # for visualization
            }
        }

def collate_fn_fixed(batch):
    B = len(batch)
    obs_len = batch[0]['target_feats'].shape[0]
    pred_len = batch[0]['gt'].shape[0]
    K = batch[0]['neighbors_dyn'].shape[0]

    target = np.zeros((B, obs_len, 7), dtype=np.float32)
    neigh_dyn = np.zeros((B, K, obs_len, 7), dtype=np.float32)
    neigh_spatial = np.zeros((B, K, obs_len, 18), dtype=np.float32)
    lane = np.zeros((B, obs_len, 1), dtype=np.float32)
    gt = np.zeros((B, pred_len, 2), dtype=np.float32)
    metas = []

    for i, s in enumerate(batch):
        target[i] = s['target_feats']
        neigh_dyn[i] = s['neighbors_dyn']
        neigh_spatial[i] = s['neighbors_spatial']
        lane[i] = s['lane_feats']
        gt[i] = s['gt']
        metas.append(s['meta'])

    return {
        'target': torch.from_numpy(target),
        'neigh_dyn': torch.from_numpy(neigh_dyn),
        'neigh_spatial': torch.from_numpy(neigh_spatial),
        'lane': torch.from_numpy(lane),
        'gt': torch.from_numpy(gt),
        'meta': metas
    }

def make_dataloader_fixed(tracks_df, batch_size=32, shuffle=True, **kwargs):
    ds = HighDDatasetFixed(tracks_df, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_fixed)