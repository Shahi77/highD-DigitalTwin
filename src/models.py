# models_improved.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SimpleSLSTM(nn.Module):
    """
    Simple Social-LSTM-like baseline.
    Input: target sequence of shape (B, obs_len, feat_dim=7)
    neighbors_dyn: (B, k, obs_len, feat_dim)
    neighbors_spatial: (B, k, obs_len, 18)
    Output: predicted displacements (B, pred_len, 2) in agent frame
    """
    def __init__(self, input_dim=7, hidden_dim=256, output_dim=2, obs_len=10, pred_len=25, k_neighbors=8):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.k = k_neighbors
        self.hidden_dim = hidden_dim

        # target encoder
        self.enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # neighbor encoder: encode each neighbor then pool (mean)
        self.nei_enc = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # decoder: simple LSTM that predicts displacement per step
        self.dec = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

        # init latent transform
        self.init_fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        # target: (B, obs_len, 7)
        B = target.shape[0]
        pred_len = self.pred_len if pred_len is None else pred_len

        _, (h_t, c_t) = self.enc(target)  # h_t: (1, B, H)
        h_t = h_t.squeeze(0)
        # neighbors: merge k x obs_len x 7 -> encode each neighbor then mean pool
        Bk = neigh_dyn.shape[0]
        if neigh_dyn.shape[1] == 0:
            neigh_pooled = torch.zeros(B, self.hidden_dim, device=target.device)
        else:
            nei = neigh_dyn.view(-1, neigh_dyn.shape[2], neigh_dyn.shape[3])  # (B*k, obs, 7)
            _, (h_n, _) = self.nei_enc(nei)
            h_n = h_n.squeeze(0).view(B, self.k, self.hidden_dim)
            neigh_pooled = h_n.mean(dim=1)

        # fuse
        h0 = torch.tanh(self.init_fc(torch.cat([h_t, neigh_pooled], dim=-1)))  # (B,H)
        h0 = h0.unsqueeze(0)  # (1,B,H)
        c0 = torch.zeros_like(h0)

        # decode autoregressively starting at zero displacement
        outputs = []
        inp = torch.zeros(B, 1, 2, device=target.device)  # start token (0,0)
        hx = (h0, c0)
        for t in range(pred_len):
            out, hx = self.dec(inp, hx)
            step = self.out(out[:, -1, :])  # (B,2)
            outputs.append(step.unsqueeze(1))
            inp = step.unsqueeze(1)  # feed own prediction
        out = torch.cat(outputs, dim=1)
        return out  # (B, pred_len, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultichannelAttention(nn.Module):
    """
    Paper's multichannel attention: separate channels for different feature types,
    then adaptive weighting
    """
    def __init__(self, in_dims_list, d_model):
        super().__init__()
        self.channels = nn.ModuleList()
        for in_dim in in_dims_list:
            self.channels.append(nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
            ))
        self.fusion = nn.Linear(d_model * len(in_dims_list), d_model)
    
    def forward(self, *inputs):
        """inputs: list of tensors with different feature dims but same (B, T, C_i)"""
        weighted = []
        for inp, channel in zip(inputs, self.channels):
            w = channel(inp)  # (B, T, d_model)
            weighted.append(inp @ self.channels[0][0].weight[:inp.size(-1), :].T * w)  # weighted projection
        concat = torch.cat(weighted, dim=-1)
        return self.fusion(concat)

class ImprovedTrajectoryTransformer(nn.Module):
    """
    Architecture:
    1. Multichannel attention for vehicle dynamics, spatial relations, map features
    2. Transformer encoder for sequence modeling
    3. Decoder that outputs ABSOLUTE positions (not deltas)
    4. Optional: teacher-forced constant-velocity residual warm-up
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8,
                 use_cv_warmup=True):   # <-- new toggle
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        self.use_cv_warmup = use_cv_warmup

        # Feature projections
        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)

        # Multichannel attention
        self.multi_att = MultichannelAttention([d_model, d_model, d_model, d_model], d_model)

        # Positional + encoder/decoder
        self.neigh_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1,
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)

    def forward(self, target, neigh_dyn, neigh_spatial, lane,
                last_obs_pos=None, pred_len=None, train_stage=None):
        """
        target: (B, T_obs, 7)
        neigh_dyn: (B, K, T_obs, 7)
        neigh_spatial: (B, K, T_obs, 18)
        lane: (B, T_obs, 1)
        last_obs_pos: (B, 2)
        pred_len: override
        train_stage: optional int for curriculum stage (1,2,3,...)
        """
        B, T_obs = target.size(0), target.size(1)
        K = neigh_dyn.size(1)
        current_pred_len = pred_len if pred_len is not None else self.pred_len

        # Project & fuse features
        target_feat = self.target_proj(target)
        lane_feat = self.lane_proj(lane)
        neigh_dyn_flat = neigh_dyn.view(B, K*T_obs, 7)
        neigh_dyn_proj = self.neigh_dyn_proj(neigh_dyn_flat).view(B, K, T_obs, self.d_model)
        neigh_dyn_agg = neigh_dyn_proj.mean(dim=1)
        neigh_spatial_flat = neigh_spatial.view(B, K*T_obs, 18)
        neigh_spatial_proj = self.neigh_spatial_proj(neigh_spatial_flat).view(B, K, T_obs, self.d_model)
        neigh_spatial_agg = neigh_spatial_proj.mean(dim=1)

        fused = self.multi_att(target_feat, neigh_dyn_agg, neigh_spatial_agg, lane_feat)
        fused = self.pos_enc(fused)
        memory = self.encoder(fused)

        # Decoder queries
        queries = self.query_embed.weight[:current_pred_len].unsqueeze(0).repeat(B, 1, 1)
        queries = self.pos_embed_dec(queries)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_pred_len).to(queries.device)
        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)
        preds = self.output_head(decoded)  # (B, T_pred, 2)

        # ---  Teacher-forced CV Residual Warm-up (3 lines) ---
        if self.training and self.use_cv_warmup and (train_stage == 1 or train_stage is None):
            # estimate constant-velocity baseline from last two obs
            v_last = target[:, -1, :2] - target[:, -2, :2]       # (B,2)
            t = torch.arange(1, current_pred_len+1, device=target.device).float().view(1, -1, 1)
            cv_baseline = last_obs_pos.unsqueeze(1) + t * v_last.unsqueeze(1)  # (B,T,2)
            preds = preds + cv_baseline  # learn residual around CV baseline
        elif last_obs_pos is not None:
            preds = preds + last_obs_pos.unsqueeze(1)
        # -------------------------------------------------------

        return preds
