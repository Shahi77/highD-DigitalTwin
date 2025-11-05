# models_improved.py
import torch
import torch.nn as nn
import math

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
    Architecture closer to paper:
    1. Multichannel attention for vehicle dynamics, spatial relations, map features
    2. Transformer encoder for sequence modeling
    3. Decoder that outputs ABSOLUTE positions (not deltas) using autoregressive attention
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, pred_len=25, k_neighbors=8):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len
        
        # Feature projections
        self.target_proj = nn.Linear(7, d_model)
        self.neigh_dyn_proj = nn.Linear(7, d_model)
        self.neigh_spatial_proj = nn.Linear(18, d_model)
        self.lane_proj = nn.Linear(1, d_model)
        
        # Multichannel attention (paper Eq. 1-4)
        self.multi_att = MultichannelAttention([d_model, d_model, d_model, d_model], d_model)
        
        # Neighbor aggregation with attention
        self.neigh_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoder (paper uses 8 heads as mentioned)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder: autoregressive with cross-attention to encoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=4*d_model, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output head: predict ABSOLUTE position at each step
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )
        
        # Learnable query embeddings for decoder
        self.query_embed = nn.Embedding(pred_len, d_model)
        self.pos_embed_dec = PositionalEncoding(d_model, max_len=pred_len)
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, last_obs_pos=None, pred_len=None):
        """
        target: (B, T_obs, 7)
        neigh_dyn: (B, K, T_obs, 7)
        neigh_spatial: (B, K, T_obs, 18)
        lane: (B, T_obs, 1)
        last_obs_pos: (B, 2) - last observed position in agent frame (typically [0,0])
        pred_len: int - override prediction length (for curriculum training)
        
        Returns: (B, T_pred, 2) ABSOLUTE positions in agent frame
        """
        B, T_obs = target.size(0), target.size(1)
        K = neigh_dyn.size(1)
        
        # Use provided pred_len or default
        current_pred_len = pred_len if pred_len is not None else self.pred_len
        
        # Project features
        target_feat = self.target_proj(target)  # (B, T, D)
        lane_feat = self.lane_proj(lane)  # (B, T, D)
        
        # Aggregate neighbors: pool dynamic and spatial separately
        neigh_dyn_flat = neigh_dyn.view(B, K*T_obs, 7)
        neigh_dyn_proj = self.neigh_dyn_proj(neigh_dyn_flat).view(B, K, T_obs, self.d_model)
        neigh_dyn_agg = neigh_dyn_proj.mean(dim=1)  # (B, T, D)
        
        neigh_spatial_flat = neigh_spatial.view(B, K*T_obs, 18)
        neigh_spatial_proj = self.neigh_spatial_proj(neigh_spatial_flat).view(B, K, T_obs, self.d_model)
        neigh_spatial_agg = neigh_spatial_proj.mean(dim=1)  # (B, T, D)
        
        # Multichannel attention fusion (paper's key innovation)
        fused = self.multi_att(target_feat, neigh_dyn_agg, neigh_spatial_agg, lane_feat)  # (B, T, D)
        
        # Add positional encoding and encode
        fused = self.pos_enc(fused)
        memory = self.encoder(fused)  # (B, T_obs, D)
        
        # Decoder: generate query sequence (use current_pred_len, not self.pred_len)
        queries = self.query_embed.weight[:current_pred_len].unsqueeze(0).repeat(B, 1, 1)  # (B, current_pred_len, D)
        queries = self.pos_embed_dec(queries)
        
        # Autoregressive decoding with causal mask (MUST match current_pred_len)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_pred_len).to(queries.device)
        decoded = self.decoder(queries, memory, tgt_mask=tgt_mask)  # (B, current_pred_len, D)
        
        # Output absolute positions
        preds = self.output_head(decoded)  # (B, current_pred_len, 2)
        
        # If last_obs_pos provided, predictions are relative to that (but typically [0,0] in agent frame)
        if last_obs_pos is not None:
            preds = preds + last_obs_pos.unsqueeze(1)
        
        return preds