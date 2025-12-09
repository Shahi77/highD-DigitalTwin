# models.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ==================== Vanilla LSTM ====================
class VanillaLSTM(nn.Module):
    """
    Basic LSTM for trajectory prediction without social context.
    Input: target sequence of shape (B, obs_len, feat_dim=7)
    Output: predicted displacements (B, pred_len, 2) in agent frame
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, obs_len=20, pred_len=25, num_layers=2):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        B = target.shape[0]
        pred_len = self.pred_len if pred_len is None else pred_len
        
        # Encode observed trajectory
        _, (h_n, c_n) = self.encoder(target)  # h_n: (num_layers, B, H)
        
        # Decode autoregressively
        outputs = []
        decoder_input = torch.zeros(B, 1, 2, device=target.device)
        h_dec, c_dec = h_n, c_n
        
        for t in range(pred_len):
            out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            pred_step = self.output_layer(out.squeeze(1))  # (B, 2)
            outputs.append(pred_step.unsqueeze(1))
            decoder_input = pred_step.unsqueeze(1)
            
        return torch.cat(outputs, dim=1)  # (B, pred_len, 2)


# ==================== CS-LSTM (Convolutional Social LSTM) ====================
class CSLSTM(nn.Module):
    """
    CS-LSTM: Uses convolutional social pooling to aggregate neighbor information.
    Based on: "Convolutional Social Pooling for Vehicle Trajectory Prediction"
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, obs_len=20, pred_len=25, 
                 k_neighbors=8, grid_size=8):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        
        # Target encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Neighbor encoder
        self.neighbor_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Convolutional social pooling
        self.social_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # Decoder
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def social_pooling(self, target_hidden, neighbor_hiddens):
        """Create spatial grid and apply convolutional pooling"""
        B = target_hidden.shape[0]
        
        # Create grid representation
        grid = torch.zeros(B, self.hidden_dim, self.grid_size, self.grid_size, 
                          device=target_hidden.device)
        
        # Place target in center
        center = self.grid_size // 2
        grid[:, :, center, center] = target_hidden
        
        # Place neighbors in grid (simplified spatial mapping)
        if neighbor_hiddens.shape[1] > 0:
            for i in range(min(neighbor_hiddens.shape[1], self.grid_size * self.grid_size - 1)):
                x = (i + 1) % self.grid_size
                y = (i + 1) // self.grid_size
                grid[:, :, y, x] = neighbor_hiddens[:, i, :]
        
        # Apply convolutional pooling
        pooled = self.social_conv(grid).squeeze(-1).squeeze(-1)  # (B, H//4)
        return pooled
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        B = target.shape[0]
        K = neigh_dyn.shape[1]
        pred_len = self.pred_len if pred_len is None else pred_len
        
        # Encode target
        _, (h_target, c_target) = self.encoder(target)
        h_target = h_target.squeeze(0)  # (B, H)
        
        # Encode neighbors
        if K > 0:
            neigh_flat = neigh_dyn.reshape(-1, neigh_dyn.shape[2], neigh_dyn.shape[3])
            _, (h_neigh, _) = self.neighbor_encoder(neigh_flat)
            h_neigh = h_neigh.squeeze(0).view(B, K, self.hidden_dim)
        else:
            h_neigh = torch.zeros(B, 0, self.hidden_dim, device=target.device)
        
        # Social pooling
        social_context = self.social_pooling(h_target, h_neigh)
        
        # Fuse target and social context
        fused = torch.tanh(self.fusion(torch.cat([h_target, social_context], dim=-1)))
        h_dec = fused.unsqueeze(0)
        c_dec = torch.zeros_like(h_dec)
        
        # Decode
        outputs = []
        decoder_input = torch.zeros(B, 1, 2, device=target.device)
        for t in range(pred_len):
            out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            pred_step = self.output_layer(out.squeeze(1))
            outputs.append(pred_step.unsqueeze(1))
            decoder_input = pred_step.unsqueeze(1)
            
        return torch.cat(outputs, dim=1)


# ==================== Social-GAN ====================
class SocialGAN(nn.Module):
    """
    Social-GAN: Generative adversarial network for socially acceptable trajectories.
    Simplified version focusing on the generator with social pooling.
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, obs_len=20, pred_len=25,
                 k_neighbors=8, noise_dim=8, pooling_type='pool_net'):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        
        # Target encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Neighbor encoder
        self.neighbor_encoder = nn.LSTM(input_dim, hidden_dim // 2, batch_first=True)
        
        # Social pooling MLP
        self.pooling_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Noise injection
        self.noise_encoder = nn.Linear(noise_dim, hidden_dim // 4)
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4, hidden_dim)
        
        # Decoder
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def social_pool(self, target_hidden, neighbor_hiddens):
        """Pool social information using MLP"""
        B = target_hidden.shape[0]
        K = neighbor_hiddens.shape[1]
        
        if K == 0:
            return torch.zeros(B, self.hidden_dim // 2, device=target_hidden.device)
        
        # Repeat target for each neighbor
        target_repeated = target_hidden.unsqueeze(1).repeat(1, K, 1)  # (B, K, H)
        
        # Concatenate and pool
        combined = torch.cat([target_repeated, neighbor_hiddens], dim=-1)  # (B, K, H + H//2)
        pooled = self.pooling_mlp(combined)  # (B, K, H//2)
        pooled = torch.max(pooled, dim=1)[0]  # Max pooling over neighbors
        
        return pooled
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None, noise=None):
        B = target.shape[0]
        K = neigh_dyn.shape[1]
        pred_len = self.pred_len if pred_len is None else pred_len
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(B, self.noise_dim, device=target.device)
        
        # Encode target
        _, (h_target, c_target) = self.encoder(target)
        h_target = h_target.squeeze(0)
        
        # Encode neighbors
        if K > 0:
            neigh_flat = neigh_dyn.reshape(-1, neigh_dyn.shape[2], neigh_dyn.shape[3])
            _, (h_neigh, _) = self.neighbor_encoder(neigh_flat)
            h_neigh = h_neigh.squeeze(0).view(B, K, self.hidden_dim // 2)
        else:
            h_neigh = torch.zeros(B, 0, self.hidden_dim // 2, device=target.device)
        
        # Social pooling
        social_context = self.social_pool(h_target, h_neigh)
        
        # Encode noise
        noise_encoded = torch.tanh(self.noise_encoder(noise))
        
        # Fuse all contexts
        fused = torch.tanh(self.fusion(torch.cat([h_target, social_context, noise_encoded], dim=-1)))
        h_dec = fused.unsqueeze(0)
        c_dec = torch.zeros_like(h_dec)
        
        # Decode
        outputs = []
        decoder_input = torch.zeros(B, 1, 2, device=target.device)
        for t in range(pred_len):
            out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            pred_step = self.output_layer(out.squeeze(1))
            outputs.append(pred_step.unsqueeze(1))
            decoder_input = pred_step.unsqueeze(1)
            
        return torch.cat(outputs, dim=1)


# ==================== GNN (Graph Neural Network) ====================
class GNNTrajectoryPredictor(nn.Module):
    """
    GNN-based trajectory prediction using graph attention networks.
    Vehicles are nodes, spatial relations are edges.
    """
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, obs_len=20, pred_len=25,
                 k_neighbors=8, num_gnn_layers=3):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        
        # Node feature encoder
        self.node_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Edge feature encoder (spatial relations)
        self.edge_encoder = nn.Sequential(
            nn.Linear(18, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        # Graph Attention Layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, hidden_dim // 4)
            for _ in range(num_gnn_layers)
        ])
        
        # Decoder
        self.decoder = nn.LSTM(output_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, target, neigh_dyn, neigh_spatial, lane, pred_len=None):
        B = target.shape[0]
        K = neigh_dyn.shape[1]
        pred_len = self.pred_len if pred_len is None else pred_len
        
        # Encode target node
        _, (h_target, c_target) = self.node_encoder(target)
        h_target = h_target.squeeze(0)  # (B, H)
        
        # Encode neighbor nodes
        if K > 0:
            neigh_flat = neigh_dyn.reshape(-1, neigh_dyn.shape[2], neigh_dyn.shape[3])
            _, (h_neigh, _) = self.node_encoder(neigh_flat)
            h_neigh = h_neigh.squeeze(0).view(B, K, self.hidden_dim)
            
            # Encode edge features (spatial relations)
            edge_feat = neigh_spatial.reshape(B, K, self.obs_len, 18)
            edge_feat = edge_feat[:, :, -1, :]  # Use last timestep
            edge_feat = self.edge_encoder(edge_feat)  # (B, K, H//4)
            
            # Stack target and neighbors
            all_nodes = torch.cat([h_target.unsqueeze(1), h_neigh], dim=1)  # (B, K+1, H)
            
            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                all_nodes = gnn_layer(all_nodes, edge_feat)
            
            # Extract updated target representation
            h_target = all_nodes[:, 0, :]
        
        # Decode
        h_dec = h_target.unsqueeze(0)
        c_dec = torch.zeros_like(h_dec)
        outputs = []
        decoder_input = torch.zeros(B, 1, 2, device=target.device)
        
        for t in range(pred_len):
            out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            pred_step = self.output_layer(out.squeeze(1))
            outputs.append(pred_step.unsqueeze(1))
            decoder_input = pred_step.unsqueeze(1)
            
        return torch.cat(outputs, dim=1)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for message passing"""
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.W_edge = nn.Linear(edge_dim, out_dim)
        self.attn = nn.Linear(out_dim * 3, 1)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, nodes, edge_features):
        """
        nodes: (B, N, in_dim) where N = K+1 (target + K neighbors)
        edge_features: (B, K, edge_dim) - features from target to each neighbor
        """
        B, N = nodes.shape[0], nodes.shape[1]
        
        # Transform node features
        h = self.W(nodes)  # (B, N, out_dim)
        
        # Target node (index 0)
        target = h[:, 0:1, :].repeat(1, N-1, 1)  # (B, K, out_dim)
        neighbors = h[:, 1:, :]  # (B, K, out_dim)
        
        if N == 1:  # No neighbors
            return self.layer_norm(h)
        
        # Transform edge features
        edge_h = self.W_edge(edge_features)  # (B, K, out_dim)
        
        # Attention scores
        attn_input = torch.cat([target, neighbors, edge_h], dim=-1)  # (B, K, 3*out_dim)
        attn_scores = torch.softmax(self.attn(attn_input), dim=1)  # (B, K, 1)
        
        # Aggregate neighbor information
        messages = (neighbors * attn_scores).sum(dim=1, keepdim=True)  # (B, 1, out_dim)
        
        # Update target node
        h_target_new = h[:, 0:1, :] + messages
        h_neighbors_new = h[:, 1:, :]
        
        # Combine
        h_new = torch.cat([h_target_new, h_neighbors_new], dim=1)
        
        return self.layer_norm(h_new)


# ==================== Original Models ====================
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
                 use_cv_warmup=True):
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

        # Teacher-forced CV Residual Warm-up
        if self.training and self.use_cv_warmup and (train_stage == 1 or train_stage is None):
            v_last = target[:, -1, :2] - target[:, -2, :2]
            t = torch.arange(1, current_pred_len+1, device=target.device).float().view(1, -1, 1)
            cv_baseline = last_obs_pos.unsqueeze(1) + t * v_last.unsqueeze(1)
            preds = preds + cv_baseline
        elif last_obs_pos is not None:
            preds = preds + last_obs_pos.unsqueeze(1)

        return preds