import torch
import torch.nn as nn

class TransformerLight(nn.Module):
    def __init__(self, input_dim=19, d_model=64, nhead=2, num_layers=1, pred_horizon=75):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 2)  # output Δx, Δy
        self.pred_horizon = pred_horizon

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        last = x[:, -1:, :]  # last timestep embedding
        preds = self.decoder(last).repeat(1, self.pred_horizon, 1)
        return preds
