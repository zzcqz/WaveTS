import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.hidden = configs.hidden
        self.lLinear = nn.Linear(self.seq_len // 2, self.pred_len)
        self.uLinear = nn.Linear(self.seq_len // 2, self.pred_len)

        self.gate = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden),           
            nn.ReLU(),
            nn.Linear(self.hidden, self.pred_len * 2),      
        )

    def haar(self, x):
        x1 = x[:, 0::2, :] / 2
        x2 = x[:, 1::2, :] / 2
        x_L = x1 + x2
        x_U = x1 - x2
        return x_L, x_U

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)  # (B, L, C)
        gate_logits = self.gate(x.permute(0, 2, 1))  # (B, C, pred_len * 2)
        gate_logits = gate_logits.view(-1, self.channels, self.pred_len, 2)  # (B, C, pred_len, 2)
        gate_w = F.softmax(gate_logits, dim=-1)  # (B, C, pred_len, 2)
        x_L, x_U = self.haar(x)  # (B, L//2, C)

        x_L = self.lLinear(x_L.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, C)
        x_U = self.uLinear(x_U.permute(0, 2, 1)).permute(0, 2, 1)  # (B, pred_len, C)

        gate_w_L = gate_w[..., 0].permute(0, 2, 1)  # (B, pred_len, C, 1)
        gate_w_U = gate_w[..., 1].permute(0, 2, 1) # (B, pred_len, C, 1)

        xy_weighted = x_L * gate_w_L + x_U * gate_w_U  # (B, pred_len, C, 1)
        xy = xy_weighted.squeeze(-1)  # (B, pred_len, C)

        xy = xy * torch.sqrt(x_var) + x_mean
        return xy
