import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class HighFreqExpert(nn.Module): 
    def __init__(self, input_size, output_size):
        super(HighFreqExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.num_low_experts = configs.num_experts  
        self.total_experts = self.num_low_experts + 1  

        
        self.low_experts = nn.ModuleList([
            Expert(self.seq_len//2, self.pred_len) for _ in range(self.num_low_experts)
        ])
       
        self.high_expert = HighFreqExpert(self.seq_len//2, self.pred_len)

        
        self.gate = nn.Sequential(
            nn.Linear(self.seq_len, 64),  
            nn.ReLU(),
            nn.Linear(64, self.pred_len * self.total_experts),
        )

    def haar(self, x):
        x1 = x[:, 0::2, :] / 2
        x2 = x[:, 1::2, :] / 2
        x_L = x1 + x2
        x_U = x1 - x2
        return x_L, x_U

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # RIN 归一化
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / torch.sqrt(x_var)
        
        B, L, C = x_norm.shape
        x_L, x_U = self.haar(x_norm)           # (B, L//2, C)
        x_L = x_L.permute(0, 2, 1)             # (B, C, L//2)
        x_U = x_U.permute(0, 2, 1)             # (B, C, L//2)

      
        x_gate_input = x_norm.permute(0, 2, 1) # (B, C, L)
        gating_logits = self.gate(x_gate_input) # (B, C, pred_len * total_experts)
        gating_logits = gating_logits.view(B, C, self.pred_len, self.total_experts) # (B, C, S, E)
        gating_weights = F.softmax(gating_logits, dim=-1) # (B, C, S, E)

       
        low_outputs = [expert(x_L) for expert in self.low_experts]  # list of (B, C, S)
        low_outputs = torch.stack(low_outputs, dim=-1)              # (B, C, S, num_low)

       
        high_output = self.high_expert(x_U).unsqueeze(-1)           # (B, C, S, 1)
        all_outputs = torch.cat([low_outputs, high_output], dim=-1) # (B, C, S, total_experts)

        output = torch.einsum('b c s e, b c s e -> b c s', all_outputs, gating_weights) # (B, C, S)

        output = output.permute(0, 2, 1)  # (B, S, C)
        final_output = output * torch.sqrt(x_var) + x_mean

        # low_weights = gating_weights[..., :self.num_low_experts]  # (B, C, S, num_low)
        # high_weight = gating_weights[..., -1:]                    # (B, C, S, 1)

        # return final_output, output, high_output.squeeze(-1), gating_weights
        return final_output
