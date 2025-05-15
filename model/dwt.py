import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Expert(nn.Module):
    def __init__(self, input_size, output_size):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.num_experts = configs.num_experts
        self.lLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.uLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.experts = nn.ModuleList([Expert(self.seq_len//2, self.pred_len) for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(self.seq_len//2, self.pred_len//2),
            nn.ReLU(),
            nn.Linear(self.pred_len//2, self.num_experts),
            nn.Softmax(dim=2)
        )
    
    def haar(self,x):
        x1 = x[:, 0::2, :] / 2
        x2 = x[:, 1::2, :] / 2
        x_L = x1 + x2
        x_U = x1 - x2
        return x_L,x_U
    
    def inverse_haar(self,x_L, x_U):
        x1 = x_L + x_U
        x2 = x_L - x_U
        x = torch.zeros((x1.shape[0], x1.shape[1]*2, x1.shape[2])).to(x_L.device)
        x[:, 0::2, :] = x1
        x[:, 1::2, :] = x2
        return x

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        B, L, C = x.shape
        x_L,x_U = self.haar(x)
        x_L = x_L.permute(0,2,1)#(B,C, L//2)
        gating_weights = self.gate(x_L)#(B,C, num_experts)
        expert_outputs = [expert(x_L) for expert in self.experts]#(B,C,num_experts*S)
        expert_outputs = torch.stack(expert_outputs, dim=2)#(B,C, num_experts, S)
        expanded_gating_weights = gating_weights.unsqueeze(3).expand(-1, -1, -1, self.pred_len)#(B,C, num_experts,S)
        output = torch.einsum('bcns,bcns->bcs', expert_outputs, expanded_gating_weights)#(B,C,S)
        
        x_L = output.permute(0,2,1)

 
        x_U = self.uLinear(x_U.permute(0,2,1)).permute(0,2,1)
        xy = x_L + x_U

        xy=(xy) * torch.sqrt(x_var) + x_mean
        
        return xy[:,-self.pred_len:,:],output,x_U

