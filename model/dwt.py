import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.xi = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.deta = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.alpha.data.fill_(0.5)
        self.beta.data.fill_(0.5)

        self.lLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.uLinear = nn.Linear(self.seq_len//2,self.pred_len)

        #self.llLinear = nn.Linear(self.seq_len//4,self.pred_len)
        #self.luLinear = nn.Linear(self.seq_len//4,self.pred_len)
        #self.ulLinear = nn.Linear(self.seq_len//4,self.pred_len)
        #self.uuLinear = nn.Linear(self.seq_len//4,self.pred_len)
    def db5(self,x):
        x = x.permute(0,2,1)
        low_pass = torch.tensor([-0.0033357253, -0.0125807520, 0.0062414902, 0.0775714938, -0.0322448696,
                         -0.2422948871, 0.1384281459, 0.7243085284, 0.6038292698, 0.1601023980], dtype=x.dtype,device=x.device)
        high_pass = torch.tensor([-0.1601023980, 0.6038292698, -0.7243085284, 0.1384281459, 0.2422948871,
                                -0.0322448696, -0.0775714938, 0.0062414902, 0.0125807520, -0.0033357253], dtype=x.dtype,device=x.device)

        low_pass_filter = low_pass.repeat(self.channels, 1).unsqueeze(1)
        high_pass_filter = high_pass.repeat(self.channels, 1).unsqueeze(1)

        low_pass_result = F.conv1d(x, low_pass_filter, padding='same', groups=self.channels)
        high_pass_result = F.conv1d(x, high_pass_filter, padding='same', groups=self.channels)

        low_pass_downsampled = low_pass_result[:, :, ::2]
        high_pass_downsampled = high_pass_result[:, :, ::2]
        return low_pass_downsampled.permute(0,2,1),high_pass_downsampled.permute(0,2,1)

    def haar(self,x):
        x1 = x[:, 0::2, :] / 2
        x2 = x[:, 1::2, :] / 2
        x_L = x1 + x2
        x_U = x1 - x2
        return x_L,x_U
    
    def inverse_haar(self,x_L, x_U):
        x1 = x_L + x_U
        x2 = x_L - x_U
        x = torch.zeros((x1.shape[0], x1.shape[1], x1.shape[2],x1.shape[3]*2)).to(x_L.device)
        x[:, 0::2, :] = x1
        x[:, 1::2, :] = x2
        return x

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)

        x_L,x_U = self.haar(x)
        #x_L,x_U = self.db5(x)
        x_L = self.lLinear(x_L.permute(0,2,1)).permute(0,2,1)
        
        #x_LL = self.llLinear(x_LL.permute(0,2,1)).permute(0,2,1)
        #x_LU = self.luLinear(x_LU.permute(0,2,1)).permute(0,2,1)
        #x_UL = self.ulLinear(x_UL.permute(0,2,1)).permute(0,2,1)
        #x_UU = self.ulLinear(x_UU.permute(0,2,1)).permute(0,2,1)
        #x_U = torch.where(torch.abs(x_U) < 0.1, torch.tensor(0.0), x_U)
        
        x_U = self.uLinear(x_U.permute(0,2,1)).permute(0,2,1)
        
        xy = self.alpha*x_L + self.beta*x_U 
        xy = x_L + x_U

        #xy = self.inverse_haar(x_L,x_U)
        
        
        
        xy=(xy) * torch.sqrt(x_var) + x_mean
        
        return xy

