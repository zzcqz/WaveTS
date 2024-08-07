import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
#WaveTS :Wavelet MLP Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.patchlen = torch.tensor(24)
        self.alpha = nn.Parameter(torch.sqrt(torch.FloatTensor([2])), requires_grad=True)
        self.isglu = configs.isGLU
        self.lLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.gluLinear = nn.Linear(self.seq_len//2,self.pred_len*2)
        self.uLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.glu = nn.GLU(dim=1)


    def haar(self,x):
        x1 = x[:, 0::2, :] / self.alpha
        x2 = x[:, 1::2, :] / self.alpha
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
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        

        x_L,x_U = self.haar(x)  
        if self.isglu == True:
            x_L = self.glu_Linear(x_L.permute(0,2,1)).permute(0,2,1)
            x_L = self.glu(x_L)   
        else x_L = self.lLinear(x_L.permute(0,2,1)).permute(0,2,1)
        x_U = self.uLinear(x_U.permute(0,2,1)).permute(0,2,1)
        
        y = x_L + x_U 
        y=(y) * torch.sqrt(x_var) + x_mean
        
        return y[:,-self.pred_len:,:]

