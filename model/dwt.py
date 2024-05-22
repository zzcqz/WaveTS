import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):


    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.patchlen = torch.tensor(24)
        self.alpha = nn.Parameter(torch.sqrt(torch.FloatTensor([2])), requires_grad=True)
        self.isglu = configs.isGLU
        

        #self.alpha.data.fill_(0.5)
        #self.beta.data.fill_(0.5)
        self.channel = nn.Linear(self.channels,64)
        self.ichannel = nn.Linear(64,self.channels)
        self.lLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.uLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.llLinear = nn.Linear(self.seq_len//4,self.pred_len)
        self.lllLinear = nn.Linear(self.seq_len//8,2*self.pred_len)
        self.lluLinear = nn.Linear(self.seq_len//8,self.pred_len)
        self.luLinear = nn.Linear(self.seq_len//4,self.pred_len)
        self.glu = nn.GLU(dim=1)
    

    def db_wavelet(self,x, low_pass_filter, high_pass_filter):

        x = x.permute(0,2,1)
        batch_size, channels, signal_length = x.shape
       
        x_padded = F.pad(x, (low_pass_filter.numel() - 1, 0), mode='reflect')
       
        x_padded = x_padded.view(batch_size * channels, 1, signal_length + low_pass_filter.numel() - 1)
        
        
        low_passed = F.conv1d(x_padded, low_pass_filter[None, None, :], stride=2)
        high_passed = F.conv1d(x_padded, high_pass_filter[None, None, :], stride=2)
        
       
        low_passed = low_passed.view(batch_size, channels, -1)
        high_passed = high_passed.view(batch_size, channels, -1)
    
        return low_passed.permute(0,2,1), high_passed.permute(0,2,1)

    def db_inverse_wavelet(self,low_coeffs, high_coeffs, low_pass_filter, high_pass_filter):

        low_coeffs = low_coeffs.permute(0,2,1)
        high_coeffs = high_coeffs.permute(0,2,1)
        batch_size, channels, _ = low_coeffs.shape
       
        low_upsampled = F.interpolate(low_coeffs.view(batch_size * channels, 1, -1), scale_factor=2, mode='nearest')
        high_upsampled = F.interpolate(high_coeffs.view(batch_size * channels, 1, -1), scale_factor=2, mode='nearest')
        
        
        low_pass_filter_reversed = torch.flip(low_pass_filter, [0])
        high_pass_filter_reversed = torch.flip(high_pass_filter, [0])
        
       
        low_convolved = F.conv1d(low_upsampled, low_pass_filter_reversed[None, None, :], stride=1)
        high_convolved = F.conv1d(high_upsampled, high_pass_filter_reversed[None, None, :], stride=1)
        
       
        reconstructed = low_convolved + high_convolved
        
        reconstructed = reconstructed[:, :, :self.pred_len]
        
        
        reconstructed = reconstructed.view(batch_size, channels, -1)
        
        return reconstructed.permute(0,2,1)


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
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        

        x_L,x_U = self.haar(x)
        
        #x_L,x_U = self.db_wavelet(x,self.db4_low,self.db4_high)
        
        
        x_L = self.lLinear(x_L.permute(0,2,1)).permute(0,2,1)
        if self.isglu == True:
            x_L = self.glu(x_L)
        
        
        x_U = self.uLinear(x_U.permute(0,2,1)).permute(0,2,1)
        
        y = x_L  + x_U 
        y=(y) * torch.sqrt(x_var) + x_mean
        
        return y[:,-self.pred_len:,:]

