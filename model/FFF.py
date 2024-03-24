import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import iTransformer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.d_model = configs.d_model

        self.dominance_freq=configs.cut_freq # 720/24
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len
        self.iTransformer = iTransformer.Model(configs)
        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq-1, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq-1, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        self.linear = nn.Linear(self.seq_len,self.pred_len)
        
        #configs.pred_len=configs.seq_len+configs.pred_len
        #self.Dlinear=DLinear.Model(configs)
        #configs.pred_len=self.pred_len


    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        up_specx = low_specx[:,self.dominance_freq:]
        up_specxy = torch.zeros([low_specx.size(0),int((self.seq_len)/2+1),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
        up_specxy[:,-up_specx.size(1):,:] = up_specx
        up_xy=torch.fft.irfft(up_specxy, dim=1)
        #up_xy = self.iTransformer(up_xy,None,None,None)
        up_xy = self.linear(up_xy.permute(0,2,1)).permute(0,2,1)

        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,1:self.dominance_freq,:] # LPF  B dominance_freq-1 N
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,1:low_specxy_.size(1)+1,:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # compemsate the length change
    
        low_xy = low_xy[:,-self.pred_len:,:] + up_xy
        # dom_x=x-low_x
        #low_xy = self.projection(low_xy.permute(0,2,1)).permute(0,2,1)
        #low_xy=self.Dlinear(low_xy)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy

