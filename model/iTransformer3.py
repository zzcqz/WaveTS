#在encoder之前加入特征提取
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class SEAttention(nn.Module):
    def __init__(self, channel,reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 池化层，将每一个通道的宽和高都变为 1 (平均值)
        self.fc = nn.Sequential(
            nn.Linear(channel,  reduction, bias=False),
            nn.ReLU(),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MSRAblock(nn.Module):
    def __init__(self):
        super(MSRAblock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=321 , out_channels= 512, kernel_size=3, stride=1,dilation=1,padding=1)
        self.conv2 = nn.Conv1d(in_channels=321 , out_channels= 512, kernel_size=3, stride=1,dilation=2,padding=2)
        self.conv3 = nn.Conv1d(in_channels=321 , out_channels= 512, kernel_size=3, stride=1,dilation=3,padding=3)
        self.conv4 = nn.Conv1d(in_channels=321 , out_channels= 512, kernel_size=3, stride=1,dilation=5,padding=5)
        
        self.conv = nn.Conv1d(in_channels=512,out_channels=321,kernel_size=1,stride=1)
        self.convx = nn.Conv1d(in_channels=321,out_channels=321,kernel_size=1,stride=1)    #给x映射扩展通道数量
        self.relu = nn.ReLU()
        self.se = SEAttention(321,256)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.conv(x2)
        x2 = self.relu(x2)

        x3 = self.conv3(x)
        x3 = self.relu(x3)
        x3 = self.conv(x3)
        x3 = self.relu(x3)

        x4 = self.conv4(x)
        x4 = self.relu(x4)
        x4 = self.conv(x4)
        x4 = self.relu(x4)
        

        h = x1+x2+x3+x4
        h = self.se(h)
        x = self.convx(x)   #scale = (batchsize,321,seq_len)
        out = x + h
        return out

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.msr = MSRAblock()
        self.conv1 = nn.Conv1d(in_channels=321 , out_channels= 321, kernel_size=3, stride=1,dilation=1,padding=1)
        self.conv2 = nn.Conv1d(in_channels=1024 , out_channels= 321, kernel_size=3, stride=1,dilation=1,padding=1)
        self.relu = nn.ReLU()
        self.norm_layer=torch.nn.LayerNorm(configs.d_model)
        self.drop = nn.Dropout(0.1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        
        x = self.enc_embedding(x_enc, None)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.conv1(enc_out)
        enc_out = self.conv1(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out += x
        enc_out = self.norm_layer(enc_out)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #enc_out = self.projector(enc_out).permute(0, 2, 1)
        #enc_out = self.drop(enc_out)        
        #enc_out = self.enc_embedding(enc_out, None)
        #enc_out = self.norm_layer(enc_out)
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, N]
