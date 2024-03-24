import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import iTransformer
from sklearn.cluster import KMeans

class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        self.lLinear = nn.Linear(self.seq_len//2,self.pred_len)
        self.g1Linear = nn.Linear(self.seq_len//2,self.pred_len)
        self.g2Linear = nn.Linear(self.seq_len//2,self.pred_len)
        self.g3Linear = nn.Linear(self.seq_len//2,self.pred_len)
        self.uLinear = nn.Linear(self.seq_len//2,self.pred_len)
    
    """def kgroup(self,data, n_clusters, max_iters=100,random_seed=1):
    # 随机初始化簇中心
        y = data
        B,L,N = y.shape
        y = y.reshape(B*L,N)
        y = y.permute(1,0)
        y = torch.adaptive_avg_pool1d(y,1)
        torch.manual_seed(random_seed)
        centroids = y[torch.randperm(len(data))[:n_clusters]]
        for _ in range(max_iters):
            # 计算每个样本到各个簇中心的距离
            distances = torch.norm(y.unsqueeze(1) - centroids.unsqueeze(0), dim=2)

            # 将每个样本分配到最近的簇
            labels = torch.argmin(distances, dim=1)

            # 更新簇中心为每个簇内样本的均值
            new_centroids = torch.stack([y[labels == i].mean(dim=0) for i in range(n_clusters)])

            # 如果簇中心变化很小，停止迭代
            if torch.norm(new_centroids - centroids) < 1e-4:
                break

            centroids = new_centroids

        # 获取每个簇中的数据索引
        clusters_indices = {i: torch.where(labels == i)[0] for i in range(n_clusters)}
        index_1,index_2,index_3 = clusters_indices[0],clusters_indices[1],clusters_indices[2]
        
        g1,g2,g3= data[:,:,index_1],data[:,:,index_2],data[:,:,index_3]
        return g1,g2,g3,index_1,index_2,index_3"""
        

    def haar(self,x):
        x1 = x[:, 0::2, :] / 2
        x2 = x[:, 1::2, :] / 2
        x_L = x1 + x2
        x_U = x1 - x2
        return x_L,x_U

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)
        
        

        x_L,x_U = self.haar(x)
        z = x_L
        B,L,N = z.shape
        z = z.reshape(B*L,N)
        z = z.permute(1,0)
        z = z.cpu().numpy()
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(z)
        
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        clusters_indices = {i: np.where(labels == i)[0] for i in range(3)}
        i1,i2,i3 = clusters_indices[0],clusters_indices[1],clusters_indices[2]
        i1,i2,i3 = torch.from_numpy(i1),torch.from_numpy(i2),torch.from_numpy(i3)
        g1,g2,g3 = x_L[:,:,i1],x_L[:,:,i2],x_L[:,:,i3]
        #g1,g2,g3,i1,i2,i3 = self.kgroup(x_L.numpy(),3)
        g1 = self.g1Linear(g1.permute(0,2,1)).permute(0,2,1)
        g2 = self.g2Linear(g2.permute(0,2,1)).permute(0,2,1)
        g3 = self.g3Linear(g3.permute(0,2,1)).permute(0,2,1)
        g = torch.zeros(x.size(0),self.pred_len,x.size(2),device=x_L.device)
        g[:,:,i1] = g1
        g[:,:,i2] = g2
        g[:,:,i3] = g3
        x_U = self.uLinear(x_U.permute(0,2,1)).permute(0,2,1)
        
        
        xy = x_U + g
        

        
        xy=(xy) * torch.sqrt(x_var) + x_mean
        
        return xy





