import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu2 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        resx1 = self.conv1(self.relu1(self.bn1(x)))
        resx2 = self.conv2(self.relu2(self.bn2(resx1)))
        
        maxpoolx = self.maxpool(x)
        return resx2 + maxpoolx
    
    
class Pre_activation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        
    def forward(self, x):
        prex1 = self.relu1(self.bn1(self.conv1(x)))
        
        maxpoolx = self.maxpool1(prex1)
        prex2 = self.conv3(self.relu2(self.bn2(self.conv2(prex1)))) 
        
        return maxpoolx + prex2
    
    
class MultiScale_encoder(nn.Module):
    def __init__(self, in_channels, scale_num):
        super().__init__()
        self.scale_num = scale_num
        
        self.pre_activate = Pre_activation(in_channels)
        self.resblocks = nn.ModuleList()
        for i in range(self.scale_num):
            self.resblocks.append(Residual(in_channels))
        self.gap = nn.AdaptiveMaxPool1d(1)
        
        self.bnfi = nn.BatchNorm1d(in_channels)
        self.relufi = nn.LeakyReLU()
        
    def forward(self, x):
        resx = self.pre_activate(x)
        res_embs = []
    
        for i in range(self.scale_num):
            resx = self.resblocks[i](resx)
            res_embs.append(self.gap(resx).squeeze(2))
        resx = self.relufi(self.bnfi(resx))
        res_embs.append(self.gap(resx).squeeze(2))
        return torch.cat(res_embs, dim=1)
    

class ReversalNet(nn.Module):
    def __init__(self, in_channels, scale_num, lead_num, attention_len, class_num):
        super().__init__()
        
        self.lead_num = lead_num
        self.leads = nn.ModuleList()
        for i in range(lead_num):
            self.leads.append(MultiScale_encoder(in_channels, scale_num))
        
        self.attn_activate = nn.Tanh()
        self.attn_linear = nn.Linear(attention_len, 1, bias=False)
        
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(attention_len, class_num)

        
    def forward(self, x):
        x = x.transpose(1,2)
        assert x.size(1) == self.lead_num
        
        leady = []
        for i in range(self.lead_num):
            leady.append(self.leads[i](x[:,i,:].unsqueeze(1)).unsqueeze(1))
        
        lead_output = torch.cat(leady, dim=1)
        
        atten_score = self.attn_linear(self.attn_activate(lead_output)).squeeze(2)
        atten_score = F.softmax(atten_score, dim=1).unsqueeze(1)
        
        lead_output = self.attn_activate(torch.bmm(atten_score, lead_output)).squeeze(1)
        
        dropout_y = self.dropout(lead_output)
        
        
        return self.linear(dropout_y) 
    
multiencoder = ReversalNet(1, 5, 2, 6, 3)
x = torch.randn(10,258,2)
# # print(x.size(2))
x = x.cuda()
multiencoder = multiencoder.cuda()
print(multiencoder(x).shape)
# print(x[:,0,:].shape)
# a = torch.Tensor([[1,2],[3,4]])
# b = torch.Tensor([0.5, 1])

# print(a * b)
