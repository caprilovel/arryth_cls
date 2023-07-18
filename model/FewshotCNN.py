import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.data_loader import *


class ConvMaxpool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvMaxpool, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)
        
    def forward(self, x):
        return self.maxpool(self.conv(x))

def l2dist(a, b):
    return ((a-b)**2).sum()/2

class FewshotCNN(nn.Module):
    def __init__(self, in_channels):
        super(FewshotCNN, self).__init__()
        self.seq = nn.Sequential()
        channel_mul = [ 2**i for i in range(4)]
        print(channel_mul)
        for i in range(4):
            print(channel_mul[i] * in_channels, 2 * channel_mul[i] * in_channels)
            self.seq.add_module('convmaxpool{}'.format(i), ConvMaxpool(channel_mul[i] * in_channels, 2 * channel_mul[i] * in_channels))
        
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        assert x.size(0) == 2
        outputx = self.seq(x)
        x = outputx.view(2,-1)
        x1 = x[0,:]
        x2 = x[1,:]
        
        
        dist = l2dist(x1, x2)
        return torch.sigmoid(dist)
    
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
DS_total = DS1 + DS2
x = torch.randn(2,2,256)
path  = "./data/mit-bih-arrhythmia-database-1.0.0"
fsdl = FewshotDataloader(DS_total, path)