import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Spatial_Block(nn.Module):
    '''
    inputsize: Batchsize x Dimension x Length
    
    input_dimension: 
    pool_kernel_size: pooling layer size 
    d_prime: fc convert size
    
    outputsize: Batchsize x Dimension x Length
    '''
    def __init__(self, input_dimension, pool_kernel_size, d_prime):
        super().__init__()
        self.pooling = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=1, padding=(pool_kernel_size-1)//2)
        self.fc1 = nn.Linear(in_features=input_dimension, out_features=d_prime)
        self.fc2 = nn.Linear(in_features=d_prime, out_features=input_dimension)
    
    def forward(self, x):
        y = self.pooling(x)
        y = y.transpose(1,2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.transpose(1,2)
        return x * y
        
class SMATE_Encoder(nn.Module):
    '''
    input_shape: batch_size x Dimension x Length
    
    input_dimension: int, the channels/dimension of the input 
    input_length: int, the length of the input 
    num_layers: int, the num of blocks 
    d_prime: List, the hidden size of Spatial block hidden layer, the length of this list should be same as the num_layers
    kernels: List, the kernel sizes of each SMATE_Encoder block(both the Convd1d/AvgPooling), the length of this list should be same as the num_layers
    out_channels: List, the channels/dimensions of the output of each block,  the length of this list should be same as the num_layers
    rnn: Str, optional, default as GRU, the type of RNN in SMATE_Encoder, one of GRU, LSTM
    
    '''
    def __init__(self, input_dimension, input_length, d_prime, kernels, out_channels, final_pool, num_layers=3, rnn="GRU"):
        super().__init__()
        if rnn is "GRU":
            self.rnn_sq = nn.GRU(input_size=input_dimension, batch_first=True, hidden_size=out_channels[-1], num_layers=num_layers)
        elif rnn is "LSTM":
            self.rnn_sq = nn.LSTM(input_size=input_dimension, batch_first=True, hidden_size=out_channels[-1], num_layers=num_layers)
        out_channels.insert(0, input_dimension)
        self.smate = nn.Sequential()
        for i in range(num_layers):
            self.smate.add_module("smate_block"+str(i), Spatial_Block(out_channels[i], pool_kernel_size=kernels[i], d_prime=d_prime))
            self.smate.add_module("conv1d"+str(i), nn.Conv1d(in_channels=out_channels[i], out_channels=out_channels[i+1], kernel_size=kernels[i], padding=(kernels[i]-1)//2))
            self.smate.add_module("batch_norm"+str(i), nn.BatchNorm1d(out_channels[i+1]))
            self.smate.add_module("relu"+str(i), nn.ReLU())
        self.t_avgpool = nn.AvgPool1d(kernel_size=final_pool, padding=(final_pool-1)//2)
        self.s_avgpool = nn.AvgPool1d(kernel_size=final_pool, padding=(final_pool-1)//2)
        self.mlp = nn.Sequential()
        self.mlp.add_module("fc1", nn.Linear(2*out_channels[-1], out_features=500))
        self.mlp.add_module("relu", nn.ReLU())
        self.mlp.add_module("fc2", nn.Linear(500, 300))
        self.mlp.add_module("relu", nn.ReLU())
    def forward(self, x):
        x_transpose = x.transpose(1,2)
        y_rnn,_ = self.rnn_sq(x_transpose)
        y_s = self.smate(x)
        y_s = y_s.transpose(1,2)
        y = torch.cat([y_s, y_rnn], dim=2)
        y = self.mlp(y)
        return y
        
