import torch.nn as nn
import torch.nn.functional as F
import torch
class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, input_scales = [3,5,7,11,13]):
        super().__init__()
        self.conv_array = []
        for i in input_scales:
            self.conv_array.append(nn.Conv1d(in_channels, out_channels,kernel_size=i, padding=(i-1)//2))
    def forward(self, x):
        y = []
        for conv in self.conv_array:
            y.append(conv(x))
        y_sum = torch.cat(y, dim=1)
        return y_sum    


class LinearAttn(nn.Module):
    '''
    
    '''
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.batch_first = batch_first
    def forward(self, x):
        # batch first:x size: BxLxD (Batch, Length,  Dimension)
        # batch first False: LxBxD
        if self.batch_first:
            x_trans = x
        else:
            x_trans = x.permute([1,0,2])        
        y = self.linear(x_trans)
        y = F.relu(y)
        y = self.avgpool(y)
        y = y.squeeze(2)
        att_score = F.softmax(y, dim=1)
        att_score = att_score.unsqueeze(2)
        att_score = att_score.expand_as(x_trans)
        att_score = att_score.permute([0,2,1])
        if not self.batch_first:
            att_score = att_score.permute([1,0,2])
        return att_score
        
class LSTMAtt(nn.Module):
    '''
    
    '''
    def __init__(self, input_size, hidden_size, num_layers = 1, bias=True, bidirectional=False, batch_first=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, bidirectional=bidirectional)
        #
        self.attn_module = LinearAttn(hidden_size, hidden_size, batch_first=batch_first)
        
    def forward(self, x):
        y,(h_0,c_0) = self.lstm(x)
        attn_score = self.attn_module(y)
        
        return y*attn_score 

class Spatial_Block(nn.Module):
    '''
    inputsize: Batchsize x Dimension x Length
    
    input_dimension: 
    pool_kernel_size: pooling layer size 
    d_prime: fc convert size
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