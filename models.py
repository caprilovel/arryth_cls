import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import * 

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
    output_shape:
    
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
    
class SMATE_Decoder(nn.Module):
    def __init__(self, num_layers, input_dimension, input_length, hidden_size, rnn="GRU"):
        '''
        input_shape: batch_size X Length X Dimension
        ouput_shape: batch_size x Dimension x Length
        
        input_dimension: int, the channels/dimension of the input 
        input_length: int, the length of the input 
        num_layers: int, the num of blocks 
        d_prime: List, the hidden size of Spatial block hidden layer, the length of this list should be same as the num_layers
        kernels: List, the kernel sizes of each SMATE_Encoder block(both the Convd1d/AvgPooling), the length of this list should be same as the num_layers
        out_channels: List, the channels/dimensions of the output of each block,  the length of this list should be same as the num_layers
        rnn: Str, optional, default as GRU, the type of RNN in SMATE_Encoder, one of GRU, LSTM
        '''
        super().__init__()
        # self.uppool = nn.Upsample()
        if rnn is "GRU":
            self.rnn_sq = nn.GRU(input_size=input_dimension, batch_first=True, hidden_size=hidden_size, num_layers=num_layers)
        elif rnn is "LSTM":
            self.rnn_sq = nn.LSTM(input_size=input_dimension, batch_first=True, hidden_size=hidden_size, num_layers=num_layers) 
    
    def forward(self, x):
        # y = self.uppool(x)
        y,(h,c) = self.rnn_sq(x)
        return y

    
class ClusterML(nn.Module):
    def __init__(self, nclass, input_size, embedding_size):
        super().__init__()
        self.centroid_List = nn.Parameter(torch.zeros([nclass, embedding_size]))
        self.flag = [0 for i in range(nclass)]
        self.nclass = nclass
        self.linear = nn.Linear(input_size ,embedding_size)
    
    def forward(self, x, y):
        output = self.linear(x)
        for i in range(self.nclass):
            idx = (y == i)
            batch_repr = output[idx].mean(0)
            if self.flag[i]:
                self.centroid_List[i] += -0.1 * self.centroid_List[i] + 0.1 * batch_repr
            else:
                self.centroid_List[i] += batch_repr
            proto_dist = euclidean_dist(self.centroid_List, self.centroid_List)
            proto_dist = torch.exp(-0.5 * proto_dist)
            dist = euclidean_dist(output, self.centroid_List)
        return F.sigmoid(torch.exp(-0.5 * dist)), proto_dist

def output_conv2d_size(in_size, kernel_size, stride=1, padding=0, dilation=1):
    output = []
    for i in range(2):
        output.append((in_size[i]-kernel_size[i]-(dilation-1)*(kernel_size[i]-1)+2*padding)//stride +1)
    return output


#-----------------------------------------------------------#

#   TDConvdBlock

#-----------------------------------------------------------#


class TDConvdBlock(nn.Module):
    def __init__(self, inchannels, outchannels, input_shape, stride, kernel_size=[2,2], use_FAM=True):
        super().__init__()
        self.out_channels = outchannels
        self.outputsize = output_conv2d_size(input_shape, kernel_size,stride=stride)
        self.convd = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, stride=stride, kernel_size=kernel_size)
        
        self.feature_att_fmap = nn.Linear(input_shape[0]*input_shape[1], self.outputsize[0]*self.outputsize[1])
        self.feature_att_kernel = nn.Linear(kernel_size[0]*kernel_size[1]*inchannels, self.outputsize[0]*self.outputsize[1])
        
    
    def forward(self, x):
        '''
        x size: batchsize x inchannel x length x width 
        '''
        y = self.convd(x)
        kernel_graphs = []
        featuremap_graphs = []
        N = x.size(0)
        
        for i in range(self.out_channels):
            #kernel_graph size: inchannels x kernel_length x kernel_width 
            kernel_graph = self.convd.state_dict()['weight'][i]
            kernel_graphs.append(torch.unsqueeze(torch.flatten(kernel_graph),dim=0))
            #featuremap_graph size: batchsize x inchannels x in_featuremap_length x in_featuremap_width 
            featuremap_graph = x[:,1,:]
            featuremap_graphs.append(torch.unsqueeze(featuremap_graph.view(N, -1),dim=1))
        # kernel_graphs cat size: out_channels x kernelshape(eq. length * width)
        # attn_kernel size: batchsize x out_channels x kernelshape(eq. length * width)
        attn_kernel =  (torch.unsqueeze(torch.cat(kernel_graphs, dim=0),dim=0)).expand(N,-1,-1)
        
        # featuremap_graphs cat size: batch_size x outchannels x featuremapshape(eq. length * width)
        attn_featuremap = torch.cat(featuremap_graphs, dim=1)
        # attn_score size:   
        
        a1 = self.feature_att_fmap(attn_featuremap)
        a2 = self.feature_att_kernel(attn_kernel) 
        print(a1.shape, a2.shape) 
        attn_score = torch.tanh(a1+a2)
        attn_weight = F.softmax(attn_score, dim=2)
        attn_weight = torch.reshape(attn_weight, [self.out_channels, self.outputsize[0], -1])
        y = y*attn_weight
        return y

class LocalAttentionModule(nn.Module):
    def __init__(self, ecg_seg_length, height, cw):
        super().__init__()
        self.ecg_seg_linear = nn.Linear(ecg_seg_length, height)
        self.featuremap = nn.Linear(cw, 1)
        self.height = height
    def forward(self, x, ecg_seg):
        '''
        input_size : batch_size x channels x width x height
        ecg_seg : batch_size x ecg_length
        '''
        N = x.size(0)
        # y size: batch_size x height x (width*channels)
        y = torch.transpose(x, 1, 3)
        y = torch.reshape(y, [N, self.height, -1])

        out = torch.squeeze(self.featuremap(y), -1)
        ecg_out = self.ecg_seg_linear(ecg_seg)

        attn = torch.tanh(out+ecg_out)
        attn_score = F.softmax(attn, dim=1)
        attn_score = torch.unsqueeze(torch.unsqueeze(attn_score, 1), 1)
        return x * attn_score

class TestConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 10, 3)
        self.linear = nn.Linear(9,1)
        self.linear2 = nn.Linear(180,2)
    
    def forward(self,x):
        N = x.size(0)
        y = self.conv(x)
        graphs = []
        for i in range(10):
            # weight size: out_channels x in_channels x kernel_size
            # kernel_graph size: in_channels x kernel_size
            kernel_graph = self.conv.state_dict()['weight'][i]
            # graph size: total_size
            graph = torch.flatten(kernel_graph)
            graphs.append(torch.unsqueeze(graph, dim=0))
            # torch.cat([graphs, ])
         
        graphs = torch.cat(graphs, dim=0)    
        att = self.linear(graphs)
        att_weihght = F.softmax(att,dim=1)
        att_weihght = att_weihght.expand_as(y)
        z= (y * att_weihght).view(N, -1)

        return  self.linear2(z)  


#-----------------------------------------------------------#

#   LSTM实现，尚未增加bidirectional条件

#   nn.Paramter与Tensor的选取：Parameter自动加入模型的可训练参数中，即在优化器中，可以通过model.parameters()直接进行迭代优化。

#-----------------------------------------------------------#


def three(num_inputs, num_hiddens):
    return (
        nn.Parameter(torch.randn(num_inputs, num_hiddens), requires_grad=True),
        nn.Parameter(torch.randn(num_hiddens, num_hiddens), requires_grad=True),
        nn.Parameter(torch.zeros(num_hiddens), requires_grad=True)
    )
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.W_xi, self.W_hi, self.b_i = three(input_size, hidden_size)  # 输入门参数
        self.W_xf, self.W_hf, self.b_f = three(input_size, hidden_size)  # 遗忘门参数
        self.W_xo, self.W_ho, self.b_o = three(input_size, hidden_size)  # 输出门参数
        self.W_xc, self.W_hc, self.b_c = three(input_size, hidden_size)  # 候选记忆单元参数
        self.W_hq = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        self.b_q = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.hidden_size = hidden_size
    def forward(self, inputs):
        N = inputs.size(1)
        (H, C) = (torch.zeros(N, self.hidden_size), torch.zeros(N, self.hidden_size))
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ self.W_xi) + (H @ self.W_hi) + self.b_i)
            F = torch.sigmoid((X @ self.W_xf) + (H @ self.W_hf) + self.b_f)
            O = torch.sigmoid((X @ self.W_xo) + (H @ self.W_ho) + self.b_o)
            C_tilda = torch.tanh((X @ self.W_xc) + (H @ self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ self.W_hq) + self.b_q
            outputs.append(torch.unsqueeze(Y, dim=0))
            
        return torch.cat(outputs, dim=0), (H, C)


#-----------------------------------------------------------#

#   MultiScaleConv1dBlock，多尺度一维卷积，采用多个不同卷积核大小的一维卷积层进行卷积，最后拼接

#-----------------------------------------------------------#


class MultiScaleConv1dBlock(nn.Module):
    '''
    docstring
    
    '''
    def __init__(self, in_channels, out_channels, input_scales = [3,5,7,11,13]):
        '''Multiscale conv1d sequential convolution module

        Multiscale conv1d sequential convolution module
        多尺度一元时序卷积模块
        
        Args:
          in_channels: Integer, the numbers of the input channels 
          out_channels: Integer, the numbers of output channels in each convolutional layer 
          input_scales: List of Integers, the scales of the convolutional kernels 
        '''
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


#-----------------------------------------------------------#

#   SK-block, 来源于论文"Selective Kernel Networks"，采用多个不同kernel大小的一维卷积层卷积后，采用注意力机制加权后将不同的feature map相加

#-----------------------------------------------------------#



class SK_MultiScaleConv1dBlock(nn.Module):
    '''
    this module is inspired by the SK Net and Multi-scale CNN
    
    rate: the size of the compressed vector is the higher value between rate * out_channels and min_length              
    
    '''
    def __init__(self, in_channels, rate, min_length=32, input_scales=[3,5,7,11,13]):
        super().__init__()
        self.scales = len(input_scales)
        self.conv_array = []
        self.linear_array = []
        z_channels = max(min_length, int(rate * in_channels))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(z_channels)
        self.linear_sum = nn.Linear(in_channels, z_channels)
        self.linear_split = nn.Linear(z_channels, self.scales * in_channels)        
        for i in input_scales:
            self.conv_array.append(nn.Conv1d(in_channels, in_channels,kernel_size=i, padding=(i-1)//2))
            

    def forward(self, x):
        N = x.size(0)
        y = []
        for conv in self.conv_array:
            y.append(torch.unsqueeze(conv(x), dim=1))
        y_cat = torch.cat(y, dim=1)
        y_sum = torch.sum(y_cat, dim=1)
        gap = torch.squeeze(self.gap(y_sum), dim=2)
        output = self.bn(self.linear_sum(gap))
        split = self.linear_split(output)
        split = torch.reshape(split, (N, self.scales, -1))
        split = F.softmax(split, dim=1)
        split = torch.unsqueeze(split, dim=3)
        return torch.sum(split * y_cat, dim=1) + x


#-----------------------------------------------------------#

#   ResNetModule, 简单的用于构建backbone架构

#-----------------------------------------------------------#


class Resblock(nn.Module):
    def __init__(self, in_channel, kernel_size, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channel, in_channel,kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size-1)//2)


    def forward(self, x):
        y = F.leaky_relu(self.conv1(x))
        y = F.leaky_relu(self.conv2(y))
        return y+x



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3),
            Resblock(32,3),
            Resblock(32,3),
            Resblock(32,3),
            Resblock(32,3),
            nn.Conv1d(32, 5, kernel_size=3),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = torch.transpose(x , 1, 2)
        return self.net(x).squeeze(2)


#-----------------------------------------------------------#

#   TemporalAttentionModule, 复现论文"A New Attention Mechanism to Classify Multivariate Time Series"模型

#-----------------------------------------------------------#


class TemporalAttention(nn.Module):
    def __init__(self, length, in_channels, qk_channels, v_channels):
        super().__init__()
        self.length = length
        self.key_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.query_conv = nn.Conv1d(in_channels, qk_channels, 1)
        self.value_conv = nn.Conv1d(in_channels, v_channels, 1)

        self.out_conv = nn.Conv1d(v_channels, in_channels, 1)



    def forward(self, x, mask=False):
        #  data shape
        #  key: batch x length x qk_channels
        #  query: batch x qk_channels x length
        #  value: batch x length x v_channels

        key = torch.transpose(self.key_conv(x), 1, 2)
        query = self.query_conv(x)
        value = torch.transpose(self.value_conv(x), 1, 2)
        # attn_map: batch x key x query
        attn_map =  torch.bmm(key, query)
        # 构建上三角矩阵
        attn_mask = torch.triu(torch.ones(self.length, self.length), diagonal=0)
        if mask:
            attn_map = attn_map * attn_mask
        attn_score = F.softmax(attn_map, dim=2)
        attn_value = torch.bmm(attn_score, value) 
        attn_value = torch.transpose(attn_value, 1, 2)
        return self.out_conv(attn_value)
