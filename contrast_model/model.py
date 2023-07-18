import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Any, List, Optional, Tuple
from torch import Tensor
from collections import OrderedDict

class SamePaddingConv1d(nn.Module):
    ''' Same padding conv1d for pytorch. Only effects when dilation == 1
    

    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, stride=1):
        super(SamePaddingConv1d, self).__init__()
        low_pad = dilation * (kernel_size - 1) + 2 - 2 * stride
        high_pad = dilation * (kernel_size - 1) + 1 - stride
        pad  = high_pad
        if pad^1 == pad-1 :
            self.l_pad = pad // 2 + 1 
            self.r_pad = pad // 2 
        else:
            self.l_pad = pad // 2 
            self.r_pad = pad // 2
        self.Conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups) 
        
        self.low_pad = low_pad
        self.high_pad = high_pad
    
    def forward(self, x):
        return self.Conv1d(F.pad(x, (self.l_pad, self.r_pad)))
    
    
class DilatedConvNet(nn.Module):
    def __init__(self, input_size, kernel_size, groups=1, layers=5, radix=2):
        super(DilatedConvNet, self).__init__()
        self.Net = nn.Sequential()
        for layer in range(layers):
            self.Net.add_module(f"conv{layer+1}", 
                                SamePaddingConv1d(
                                    input_size, input_size, kernel_size,
                                    2**layer, groups=1
                                ))
            self.Net.add_module(f"bn{layer+1}", 
                                nn.BatchNorm1d(input_size)
                               )
            self.Net.add_module(f"actv{layer+1}", 
                                nn.ReLU()
                               )
    def forward(self, x):
        return self.Net(x)
    
class Bottleneck1d(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        '''
        
        Bottleneck is consisted of three convolutional layers. 
        '''
        super().__init__()
        self.Net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
        )
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.actv = nn.ReLU()
        
    def forward(self, x):
        y = self.Net(x) + x
        y = self.actv(self.bn(y))
        return y
    
class DownSampleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2) -> None:
        super().__init__()
        self.downsampleconv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) 
        self.bn = nn.BatchNorm1d(out_channels)
        self.actv = nn.ReLU()
    
    def forward(self, x):
        x = self.downsampleconv(x)
        x = self.bn(x)
        x = self.actv(x)
        return x
        

class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_resdual, ) -> None:
        super().__init__()    
        blk = []
        for i in range(num_resdual):
            blk.append(Bottleneck1d(in_channels, hidden_channels))
        self.block = nn.Sequential(*blk)
            
    def forward(self, x):
        return self.block(x)
    
    
    
    
class ResNet19(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        
        channels_multip = [in_channels * 2 ** i for i in range(7)]
        
        self.resblock1 = ResBlock(in_channels=channels_multip[0], hidden_channels= int(1.5 * channels_multip[0]), num_resdual=2)
        self.downsample1 = DownSampleConv1d(channels_multip[0], channels_multip[1])
        
        self.resblock2 = ResBlock(in_channels=channels_multip[1], hidden_channels= int(1.5 * channels_multip[1]), num_resdual=2)
        self.downsample2 = DownSampleConv1d(channels_multip[1], channels_multip[2])
        
        self.resblock3 = ResBlock(in_channels=channels_multip[2], hidden_channels= int(1.5 * channels_multip[2]), num_resdual=4)
        self.downsample3 = DownSampleConv1d(channels_multip[2], channels_multip[3])
        
        self.resblock4 = ResBlock(in_channels=channels_multip[3], hidden_channels= int(1.5 * channels_multip[0]), num_resdual=4)
        self.downsample4 = DownSampleConv1d(channels_multip[3], channels_multip[4])
        
        self.resblock5 = ResBlock(in_channels=channels_multip[4], hidden_channels= int(1.5 * channels_multip[0]), num_resdual=4)
        self.downsample5 = DownSampleConv1d(channels_multip[4], channels_multip[5])
        
    def forward(self, x):
        x = self.resblock1(x)
        x = self.downsample1(x)
        
        x = self.resblock2(x)
        x = self.downsample2(x)
        
        x = self.resblock3(x)
        x = self.downsample3(x)
        
        x = self.resblock4(x)
        x = self.downsample4(x)
        
        x = self.resblock5(x)
        x = self.downsample5(x)
        
        return x 






class LengthNLM_Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, hidden_rate=0.6):
        
        super(LengthNLM_Residual, self).__init__()
        self.conv1 = SamePaddingConv1d(in_channels, in_channels, kernel_size,
                 stride=1, dilation=1)
        self.gn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = SamePaddingConv1d(in_channels, in_channels, kernel_size,
                 stride=1, dilation=1,)
        self.gn2 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        
        # SE attention 
        hidden_size = int(hidden_rate * in_channels)
        self.squeeze =nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, in_channels),
            nn.Sigmoid(),
        )
        
        
        
        
        
        
        

    def forward(self, x):
        return

    
class Length_NLM(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Length_NLM, self).__init__()
        
    def forward(self, x):
        pass
        
        
        

    
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False 
        ):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        
        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)
        
        self.memory_efficient = memory_efficient
        
    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output
    
    def any_requires_grad(self, input: List[Tensor]) -> Tensor:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False
    

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)
        
        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method
    def forward(self, input: List[Tensor]) -> Tensor:
        pass
    
    @torch.jit._overload_method
    def forward(self, input: Tensor) -> Tensor:
        pass
    
    
    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        
        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
            
        else:
            bottleneck_output = self.bn_function(prev_features)
            
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features 
            

class _DenseBlock(nn.ModuleDict):
    _version = 2
    
    def __init__(
        self, 
        num_layers: int,
        num_input_features:int,
        bn_size: int,
        growth_rate: int,
        drop_rate: int,
        memory_efficient: bool = False,
                ) -> None:
        super().__init__() 
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient, 
            )
            self.add_module(f"denselayer{i+1}", layer)


    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    
    
class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features:int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
    

class DenseNet(nn.Module):
    r"""Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    def __init__(
         self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        
        # _log_api_usage_once(self)
        
        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm1d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out 
    

    
