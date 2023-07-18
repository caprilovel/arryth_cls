import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
from torch import optim
class Data_iter():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.length = self.data.size(0)
        self.idx = [i for i in range(self.length)]
        self.train_size = 0
    def shuffle(self,seed=None):
        if  seed is not None:
            random.seed(seed)
        random.shuffle(self.idx)
    def train_iter(self, total_size, batch_size):
        self.train_size = total_size
        for i in range((total_size+batch_size-1)//batch_size):
            if batch_size*(1+i)-1 < total_size:
                batch_idx = self.idx[i*batch_size:(1+i)*batch_size]
            else:
                batch_idx = self.idx[i*batch_size:total_size]
            yield self.data[batch_idx], self.labels[batch_idx]
    
    def eval_iter(self, total_size, batch_size):
        assert total_size + self.train_size <= self.length
        for i in range((total_size+batch_size-1)//batch_size):
            if batch_size*(1+i)-1 < total_size:
                batch_idx = self.idx[i*batch_size+self.train_size:(1+i)*batch_size+self.train_size]
            else:
                batch_idx = self.idx[i*batch_size+self.train_size:total_size+self.train_size]
            yield self.data[batch_idx], self.labels[batch_idx]
class Easynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,2)
    def forward(self, x):
        return self.linear(x)
# dataT = torch.randn(20,3)            
# labelT = torch.LongTensor([i for i in range(20)])
# data_iter = Data_iter(dataT, labelT)
# data_iter.shuffle()
# for i,j in data_iter.train_iter(10, 3):
#     print(i.size(), j)
# for i,j in data_iter.eval_iter(5,2):

#-----------------------------------------------------------#

#   该函数用于triplet loss,将

#-----------------------------------------------------------#
def triplet_loss(embeddings, nclass):
    """Choose the hard positive and negative samples to generate the triplet 选取难正样本和负样本,用于生成triplet_loss

    This function is delighted by the paper 'FaceNet: A Unified Embedding for Face Recognition and Clustering' to find the proper triplets in a batch. Hence, the batch should contain the different class samples. The input is the number of class mul the number of samples contained in each class. 这个函数受启发于论文'FaceNet: A Unified Embedding for Face Recognition and Clustering',用于寻找一个batch中的不同triplets.因此输入的每一个batch应该包含每一个种类的样本,输入大小等同于种类数*每个种类包含的样本数.

    Args:
      embeddings:
        the embeddings, which size is batch_size x samples   
      
    Returns:


    """
    batch_size = embeddings[0]
    x = embeddings.unsqueeze(0).expand(embeddings.size(0), embeddings.size(0), embeddings.size(1))
    y = embeddings.unsqueeze(1).expand(embeddings.size(0), embeddings.size(0), embeddings.size(1))

    distance_metric = torch.pow(x - y, 2).sum(2)
    return distance_metric
model = Easynet()
a = torch.randn(4, 2)
model.train()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

optimizer.zero_grad()

y = model(a)
loss_metric = triplet_loss(y, 2)

loss = loss_metric[0][1] + loss_metric[1][0]

print(loss)
loss.backward()
optimizer.step()