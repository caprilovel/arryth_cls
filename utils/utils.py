
import random
import sys
import os 
import time 
import wfdb
import os
import torch 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import defaultdict

#-----------------------------------------------------------#

#   label_select,用于样本不均衡的数据集，每一次的输入样本为每一类相同数目的样本

#-----------------------------------------------------------#


def label_select(labels, sampling):
    '''
    this function is used for selecting the same amount train samples in each class

    input:
    labels: List, the label list
    samling: the size of the samples in each class
    
    output:
     the list of initialize train dataset index
    
    '''
    if type(labels) is list:
        labels = np.array(labels)
    classes = np.unique(labels)
    # n_class = len(classes)
    class_dict = {}
    sample_list = []   
    for i in classes:
        class_dict[i] = [j for j,x in enumerate(labels) if x==i]
        np.random.shuffle(class_dict[i])
        sample_list.append(class_dict[i][0:sampling])
    return np.concatenate(np.array(sample_list))

def predict (output):
    '''
    输入为batch * 预测向量,返回batch * 预测结果
    '''
    return output.max(1)[1].cpu().numpy()

def save_confusion_matrix(cm, path, title=None, labels_name=None,   cmap=plt.cm.Blues):
    # if not labels_name:
    #     labels_name = [i for i in range(len(np.unique(labels)))]
    
    plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小
    # plt.colorbar()
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels_name, yticklabels=labels_name,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(path + 'cm.jpg', dpi=300)


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


#-----------------------------------------------------------#

#   random_seed，用于torch训练随机种子

#-----------------------------------------------------------#


def random_seed(seed=2020):
    # determine the random seed
    random.seed(seed)
    # hash, save the random seed                   
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, labels):
    pred = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, pred)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

#-----------------------------------------------------------#

#   将时间（秒）转换为 时间（天，小时，分钟，秒）

#   用法示例：day,hour,minute,second = second2time(30804.7)

#   print("time: {:02d}d{:02d}h{:02d}m{:.2f}s".format(day, hour, minute, second))

#   用于显示训练时间

#-----------------------------------------------------------#

def second2time(second):
    intsecond = int(second)
    day = int(second) // (24 * 60 * 60)
    intsecond -= day * (24 * 60 * 60)
    hour = intsecond // (60 * 60)
    intsecond -= hour * (60 * 60)
    minute = intsecond // 60
    intsecond -= 60 * minute
    return (day, hour, minute, second - int(second) + intsecond)


#-----------------------------------------------------------#

#   使用方法：在开始加入此行代码sys.stdout = Logger('log.log')

#-----------------------------------------------------------#


class Logger(object):
    def __init__(self, logFile='Default.log'):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

def get_time_str(style='Nonetype'):
    t = time.localtime()
    if style is 'Nonetype':
        return ("{}{}{}{}{}{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    elif style is 'underline':
        return ("{}_{}_{}_{}_{}_{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_focal_loss(path, length, DataSet):
    default_dict_ecg = defaultdict(lambda: 4)
    dict_ecg_num =  {"N":0, "L":0, "R":0, "e":0, "j":0,      #"B":0,
                    "A":1, "a":1, "J":0, "S":1, 
                    "V":2, "E":2,                   #"r":2,
                    "F":3,                          #"n":3,
                    "/":4, "f":4, "Q":4, "|":4             #"?":4
                    }  
    for i in dict_ecg_num:
        default_dict_ecg[i] = dict_ecg_num[i]

    data_distribution = [0 for i in range(6)]
    for i in DataSet:    
        total_path = "{}/{}".format(path, i)
        atr = wfdb.rdann(total_path, extension="atr")
        atr_sample = atr.sample
        atr_symbol = atr.symbol
        R_times = len(atr_symbol)
        for i in range(R_times - 2):
            data_distribution[5] += 1
            if atr_sample[i+1] - atr_sample[i] > length:
                data_distribution[default_dict_ecg[atr_symbol[i+1]]] += 1
    fraction = np.array(data_distribution)/sum(data_distribution[0:5])
    return fraction