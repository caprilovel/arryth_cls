import numpy  as np
import wfdb
from collections import defaultdict


DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
DS = DS1 + DS2

path  = "./data/mit-bih-arrhythmia-database-1.0.0"
# default_dict_ecg = defaultdict(lambda: 4)
# dict_ecg_num =  {"N":0, "L":0, "R":0, "e":0, "j":0,      #"B":0,
#                 "A":1, "a":1, "J":0, "S":1, 
#                 "V":2, "E":2,                   #"r":2,
#                 "F":3,                          #"n":3,
#                 "/":4, "f":4, "Q":4, "|":4             #"?":4
#                 }  
# for i in dict_ecg_num:
#     default_dict_ecg[i] = dict_ecg_num[i]

# data_distribution = [0 for i in range(6)]
# for i in DS:    
#     total_path = "{}/{}".format(path, i)
#     atr = wfdb.rdann(total_path, extension="atr")
#     atr_sample = atr.sample
#     atr_symbol = atr.symbol
#     R_times = len(atr_symbol)
#     for i in range(R_times - 2):
#         data_distribution[5] += 1
#         if atr_sample[i+1] - atr_sample[i] > 128:
#             data_distribution[default_dict_ecg[atr_symbol[i+1]]] += 1

# print(data_distribution, sum(data_distribution[0:5]))
# Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题。该损失函数降低了大量简单负样本在训练中所占的权重，也可理解为一种困难样本挖掘
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
        total_path = "{}/{}".format(path, i) # 格式化字符串
        atr = wfdb.rdann(total_path, extension="atr") 
        atr_sample = atr.sample
        atr_symbol = atr.symbol
        R_times = len(atr_symbol)
        for i in range(R_times - 2):
            data_distribution[5] += 1
            if atr_sample[i+1] - atr_sample[i] > length:
                data_distribution[default_dict_ecg[atr_symbol[i+1]]] += 1
    return np.array(data_distribution)/sum(data_distribution[0:5])
print(get_focal_loss(path, 128, DS))