import numpy as np
import wfdb
from collections import defaultdict
import torch

default_dict_ecg = defaultdict(lambda: 4)
dict_ecg_num =  {"N":0, "L":0, "R":0, "e":0, "j":0,      #"B":0,
                "A":1, "a":1, "J":0, "S":1, 
                "V":2, "E":2,                   #"r":2,
                "F":3,                          #"n":3,
                "/":4, "f":4, "Q":4, "|":4             #"?":4
                }  
for i in dict_ecg_num:
    default_dict_ecg[i] = dict_ecg_num[i]


# numpy归一化
# 平均归一化
def mean_standard(x):
    Min = np.min(x)
    Max = np.max(x)
    return (x-Min)/(Max-Min)
# 标准化归一化
def std_standard(x):
    mu = np.expand_dims(np.average(x, axis=1), axis=1)
    sigma = np.expand_dims(np.std(x, axis=1), axis=1)
    x = (x - mu)/ sigma
    return x
# sigmoid归一化
def sigmoid_standard(x):
    return 
def arrhythmia_iter(num, path):
    total_path = path + '/' + str(num)
    data = (wfdb.rdrecord(total_path, physical=False)).d_signal
    # data = std_standard(data)
    atr = wfdb.rdann(total_path, extension="atr")
    atr_sample = atr.sample
    atr_symbol = atr.symbol
    R_times = len(atr_symbol)
    for i in range(R_times - 2):
        yield data[atr_sample[i]:atr_sample[i+2]][:].transpose(), atr_symbol[i+1]

def mitdb_aryth_batchiter(batch_size, nums, path, use_last_batch=True):
    '''
    ###INPUT:
    batch_size: int, the batch_size of the iter
    nums: List, the ecg data sets to subsection
    path: str, the ecg data storage path
    use_last_batch: bool, if True, the final batch of an ecg data is not same as the batch_size; else, drop the final ecg data batch.

    ###FUNCTION:
    read a single ecg data, realize automatic segmentation
    a single heartbeat is consist of 258 
    
    ###OUTPUT:
    batch_x: 
    shape (batch_size x length x dimension, precisely equals to batch_size x 258 x 2)
    type  tensor.FloatTensor
    batch_y: shape (batch_size) 
    type  tensor.LongTensor
    
    '''
    # default dictionary for aryth classify
    default_dict_ecg = defaultdict(lambda: 4)
    dict_ecg_num =  {"N":0, "L":0, "R":0, "e":0, "j":0,      #"B":0,
                    "A":1, "a":1, "J":0, "S":1, 
                    "V":2, "E":2,                   #"r":2,
                    "F":3,                          #"n":3,
                    "/":4, "f":4, "Q":4, "|":4             #"?":4
                    }  
    for i in dict_ecg_num:
        default_dict_ecg[i] = dict_ecg_num[i]

    for num in nums:
        total_path = path + '/' + str(num)
        data = (wfdb.rdrecord(total_path, physical=False)).d_signal
        # data = std_standard(data)
        atr = wfdb.rdann(total_path, extension="atr")
        atr_sample = atr.sample
        atr_symbol = atr.symbol
        R_times = len(atr_symbol)
        batch_x = []
        batch_y = []
        count = 0
        for i in range(R_times - 2):
            if atr_sample[i+1] - atr_sample[i]>128:
                batch_x.append(torch.FloatTensor(data[atr_sample[i+1]-128:atr_sample[i+1]+128]).unsqueeze(dim=0)) # unsqueeze(dim=0): 增加第零维度
                batch_y.append(default_dict_ecg[atr_symbol[i+1]])
                count += 1
                if count == batch_size:
                    yield torch.cat(batch_x,dim=0), torch.LongTensor(batch_y)
                    count = 0
                    batch_x = []
                    batch_y = []
        if  count is not 0 and use_last_batch:
            yield torch.cat(batch_x,dim=0), torch.LongTensor(batch_y)
                          
class FewshotDataloader():
    def __init__(self, nums, path) -> None:
        default_dict_ecg = defaultdict(lambda: 4)
        dict_ecg_num =  {"N":0, "L":0, "R":0, "e":0, "j":0,      #"B":0,
                    "A":1, "a":1, "J":0, "S":1, 
                    "V":2, "E":2,                   #"r":2,
                    "F":3,                          #"n":3,
                    "/":4, "f":4, "Q":4, "|":4             #"?":4
                    }
        for i in dict_ecg_num:
            default_dict_ecg[i] = dict_ecg_num[i]  
        self.atr_array = [ [] for i in range(5)]
        for num in nums:
            total_path = path + '/' + str(num)
            data = (wfdb.rdrecord(total_path, physical=False)).d_signal
            # data = std_standard(data)
            atr = wfdb.rdann(total_path, extension="atr")
            atr_sample = atr.sample
            atr_symbol = atr.symbol
            
            for i in range(len(atr_symbol)):
                self.atr_array[default_dict_ecg[atr_symbol[i]]].append(atr_sample[i])
        print('N:{}\n A:{}\n V:{}\n F:{}\n Q:{}\n'.format(len(self.atr_array[0]), len(self.atr_array[1]), len(self.atr_array[2]), len(self.atr_array[3]), len(self.atr_array[4])))       
        
        
        
def read_ecg_text(path):
    ''' read the ecg data annotations and labels
    
    args:
        path: ecg data path.
        
    returns:
        peak_index:
        
        settings:
    '''
    dataname = path + "_info.txt"
    import os, json 
    if not os.path.exists(dataname):
        raise("no such file!")
    with open(dataname, 'r') as f:
        line1 = f.readline()
        line2 = f.readline()
    peak_index = eval(line1)
    settings = json.loads(line2)
    return peak_index, settings