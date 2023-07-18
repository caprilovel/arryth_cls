import argparse
import time
from sklearn.metrics import accuracy_score, confusion_matrix
# from models import *
from utils.loss_func import focal_loss 
from utils.utils import * 
from utils.data_loader import mitdb_aryth_batchiter
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import sys
import os 
from demo1 import *

# 设置log
mkdir('./log/')
sys.stdout = Logger('./log/{}log.log'.format(get_time_str()))


# argparse是python用于解析命令行参数和选项的标准模块, argparse模块的作用是用于解析命令行参数 argparse模块可以让人轻松编写用户友好的命令行接口。
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help="random_seed")
parser.add_argument('--use_arg', type=boolean_string, default=False, help='whether use the args')



args = parser.parse_args()


if args.use_arg:
    seed = args.seed
else: 
    seed = 49
    
random_seed(seed)
dataset_divide = [0.6, 0.2, 0.2]

# class SKnet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.skseq = nn.Sequential(
#             TemporalAttention(258, 2, 8, 8),
#             nn.Conv1d(2, 32, 3, padding=1),
#             SK_MultiScaleConv1dBlock(32, 0.8),
#             nn.ReLU(),
#             TemporalAttention(258, 32, 32, 32),
#             SK_MultiScaleConv1dBlock(32, 0.8),
#             nn.ReLU(),
#             SK_MultiScaleConv1dBlock(32, 0.8),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.linear = nn.Linear(32, 5)
#     def forward(self, x):
#         x = torch.transpose(x, 1, 2)
#         y = self.skseq(x)
#         y = y.squeeze(2)
#         return self.linear(y)


DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
Total_DS = DS1 + DS2
random.shuffle(Total_DS)
DS_train = Total_DS[0:int(dataset_divide[0] * len(Total_DS))]
DS_eval = Total_DS[int(dataset_divide[0] * len(Total_DS)):int((dataset_divide[1] + dataset_divide[0]) * len(Total_DS))]
DS_test = Total_DS[int((dataset_divide[1] + dataset_divide[0]) * len(Total_DS)):]




path  = "./data/mit-bih-arrhythmia-database-1.0.0"
focal_weight = get_focal_loss(path, 128, DataSet=DS_train)[0:5]
focal_weight = (1 - focal_weight) ** 1.5
focal_weight = torch.FloatTensor(focal_weight).cuda()
model = ReversalNet(1, 5, 2, 6, 5)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)


model = model.cuda()
def train(epochs, use_gpu=True):
    loss_list=[sys.maxsize]
    ttime = time.time()
    for epoch in range(epochs):
        epoch_time = time.time()
        train_batch_iter = mitdb_aryth_batchiter(10, DS_train, path, use_last_batch=False)
        
        

        model.train()
        loss_total = []
        for inputs, labels in train_batch_iter:
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(inputs)
            loss = F.cross_entropy(output, labels, weight=focal_weight)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            loss_total.append(loss.item())

        model.eval()
        eval_output = []
        eval_label = []
        eval_batch_iter = mitdb_aryth_batchiter(10, DS_eval, path)
        for inputs, labels in eval_batch_iter:
            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            eval_output.append(output.detach().cpu())
            eval_label.append(labels.detach().cpu())

        # eval_pred = torch.cat(eval_output, dim=0)
        # eval_pred = eval_pred.max(1)[1].cpu().numpy()
        output_cat = torch.cat(eval_output, dim=0)
        label_cat = torch.cat(eval_label, dim=0)
        acc_val = accuracy(output_cat, label_cat)
        total_time = time.time() - ttime
        day, hour, minute, second = second2time(total_time)
        if epoch%2==0:
            print('Epoch: {:04d}'.format(epoch + 1),
            'acc_val:{:.4f}'.format(acc_val),
            'epoch_time: {:.2f}s'.format(time.time() - epoch_time),
            'total_time: {:02d}d{:02d}h{:02d}m{:.1f}s'.format(day, hour, minute, second)
            )




train(300)
save_dir = "./model_data/{}{}".format(str(type(model).__name__), str(50))
mkdir(save_dir)
torch.save(model, save_dir + '/' + 'model.pth')


def test(use_gpu=True):
    model.eval()
    test_output = []
    test_label = []
    test_batch_iter = mitdb_aryth_batchiter(10, DS_test, path)
    for inputs, labels in test_batch_iter:
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        output = model(inputs)
        test_output.append(output.detach().cpu())
        test_label.append(labels.detach().cpu())
    acc_val = accuracy(torch.cat(test_output,dim=0), torch.cat(test_label, dim=0))
    labels = torch.cat(test_label,dim=0).cpu().numpy()
    pred = torch.cat(test_output,dim=0).max(1)[1].cpu().numpy()
    print(acc_val)
    print(confusion_matrix(labels, pred))
test()
