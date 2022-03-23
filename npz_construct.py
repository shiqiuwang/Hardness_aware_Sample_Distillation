'''
Description: Wrap the dataset in .npz format
version: 1.0
Author: wangqiushi
Date: 2022-02-25 21:10:17
LastEditors: Wang Qiushi
LastEditTime: 2022-02-25 23:34:16
'''

"""
alt+shift+下箭头:复制上一行代码到一行
"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from my_dataset import RMBDataset
import warnings

warnings.filterwarnings("ignore")




def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



set_seed()  # 设置随机种子

# 参数设置
TRAIN_BATCH_SIZE = 83067
VALID_BATCH_SIZE = 11256
TEST_BATCH_SIZE = 11261


# ============================ step 1/5 数据 ============================

train_dir = "./sop_split_data/train" # 训练集路径
#valid_dir = "./sop_split_data/valid" # 验证集路径
#test_dir = "./sop_split_data/test" # 测试集路径


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 对训练数据集的transform
train_transform = transforms.Compose([
    transforms.Resize((256)),  
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 对验证数据集的transform
valid_transform = transforms.Compose([
    transforms.Resize((256)),  
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 对测试数据集的transform
test_transform = transforms.Compose([
    transforms.Resize((256)), 
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
train_data = RMBDataset(data_dir=train_dir,transform=train_transform)
# valid_data = RMBDataset(data_dir=valid_dir,transform=valid_transform)
# test_data = RMBDataset(data_dir=test_dir,transform=test_transform)

# 构建train_DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE)
print(len(train_data))

# 构建valid_DataLoder
# valid_loader = DataLoader(dataset=valid_data, batch_size=VALID_BATCH_SIZE)
# print(len(valid_data))

# 构建test_DataLoder
# test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE)
# print(len(test_data))

# 将训练数据集构造为.npz
for _, data in enumerate(train_loader):
    x_train,y_train = data
x_train = x_train.transpose(1,3)
print(x_train.shape)
print(y_train.shape)

print(np.array(x_train).shape)
print(np.array(y_train).shape)

# 将验证数据集构造为.npz
# for _, data in enumerate(valid_loader):
#     x_valid,y_valid = data
# x_valid=x_valid.transpose(1,3)
# print(x_valid.shape)
# print(y_valid.shape)

# print(np.array(x_valid).shape)
# print(np.array(y_valid).shape)

# 将测试数据集构造为.npz
# for _, data in enumerate(test_loader):
#     x_test,y_test = data
# x_test = x_test.transpose(1,3)
# print(x_test.shape)
# print(y_test.shape)

# print(np.array(x_test).shape)
# print(np.array(y_test).shape)


if not os.path.exists('new_data_files'):
    os.makedirs('new_data_files')

np.savez_compressed('./new_data_files/train.npz',
                      x_train=x_train, y_train=y_train)
# np.savez_compressed('./data_files/valid.npz',
#                       x_valid=x_valid, y_valid=y_valid)
# np.savez_compressed('./data_files/test.npz',
#                       x_test=x_test, y_test=y_test)