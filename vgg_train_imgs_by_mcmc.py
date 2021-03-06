# -*- coding: utf-8 -*-

'''
@Time    : 2021/10/21 15:32
@Author  : Qiushi Wang
@FileName: vgg_train_imgs_by_mcmc.py
@Software: PyCharm
'''


import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
# from matplotlib import pyplot as plt
# from model.lenet import LeNet
import torchvision.models as models
from my_dataset import RMBDataset
from my_dataset_for_mcmc import RMBDatasetMCMC
import time
# from SPLD import spld
# from mcmc import MC
from early_stopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import energyusage

warnings.filterwarnings("ignore")


def train_model(MAX_EPOCH):
    BATCH_SIZE = 64
    LR = 0.001
    log_interval = 1
    val_interval = 1

    num_classes = 12

    # device = torch.device("cuda:2")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # ============================ step 1/5 数据 ============================
    train_dir = "./results/selected_imgs_by_mcmc/new_selected_0.3_images.txt"
    # train_dir = "./data/std_split_data/train"
    valid_dir = "./data/std_split_data/valid"

    path_state_dict = './data/vgg16-397923af.pth'

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256)),  # (256, 256) 区别
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = RMBDatasetMCMC(data_dir=train_dir, transform=train_transform)
    valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    print(len(train_data))
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    print(len(valid_data))

    alexnet_model = models.vgg16()
    pretrained_state_dict = torch.load(path_state_dict)
    alexnet_model.load_state_dict(pretrained_state_dict)

    num_ftrs = alexnet_model.classifier._modules["6"].in_features
    alexnet_model.classifier._modules["6"] = nn.Linear(num_ftrs, num_classes)

    alexnet_model = alexnet_model.cuda(1)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(alexnet_model.parameters(), lr=LR, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)  # 设置学习率下降策略

    # ============================ step 5/5 训练 ============================

    valid_acc = []
    valid_precision = []
    valid_recall = []
    valid_f1 = []
    time_record = []

    train_acc = []
    train_precision = []
    train_recall = []
    train_f1 = []
    train_time_record = []

    start_time = time.time()
    for epoch in range(MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        y_true = []
        y_pred = []
        y_valid_true = []
        y_valid_pred = []
        alexnet_model.train()

        for i, data in enumerate(train_loader):
            inputs, labels = data

            inputs = inputs.cuda(1)
            labels = labels.cuda(1)

            outputs = alexnet_model(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            for la in labels.cpu().numpy():
                y_true.append(la)
            for pre in predicted.cpu().numpy():
                y_pred.append(pre)
            # 打印训练信息
            loss_mean += loss.item()
            # train_curve.append(loss.item())
            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                train_acc.append(correct / total)
                train_precision.append(precision_score(y_true, y_pred, average='macro'))
                train_recall.append(recall_score(y_true, y_pred, average='macro'))
                train_f1.append(f1_score(y_true, y_pred, average='macro'))
                train_time_record.append(time.time() - start_time)
                print(
                    "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                        epoch, MAX_EPOCH, i + 1, len(train_loader), loss_mean, correct / total,
                        precision_score(y_true, y_pred, average='macro'),
                        recall_score(y_true, y_pred, average='macro'),
                        f1_score(y_true, y_pred, average='macro')))
                loss_mean = 0.
        scheduler.step()

        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            alexnet_model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.cuda(1), labels.cuda(1)

                    bs, ncrops, c, h, w = inputs.size()  # [4, 10, 3, 224, 224
                    outputs = alexnet_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)

                    _, predicted = torch.max(outputs_avg.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    for la in labels.cpu().numpy():
                        y_valid_true.append(la)
                    for pre in predicted.cpu().numpy():
                        y_valid_pred.append(pre)

                    loss_val += loss.item()

                loss_val_mean = loss_val / len(valid_loader)

                valid_acc.append(correct_val / total_val)
                valid_precision.append(precision_score(y_valid_true, y_valid_pred, average='macro'))
                valid_recall.append(recall_score(y_valid_true, y_valid_pred, average='macro'))
                valid_f1.append(f1_score(y_valid_true, y_valid_pred, average='macro'))
                time_record.append(time.time() - start_time)
                print(
                    "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} precision:{:.2%} recall:{:.2%} f1:{:.2%}".format(
                        epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val_mean, correct_val / total_val,
                        precision_score(y_valid_true, y_valid_pred, average='macro'),
                        recall_score(y_valid_true, y_valid_pred, average='macro'),
                        f1_score(y_valid_true, y_valid_pred, average='macro')))
                # early_stopping(loss_val_mean, alexnet_model)
                # if early_stopping.early_stop:
                #     print("Early stopping")
                #     # 结束模型训练
                #     break
    end_time = time.time()
    print("running time is:", end_time - start_time)

    return end_time - start_time


def main():
    energyusage.evaluate(train_model, 20, pdf=True, energyOutput=False)


if __name__ == '__main__':
    main()