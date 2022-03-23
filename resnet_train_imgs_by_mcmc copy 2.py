# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/18 17:36
@Author  : Qiushi Wang
@FileName: alexnet_train_imgs_by_mcmc.py
@Software: PyCharm
'''
import os
import random
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
# from matplotlib import pyplot as plt
# from model.lenet import LeNet
import torchvision.models as models
from my_dataset import RMBDataset
from my_dataset import RMBDataset_subsampling
from my_dataset_for_mcmc import RMBDatasetMCMC
import time
# from SPLD import spld
# from mcmc import MC
from early_stopping import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
import energyusage
import pytorch_influence_functions as plf

warnings.filterwarnings("ignore")


def select_from_one_class(y_train,prob_pi,label,ratio):
    # select positive and negative samples respectively
    x,y=zip(*y_train.data_info)
    y_train=np.array(y)
    num_sample = y_train[y_train==label].shape[0]
    all_idx = np.arange(y_train.shape[0])[y_train==label]
    label_prob_pi = prob_pi[all_idx]
    obj_sample_size = int(ratio * num_sample)

    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < label_prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)#没抽够，直接并入
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)#随机抽剩余个数的样本
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:#如果找了一百次还没有找到，则按IF值进行排序或者随机抽样
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx

def select_from_one(y_train,prob_pi,ratio):
# select positive and negative samples respectively
    num_sample = len(y_train)
    all_idx = np.arange(num_sample)
    label_prob_pi = prob_pi[all_idx]
    obj_sample_size = int(ratio * num_sample)

    sb_idx = None
    iteration = 0
    while True:
        rand_prob = np.random.rand(num_sample)
        iter_idx = all_idx[rand_prob < label_prob_pi]
        if sb_idx is None:
            sb_idx = iter_idx
        else:
            new_idx = np.setdiff1d(iter_idx, sb_idx)
            diff_size = obj_sample_size - sb_idx.shape[0]
            if new_idx.shape[0] < diff_size:
                sb_idx = np.union1d(iter_idx, sb_idx)#没抽够，直接并入
            else:
                new_idx = np.random.choice(new_idx, diff_size, replace=False)
                sb_idx = np.union1d(sb_idx, new_idx)#随机抽剩余个数的样本
        iteration += 1
        if sb_idx.shape[0] >= obj_sample_size:
            sb_idx = np.random.choice(sb_idx,obj_sample_size,replace=False)
            return sb_idx

        if iteration > 100:#如果找了一百次还没有找到，则按IF值进行排序或者随机抽样
            diff_size = obj_sample_size - sb_idx.shape[0]
            leave_idx = np.setdiff1d(all_idx, sb_idx)
            # left samples are sorted by their IF
            # leave_idx = leave_idx[np.argsort(prob_pi[leave_idx])[-diff_size:]]
            leave_idx = np.random.choice(leave_idx,diff_size,replace=False)
            sb_idx = np.union1d(sb_idx, leave_idx)
            return sb_idx

def train_model(MAX_EPOCH):
    BATCH_SIZE = 16
    LR = 0.001
    log_interval = 1
    val_interval = 1

    num_classes = 12

    # device = torch.device("cuda:2")
    device = torch.device("cuda:2")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # ============================ step 1/5 数据 ============================
    #train_dir = "./results/selected_imgs_by_mcmc/new_new_new_selected_0.5_images.txt"
    train_dir = "./data/std_split_data/train"
    valid_dir = "./data/std_split_data/valid"

    path_state_dict = './resnet_2_model.pth'
    #path_state_dict = './data/resnet50-19c8e357.pth'
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
    train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
    valid_data = RMBDataset(data_dir=valid_dir, transform=train_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    print(len(train_data))
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    print(len(valid_data))

    #alexnet_model = models.resnet50()
    #alexnet_model.load_state_dict( torch.load(path_state_dict))
    alexnet_model=torch.load(path_state_dict)

    num_ftrs = alexnet_model.fc.in_features
    alexnet_model.fc = nn.Linear(num_ftrs, num_classes)
    plf.init_logging()
    config = plf.get_default_config()
    config["num_classes"]=12
    config["gpu"]=1
    trainloader = DataLoader(dataset=train_data, batch_size=1)
    testloader = DataLoader(dataset=valid_data, batch_size=1)
    import pickle
    ifl=plf.calc_img_wise(config, alexnet_model, trainloader, testloader)
    pickle.dump(ifl,open("./influence factor.","wb"))
    #ifl=pickle.load(open("./influence factor.","rb"))
  
    

    #print(train_data)
    # ifl=plf.calc_img_wise(config, alexnet_model, trainloader, testloader)
    # pickle.dump(ifl,open("./influence factor.","wb"))
    # a=2
    # _1l=[]
    # _2l=[]
    # for i in range(len(ifl['0']["influence"])):
    #     if ifl['0']["influence"][i]>0:
    #         _1l.append(i)
    #     else:
    #         _2l.append(i)
    # print(type(ifl['0']["influence"]))
    # sigmoid_k = 10
    # build sampling probability
    phi_ar=- np.array(ifl['0']["influence"])
    #print(phi_ar)
    IF_interval =phi_ar.max() - phi_ar.min()
    alpha=12
    a_param = (alpha / IF_interval)
    prob_pi=1 / (1 + np.exp(a_param * phi_ar))
    #sub_sample_index=select_from_one(trainloader,prob_pi,0.9)
    idx_list=[]
    for i in range(12):
        idx_list.append( select_from_one_class(train_data,prob_pi,i,0.9))
    sub_idx=idx_list[0]
    for i in range(1,12):
        sub_idx=np.union1d(sub_idx,idx_list[i])
    train_data_sub=RMBDataset_subsampling(data_dir=train_dir, transform=train_transform,subindex=sub_idx)
    #prob_pi = 1 / (1 + np.exp(a_param * phi_ar))
   # print(prob_pi)




    
    # train_data_sub = RMBDataset_subsampling(data_dir=train_dir, transform=train_transform,subindex=ifl["0"]["helpful"])
    print(len(train_data_sub))
    trainloader = DataLoader(dataset=train_data_sub, batch_size=BATCH_SIZE,shuffle=True)


    
    #alexnet_model = alexnet_model.cuda()
    alexnet_model = alexnet_model.to(device)
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
    for epoch in range(1):

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

            #inputs = inputs.cuda()
            #labels = labels.cuda()
            inputs = inputs.to(device)
            labels = labels.to(device)

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
                    #inputs, labels = inputs.cuda(), labels.cuda()
                    inputs, labels = inputs.to(device), labels.to(device)
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
    torch.save(alexnet_model, './resnet_2andifl_model.pth')
    return end_time - start_time


def main():
    #energyusage.evaluate(train_model, 20, pdf=True, energyOutput=False)
    train_model(1)

if __name__ == '__main__':
    main()



