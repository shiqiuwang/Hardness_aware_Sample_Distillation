# -*- coding: utf-8 -*-

'''
@Time    : 2021/5/11 19:07
@Author  : Qiushi Wang
@FileName: create_folders.py
@Software: PyCharm
'''
import os


def create(addr):
    for i in range(0, 12):  # 范围是你需要的数字范围，我创建的是名为1~50的文件夹
        os.makedirs(addr + '/' + str(i)+'/'+'no_selected') # str(i)前后可加你需要的前缀和后缀


# for i in range(10,100):
#    addr = './results/easy_'+str(i)  # 你需要创建文件夹的目录
#    create(addr)
# for i in range(10,100):
#    addr = './results/hard_'+str(i)  # 你需要创建文件夹的目录
#    create(addr)
# addr1 = './results/confusion_dataset/alexnet/0_10'
# addr2 = './results/confusion_dataset/alexnet/10_20'
# addr3 = './results/confusion_dataset/alexnet/20_30'
# addr4 = './results/confusion_dataset/alexnet/30_40'
# addr5 = './results/confusion_dataset/alexnet/40_50'
# addr6 = './results/confusion_dataset/alexnet/50_60'
# addr7 = './results/confusion_dataset/alexnet/60_70'
# addr8 = './results/confusion_dataset/alexnet/70_80'
# addr9 = './results/confusion_dataset/alexnet/80_90'
# addr10 = './results/confusion_dataset/alexnet/90_100'
# addr1 = './results/webvision/hard_90/train'
# addr1 = './result/train3'
addr1 = './results/case_study'

create(addr1)
# create(addr2)
# create(addr3)
# create(addr4)
# create(addr5)
# create(addr6)
# create(addr7)
# create(addr8)
# create(addr9)
# create(addr10)
