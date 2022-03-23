# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/21 14:44
@Author  : Qiushi Wang
@FileName: cal_source_ie.py
@Software: PyCharm
'''

from math import log2

category = [0] * 12

with open("./results/spld_loss/alexnet_train_loss_file.txt", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split(":")[0]
        category[int(line.split("/")[4])] += 1

n = sum(category)

res = 0

for val in category:
    res += (-1 * (val / n) * log2(val / n))

print(res)
