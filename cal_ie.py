# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/21 14:20
@Author  : Qiushi Wang
@FileName: cal_ie.py
@Software: PyCharm
'''
from math import log2

category = [0] * 12

with open("./results/selected_imgs_by_mcmc/selected_0.78_images.txt", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        category[int(line.split("/")[4])] += 1

n = sum(category)

res = 0

for val in category:
    res += (-1 * (val / n) * log2(val / n))

print(res)
