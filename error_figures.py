# -*- coding: utf-8 -*-

'''
@Time    : 2021/10/29 11:17
@Author  : Qiushi Wang
@FileName: error_figures.py
@Software: PyCharm
'''

# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/17 16:37
@Author  : Qiushi Wang
@FileName: selected_by_mcmc.py
@Software: PyCharm
'''

import numpy as np
from sklearn.cluster import MeanShift
from collections import Counter
from mcmc import MC
from sklearn.ensemble import IsolationForest

# T = 1000  # 模拟次数



label_0_loss_to_img = {}
label_1_loss_to_img = {}
label_2_loss_to_img = {}
label_3_loss_to_img = {}
label_4_loss_to_img = {}
label_5_loss_to_img = {}
label_6_loss_to_img = {}
label_7_loss_to_img = {}
label_8_loss_to_img = {}
label_9_loss_to_img = {}
label_10_loss_to_img = {}
label_11_loss_to_img = {}
label_12_loss_to_img = {}
label_13_loss_to_img = {}
label_14_loss_to_img = {}
label_15_loss_to_img = {}
label_16_loss_to_img = {}
label_17_loss_to_img = {}
label_18_loss_to_img = {}
label_19_loss_to_img = {}

with open("./results/spld_loss/alexnet_train_loss_file.txt", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        img, loss = line.split(":")
        loss = float(loss)
        label = int(img.split("/")[4])

        if label == 0:
            if loss in label_0_loss_to_img.keys():
                label_0_loss_to_img[loss].append(img)
            else:
                label_0_loss_to_img[loss] = [img]
        if label == 1:
            if loss in label_1_loss_to_img.keys():
                label_1_loss_to_img[loss].append(img)
            else:
                label_1_loss_to_img[loss] = [img]
        if label == 2:
            if loss in label_2_loss_to_img.keys():
                label_2_loss_to_img[loss].append(img)
            else:
                label_2_loss_to_img[loss] = [img]
        if label == 3:
            if loss in label_3_loss_to_img.keys():
                label_3_loss_to_img[loss].append(img)
            else:
                label_3_loss_to_img[loss] = [img]
        if label == 4:
            if loss in label_4_loss_to_img.keys():
                label_4_loss_to_img[loss].append(img)
            else:
                label_4_loss_to_img[loss] = [img]
        if label == 5:
            if loss in label_5_loss_to_img.keys():
                label_5_loss_to_img[loss].append(img)
            else:
                label_5_loss_to_img[loss] = [img]
        if label == 6:
            if loss in label_6_loss_to_img.keys():
                label_6_loss_to_img[loss].append(img)
            else:
                label_6_loss_to_img[loss] = [img]
        if label == 7:
            if loss in label_7_loss_to_img.keys():
                label_7_loss_to_img[loss].append(img)
            else:
                label_7_loss_to_img[loss] = [img]
        if label == 8:
            if loss in label_8_loss_to_img.keys():
                label_8_loss_to_img[loss].append(img)
            else:
                label_8_loss_to_img[loss] = [img]
        if label == 9:
            if loss in label_9_loss_to_img.keys():
                label_9_loss_to_img[loss].append(img)
            else:
                label_9_loss_to_img[loss] = [img]

        if label == 10:
            if loss in label_10_loss_to_img.keys():
                label_10_loss_to_img[loss].append(img)
            else:
                label_10_loss_to_img[loss] = [img]
        if label == 11:
            if loss in label_11_loss_to_img.keys():
                label_11_loss_to_img[loss].append(img)
            else:
                label_11_loss_to_img[loss] = [img]

label_0_loss_cur = np.array(list(map(float, list(label_0_loss_to_img.keys())))).reshape(-1, 1)
label_1_loss_cur = np.array(list(map(float, list(label_1_loss_to_img.keys())))).reshape(-1, 1)
label_2_loss_cur = np.array(list(map(float, list(label_2_loss_to_img.keys())))).reshape(-1, 1)
label_3_loss_cur = np.array(list(map(float, list(label_3_loss_to_img.keys())))).reshape(-1, 1)
label_4_loss_cur = np.array(list(map(float, list(label_4_loss_to_img.keys())))).reshape(-1, 1)
label_5_loss_cur = np.array(list(map(float, list(label_5_loss_to_img.keys())))).reshape(-1, 1)
label_6_loss_cur = np.array(list(map(float, list(label_6_loss_to_img.keys())))).reshape(-1, 1)
label_7_loss_cur = np.array(list(map(float, list(label_7_loss_to_img.keys())))).reshape(-1, 1)
label_8_loss_cur = np.array(list(map(float, list(label_8_loss_to_img.keys())))).reshape(-1, 1)
label_9_loss_cur = np.array(list(map(float, list(label_9_loss_to_img.keys())))).reshape(-1, 1)

label_10_loss_cur = np.array(list(map(float, list(label_10_loss_to_img.keys())))).reshape(-1, 1)
label_11_loss_cur = np.array(list(map(float, list(label_11_loss_to_img.keys())))).reshape(-1, 1)

# =========================================去除孤立点===============================================
label_0_loss = []
label_1_loss = []
label_2_loss = []
label_3_loss = []
label_4_loss = []
label_5_loss = []
label_6_loss = []
label_7_loss = []
label_8_loss = []
label_9_loss = []
label_10_loss = []
label_11_loss = []

# 第0类
clf_0 = IsolationForest(random_state=0).fit(label_0_loss_cur)
pre_0 = clf_0.predict(label_0_loss_cur)

for i in range(len(pre_0)):
    if pre_0[i] == -1:
        label_0_loss.append(label_0_loss_cur[i])

# 第1类
clf_1 = IsolationForest(random_state=0).fit(label_1_loss_cur)
pre_1 = clf_1.predict(label_1_loss_cur)

for i in range(len(pre_1)):
    if pre_1[i] == -1:
        label_1_loss.append(label_1_loss_cur[i])

# 第2类
clf_2 = IsolationForest(random_state=0).fit(label_2_loss_cur)
pre_2 = clf_2.predict(label_2_loss_cur)

for i in range(len(pre_2)):
    if pre_2[i] == -1:
        label_2_loss.append(label_2_loss_cur[i])

# 第3类
clf_3 = IsolationForest(random_state=0).fit(label_3_loss_cur)
pre_3 = clf_3.predict(label_3_loss_cur)

for i in range(len(pre_3)):
    if pre_3[i] == -1:
        label_3_loss.append(label_3_loss_cur[i])

# 第4类
clf_4 = IsolationForest(random_state=0).fit(label_4_loss_cur)
pre_4 = clf_4.predict(label_4_loss_cur)

for i in range(len(pre_4)):
    if pre_4[i] == -1:
        label_4_loss.append(label_4_loss_cur[i])

# 第5类
clf_5 = IsolationForest(random_state=0).fit(label_5_loss_cur)
pre_5 = clf_5.predict(label_5_loss_cur)

for i in range(len(pre_5)):
    if pre_5[i] == -1:
        label_5_loss.append(label_5_loss_cur[i])

# 第6类
clf_6 = IsolationForest(random_state=0).fit(label_6_loss_cur)
pre_6 = clf_6.predict(label_6_loss_cur)

for i in range(len(pre_6)):
    if pre_6[i] == -1:
        label_6_loss.append(label_6_loss_cur[i])

# 第7类
clf_7 = IsolationForest(random_state=0).fit(label_7_loss_cur)
pre_7 = clf_7.predict(label_7_loss_cur)

for i in range(len(pre_7)):
    if pre_7[i] == -1:
        label_7_loss.append(label_7_loss_cur[i])

# 第8类
clf_8 = IsolationForest(random_state=0).fit(label_8_loss_cur)
pre_8 = clf_8.predict(label_8_loss_cur)

for i in range(len(pre_8)):
    if pre_8[i] == -1:
        label_8_loss.append(label_8_loss_cur[i])

# 第9类
clf_9 = IsolationForest(random_state=0).fit(label_9_loss_cur)
pre_9 = clf_9.predict(label_9_loss_cur)

for i in range(len(pre_9)):
    if pre_9[i] == -1:
        label_9_loss.append(label_9_loss_cur[i])

# 第10类
clf_10 = IsolationForest(random_state=0).fit(label_10_loss_cur)
pre_10 = clf_10.predict(label_10_loss_cur)

for i in range(len(pre_10)):
    if pre_10[i] == -1:
        label_10_loss.append(label_10_loss_cur[i])

# 第11类
clf_11 = IsolationForest(random_state=0).fit(label_11_loss_cur)
pre_11 = clf_11.predict(label_11_loss_cur)

for i in range(len(pre_11)):
    if pre_11[i] == -1:
        label_11_loss.append(label_11_loss_cur[i])

# ==============================================保存选中的图片=========================================
imgs = []
# ==================================================针对第0类的采样=================================================

print(len(label_0_loss))

for loss in label_0_loss:
    for img in label_0_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第1类的采样=================================================
print(len(label_1_loss))

for loss in label_1_loss:
    for img in label_1_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第2类的采样=================================================
print(len(label_2_loss))

for loss in label_2_loss:
    for img in label_2_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第3类的采样=================================================
print(len(label_3_loss))

for loss in label_3_loss:
    for img in label_3_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第4类的采样=================================================
print(len(label_4_loss))

for loss in label_4_loss:
    for img in label_4_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第5类的采样=================================================
print(len(label_5_loss))

for loss in label_5_loss:
    for img in label_5_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第6类的采样=================================================
print(len(label_6_loss))

for loss in label_6_loss:
    for img in label_6_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第7类的采样=================================================
print(len(label_7_loss))

for loss in label_7_loss:
    for img in label_7_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))

# ==================================================针对第8类的采样=================================================
print(len(label_8_loss))

for loss in label_8_loss:
    for img in label_8_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第9类的采样=================================================
print(len(label_9_loss))

for loss in label_9_loss:
    for img in label_9_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第10类的采样=================================================
print(len(label_10_loss))

for loss in label_10_loss:
    for img in label_10_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))


# ==================================================针对第11类的采样=================================================
print(len(label_0_loss))

for loss in label_0_loss:
    for img in label_0_loss_to_img[loss[0]]:
        imgs.append(img)

print(len(imgs))

f = open("./results/selected_imgs_by_mcmc/selected_" + str(round(len(set(imgs)) / 108032, 2)) + "_images_error.txt",
         mode='w+',
         encoding='utf-8')
for img_name in set(imgs):
    f.write(img_name)
    f.write('\n')
f.close()
