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

#T = 1000  # 模拟次数

for T in range(10000,20000,1000):
    
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
    clf_0 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_0_loss_cur)
    pre_0 = clf_0.predict(label_0_loss_cur)

    for i in range(len(pre_0)):
        if pre_0[i] == 1:
            label_0_loss.append(label_0_loss_cur[i])

    # 第1类
    clf_1 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_1_loss_cur)
    pre_1 = clf_1.predict(label_1_loss_cur)

    for i in range(len(pre_1)):
        if pre_1[i] == 1:
            label_1_loss.append(label_1_loss_cur[i])

    # 第2类
    clf_2 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_2_loss_cur)
    pre_2 = clf_2.predict(label_2_loss_cur)

    for i in range(len(pre_2)):
        if pre_2[i] == 1:
            label_2_loss.append(label_2_loss_cur[i])

    # 第3类
    clf_3 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_3_loss_cur)
    pre_3 = clf_3.predict(label_3_loss_cur)

    for i in range(len(pre_3)):
        if pre_3[i] == 1:
            label_3_loss.append(label_3_loss_cur[i])

    # 第4类
    clf_4 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_4_loss_cur)
    pre_4 = clf_4.predict(label_4_loss_cur)

    for i in range(len(pre_4)):
        if pre_4[i] == 1:
            label_4_loss.append(label_4_loss_cur[i])

    # 第5类
    clf_5 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_5_loss_cur)
    pre_5 = clf_5.predict(label_5_loss_cur)

    for i in range(len(pre_5)):
        if pre_5[i] == 1:
            label_5_loss.append(label_5_loss_cur[i])

    # 第6类
    clf_6 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_6_loss_cur)
    pre_6 = clf_6.predict(label_6_loss_cur)

    for i in range(len(pre_6)):
        if pre_6[i] == 1:
            label_6_loss.append(label_6_loss_cur[i])

    # 第7类
    clf_7 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_7_loss_cur)
    pre_7 = clf_7.predict(label_7_loss_cur)

    for i in range(len(pre_7)):
        if pre_7[i] == 1:
            label_7_loss.append(label_7_loss_cur[i])

    # 第8类
    clf_8 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_8_loss_cur)
    pre_8 = clf_8.predict(label_8_loss_cur)

    for i in range(len(pre_8)):
        if pre_8[i] == 1:
            label_8_loss.append(label_8_loss_cur[i])

    # 第9类
    clf_9 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_9_loss_cur)
    pre_9 = clf_9.predict(label_9_loss_cur)

    for i in range(len(pre_9)):
        if pre_9[i] == 1:
            label_9_loss.append(label_9_loss_cur[i])

    # 第10类
    clf_10 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_10_loss_cur)
    pre_10 = clf_10.predict(label_10_loss_cur)

    for i in range(len(pre_10)):
        if pre_10[i] == 1:
            label_10_loss.append(label_10_loss_cur[i])

    # 第11类
    clf_11 = IsolationForest(random_state=0,contamination=0.04,max_samples=512).fit(label_11_loss_cur)
    pre_11 = clf_11.predict(label_11_loss_cur)

    for i in range(len(pre_11)):
        if pre_11[i] == 1:
            label_11_loss.append(label_11_loss_cur[i])

    # ==============================================保存选中的图片=========================================
    imgs = []
    # ==================================================针对第0类的采样=================================================


    selected_sampled_losses_0 = []

    ms_0 = MeanShift()
    ms_0.fit(label_0_loss)

    labels_0 = ms_0.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_0 = len(np.unique(labels_0))  # 第0类样本有几个簇

    need_sampled_cluster_0 = []  # 需要下采样的簇号
    need_drop_cluster_0 = []

    cluster_to_num_0 = Counter(labels_0)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_0:", cluster_to_num_0)

    for k in cluster_to_num_0.keys():
        if cluster_to_num_0[k] > len(labels_0) // n_clusters_0:
            need_sampled_cluster_0.append(k)

    need_sampled_losses_0 = [[] for _ in range(len(need_sampled_cluster_0))]

    for i in range(len(need_sampled_cluster_0)):
        for j in range(len(labels_0)):
            if labels_0[j] == need_sampled_cluster_0[i]:
                need_sampled_losses_0[i].append(label_0_loss[j][0])

    for j in range(len(labels_0)):
        if (labels_0[j] not in need_sampled_cluster_0):
            selected_sampled_losses_0.append(label_0_loss[j][0])

    for loss in need_sampled_losses_0:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_0.append(lo)
    print(len(label_0_loss))
    print(len(set(selected_sampled_losses_0)))
    print("raw 0:", max(label_0_loss), min(label_0_loss))
    print("sel 0:", max(set(selected_sampled_losses_0)), min(set(selected_sampled_losses_0)))

    for loss in list(set(selected_sampled_losses_0)):
        for img in label_0_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第1类的采样=================================================
    selected_sampled_losses_1 = []

    ms_1 = MeanShift()
    ms_1.fit(label_1_loss)

    labels_1 = ms_1.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_1 = len(np.unique(labels_1))  # 第0类样本有几个簇

    need_sampled_cluster_1 = []  # 需要下采样的簇号
    need_drop_cluster_1 = []

    cluster_to_num_1 = Counter(labels_1)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_1:", cluster_to_num_1)

    for k in cluster_to_num_1.keys():
        if cluster_to_num_1[k] > len(labels_1) // n_clusters_1:
            need_sampled_cluster_1.append(k)

    need_sampled_losses_1 = [[] for _ in range(len(need_sampled_cluster_1))]

    for i in range(len(need_sampled_cluster_1)):
        for j in range(len(labels_1)):
            if labels_1[j] == need_sampled_cluster_1[i]:
                need_sampled_losses_1[i].append(label_1_loss[j][0])

    for j in range(len(labels_1)):
        if (labels_1[j] not in need_sampled_cluster_1):
            selected_sampled_losses_1.append(label_1_loss[j][0])

    for loss in need_sampled_losses_1:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_1.append(lo)
    print(len(label_1_loss))
    print(len(set(selected_sampled_losses_1)))
    print("raw 1:", max(label_1_loss), min(label_1_loss))
    print("sel 1:", max(set(selected_sampled_losses_1)), min(set(selected_sampled_losses_1)))

    for loss in list(set(selected_sampled_losses_1)):
        for img in label_1_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第2类的采样=================================================
    selected_sampled_losses_2 = []

    ms_2 = MeanShift()
    ms_2.fit(label_2_loss)

    labels_2 = ms_2.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_2 = len(np.unique(labels_2))  # 第0类样本有几个簇

    need_sampled_cluster_2 = []  # 需要下采样的簇号
    need_drop_cluster_2 = []

    cluster_to_num_2 = Counter(labels_2)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_2:", cluster_to_num_2)

    for k in cluster_to_num_2.keys():
        if cluster_to_num_2[k] > len(labels_2) // n_clusters_2:
            need_sampled_cluster_2.append(k)

    need_sampled_losses_2 = [[] for _ in range(len(need_sampled_cluster_2))]

    for i in range(len(need_sampled_cluster_2)):
        for j in range(len(labels_2)):
            if labels_2[j] == need_sampled_cluster_2[i]:
                need_sampled_losses_2[i].append(label_2_loss[j][0])

    for j in range(len(labels_2)):
        if (labels_2[j] not in need_sampled_cluster_2):
            selected_sampled_losses_2.append(label_2_loss[j][0])

    for loss in need_sampled_losses_2:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_2.append(lo)
    print(len(label_2_loss))
    print(len(set(selected_sampled_losses_2)))
    print("raw 2:", max(label_2_loss), min(label_2_loss))
    print("sel 2:", max(set(selected_sampled_losses_2)), min(set(selected_sampled_losses_2)))

    for loss in list(set(selected_sampled_losses_2)):
        for img in label_2_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第3类的采样=================================================
    selected_sampled_losses_3 = []

    ms_3 = MeanShift()
    ms_3.fit(label_3_loss)

    labels_3 = ms_3.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_3 = len(np.unique(labels_3))  # 第0类样本有几个簇

    need_sampled_cluster_3 = []  # 需要下采样的簇号
    need_drop_cluster_3 = []

    cluster_to_num_3 = Counter(labels_3)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_3:", cluster_to_num_3)

    for k in cluster_to_num_3.keys():
        if cluster_to_num_3[k] > len(labels_3) // n_clusters_3:
            need_sampled_cluster_3.append(k)

    need_sampled_losses_3 = [[] for _ in range(len(need_sampled_cluster_3))]

    for i in range(len(need_sampled_cluster_3)):
        for j in range(len(labels_3)):
            if labels_3[j] == need_sampled_cluster_3[i]:
                need_sampled_losses_3[i].append(label_3_loss[j][0])

    for j in range(len(labels_3)):
        if (labels_3[j] not in need_sampled_cluster_3):
            selected_sampled_losses_3.append(label_3_loss[j][0])

    for loss in need_sampled_losses_3:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_3.append(lo)
    print(len(label_3_loss))
    print(len(set(selected_sampled_losses_3)))
    print("raw 3:", max(label_3_loss), min(label_3_loss))
    print("sel 3:", max(set(selected_sampled_losses_3)), min(set(selected_sampled_losses_3)))

    for loss in list(set(selected_sampled_losses_3)):
        for img in label_3_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第4类的采样=================================================
    selected_sampled_losses_4 = []

    ms_4 = MeanShift()
    ms_4.fit(label_4_loss)

    labels_4 = ms_4.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_4 = len(np.unique(labels_4))  # 第0类样本有几个簇

    need_sampled_cluster_4 = []  # 需要下采样的簇号
    need_drop_cluster_4 = []

    cluster_to_num_4 = Counter(labels_4)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_4:", cluster_to_num_4)

    for k in cluster_to_num_4.keys():
        if cluster_to_num_4[k] > len(labels_4) // n_clusters_4:
            need_sampled_cluster_4.append(k)

    need_sampled_losses_4 = [[] for _ in range(len(need_sampled_cluster_4))]

    for i in range(len(need_sampled_cluster_4)):
        for j in range(len(labels_4)):
            if labels_4[j] == need_sampled_cluster_4[i]:
                need_sampled_losses_4[i].append(label_4_loss[j][0])

    for j in range(len(labels_4)):
        if (labels_4[j] not in need_sampled_cluster_4):
            selected_sampled_losses_4.append(label_4_loss[j][0])

    for loss in need_sampled_losses_4:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_4.append(lo)
    print(len(label_4_loss))
    print(len(set(selected_sampled_losses_4)))
    print("raw 4:", max(label_4_loss), min(label_4_loss))
    print("sel 4:", max(set(selected_sampled_losses_4)), min(set(selected_sampled_losses_4)))

    for loss in list(set(selected_sampled_losses_4)):
        for img in label_4_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第5类的采样=================================================
    selected_sampled_losses_5 = []

    ms_5 = MeanShift()
    ms_5.fit(label_5_loss)

    labels_5 = ms_5.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_5 = len(np.unique(labels_5))  # 第0类样本有几个簇

    need_sampled_cluster_5 = []  # 需要下采样的簇号
    need_drop_cluster_5 = []

    cluster_to_num_5 = Counter(labels_5)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_5:", cluster_to_num_5)

    for k in cluster_to_num_5.keys():
        if cluster_to_num_5[k] > len(labels_5) // n_clusters_5:
            need_sampled_cluster_5.append(k)

    need_sampled_losses_5 = [[] for _ in range(len(need_sampled_cluster_5))]

    for i in range(len(need_sampled_cluster_5)):
        for j in range(len(labels_5)):
            if labels_5[j] == need_sampled_cluster_5[i]:
                need_sampled_losses_5[i].append(label_5_loss[j][0])

    for j in range(len(labels_5)):
        if (labels_5[j] not in need_sampled_cluster_5):
            selected_sampled_losses_5.append(label_5_loss[j][0])

    for loss in need_sampled_losses_5:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_5.append(lo)
    print(len(label_5_loss))
    print(len(set(selected_sampled_losses_5)))
    print("raw 5:", max(label_5_loss), min(label_5_loss))
    print("sel 5:", max(set(selected_sampled_losses_5)), min(set(selected_sampled_losses_5)))

    for loss in list(set(selected_sampled_losses_5)):
        for img in label_5_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第6类的采样=================================================
    selected_sampled_losses_6 = []

    ms_6 = MeanShift()
    ms_6.fit(label_6_loss)

    labels_6 = ms_6.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_6 = len(np.unique(labels_6))  # 第0类样本有几个簇

    need_sampled_cluster_6 = []  # 需要下采样的簇号
    need_drop_cluster_6 = []

    cluster_to_num_6 = Counter(labels_6)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_6:", cluster_to_num_6)

    for k in cluster_to_num_6.keys():
        if cluster_to_num_6[k] > len(labels_6) // n_clusters_6:
            need_sampled_cluster_6.append(k)

    need_sampled_losses_6 = [[] for _ in range(len(need_sampled_cluster_6))]

    for i in range(len(need_sampled_cluster_6)):
        for j in range(len(labels_6)):
            if labels_6[j] == need_sampled_cluster_6[i]:
                need_sampled_losses_6[i].append(label_6_loss[j][0])

    for j in range(len(labels_6)):
        if (labels_6[j] not in need_sampled_cluster_6):
            selected_sampled_losses_6.append(label_6_loss[j][0])

    for loss in need_sampled_losses_6:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_6.append(lo)
    print(len(label_6_loss))
    print(len(set(selected_sampled_losses_6)))
    print("raw 6:", max(label_6_loss), min(label_6_loss))
    print("sel 6:", max(set(selected_sampled_losses_6)), min(set(selected_sampled_losses_6)))

    for loss in list(set(selected_sampled_losses_6)):
        for img in label_6_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第7类的采样=================================================
    selected_sampled_losses_7 = []

    ms_7 = MeanShift()
    ms_7.fit(label_7_loss)

    labels_7 = ms_7.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_7 = len(np.unique(labels_7))  # 第0类样本有几个簇

    need_sampled_cluster_7 = []  # 需要下采样的簇号
    need_drop_cluster_7 = []

    cluster_to_num_7 = Counter(labels_7)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_7:", cluster_to_num_7)

    for k in cluster_to_num_7.keys():
        if cluster_to_num_7[k] > len(labels_7) // n_clusters_7:
            need_sampled_cluster_7.append(k)

    need_sampled_losses_7 = [[] for _ in range(len(need_sampled_cluster_7))]

    for i in range(len(need_sampled_cluster_7)):
        for j in range(len(labels_7)):
            if labels_7[j] == need_sampled_cluster_7[i]:
                need_sampled_losses_7[i].append(label_7_loss[j][0])

    for j in range(len(labels_7)):
        if (labels_7[j] not in need_sampled_cluster_7):
            selected_sampled_losses_7.append(label_7_loss[j][0])

    for loss in need_sampled_losses_7:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_7.append(lo)
    print(len(label_7_loss))
    print(len(set(selected_sampled_losses_7)))
    print("raw 7:", max(label_7_loss), min(label_7_loss))
    print("sel 7:", max(set(selected_sampled_losses_7)), min(set(selected_sampled_losses_7)))

    for loss in list(set(selected_sampled_losses_7)):
        for img in label_7_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第8类的采样=================================================
    selected_sampled_losses_8 = []

    ms_8 = MeanShift()
    ms_8.fit(label_8_loss)

    labels_8 = ms_8.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_8 = len(np.unique(labels_8))  # 第0类样本有几个簇

    need_sampled_cluster_8 = []  # 需要下采样的簇号
    need_drop_cluster_8 = []

    cluster_to_num_8 = Counter(labels_8)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_8:", cluster_to_num_8)

    for k in cluster_to_num_8.keys():
        if cluster_to_num_8[k] > len(labels_8) // n_clusters_8:
            need_sampled_cluster_8.append(k)

    need_sampled_losses_8 = [[] for _ in range(len(need_sampled_cluster_8))]

    for i in range(len(need_sampled_cluster_8)):
        for j in range(len(labels_8)):
            if labels_8[j] == need_sampled_cluster_8[i]:
                need_sampled_losses_8[i].append(label_8_loss[j][0])

    for j in range(len(labels_8)):
        if (labels_8[j] not in need_sampled_cluster_8):
            selected_sampled_losses_8.append(label_8_loss[j][0])

    for loss in need_sampled_losses_8:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_8.append(lo)
    print(len(label_8_loss))
    print(len(set(selected_sampled_losses_8)))
    print("raw 8:", max(label_8_loss), min(label_8_loss))
    print("sel 8:", max(set(selected_sampled_losses_8)), min(set(selected_sampled_losses_8)))

    for loss in list(set(selected_sampled_losses_8)):
        for img in label_8_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第9类的采样=================================================
    selected_sampled_losses_9 = []

    ms_9 = MeanShift()
    ms_9.fit(label_9_loss)

    labels_9 = ms_9.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_9 = len(np.unique(labels_9))  # 第0类样本有几个簇

    need_sampled_cluster_9 = []  # 需要下采样的簇号
    need_drop_cluster_9 = []

    cluster_to_num_9 = Counter(labels_9)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_9:", cluster_to_num_9)

    for k in cluster_to_num_9.keys():
        if cluster_to_num_9[k] > len(labels_9) // n_clusters_9:
            need_sampled_cluster_9.append(k)

    need_sampled_losses_9 = [[] for _ in range(len(need_sampled_cluster_9))]

    for i in range(len(need_sampled_cluster_9)):
        for j in range(len(labels_9)):
            if labels_9[j] == need_sampled_cluster_9[i]:
                need_sampled_losses_9[i].append(label_9_loss[j][0])

    for j in range(len(labels_9)):
        if (labels_9[j] not in need_sampled_cluster_9):
            selected_sampled_losses_9.append(label_9_loss[j][0])

    for loss in need_sampled_losses_9:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_9.append(lo)
    print(len(label_9_loss))
    print(len(set(selected_sampled_losses_9)))
    print("raw 9:", max(label_9_loss), min(label_9_loss))
    print("sel 9:", max(set(selected_sampled_losses_9)), min(set(selected_sampled_losses_9)))

    for loss in list(set(selected_sampled_losses_9)):
        for img in label_9_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第10类的采样=================================================
    selected_sampled_losses_10 = []

    ms_10 = MeanShift()
    ms_10.fit(label_10_loss)

    labels_10 = ms_10.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_10 = len(np.unique(labels_10))  # 第0类样本有几个簇

    need_sampled_cluster_10 = []  # 需要下采样的簇号
    need_drop_cluster_10 = []

    cluster_to_num_10 = Counter(labels_10)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_10:", cluster_to_num_10)

    for k in cluster_to_num_10.keys():
        if cluster_to_num_10[k] > len(labels_10) // n_clusters_10:
            need_sampled_cluster_10.append(k)

    need_sampled_losses_10 = [[] for _ in range(len(need_sampled_cluster_10))]

    for i in range(len(need_sampled_cluster_10)):
        for j in range(len(labels_10)):
            if labels_10[j] == need_sampled_cluster_10[i]:
                need_sampled_losses_10[i].append(label_10_loss[j][0])

    for j in range(len(labels_10)):
        if (labels_10[j] not in need_sampled_cluster_10):
            selected_sampled_losses_10.append(label_10_loss[j][0])

    for loss in need_sampled_losses_10:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_10.append(lo)
    print(len(label_10_loss))
    print(len(set(selected_sampled_losses_10)))
    print("raw 10:", max(label_10_loss), min(label_10_loss))
    print("sel 10:", max(set(selected_sampled_losses_10)), min(set(selected_sampled_losses_10)))

    for loss in list(set(selected_sampled_losses_10)):
        for img in label_10_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))

    # ==================================================针对第11类的采样=================================================
    selected_sampled_losses_11 = []

    ms_11 = MeanShift()
    ms_11.fit(label_11_loss)

    labels_11 = ms_11.labels_  # 第0类中的每个样本对应的簇号

    n_clusters_11 = len(np.unique(labels_11))  # 第0类样本有几个簇

    need_sampled_cluster_11 = []  # 需要下采样的簇号
    need_drop_cluster_11 = []

    cluster_to_num_11 = Counter(labels_11)  # 每一个簇对应的个数，字典形式
    print("cluster_to_num_11:", cluster_to_num_11)

    for k in cluster_to_num_11.keys():
        if cluster_to_num_11[k] > len(labels_11) // n_clusters_11:
            need_sampled_cluster_11.append(k)

    need_sampled_losses_11 = [[] for _ in range(len(need_sampled_cluster_11))]

    for i in range(len(need_sampled_cluster_11)):
        for j in range(len(labels_11)):
            if labels_11[j] == need_sampled_cluster_11[i]:
                need_sampled_losses_11[i].append(label_11_loss[j][0])

    for j in range(len(labels_11)):
        if (labels_11[j] not in need_sampled_cluster_11):
            selected_sampled_losses_11.append(label_11_loss[j][0])

    for loss in need_sampled_losses_11:
        loss = np.array(loss)
        # print(len(loss))
        sampled_loss = MC(loss, T, loss.mean(), loss.std()).MCMC()  # 采样出来的loss
        # print(len(set(sampled_loss)))

        for lo in sampled_loss:
            selected_sampled_losses_11.append(lo)
    print(len(label_11_loss))
    print(len(set(selected_sampled_losses_11)))
    print("raw 11:", max(label_11_loss), min(label_11_loss))
    print("sel 11:", max(set(selected_sampled_losses_11)), min(set(selected_sampled_losses_11)))

    for loss in list(set(selected_sampled_losses_11)):
        for img in label_11_loss_to_img[loss]:
            imgs.append(img)

    print(len(imgs))
    f = open("./results/selected_imgs_by_mcmc/new_new_new_selected_" + str(round(len(set(imgs)) / 108032, 2)) + "_images.txt", mode='w+',
             encoding='utf-8')
    for img_name in set(imgs):
        f.write(img_name)
        f.write('\n')
    f.close()
