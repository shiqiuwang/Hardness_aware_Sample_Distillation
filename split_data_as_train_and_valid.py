# -*- coding: utf-8 -*-

'''
@Time    : 2021/3/21 12:56
@Author  : Qiushi Wang
@FileName: split_imagenet_data.py
@Software: PyCharm
'''

import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    # dataset_dir = os.path.join("..", "..", "data", "imagenet_data")
    dataset_dir = "./data/product_source_data"
    split_dir = "./data/std_split_data"
    # split_dir = os.path.join("..", "..", "data", "imagenet_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.9
    valid_pct = 0.1
    test_pct = 0.0

    for root, dirs, files in os.walk(dataset_dir):
        # print(root, dirs, files)
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.JPG'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point - train_point,
                                                                 img_count - valid_point))
