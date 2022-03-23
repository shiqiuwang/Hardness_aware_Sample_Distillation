# -*- coding: utf-8 -*-

'''
@Time    : 2021/8/22 10:29
@Author  : Qiushi Wang
@FileName: count_selected_and_no_selected.py
@Software: PyCharm
'''
from PIL import Image

selected_imgs = []
all_imgs = []

all_imgs_loss = {}

with open("./results/spld_loss/alexnet_train_loss_file.txt", mode='r', encoding='utf-8') as f:
    for line in f.readlines():
        img, loss = line.strip().split(":")


        all_imgs.append(img)
        all_imgs_loss[img] = float(loss)

with open("./results/selected_imgs_by_mcmc/selected_0.51_images.txt", mode='r', encoding='utf-8') as f:
    for img in f.readlines():
        selected_imgs.append(img.strip())

no_selected_imgs = list(set(all_imgs) - set(selected_imgs))

# ===============================================选中的样本=============================================

selected_imgs_0 = []
selected_imgs_1 = []
selected_imgs_2 = []
selected_imgs_3 = []
selected_imgs_4 = []
selected_imgs_5 = []
selected_imgs_6 = []
selected_imgs_7 = []
selected_imgs_8 = []
selected_imgs_9 = []
selected_imgs_10 = []
selected_imgs_11 = []


for img in selected_imgs:
    if img.split("/")[4] == '0':
        selected_imgs_0.append(img)
    if img.split("/")[4] == '1':
        selected_imgs_1.append(img)
    if img.split("/")[4] == '2':
        selected_imgs_2.append(img)
    if img.split("/")[4] == '3':
        selected_imgs_3.append(img)
    if img.split("/")[4] == '4':
        selected_imgs_4.append(img)
    if img.split("/")[4] == '5':
        selected_imgs_5.append(img)
    if img.split("/")[4] == '6':
        selected_imgs_6.append(img)
    if img.split("/")[4] == '7':
        selected_imgs_7.append(img)
    if img.split("/")[4] == '8':
        selected_imgs_8.append(img)
    if img.split("/")[4] == '9':
        selected_imgs_9.append(img)

    if img.split("/")[4] == '10':
        selected_imgs_10.append(img)
    if img.split("/")[4] == '11':
        selected_imgs_11.append(img)


selected_imgs_0.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_1.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_2.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_3.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_4.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_5.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_6.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_7.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_8.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_9.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_10.sort(key=lambda x: all_imgs_loss[x])
selected_imgs_11.sort(key=lambda x: all_imgs_loss[x])


for img_path in selected_imgs_0:
    img = Image.open(img_path)
    img.save('./results/case_study/0/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_1:
    img = Image.open(img_path)
    img.save('./results/case_study/1/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_2:
    img = Image.open(img_path)
    img.save('./results/case_study/2/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_3:
    img = Image.open(img_path)
    img.save('./results/case_study/3/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_4:
    img = Image.open(img_path)
    img.save('./results/case_study/4/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_5:
    img = Image.open(img_path)
    img.save('./results/case_study/5/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_6:
    img = Image.open(img_path)
    img.save('./results/case_study/6/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_7:
    img = Image.open(img_path)
    img.save('./results/case_study/7/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_8:
    img = Image.open(img_path)
    img.save('./results/case_study/8/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_9:
    img = Image.open(img_path)
    img.save('./results/case_study/9/selected/' + img_path.split("/")[5])
    
    
for img_path in selected_imgs_10:
    img = Image.open(img_path)
    img.save('./results/case_study/10/selected/' + img_path.split("/")[5])
for img_path in selected_imgs_11:
    img = Image.open(img_path)
    img.save('./results/case_study/11/selected/' + img_path.split("/")[5])

# ==============================================未选中的样本=============================================

no_selected_imgs_0 = []
no_selected_imgs_1 = []
no_selected_imgs_2 = []
no_selected_imgs_3 = []
no_selected_imgs_4 = []
no_selected_imgs_5 = []
no_selected_imgs_6 = []
no_selected_imgs_7 = []
no_selected_imgs_8 = []
no_selected_imgs_9 = []
no_selected_imgs_10 = []
no_selected_imgs_11 = []


for img in no_selected_imgs:
    if img.split("/")[4] == '0':
        no_selected_imgs_0.append(img)
    if img.split("/")[4] == '1':
        no_selected_imgs_1.append(img)
    if img.split("/")[4] == '2':
        no_selected_imgs_2.append(img)
    if img.split("/")[4] == '3':
        no_selected_imgs_3.append(img)
    if img.split("/")[4] == '4':
        no_selected_imgs_4.append(img)
    if img.split("/")[4] == '5':
        no_selected_imgs_5.append(img)
    if img.split("/")[4] == '6':
        no_selected_imgs_6.append(img)
    if img.split("/")[4] == '7':
        no_selected_imgs_7.append(img)
    if img.split("/")[4] == '8':
        no_selected_imgs_8.append(img)
    if img.split("/")[4] == '9':
        no_selected_imgs_9.append(img)

    if img.split("/")[4] == '10':
        no_selected_imgs_10.append(img)
    if img.split("/")[4] == '11':
        no_selected_imgs_11.append(img)


no_selected_imgs_0.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_1.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_2.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_3.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_4.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_5.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_6.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_7.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_8.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_9.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_10.sort(key=lambda x: all_imgs_loss[x])
no_selected_imgs_11.sort(key=lambda x: all_imgs_loss[x])


for img_path in no_selected_imgs_0:
    img = Image.open(img_path)
    img.save('./results/case_study/0/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_1:
    img = Image.open(img_path)
    img.save('./results/case_study/1/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_2:
    img = Image.open(img_path)
    img.save('./results/case_study/2/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_3:
    img = Image.open(img_path)
    img.save('./results/case_study/3/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_4:
    img = Image.open(img_path)
    img.save('./results/case_study/4/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_5:
    img = Image.open(img_path)
    img.save('./results/case_study/5/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_6:
    img = Image.open(img_path)
    img.save('./results/case_study/6/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_7:
    img = Image.open(img_path)
    img.save('./results/case_study/7/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_8:
    img = Image.open(img_path)
    img.save('./results/case_study/8/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_9:
    img = Image.open(img_path)
    img.save('./results/case_study/9/no_selected/' + img_path.split("/")[5])

for img_path in no_selected_imgs_10:
    img = Image.open(img_path)
    img.save('./results/case_study/10/no_selected/' + img_path.split("/")[5])
for img_path in no_selected_imgs_11:
    img = Image.open(img_path)
    img.save('./results/case_study/11/no_selected/' + img_path.split("/")[5])
