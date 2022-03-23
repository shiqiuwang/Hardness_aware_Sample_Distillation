# sample_distillation

source code on SOP dataset in paper:Hardness-Aware Sample Distillation for Efficient Model Training

## Instructions for use

### data preparation

1. The original dataset and pre-training related files are in the release. First, you need to clone the source code to the local, and then go to release to download the relevant data and files and unzip them to the root directory.
2. you need to create the new folder named spld_pkl,then put the resnet_84.98.pkl and resnet_85.07.pkl which are uploaded in the realease.
3. you need to create the new folder named datal,then put the alexnet-owt-4df8aa71.pth,inception_v3_google-1a9a5a14.pth,vgg16-397923af.pth, resnet50-19c8e357.pth which are uploaded in the realease. meanwhile,you'd better create the new folder named std_split_data,and unzip the train_0_5.rar,train_6_11.rar and valid.rar,after that,merge train_0-5 and train_6_11 to the folder train,and put the train and valid floder to the std_split_data floder.


### Related code file introduction

focalloss.py :It is the implementation code of focalloss function

SPLD.py:it is the implementation code of Self-paced learning function

mcmc.py:it is the Implementation of Monte Carlo Algorithm

selected_by_mcmc.py: the source code of Sampling the original data set with Monte Carlo

spld_train.py:Use reverse self-paced learning and focalloss for model training on the original data set

alexnet_train_imgs_by_mcmc.py:Train alexnet with the sampled data set

### run

training the model with reverse self-learning,run the following command

```python
python spld_train.py
```



sampleing data, run the following command

```python
python selected_by_mcmc.py
```

Useing the sampled data to train the model run the following command

```python
python alexnet_train_imgs_by_mcmc.py
```
