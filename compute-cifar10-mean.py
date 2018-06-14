#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torchvision.models as models

import os
import sys
import math

from myDataset import myImageFolder

import numpy as np

from os.path import join as ospj

dataPath = '/home/huangzesang/data/CIFAR10/png'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
traindata =  myImageFolder(
                root=ospj(dataPath,'train'),
                classes=classes,
                label=ospj(dataPath,'../train.list'),
                transform=transforms.ToTensor())
data = np.stack([t.numpy() for t,c in traindata])
print(data.shape)
means = []
stdevs = []
for i in range(3):
    pixels = data[:,i,:,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))


# data = data.astype(np.float32)/255.
# data = dset.CIFAR10(root='cifar', train=True, download=True,
#                     transform=transforms.ToTensor()).train_data
# means = []
# stdevs = []
# for i in range(3):
#     pixels = data[:,i,:,:].ravel()
#     means.append(np.mean(pixels))
#     stdevs.append(np.std(pixels))

# print("means: {}".format(means))
# print("stdevs: {}".format(stdevs))
# print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
