import os
import sys
import time
import random
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    # init 函数用于载入numpy数据并将其转化为相应的tensor
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label)
        self.length = label.shape[0]

    # __getitem__函数用于定义训练时会返回的单个数据与标签
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    # __len__表示数据数量m
    def __len__(self):
        return self.length

