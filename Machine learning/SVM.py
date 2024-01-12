
from sklearn.datasets import make_blobs
from sklearn import svm
import numpy as np
import torch
import numpy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
from torchvision import transforms, utils



class SVMDataset(Dataset):
    def __init__(self,path):
        super(SVMDataset,self).__init__()


    def __getitem__(self, index):
        pass