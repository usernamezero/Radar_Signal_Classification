import numpy as np
import os
import sys
import torch
sys.path.append("..")

from util import util

def label_statistics(labels):
    labels = (np.array(labels)).astype(np.int64)
    label_num = np.max(labels)+1
    label_cnt = np.zeros(label_num,dtype=np.int64)
    for i in range(len(labels)):
        label_cnt[labels[i]] += 1
    label_cnt_per = label_cnt/len(labels)
    return label_cnt,label_cnt_per,label_num