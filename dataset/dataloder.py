import os
import sys
import time
import random
import scipy.io as sio
import numpy as np
sys.path.append("..")
from util import dsp, util
from util import array_operation as arr
from dataset import transforms
import statistics


def segment_train_eval_dataset(signals,labels,a=0.8,random=True):
    length = len(labels)
    if random:
        transforms.shuffledata(signals, labels)
        signals_train = signals[:int(a*length)]
        labels_train = labels[:int(a*length)]
        signals_eval = signals[int(a*length):]
        labels_eval = labels[int(a*length):]
    else:
        label_cnt,label_cnt_per,label_num = statistics.label_statistics(labels)
        cnt = 0
        for i in range(label_num):
            if i ==0:
                signals_train = signals[cnt:cnt+int(label_cnt[i]*0.8)]
                labels_train =  labels[cnt:cnt+int(label_cnt[i]*0.8)]
                signals_eval =  signals[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]
                labels_eval =   labels[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]
            else:
                signals_train = np.concatenate((signals_train, signals[cnt:cnt+int(label_cnt[i]*0.8)]))
                labels_train = np.concatenate((labels_train, labels[cnt:cnt+int(label_cnt[i]*0.8)]))

                signals_eval = np.concatenate((signals_eval, signals[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]))
                labels_eval = np.concatenate((labels_eval, labels[cnt+int(label_cnt[i]*0.8):cnt+label_cnt[i]]))
            cnt += label_cnt[i]
    return signals_train,labels_train,signals_eval,labels_eval


# 预处理
def preprocess(signals,index, use_filter=False,filter="wave_filter"):
    num, ch = signals.shape[:2]
    signal = signals[index].copy()
    fs = 1000

    # normliaze
    for i in range(ch):
        signal[i] = arr.normliaze(signal[i], mode='z-score', truncated=1e2)
    # # 滤波器
    # for i in range(ch):
    #     signal[i] = dsp.fft_filter(signal[i], fs, fc=[])
    
    if use_filter != False:
        match filter:
        
            case "wave_filter":
                for i in range(ch):
                    signal[i] = dsp.wave_filter(signal=signal[i], wave='db1',level=2,usedcoeffs=[1,1])


    return signal


def loaddastset(data_path, label_path):
    signals = np.load(data_path)
    labels = np.load(label_path)

    for i in range(signals.shape[0]):
        signals[i] = preprocess(signals, i)

    transforms.shuffledata(signals, labels)
    return signals.astype(np.float32), labels.astype(np.int64)
