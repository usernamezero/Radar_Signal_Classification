import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from util import util,options
from data import augmenter,transforms,dataloader,statistics
from models import creatnet


opt = options.Options().getparse()
net,exp = creatnet.creatnet(opt)

#load data
signals = np.load('./datasets/simple_test/signals.npy')
labels = np.load('./datasets/simple_test/labels.npy')

#load prtrained_model
net.load_state_dict(torch.load('./checkpoints/pretrained/micro_multi_scale_resnet_1d_50class.pth'))
net.eval()
if opt.gpu_id != '-1' and len(opt.gpu_id) == 1:
    net.cuda()
elif opt.gpu_id != '-1' and len(opt.gpu_id) > 1:
    net = nn.DataParallel(net)
    net.cuda()

for signal,true_label in zip(signals, labels):
    signal = signal.reshape(1,1,-1).astype(np.float32) #batchsize,ch,length
    true_label = true_label.reshape(1).astype(np.int64) #batchsize
    signal,true_label = transforms.ToTensor(signal,true_label,gpu_id =opt.gpu_id)
    out = net(signal)
    pred_label = torch.max(out, 1)[1]
    pred_label=pred_label.data.cpu().numpy()
    true_label=true_label.data.cpu().numpy()
    print(("true:{0:d} predict:{1:d}").format(true_label[0],pred_label[0]))
