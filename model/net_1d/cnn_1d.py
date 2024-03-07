import torch
from torch import nn
import torch.nn.functional as F

'''
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度
out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
stride(int or tuple, optional) - 卷积步长
padding (int or tuple, optional)- 输入的每一条边补充0的层数
dilation(int or tuple, `optional``) – 卷积核元素之间的间距
groups(int, optional) – 从输入通道到输出通道的阻塞连接数
bias(bool, optional) - 如果bias=True，添加偏置
'''


# print(model.summary()) # 显示网络结构

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 8, 1, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 16, 1, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 32, 1, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, 64, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv1d(512, 1024, 64, 1, 0, bias=False),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(inplace=True),
        # )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.avgpool(x)
        # 将前面多维度的tensor展成一维
        x = x.view(x.size(0), -1)
        # x = x.flatten(x, 1)
        x = self.out(x)
        return x


# cnn = CNN()
# print(cnn)
# input = torch.ones(1, 2000)
# output = cnn(input)
# print(output.shape)
