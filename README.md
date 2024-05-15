<div align="center">    

</div>

# Radar_Signal_Classification


一个通用的一维时序信号分析,分类框架.<br>
它将包含多种网络结构，并提供数据预处理,数据增强,训练,评估,测试等功能.<br>

## 支持的功能
### 数据预处理
通用的数据预处理方法
* Normliaze  :   5_95 | maxmin | None
* Filter           :   fft | fir | iir | wavelet | None

### 数据增强
多种多样的数据增强方法.注意:使用时应该结合数据的物理特性进行选择.<br>[[Time Series Data Augmentation for Deep Learning: A Survey]](https://arxiv.org/pdf/2002.12478.pdf)
* Base     :  scale, warp, app, aaft, iaaft, filp, crop
* Noise   :  spike, step, slope, white, pink, blue, brown, violet
* Gan      :  dcgan

### 网络
提供多种用于评估的网络.
>1d
>
>>lstm, cnn_1d, resnet18_1d, resnet34_1d, multi_scale_resnet_1d, micro_multi_scale_resnet_1d,autoencoder,mlp

>2d(stft spectrum)
>
>>mobilenet, resnet18, resnet50, resnet101, densenet121, densenet201, squeezenet, dfcnn, multi_scale_resnet,

### K-fold
使用K-fold使得结果更加可靠．
```--k_fold```&```--fold_index```<br>

* --k_fold
```python
# fold_num of k-fold. If 0 or 1, no k-fold and cut 80% to train and other to eval.
```
* --fold_index
```python
"""--fold_index
When --k_fold != 0 or 1:
Cut dataset into sub-set using index , and then run k-fold with sub-set
If input 'auto', it will shuffle dataset and then cut dataset equally
If input: [2,4,6,7]
when len(dataset) == 10
sub-set: dataset[0:2],dataset[2:4],dataset[4:6],dataset[6:7],dataset[7:]
-------
When --k_fold == 0 or 1:
If input 'auto', it will shuffle dataset and then cut 80% dataset to train and other to eval
If input: [5]
when len(dataset) == 10
train-set : dataset[0:5]  eval-set : dataset[5:]
"""
```

## 入门
### 前提要求
- Linux, Windows,mac
- CPU or NVIDIA GPU + CUDA CuDNN
- Python 3.10
- Pytroch 2.0+
### 依赖
This code depends on torchvision, numpy, scipy , PyWavelets, matplotlib, available via pip install.<br>
For example:<br>

```bash
pip3 install matplotlib
```
### 克隆仓库:
```bash
git clone https://github.com/usernamezero/Radar_Signal_Classification
```

### 训练
```bash
python3 train.py --label 50 --input_nc 1 --dataset_dir ./datasets/simple_test --save_dir ./checkpoints/simple_test --model_name micro_multi_scale_resnet_1d --gpu_id 0 --batchsize 64 --k_fold 5
# 如果需要使用cpu进行训练, 请输入 --gpu_id -1
```

## 使用自己的数据进行训练
* step1: 按照如下格式生成 signals.npy 以及 labels.npy.
```python
#1.type:numpydata   signals:np.float64   labels:np.int64
#2.shape  signals:[num,ch,length]    labels:[num]
#num:batch_size, ch :channel_num,  length:length of siganl
#for example:
signals = np.zeros((10,1,3000),dtype='np.float64')
labels = np.array([0,0,0,0,0,1,1,1,1,1])      #0->class0    1->class1
```
* step2: 新增参数  ```--dataset_dir "your_dataset_dir"``` 


* 更多可选参数 [options](./util/options.py).
### 测试
```bash
python3 simple_test.py --label 50 --input_nc 1 --model_name micro_multi_scale_resnet_1d --gpu_id 0
# 如果需要使用cpu进行训练, 请输入 --gpu_id -1
```
### how to load results
Use```torch.load(path)```to load results.pth<br>
Just like a Dict.<br>

```python
    results:{
        0:{                                    #dict,index->fold
            'F1':[0.1,0.2...],                 #list,index->epoch
            'err':[0.9,0.8...],                #list,index->epoch
            'loss':[1.1,1.0...],               #list,index->epoch
            'confusion_mat':[
                [[1204  133  763  280]
                 [ 464  150  477  152]
                 [ 768   66 1276  308]
                 [ 159   23  293 2145]],
                 [[2505  251 1322  667]
                 [1010  283  834  353]
                 [1476  174 2448  766]
                 [ 376   46  446 4365]],
                 ......
            ],                                 #list,index->epoch
            'eval_detail':[                    #list,index->epoch
                {
                    'sequences':[],
                    'ture_labels':[],
                    'pre_labels':[]
                },
                {
                    'sequences':[],
                    'ture_labels':[],
                    'pre_labels':[]
                }
                ...
            ], 
            'best_epoch':0                     #int
        }
        1:{

        ...

        }
    }
```



### [ More options](./util/options.py).