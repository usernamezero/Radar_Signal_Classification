import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
# 数据路径
data_path = r"X:\new_radar\data\ML\SNR1\data.csv"
data = pd.read_csv(data_path)
# 数据处理，获取数据集的SNR及label列表，同时在训练数据中丢弃他们
SNR = data.SNR[0]
label = data.label
data_drop_SNR = data.drop(['SNR'], axis=1)
train_data_drop_label = data_drop_SNR.drop(['label'], axis=1)
# 划分训练集，测试集，train:test = 0.75:0.25，是否在分割前对完整数据进行洗牌（打乱），默认为True，打乱
X_train, X_test, y_train, y_test = train_test_split(train_data_drop_label, label, test_size=0.25)



# def get_acc_rec_pre_f(y_true, y_pred, beta=1.0):
#     (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
#     p1, p2 = tp / (tp + fp), tn / (tn + fn)
#     r1, r2 = tp / (tp + fn), tn / (tn + fp)
#     f_beta1 = (1 + beta ** 2) * p1 * r1 / (beta ** 2 * p1 + r1)
#     f_beta2 = (1 + beta ** 2) * p2 * r2 / (beta ** 2 * p2 + r2)
#     m_p, m_r, m_f = 0.5 * (p1+p2), 0.5 * (r1+r2), 0.5 * (f_beta1 + f_beta2)
#     count = np.bincount(y_true)
#     w1, w2 = count[1]/sum(count), count[0]/sum(count)# 计算加权平均
#     w_p, w_r,w_f=w1 * p1+w2 * p2, w1 * r1+w2 * r2, w1 *f_beta1+w2 * f_beta2
#     print(f"算术平均： 精确率：{m_p},召回率：{m_r},F值：{m_f}")
#     print(f"加权平均：精确率：{w_p},召回率：{w_r},F值：{w_f}")


start=time.time()
# solver=拟牛顿法
logistic = LogisticRegression(penalty='l2',C=1,solver='lbfgs',max_iter=1000,multi_class='ovr')
train_prediction = logistic.fit(X_train,y_train).predict(X_train)
# 添加一些指标
y_pred = logistic.predict(X_test)
# predict_proba的返回值是一个矩阵，
# 矩阵的index是对应第几个样本，columns对应第几个标签，矩阵中的数字则是第i个样本的标签为j的概率值。区别于predict直接返回标签值。
y_probas = logistic.predict_proba(X_test)
# skplt.metrics.plot_roc(y_test,y_probas.astype('int'))
skplt.metrics.plot_precision_recall(y_test,y_probas)
# skplt.metrics.plot_roc(y_test,y_probas)
# plt.savefig("../imgs/roc.jpg")
plt.savefig("../imgs/PR.jpg")
plt.show()
# skplt.metrics.plot_confusion_matrix(y_test,y_pred,normalize=True)
# plt.savefig("../imgs/confusion_matrix.jpg")
# plt.show()

print("准确率：",logistic.score(X_train,y_train))
# get_acc_rec_pre_f(y_train, y_pred)
print(y_train.shape,y_pred.shape)
test_prediction=logistic.fit(X_train,y_train).predict(X_test)
print("Logistic训练模型评分："+str(accuracy_score(y_train,train_prediction)))
print("Logistic待测模型评分："+str(accuracy_score(y_test,test_prediction)))
end=time.time()
print ("运行时间："+str(end-start))