import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB  # 伯努利贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯分类器
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB  # 高斯分类器
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif': ['SimHei', 'Arial']})  # 设定汉字字体，防止出现方框

# 数据路径
data_path = r"../data/ML/SNR1/data.csv"
data = pd.read_csv(data_path)
# 数据处理，获取数据集的SNR及label列表，同时在训练数据中丢弃他们
SNR = data.SNR[0]
label = data.label
data_drop_SNR = data.drop(['SNR'], axis=1)
train_data_drop_label = data_drop_SNR.drop(['label'], axis=1)
# 划分训练集，测试集，train:test = 0.75:0.25，是否在分割前对完整数据进行洗牌（打乱），默认为True，打乱
X_train, X_test, y_train, y_test = train_test_split(train_data_drop_label, label, test_size=0.25)


# 引入伯努利分类器，开始训练
def Bernouli():
    start = time.time()
    clf = BernoulliNB(alpha=10)
    train_prediction = clf.fit(X_train, y_train).predict(X_train)
    test_prediction = clf.fit(X_train, y_train).predict(X_test)
    print("伯努利贝叶斯训练模型评分：" + str(accuracy_score(y_train, train_prediction)))
    print("伯努利贝叶斯待测模型评分：" + str(accuracy_score(y_test, test_prediction)))
    end = time.time()
    print("运行时间：" + str(end - start))  # 时间单位是秒


def Multinomials():
    start = time.time()
    mlt = MultinomialNB(alpha=150)
    train_p = mlt.fit(X_train, y_train).predict(X_train)
    test_p = mlt.fit(X_train, y_train).predict(X_train)
    print("多项式贝叶斯训练模型评分：" + str(accuracy_score(y_train, train_p)))
    print("多项式贝叶斯测试模型评分：" + str(accuracy_score(y_train, test_p)))
    end = time.time()
    print("运行时间：" + str(end - start))  # 时间单位是秒


Bernouli()
Multinomials()

# 对比伯努利和多项式分类器的在不同参数下的模型得分
result = pd.DataFrame(
    columns=["参数", "伯努利训练模型得分", "伯努利待测模型得分", "多项式训练模型得分", "多项式待测模型得分"])
for i in range(1, 300):
    Bernoulli = BernoulliNB(alpha=i).fit(X_train, y_train)
    Multinomial = MultinomialNB(alpha=i).fit(X_train, y_train)
    result = result._append([{"参数": i, "伯努利训练模型得分": accuracy_score(y_train, Bernoulli.predict(X_train)),
                             "伯努利待测模型得分": accuracy_score(y_test, Bernoulli.predict(X_test)),
                             "多项式训练模型得分": accuracy_score(y_train, Multinomial.predict(X_train)),
                             "多项式待测模型得分": accuracy_score(y_test, Multinomial.predict(X_test))}])
result.to_csv('../result/NaiveBayes.csv', index=False, sep=',')
# 画折线图
fig = plt.subplots(figsize=(15, 5))
plt.plot(result["参数"], result["伯努利训练模型得分"], label="伯努利训练模型得分")
plt.plot(result["参数"], result["伯努利待测模型得分"], label="伯努利待测模型得分")
plt.plot(result["参数"], result["多项式训练模型得分"], label="多项式训练模型得分")
plt.plot(result["参数"], result["多项式待测模型得分"], label="多项式待测模型得分")
plt.rcParams.update({'font.size': 15})
plt.legend()  # 创建图例
plt.xticks(fontsize=15)  # 设置坐标轴上的刻度字体大小
plt.yticks(fontsize=15)
plt.xlabel("参数", fontsize=15)  # 设置坐标轴上的标签内容和字体
plt.ylabel("得分", fontsize=15)
plt.savefig('../imgs/Bayes1.jpg')
plt.show()
